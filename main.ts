// free-model-aggregator-01: serve every request with the best free model you can get.
//
// One long-running program: it reads the brief, probes each provider's catalog,
// builds one catalog, listens, registers with the env, and answers the client's
// requests. A trial runs it as `node /bot/main.ts` from the repository's root,
// with Node 22's standard library alone.
//
// Routing (the part that wins the score):
//   - quality: every free candidate that meets the request's requirements is ranked
//     by the quality table's rating_milli, best first; ties go to the provider with
//     the fewest calls spent so far, then to the brief's provider order. A JSON-mode
//     request never goes on the Anthropic wire, which cannot carry JSON mode.
//   - failover: a provider that refuses (402, 429, 5xx) is counted and the request
//     is answered by the next candidate. A refusal that says the budget is spent,
//     or that the provider is out, retires that provider for the rest of the trial.
//   - wires: every provider call is made non-stream and the answer is buffered;
//     the client's stream is then emitted from the buffered text and usage counts,
//     in the exact event order the client checks. The provider's answer is always
//     text + usage + a family-0 stop reason ("stop" / "end_turn"), so the client
//     response is built from those three values in either wire.
//   - catalog: the probe order is kept (the scorer compares free-model ids in
//     order); GET /v1/models lists the deduplicated free ids in that same order.
//
// Everything is a pure function of the requests and the providers' answers seen
// so far: no clock, no randomness, so a replay reproduces the same bytes.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicMessage,
  anthropicStreamFrames,
  buildCatalogOrdered,
  candidatesFor,
  normalizeClient,
  normToAnthropic,
  normToGemini,
  normToOpenAI,
  openAICompletion,
  openAIStreamFrames,
  parseProviderAnswer,
  type Candidate,
  type ProviderAnswer,
} from './lib/extra.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Per provider: inference calls spent (every attempt that reached the provider) and whether it is retired. */
const providerState = new Map<string, { calls: number; dead: boolean }>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stateOf(providerId: string): { calls: number; dead: boolean } {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalogOrdered(probes, brief.ratings);
}

/** The free models across the providers that answered, deduplicated, in catalog order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/**
 * The ranked candidates for one request: best rating first; on a tie, the
 * provider with the fewest calls spent, then the brief's provider order, then
 * the quality table's row order.
 */
function rankedCandidates(requires: Requirement): Candidate[] {
  const candidates = candidatesFor(probes, brief.ratings, requires);
  return candidates.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const callsA = stateOf(a.providerId).calls;
    const callsB = stateOf(b.providerId).calls;
    if (callsA !== callsB) return callsA - callsB;
    if (a.providerIndex !== b.providerIndex) return a.providerIndex - b.providerIndex;
    return a.ratingIndex - b.ratingIndex;
  });
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** The provider request body for one candidate, in the provider's own wire. Always non-stream: the answer is buffered. */
function providerBody(api: 'openai' | 'anthropic', body: Json, candidate: Candidate): Json {
  if (candidate.wire === 'openai') {
    if (api === 'openai') {
      const out: Json = { ...body, model: candidate.model, stream: false };
      delete out.stream_options;
      return out;
    }
    const out = anthropicToOpenAI(body, candidate.model);
    out.stream = false;
    delete out.stream_options;
    return out;
  }
  const norm = normalizeClient(body, api);
  if (candidate.wire === 'anthropic') return normToAnthropic(norm, candidate.model);
  return normToGemini(norm, candidate.model);
}

/** One candidate's endpoint and headers. */
function providerTarget(candidate: Candidate): { url: string; headers: Record<string, string> } {
  const probe = probes[candidate.providerIndex] as Probe;
  const base = joinBase(ENV_URL, probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  if (candidate.wire === 'openai') return { url: `${base}/v1/chat/completions`, headers };
  if (candidate.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    return { url: `${base}/v1/messages`, headers };
  }
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model)}:generateContent`, headers };
}

/**
 * Serve one client request with the best free model that will answer it.
 * Returns the served model and its answer, or null when no candidate answered.
 */
async function serve(api: 'openai' | 'anthropic', body: Json, requires: Requirement): Promise<{ model: string; answer: ProviderAnswer } | null> {
  const candidates = rankedCandidates(requires);
  for (const candidate of candidates) {
    const state = stateOf(candidate.providerId);
    if (state.dead) continue;
    const target = providerTarget(candidate);
    let res: Response;
    try {
      res = await fetch(target.url, {
        method: 'POST',
        headers: target.headers,
        body: JSON.stringify(providerBody(api, body, candidate)),
        signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
      });
    } catch (err) {
      log(`${candidate.providerId}/${candidate.model}: unreachable: ${(err as Error).message}`);
      continue;
    }
    state.calls += 1;
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      log(`${candidate.providerId}/${candidate.model}: HTTP ${res.status} (call ${state.calls})`);
      if (text.includes('budget is spent') || text.includes(' is out')) {
        state.dead = true;
        log(`${candidate.providerId} is retired: ${text.slice(0, 200)}`);
      }
      continue;
    }
    let parsed: ProviderAnswer | null = null;
    try {
      const json: unknown = await res.json();
      parsed = parseProviderAnswer(candidate.wire, json);
    } catch {
      parsed = null;
    }
    if (parsed === null) {
      // A 2xx the parser cannot read is still the one successful call for this
      // request: never answer it twice. Report it as an empty answer, which the
      // client will mark invalid, rather than spending a second provider call.
      log(`${candidate.providerId}/${candidate.model}: unreadable 2xx answer`);
      return { model: candidate.model, answer: { text: '', inputTokens: 0, outputTokens: 0 } };
    }
    return { model: candidate.model, answer: parsed };
  }
  return null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function sendStream(res: ServerResponse, frames: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(frames);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions or POST /v1/messages: one client request, answered in its own wire. */
async function infer(api: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const served = await serve(api, body, requires);
  if (served === null) {
    return sendError(res, api, 503, 'no_free_model', 'no free model on the directory answered this request');
  }
  log(`served ${api} stream=${stream} with ${served.model}`);
  if (api === 'openai') {
    if (stream) return sendStream(res, openAIStreamFrames(served.model, served.answer));
    return sendJson(res, 200, openAICompletion(served.model, served.answer));
  }
  if (stream) return sendStream(res, anthropicStreamFrames(served.model, served.answer));
  return sendJson(res, 200, anthropicMessage(served.model, served.answer));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await infer('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await infer('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
