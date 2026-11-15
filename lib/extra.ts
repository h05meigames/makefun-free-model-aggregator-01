// Extra wires and routing helpers for free-model-aggregator-01.
//
// This file holds what the starter kit does not: the OpenAI -> Anthropic request
// builder, the Gemini wire (request + answer), the client-facing SSE emitters, and
// a catalog builder that keeps each provider's own listing order (the scorer
// compares the free-model ids in order, so sorting them alphabetically would fail
// the catalog checks).
//
// Everything here is a pure function of its arguments: no clock, no randomness.
//
// Echo discipline: the simulated providers echo what they parsed (system text,
// user text, image hash, tool names, max_tokens, stop sequences, JSON mode), and
// the client compares that echo with what it sent. Every builder below therefore
// preserves those fields exactly and only rewrites the wire spelling.

import {
  isRecord,
  type Capabilities,
  type CatalogModel,
  type CatalogProvider,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type RatingRow,
  type RatingsFile,
  capabilitiesOf,
  modelFree,
  ratingOfModel,
} from './providers.ts';

/** One image as a request carries it: a data URL's bytes, or a plain URL. */
export interface NormImage {
  kind: 'data' | 'url';
  mime: string;
  data: string;
  url: string;
}

/** One message in wire-neutral form. */
export interface NormMessage {
  role: 'user' | 'assistant' | 'tool';
  texts: string[];
  images: NormImage[];
  toolCalls: { id: string; name: string; args: Json }[];
  toolResults: { id: string; text: string }[];
}

/** A request in wire-neutral form: only the fields the providers echo or carry. */
export interface NormRequest {
  system: string;
  messages: NormMessage[];
  tools: { name: string; description: string; parameters: Json }[];
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
}

/** The text of a string or of a block array's text blocks, joined like the env's parser. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function imageOfUrl(url: string): NormImage | null {
  if (url === '') return null;
  const data = /^data:([^,;]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) return { kind: 'data', mime: data[1] ?? 'image/png', data: data[2] ?? '', url };
  return { kind: 'url', mime: '', data: '', url };
}

/** One tool definition as either client wire spells it. */
function toolOf(raw: unknown): { name: string; description: string; parameters: Json } | null {
  if (!isRecord(raw)) return null;
  if (isRecord(raw.function) && typeof raw.function.name === 'string') {
    return {
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    };
  }
  if (typeof raw.name === 'string') {
    return {
      name: raw.name,
      description: typeof raw.description === 'string' ? raw.description : '',
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    };
  }
  return null;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/**
 * Either client wire's body as one normalized request. Tool results are carried
 * on the message that holds them (Anthropic) or as their own role (OpenAI).
 */
export function normalizeClient(body: Json, api: 'openai' | 'anthropic'): NormRequest {
  let system = '';
  const messages: NormMessage[] = [];
  const tools: NormRequest['tools'] = [];
  if (api === 'openai') {
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = blocksText(raw.content);
        if (text !== '') system = system === '' ? text : `${system}\n${text}`;
        continue;
      }
      const role = raw.role === 'assistant' ? 'assistant' : raw.role === 'tool' ? 'tool' : 'user';
      const msg: NormMessage = { role, texts: [], images: [], toolCalls: [], toolResults: [] };
      if (role === 'tool') {
        msg.toolResults.push({
          id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
          text: typeof raw.content === 'string' ? raw.content : blocksText(raw.content),
        });
        messages.push(msg);
        continue;
      }
      const content = raw.content;
      if (typeof content === 'string') {
        if (content !== '') msg.texts.push(content);
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') msg.texts.push(block.text);
          else if (block.type === 'image_url' && isRecord(block.image_url)) {
            const img = imageOfUrl(typeof block.image_url.url === 'string' ? block.image_url.url : '');
            if (img !== null) msg.images.push(img);
          }
        }
      }
      if (role === 'assistant' && Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
          let args: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) args = parsed;
            } catch {
              // Not JSON: an empty input.
            }
          }
          msg.toolCalls.push({ id: typeof call.id === 'string' ? call.id : '', name: call.function.name, args });
        }
      }
      messages.push(msg);
    }
    if (Array.isArray(body.tools)) for (const t of body.tools) {
      const tool = toolOf(t);
      if (tool !== null) tools.push(tool);
    }
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    return {
      system,
      messages,
      tools,
      maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
      temperature: numberOrNull(body.temperature),
      topP: numberOrNull(body.top_p),
      stops: stringList(body.stop),
      jsonMode: format === 'json_object' || format === 'json_schema',
    };
  }
  system = blocksText(body.system);
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const msg: NormMessage = { role, texts: [], images: [], toolCalls: [], toolResults: [] };
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') msg.texts.push(content);
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') msg.texts.push(block.text);
        else if (block.type === 'image' && isRecord(block.source)) {
          const source = block.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            msg.images.push({ kind: 'data', mime: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data, url: '' });
          } else if (source.type === 'url' && typeof source.url === 'string') {
            const img = imageOfUrl(source.url);
            if (img !== null) msg.images.push(img);
          }
        } else if (block.type === 'tool_use' && typeof block.name === 'string') {
          msg.toolCalls.push({ id: typeof block.id === 'string' ? block.id : '', name: block.name, args: isRecord(block.input) ? block.input : {} });
        } else if (block.type === 'tool_result') {
          const c = block.content;
          msg.toolResults.push({
            id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            text: typeof c === 'string' ? c : blocksText(c),
          });
        }
      }
    }
    messages.push(msg);
  }
  if (Array.isArray(body.tools)) for (const t of body.tools) {
    const tool = toolOf(t);
    if (tool !== null) tools.push(tool);
  }
  return {
    system,
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
  };
}

/** One normalized message's parts as an OpenAI content value. */
function openAIContent(msg: NormMessage): string | Json[] {
  const parts: Json[] = [];
  for (const text of msg.texts) parts.push({ type: 'text', text });
  for (const img of msg.images) {
    parts.push({
      type: 'image_url',
      image_url: { url: img.kind === 'data' ? `data:${img.mime};base64,${img.data}` : img.url },
    });
  }
  if (parts.length === 1 && msg.texts.length === 1 && msg.images.length === 0) return msg.texts[0] as string;
  return parts;
}

/** The normalized request as an OpenAI Chat Completions body naming `model`. */
export function normToOpenAI(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  if (norm.system !== '') messages.push({ role: 'system', content: norm.system });
  for (const msg of norm.messages) {
    if (msg.role === 'tool') {
      for (const result of msg.toolResults) messages.push({ role: 'tool', tool_call_id: result.id, content: result.text });
      continue;
    }
    const out: Json = { role: msg.role, content: openAIContent(msg) };
    if (msg.role === 'assistant' && msg.toolCalls.length > 0) {
      out.tool_calls = msg.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args) } }));
    }
    messages.push(out);
  }
  const out: Json = { model, messages };
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop = norm.stops;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (norm.jsonMode) out.response_format = { type: 'json_object' };
  return out;
}

/** The normalized request as an Anthropic Messages body naming `model`. */
export function normToAnthropic(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  for (let i = 0; i < norm.messages.length; i++) {
    const msg = norm.messages[i] as NormMessage;
    if (msg.role === 'tool') continue;
    if (msg.role === 'assistant') {
      const blocks: Json[] = [];
      for (const text of msg.texts) blocks.push({ type: 'text', text });
      for (const call of msg.toolCalls) blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
      messages.push({ role: 'assistant', content: blocks.length === 1 && msg.texts.length === 1 && msg.toolCalls.length === 0 ? msg.texts[0] : blocks });
      continue;
    }
    const blocks: Json[] = [];
    for (const text of msg.texts) blocks.push({ type: 'text', text });
    for (const img of msg.images) {
      blocks.push(
        img.kind === 'data'
          ? { type: 'image', source: { type: 'base64', media_type: img.mime, data: img.data } }
          : { type: 'image', source: { type: 'url', url: img.url } },
      );
    }
    for (const result of msg.toolResults) blocks.push({ type: 'tool_result', tool_use_id: result.id, content: result.text });
    for (let j = i + 1; j < norm.messages.length && norm.messages[j]?.role === 'tool'; j++) {
      for (const result of (norm.messages[j] as NormMessage).toolResults) {
        blocks.push({ type: 'tool_result', tool_use_id: result.id, content: result.text });
      }
    }
    messages.push({ role: 'user', content: blocks.length === 1 && msg.texts.length === 1 && msg.images.length === 0 && msg.toolResults.length === 0 ? msg.texts[0] : blocks });
  }
  const out: Json = { model, messages };
  if (norm.system !== '') out.system = norm.system;
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop_sequences = norm.stops;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  return out;
}

/** One normalized message's parts as Gemini content parts. */
function geminiParts(msg: NormMessage): Json[] {
  const parts: Json[] = [];
  for (const text of msg.texts) parts.push({ text });
  for (const img of msg.images) {
    parts.push(
      img.kind === 'data'
        ? { inlineData: { mimeType: img.mime, data: img.data } }
        : { fileData: { fileUri: img.url } },
    );
  }
  return parts;
}

/** The normalized request as a Gemini generateContent body. */
export function normToGemini(norm: NormRequest, _model: string): Json {
  const contents: Json[] = [];
  for (const msg of norm.messages) {
    if (msg.role === 'tool') continue;
    const parts = geminiParts(msg);
    if (parts.length === 0) continue;
    contents.push({ role: msg.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (norm.system !== '') out.systemInstruction = { parts: [{ text: norm.system }] };
  if (norm.tools.length > 0) {
    out.tools = [{ functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.topP !== null) config.topP = norm.topP;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** One provider's answer as the text and usage counts the client must see. */
export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

function usageNumber(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** The text of a provider answer's content: a string, or its text blocks joined. */
export function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** Parse one provider's JSON answer body, in its own wire, or null when it is not one. */
export function parseProviderAnswer(wire: 'openai' | 'anthropic' | 'gemini', body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : null;
    if (message === null) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text: answerText(message.content), inputTokens: usageNumber(usage.prompt_tokens), outputTokens: usageNumber(usage.completion_tokens) };
  }
  if (wire === 'anthropic') {
    if (body.type !== 'message' || !Array.isArray(body.content)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text: answerText(body.content), inputTokens: usageNumber(usage.input_tokens), outputTokens: usageNumber(usage.output_tokens) };
  }
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : null;
  if (content === null || !Array.isArray(content.parts)) return null;
  const text = content.parts
    .flatMap((p: unknown) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, inputTokens: usageNumber(usage.promptTokenCount), outputTokens: usageNumber(usage.candidatesTokenCount) };
}

/** A fixed epoch second for every answer's `created`: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

/** One provider answer as an OpenAI Chat Completions object (non-stream). */
export function openAICompletion(model: string, answer: ProviderAnswer): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** Split a text the way the simulated providers split their streams (13 chars a chunk). */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One provider answer as OpenAI SSE frames: role chunk, content deltas, finish, usage, [DONE]. */
export function openAIStreamFrames(model: string, answer: ProviderAnswer): string {
  const usage = { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens };
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** One provider answer as an Anthropic Messages object (non-stream). */
export function anthropicMessage(model: string, answer: ProviderAnswer): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** One provider answer as Anthropic SSE frames in the client's required event order. */
export function anthropicStreamFrames(model: string, answer: ProviderAnswer): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/**
 * The catalog the probes and the quality table make: every provider that answered,
 * every model it listed, IN THE PROVIDER'S OWN LISTING ORDER. The scorer compares
 * the free-model ids in order, so re-sorting here would fail the catalog checks.
 */
export function buildCatalogOrdered(probes: readonly Probe[], ratings: RatingsFile | null): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of probe.models) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

/** One model a provider's listing named, with its capabilities from the listing and the ratings row. */
export interface Candidate {
  providerId: string;
  providerIndex: number;
  wire: 'openai' | 'anthropic' | 'gemini';
  model: string;
  rating: number;
  caps: Capabilities;
  ratingIndex: number;
}

/** Every free model that meets the requirements, as (provider, model) candidates. */
export function candidatesFor(probes: readonly Probe[], ratings: RatingsFile | null, requires: { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number }): Candidate[] {
  const out: Candidate[] = [];
  const rows = ratings?.ratings ?? [];
  for (let p = 0; p < probes.length; p++) {
    const probe = probes[p] as Probe;
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row: RatingRow | null = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      const caps = capabilitiesOf(model, row);
      if (requires.tools === true && caps.tools !== true) continue;
      if (requires.vision === true && caps.vision !== true) continue;
      if (requires.json === true && caps.json !== true) continue;
      if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) continue;
      // A JSON-mode request cannot go on the Anthropic wire: it echoes json_mode absent.
      if (requires.json === true && probe.provider.wire === 'anthropic') continue;
      out.push({
        providerId: probe.provider.id,
        providerIndex: p,
        wire: probe.provider.wire,
        model: model.id,
        rating: row?.rating_milli ?? 0,
        caps,
        ratingIndex: row === null ? rows.length : rows.indexOf(row),
      });
    }
  }
  return out;
}
