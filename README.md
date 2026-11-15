# free-model-aggregator-01

One long-running program that serves every request of the trial with the best
free model it can get. Run as `node main.ts` from this repository's root, with
Node 22's standard library alone (TypeScript runs directly).

## How it works

1. Reads the brief at `$MAKEFUN_ENV_URL`'s sibling path `$MAKEFUN_PROMPT_FILE`:
   the provider directory, the quality table (ratings + capabilities), and the
   workload manifest.
2. Probes each provider's catalog route in the provider's own wire and builds
   one catalog: every provider that answered, every model it listed, **in the
   provider's own listing order** (the scorer compares the free-model ids in
   order).
3. Listens on `0.0.0.0` at an OS-assigned port, registers with the env
   (`POST /register {protocol_version: 1, port, catalog}`), then serves:
   - `GET /v1/models` — the deduplicated free model ids, OpenAI shape for
     `Authorization: Bearer`, Anthropic shape for `x-api-key`.
   - `POST /v1/chat/completions` and `POST /v1/messages` — in the wire the
     client used, JSON or SSE as asked.
4. Polls `GET /work` and exits 0 when the client is done.

## Routing

- **Quality:** every free candidate meeting the request's requirements is ranked
  by `rating_milli`, best first; ties go to the provider with the fewest calls
  spent so far, then the brief's provider order. A JSON-mode request never goes
  on the Anthropic wire, which cannot carry JSON mode.
- **Failover:** a provider that refuses (402, 429, 5xx) is counted and the
  request is answered by the next candidate. A refusal that says the budget is
  spent, or that the provider is out, retires that provider for the trial.
- **Wires:** every provider call is made non-stream and the answer is buffered
  (text + usage counts). The client's stream is then emitted from the buffered
  answer, in the exact event order the client checks. Request translation
  preserves the fields the providers echo (system text, user text, images,
  tool names, max_tokens, stop sequences, JSON mode) exactly.

## Files

- `main.ts` — the server, the routing, the register and the poll.
- `lib/providers.ts` — the brief's three documents and the catalog probe
  (from the kit's starter).
- `lib/wire.ts` — the OpenAI <-> Anthropic translations and model listings
  (from the kit's starter).
- `lib/extra.ts` — the Gemini wire, the OpenAI -> Anthropic request builder,
  the client-facing SSE emitters, and the order-preserving catalog builder.
- `LICENSE` — MIT.

## Local grading

```sh
node cli/makefun.mjs grade-local --kit <kit-dir> --bot . --driver process --worktree --instances 3
```
