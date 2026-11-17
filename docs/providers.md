# Providers

`llm-bench` talks to one wire format: OpenAI Chat Completions
(`POST {base_url}/chat/completions`, with server-sent events for streaming). Any endpoint that
implements it can be benchmarked, whether a hosted API, a gateway, or a local server.

## Built-in presets

`llm-bench providers` lists them with their notes.

| Preset | Base URL | Key variable | Checked how |
|---|---|---|---|
| `openai-compatible` | you set `base_url` | you set `api_key_env` | CI runs it against in-process fixtures, a real local HTTP server and the MolochAI mock server. Each provider's details (URL, model ids, limits) are yours to check. |
| `molochai` | `https://api.molochai.dev/api/v1`, or `MOLOCHAI_BASE_URL` | `MOLOCHAI_API_KEY` | Against the public docs, against the official mock server in CI (reasoning, keep-alive and queue comments, usage chunk, in-stream errors, error shapes), and with an unauthenticated request to the live API on 2026-11-17. A live run with a key is a manual workflow (`.github/workflows/live.yml`). |
| `synthetic` | in-process | none | Not a provider. It simulates an endpoint for demos and tests, and every report marks its numbers as synthetic. |

A preset only lists defaults. Every field can be overridden per endpoint
([configuration.md](configuration.md#endpoint-settings)).

### MolochAI notes

These behaviors come from the MolochAI documentation
([API reference](https://molochai.dev/docs/api-reference?utm_source=github&utm_medium=oss&utm_campaign=llm-api-benchmark),
[streaming](https://molochai.dev/docs/streaming?utm_source=github&utm_medium=oss&utm_campaign=llm-api-benchmark)).
The preset accounts for them:

- `max_tokens` is accepted but has no effect. Each reply is capped by the plan's output limit, so
  the preset sets `max_tokens_honored: false` and assumes up to 9,000 output tokens per reply for
  worst-case estimates. That is the documented cap on most plans; set
  `assumed_max_output_tokens` to your plan's cap.
- `stream_options` is dropped, but a usage chunk is always sent at the end of a stream. The
  preset sets `stream_usage: false`.
- Reasoning models send `delta.reasoning_content` before the answer, so TTFT and TTFC differ.
  Keep-alive comment lines (`: ka`, `: hb`) and queue comments (`: queued ...`) are counted in
  `comment_lines`.
- Errors after a stream has opened arrive as an `{"error": ...}` event and are recorded as
  `stream_error:<type>`. A quota 429 carries `X-Should-Retry: false`, which stops the endpoint.
- Rate limit: 60 requests per minute per account. The preset spaces requests accordingly.
- Plain (non-streamed) requests pass through Cloudflare, which ends them after about 100 seconds
  with a 524. Prefer `modes: [stream]` for long workloads.
- Model ids: `llm-bench models --preset molochai` (one `GET /models` request; it needs a plan or
  credit).
- Pricing is not in the preset. Copy it from
  [molochai.dev/pricing](https://molochai.dev/pricing?utm_source=github&utm_medium=oss&utm_campaign=llm-api-benchmark)
  into `pricing` if you want cost tracking or a budget.

## Any other OpenAI-compatible endpoint

```yaml
providers:
  - name: my-provider
    preset: openai-compatible
    base_url: https://api.example.com/v1     # the part before /chat/completions
    api_key_env: MY_PROVIDER_API_KEY
    model: ${MY_MODEL}                       # or the literal id
    rate_limit_rpm: 60                       # the provider's published limit
```

Before you run it:

1. Find the base URL and model ids in the provider's documentation, or run
   `llm-bench models --base-url <url> --api-key-env <VAR>`.
2. Run `llm-bench run --config file.yaml --dry-run` to check the plan and that the key is set.
3. Start with `--repetitions 1 --workloads short-answer` and look at the per-request lines. A
   400 or 422 usually means a parameter the provider does not accept (see below).

Common differences between "OpenAI-compatible" APIs, and the setting that handles each:

| Symptom | Setting |
|---|---|
| `max_tokens` is rejected and `max_completion_tokens` is expected (newer OpenAI models) | `max_tokens_param: max_completion_tokens` |
| `stream_options` is rejected | `stream_usage: false` (token counts may then be estimated) |
| `temperature` is rejected for a model | `params: {temperature: null}` |
| The key goes in a header other than `Authorization: Bearer` | `auth_scheme: header`, `auth_header: api-key` |
| An `api-version` query parameter is required | `query: {api-version: "..."}` |
| Local server without auth | `auth_scheme: none` |
| The output limit is ignored | `max_tokens_honored: false`, `assumed_max_output_tokens: N` |

### Examples (not verified by the maintainers)

These follow each project's documentation. The maintainers have not run them, so check the
current docs before you rely on them.

```yaml
  # OpenAI (benchmarks/openai-compatible.yaml)
  - name: openai
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    model: ${OPENAI_MODEL}
    max_tokens_param: max_completion_tokens

  # A local Ollama server (its OpenAI-compatible endpoint)
  - name: ollama-local
    base_url: http://localhost:11434/v1
    auth_scheme: none
    model: ${OLLAMA_MODEL}

  # A local vLLM server (vllm serve <model>)
  - name: vllm-local
    base_url: http://localhost:8000/v1
    auth_scheme: none
    model: ${VLLM_MODEL}
```

If you verify an endpoint, open a pull request that adds its configuration to `benchmarks/`
with the date and how you checked it. Leave measurements out of the pull request.

## Adding a different wire format

Adapters live in `src/llm_api_benchmark/providers/`. An adapter subclasses `base.Adapter` and
implements four methods:

- `build_request(messages, stream, max_tokens, temperature)` returns the URL, headers, query and
  body.
- `parse_chunk(event)` returns the text, reasoning, finish reason, usage, model or error carried
  by one streamed event.
- `parse_completion(body)` does the same for a plain response.
- `parse_error(status, body, content_type)` returns the error type and message.

Register the adapter in `providers/__init__.py:ADAPTERS` and add a preset with `api: <name>`.
Timing, SSE parsing, the outcome classification and every report are shared. An adapter only
translates the format. Add fixture tests in the style of `tests/test_openai_chat.py`.
