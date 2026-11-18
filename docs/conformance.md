# OpenAI-compatible API test suite (`llm-bench conformance`)

`llm-bench conformance` checks how closely an endpoint follows the OpenAI Chat Completions API.
It sends about 20 small requests to each endpoint in a configuration file. It reports which
behaviors match, which differ, and which parameters the endpoint accepts and then ignores.

Use it to check a provider before you switch to it, to test a gateway or proxy you run, or as a
CI gate for your own server.

```bash
llm-bench conformance --list                                  # the tests
llm-bench conformance -c synthetic-conformance.yaml           # offline demo, no key
llm-bench conformance -c my-endpoint.yaml --dry-run           # what would be sent
llm-bench conformance -c my-endpoint.yaml --formats json,md,junit
```

The suite uses the same configuration files as `llm-bench run`, and only reads their `providers`.
Any endpoint that works with `run` works here: a preset, `base_url`, `model`, `api_key_env`,
headers, query parameters, `params` and `rate_limit_rpm`.

## Outcomes

| Outcome | Meaning |
|---|---|
| **PASS** | The endpoint behaved as the OpenAI API does. |
| **WARN** | It works, with a difference worth knowing. Examples: an unsupported parameter rejected with a clear 400; an error body that is not `{"error": {...}}`; an inconclusive reply. |
| **FAIL** | A difference that breaks clients: a parameter accepted and silently ignored, a missing field, a malformed stream, an unknown model served anyway, a 5xx for a client mistake. |
| **skip** | Not applicable, for example the auth tests on an endpoint without auth. |
| **ERROR** | Could not be judged: network error, rate limit, a 5xx, or the run stopped. |

Rejecting a parameter is a WARN, not a FAIL. A client that sends `n=2` and gets a 400 knows it
did not get two choices. A client that gets HTTP 200 with one choice does not.

## Tests

| Test | Request | PASS when | Typical FAIL |
|---|---|---|---|
| `models.list` | `GET /models` | 200, `data[]` of objects with string `id`. WARN if `object` fields are missing or the configured model is not listed. | No list |
| `chat.basic` | A system and a user message | 200; `choices[0].message.role` is `assistant` with non-empty `content`; `finish_reason` and `id` are present. WARN if `object`, `created` or `model` is missing. | Missing fields |
| `chat.usage` | none (reads `chat.basic`) | `usage` has integer `prompt_tokens`, `completion_tokens` and `total_tokens`. WARN if the total is not their sum. | No usage |
| `stream.basic` | `stream: true` | `text/event-stream`, JSON chunks, a content delta, a `finish_reason`, and `data: [DONE]` at the end | No `[DONE]`, non-JSON events, an error event |
| `stream.usage` | `stream_options.include_usage` | A final chunk with `usage` and an empty `choices` list. WARN if usage rides on a chunk that has choices. | No usage |
| `params.max_tokens` | Count to 60 with `max_tokens: 8` | At most 9 completion tokens. WARN without `finish_reason: length`. | Ignored |
| `params.max_completion_tokens` | The same with `max_completion_tokens` | As above | Ignored |
| `params.stop` | Numbers 1 to 12 with `stop: ["7"]` | The reply has 6 and nothing from 7 on | Reply runs past 7 |
| `params.n` | `n: 2` | Two choices with indexes 0 and 1 | One choice |
| `params.response_format.json_object` | Asks for JSON, with JSON mode | The reply is a bare JSON object. WARN if the JSON is only inside a code fence: the mode was likely not applied. | Prose |
| `params.response_format.json_schema` | A prompt that does not mention JSON, with a strict schema | The reply is JSON that validates against the schema | Prose, or JSON that breaks the schema |
| `params.logprobs` | `logprobs: true, top_logprobs: 1` | `choices[0].logprobs.content[]` with `token` and `logprob` | Missing |
| `params.temperature_range` | `temperature: 3.5` | Rejected with 400 or 422. A 200 is a WARN: the value was clamped or dropped. | 5xx |
| `tools.call` | One tool, forced with `tool_choice` | `tool_calls[0]` has an `id`, `type: function`, the right name, and `arguments` as a JSON string. Arguments that break the schema only add a note, since that is the model, not the format. | No call, or `arguments` that is an object or not JSON |
| `tools.stream` | The same, streamed | `delta.tool_calls` with an integer `index` rebuild a valid call. The note says whether the arguments came in one piece or in fragments. | No deltas, no `[DONE]` |
| `tools.result` | An assistant tool call and a `tool` message with the result | A text answer that uses the result's value. WARN if the answer ignores the value, or the model calls the tool again. | Error, or an empty reply |
| `tools.choice_none` | `tool_choice: "none"` | No tool call | A tool call |
| `errors.invalid_key` | A made-up key | 401 with `{"error": {"message": ...}}`. Another JSON body is a WARN. | 200 |
| `errors.missing_key` | No key | As above | 200 |
| `errors.unknown_model` | `model: conformance-check-unknown-model` | 404, 400 or 422 with an OpenAI error body | 200: another model answered |
| `errors.invalid_request` | `messages: []` | 400 or 422 with an OpenAI error body | 200 |
| `errors.malformed_json` | A truncated JSON body | 400 or 422 | 5xx |

The prompts are short and harmless. Each one bounds its own reply even where `max_tokens` is
ignored: counting to 60, a one-word answer, one sentence. A run of all tests sends 21 requests
and costs a few hundred tokens. The error tests are rejected before any model runs.

## Request handling

- Requests are spaced to the endpoint's `rate_limit_rpm`.
- A 429 with `Retry-After` of 60 seconds or less is waited out and retried once. Any other 429
  is an ERROR.
- A 401, 402 or 403 with the configured key stops that endpoint's suite. The remaining tests are
  marked ERROR, so a bad key or an empty account costs one request.
- Keys are read from environment variables and redacted from every output. The invalid-key test
  uses the fixed string `invalid-key-for-a-conformance-check`.
- More than 20 real requests ask for confirmation, as in `run`; `--yes` skips it.

## Output

- **Terminal:** a results table. With several endpoints it is a matrix, followed by every
  non-PASS result with its reason.
- **JSON** (`schema: llm-api-benchmark/conformance/v1`): every test's outcome, summary, notes,
  HTTP status, latency and request count, the endpoint's public configuration, and the response
  headers seen (`x-request-id`, `x-ratelimit-*`, `retry-after`, `x-should-retry`).
- **Markdown:** a compatibility matrix and one table of results per endpoint.
- **JUnit XML:** one test suite per endpoint. FAIL becomes a failure, ERROR an error, and skip
  skipped. WARN passes, with a note in `system-out`.

`llm-bench report FILE --format table|markdown|junit|json` renders a saved result again.

The exit code is 1 when any test FAILs, which makes the suite usable as a CI gate:

- `--fail-on warn` also fails on WARN;
- `--fail-on never` always exits 0.

## The offline reference endpoint

With `preset: synthetic`, the suite runs against a reference endpoint inside the process.

- **Without quirks** it behaves as the OpenAI API documents, and passes every test.
- **With `synthetic.quirks`** it misbehaves on purpose, one way per quirk. Each quirk is a
  documented way real endpoints differ, so every kind of failure can be seen offline.

| Quirk | Effect |
|---|---|
| `models_missing` | `GET /models` answers 404 |
| `bare_errors` | Error bodies are `{"message": ...}` |
| `accept_any_key` | Any key, or none, is accepted |
| `substitute_model` | An unknown model id is served by the configured model |
| `accept_empty_messages` | An empty `messages` list gets a reply |
| `server_error_on_bad_json` | A malformed body gets HTTP 500 |
| `accept_out_of_range` | A temperature outside 0 to 2 is accepted |
| `ignore_max_tokens` | `max_tokens` and `max_completion_tokens` are ignored |
| `ignore_stop`, `ignore_n`, `ignore_response_format`, `ignore_logprobs` | That parameter is ignored |
| `ignore_tool_choice` | `tool_choice: "none"` is ignored |
| `tool_args_object` | Tool-call `arguments` is sent as an object instead of a JSON string |
| `no_stream_usage` | Streams never carry usage |
| `no_done` | Streams end without `data: [DONE]` |

The reference endpoint runs no model. It builds replies from the prompt with fixed rules, enough
for these tests. Its results are synthetic fixture data, and every report says so.

## Reading and publishing results

- A result describes one endpoint, one model and one moment. Providers change servers and
  models without notice. Date every result you publish and keep the JSON with it.
- Parameter tests judge a model's reply to a short prompt. An unusual reply can look like a
  server problem, for example a model that will not count. Rerun a FAIL before you report it.
- A FAIL on a parameter you never send does not matter to you. A WARN on error bodies matters if
  your client reads `error.message` or `error.code`.
- The suite does not measure speed or answer quality. Use `llm-bench run` for speed. For tool
  calls on your own tools, use a dataset with `tools` (see
  [configuration.md](configuration.md#tool-workloads)).
- Compare providers by running them from one config file, so they are tested at the same time.
