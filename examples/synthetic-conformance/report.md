# OpenAI-compatible API conformance

> **SYNTHETIC FIXTURE DATA: at least one endpoint is the in-process reference endpoint. Its results show how the suite reports, not how any real API behaves.**

Tool: llm-api-benchmark 0.1.0, Python 3.12.4, httpx 0.28.1.

## Endpoints

| Name | Base URL | Model | Tested (UTC) | Pass | Warn | Fail | Skip | Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | synthetic://local | reference-model | 2026-11-18T11:09:45+00:00 | 22 | 0 | 0 | 0 | 0 |
| quirky | synthetic://local | reference-model | 2026-11-18T11:09:45+00:00 | 11 | 5 | 6 | 0 | 0 |

## Results

| Test | What it checks | reference | quirky |
| --- | --- | --- | --- |
| `models.list` | GET /models lists models | PASS | PASS |
| `chat.basic` | A plain chat completion has the standard shape | PASS | PASS |
| `chat.usage` | usage has prompt, completion and total tokens | PASS | PASS |
| `stream.basic` | SSE chunks, finish_reason and [DONE] | PASS | PASS |
| `stream.usage` | stream_options.include_usage sends usage | PASS | FAIL |
| `params.max_tokens` | max_tokens limits the reply | PASS | FAIL |
| `params.max_completion_tokens` | max_completion_tokens limits the reply | PASS | FAIL |
| `params.stop` | stop ends the reply at the sequence | PASS | PASS |
| `params.n` | n=2 returns two choices | PASS | FAIL |
| `params.response_format.json_object` | JSON mode returns a bare JSON object | PASS | WARN |
| `params.response_format.json_schema` | Structured output follows a JSON schema | PASS | FAIL |
| `params.logprobs` | logprobs are returned | PASS | PASS |
| `params.temperature_range` | An out-of-range temperature is rejected | PASS | PASS |
| `tools.call` | A forced tool call has the standard shape | PASS | PASS |
| `tools.stream` | Streamed tool-call deltas rebuild a valid call | PASS | PASS |
| `tools.result` | A tool result message is accepted and used | PASS | PASS |
| `tools.choice_none` | tool_choice "none" prevents calls | PASS | PASS |
| `errors.invalid_key` | An invalid key gets 401 with an error body | PASS | WARN |
| `errors.missing_key` | A missing key gets 401 with an error body | PASS | WARN |
| `errors.unknown_model` | An unknown model is an error, not a substitute | PASS | FAIL |
| `errors.invalid_request` | An empty messages list gets 400 or 422 | PASS | WARN |
| `errors.malformed_json` | A malformed JSON body gets 400, not a 5xx | PASS | WARN |

PASS: behaves like the OpenAI API. WARN: works, with a difference worth knowing (for example an explicit 4xx for an unsupported parameter, or a non-OpenAI error body). FAIL: a difference that breaks clients, such as a parameter that is accepted and silently ignored. skip: not applicable. ERROR: could not be judged (network, rate limit, 5xx).

## reference

| Test | Result | What happened | Notes |
| --- | --- | --- | --- |
| `models.list` | PASS | 1 model(s) listed |  |
| `chat.basic` | PASS | answered 'ready' |  |
| `chat.usage` | PASS | prompt 14, completion 1, total 15 |  |
| `stream.basic` | PASS | 3 chunks, answered 'ready' |  |
| `stream.usage` | PASS | final usage chunk: prompt 9, completion 1, total 10 |  |
| `params.max_tokens` | PASS | 8 completion tokens (limit 8) |  |
| `params.max_completion_tokens` | PASS | 8 completion tokens (limit 8) |  |
| `params.stop` | PASS | stopped before '7' ('1 2 3 4 5 6') |  |
| `params.n` | PASS | two choices returned |  |
| `params.response_format.json_object` | PASS | the reply is a bare JSON object |  |
| `params.response_format.json_schema` | PASS | the reply matches the schema |  |
| `params.logprobs` | PASS | logprobs for 1 token(s) |  |
| `params.temperature_range` | PASS | rejected with HTTP 400 (temperature must be between 0 and 2.) |  |
| `tools.call` | PASS | get_weather({"city": "Paris", "unit": "celsius"}) |  |
| `tools.stream` | PASS | get_weather({"city": "Paris", "unit": "celsius"}) rebuilt from deltas | arguments arrived in 5 fragments |
| `tools.result` | PASS | answered from the tool result ('It is 18 degrees in Paris right now.') |  |
| `tools.choice_none` | PASS | answered in text |  |
| `errors.invalid_key` | PASS | HTTP 401 with an OpenAI error body (Incorrect API key provided.) |  |
| `errors.missing_key` | PASS | HTTP 401 with an OpenAI error body (Incorrect API key provided.) |  |
| `errors.unknown_model` | PASS | HTTP 404 with an OpenAI error body (The model `conformance-check-unknown-model` does not exist.) |  |
| `errors.invalid_request` | PASS | HTTP 400 with an OpenAI error body (messages must be a non-empty array.) |  |
| `errors.malformed_json` | PASS | HTTP 400 with an OpenAI error body (We could not parse the JSON body of your request.) |  |

## quirky

| Test | Result | What happened | Notes |
| --- | --- | --- | --- |
| `models.list` | PASS | 1 model(s) listed |  |
| `chat.basic` | PASS | answered 'ready' |  |
| `chat.usage` | PASS | prompt 14, completion 1, total 15 |  |
| `stream.basic` | PASS | 3 chunks, answered 'ready' |  |
| `stream.usage` | FAIL | include_usage was requested but no chunk carries usage |  |
| `params.max_tokens` | FAIL | ignored: 60 completion tokens with max_tokens=8 | finish_reason 'stop' |
| `params.max_completion_tokens` | FAIL | ignored: 60 completion tokens with max_completion_tokens=8 | finish_reason 'stop' |
| `params.stop` | PASS | stopped before '7' ('1 2 3 4 5 6') |  |
| `params.n` | FAIL | ignored: 1 choice(s) returned for n=2 |  |
| `params.response_format.json_object` | WARN | JSON only inside a code fence: JSON mode was likely not applied | The prompt asks for JSON, so a reply can look right without JSON mode. |
| `params.response_format.json_schema` | FAIL | ignored: the reply is not JSON ('A typical house cat is small, curious and fond of ') | The prompt does not mention JSON, so only the schema could make the reply JSON. |
| `params.logprobs` | PASS | logprobs for 1 token(s) |  |
| `params.temperature_range` | PASS | rejected with HTTP 400 (temperature must be between 0 and 2.) |  |
| `tools.call` | PASS | get_weather({"city": "Paris", "unit": "celsius"}) |  |
| `tools.stream` | PASS | get_weather({"city": "Paris", "unit": "celsius"}) rebuilt from deltas | arguments arrived in 5 fragments |
| `tools.result` | PASS | answered from the tool result ('It is 18 degrees in Paris right now.') |  |
| `tools.choice_none` | PASS | answered in text |  |
| `errors.invalid_key` | WARN | HTTP 401, but the body is other-json: Incorrect API key provided. | OpenAI clients read {"error": {"message", "type", "code"}}; other shapes lose detail |
| `errors.missing_key` | WARN | HTTP 401, but the body is other-json: Incorrect API key provided. | OpenAI clients read {"error": {"message", "type", "code"}}; other shapes lose detail |
| `errors.unknown_model` | FAIL | HTTP 200: an unknown model was served (reply says model 'reference-model') | A typo in a model id should fail, not run on another model. |
| `errors.invalid_request` | WARN | HTTP 400, but the body is other-json: messages must be a non-empty array. | OpenAI clients read {"error": {"message", "type", "code"}}; other shapes lose detail |
| `errors.malformed_json` | WARN | HTTP 400, but the body is other-json: We could not parse the JSON body of your request. | OpenAI clients read {"error": {"message", "type", "code"}}; other shapes lose detail |

## Limitations

- Each test is one request at one moment. Parameter tests judge the model's reply to a short prompt, so an unusual reply can look like a server problem; rerun a FAIL before reporting it.
- The suite checks the wire format and parameter handling, not answer quality or speed.
- A result describes one endpoint and one model. Providers change servers and models without notice; date every published result.

---

Generated by [llm-api-benchmark](https://github.com/Moloch-AI/llm-api-benchmark) (`llm-bench conformance`). Test definitions: [docs/conformance.md](https://github.com/Moloch-AI/llm-api-benchmark/blob/main/docs/conformance.md).
