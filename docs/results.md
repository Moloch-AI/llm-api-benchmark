# Result files

`llm-bench run` writes `<name>-<UTC timestamp>.json`, and with `--formats`, also:

- `.requests.csv`: one row per request.
- `.summary.csv`: one row per endpoint × workload × mode.
- `.md`: the Markdown report.
- `.html`: a self-contained HTML report with inline SVG charts. It loads nothing from the network.

`llm-bench report <file.json> --format table|markdown|html|csv|requests-csv|json` re-renders a
results file. Statistics are recomputed from the stored requests, so reports always match the raw
data.

Result files contain no credentials. Endpoints are listed by `api_key_env` name only; user and
password parts of URLs, key-like query values and anything that looks like a bearer token are
redacted. Error messages pass through the same filter, in case a provider echoes a key back.

## JSON layout (`schema: llm-api-benchmark/results/v1`)

| Field | Content |
|---|---|
| `tool` | Name and version of the tool. |
| `run` | `name`, `started_at`, `finished_at` (UTC), `duration_s`, `interrupted`, `stop_reason`, `synthetic`, `planned_requests`, `spent`. |
| `environment` | Python version and implementation, platform, httpx version. |
| `config` | The run settings that were in effect, after command-line overrides. |
| `dataset` | `id`, `version`, `sha256`, `source`, and per workload `max_tokens`, prompt count and description. |
| `providers` | Each endpoint: preset, base URL, model, key variable name, header names, params, pricing with source and date, rate limit, preset notes. |
| `warnings` | For example, stale pricing. |
| `limitations` | Plain statements about what the run can and cannot show (also printed in the reports). |
| `summary` | Per-group statistics (the same as `.summary.csv`, with full distributions). Includes `check_types` (passed and run per check type) and, for tool workloads, `tool_call_rate`, `tool_agreement` and `tool_agreement_prompts`. |
| `provider_totals` | Per-endpoint reliability across all workloads, warm-ups included. |
| `requests` | Every planned request, in plan order (fields below). |

## Request fields

| Field | Meaning |
|---|---|
| `index`, `provider`, `model`, `workload`, `prompt_id`, `mode`, `repetition`, `warmup` | Which planned request this is. |
| `started_at` | UTC time the request was sent. |
| `status`, `error_kind`, `http_status`, `error_type`, `error_message`, `skip_reason` | Outcome ([methodology](methodology.md#outcomes)). |
| `headers_s`, `ttfb_s`, `ttft_s`, `ttfc_s`, `last_token_s`, `latency_s` | Seconds from sending: headers, first body line, first text, first answer text, last text chunk, end. |
| `output_tps`, `decode_tps` | Derived throughput ([definitions](methodology.md#what-is-measured)). |
| `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `usage_reported`, `tokens_estimated` | Token counts and where they came from. |
| `output_chars`, `reasoning_chars`, `text_chunks`, `comment_lines` | Size and shape of the stream. |
| `finish_reason`, `response_model`, `done_marker` | As reported by the provider. |
| `retry_after_s`, `should_retry` | `Retry-After` and `X-Should-Retry` headers on errors. |
| `cost` | With pricing, from the reported (or estimated) tokens. |
| `checks`, `checks_passed` | Check results, when the prompt has checks. |
| `tool_call_names`, `tool_signature` | Tool workloads: the functions called (`[]` = none) and a hash of the calls with parsed arguments ([methodology](methodology.md#tool-calls)). |
| `output` | Reply text, only with `save_outputs: true`. |
| `tool_calls` | Each call's `name` and `arguments` text, only with `save_outputs: true`. |
