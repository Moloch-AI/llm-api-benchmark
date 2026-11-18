# Changelog

## 0.1.0 (unreleased)

First version.

- `llm-bench` commands: `run`, `providers`, `datasets`, `models`, `report`, `compare` and `init`.
- Endpoints: OpenAI Chat Completions, streamed and plain. Presets for `openai-compatible`,
  `molochai` and an in-process `synthetic` endpoint.
- Metrics per request:
  - timing: headers, TTFT, TTFC and latency;
  - throughput: output and decode tokens/s;
  - tokens, from usage or estimated and labeled;
  - outcome, by type;
  - cost from user-supplied prices;
  - deterministic checks.
- Distributions per endpoint, workload and mode: min, P25, median, P75, P90, P95, max and mean,
  with sample sizes.
- Bounds: `max_requests`, an optional budget with worst-case reservation, `--dry-run`,
  confirmation above 20 requests, concurrency 1 by default (at most 8), `rate_limit_rpm`
  pacing, `Retry-After` handling, and stopping an endpoint after permanent errors.
- Built-in dataset `core-v1`: 8 workloads × 3 prompts, with checks.
- Tool calling:
  - the `tools-v1` dataset: 4 workloads × 3 prompts for picking a function, typed arguments,
    several calls in one reply, and `tool_choice`;
  - `tools` and `tool_choice` in your own datasets;
  - checks `tool_call`, `no_tool_call`, `tool_args_valid` (a built-in JSON Schema subset) and
    `tool_call_count`;
  - streamed calls rebuilt from fragments;
  - "Same calls", a figure for agreement across repetitions;
  - per-check pass counts in the reports.
- Templates `synthetic-tools` and `molochai-tools`, and `run --dataset`.
- `llm-bench conformance`: an OpenAI Chat Completions conformance suite.
  - 22 tests: model listing, the plain and streamed shape, usage, `max_tokens`,
    `max_completion_tokens`, `stop`, `n`, JSON mode and JSON schema, `logprobs`, temperature
    validation, tool calls (plain, streamed, tool results and `tool_choice: "none"`), and error
    bodies.
  - Outcomes PASS, WARN, FAIL, skip and ERROR. A silently ignored parameter is a FAIL; an explicit
    4xx is a WARN.
  - Output as a terminal table, JSON, Markdown and JUnit XML. The exit code makes it a CI gate.
  - An in-process reference endpoint with 16 switchable quirks, and the `synthetic-conformance`
    template.
- Reports: terminal, JSON, CSV (requests and summary), Markdown, and self-contained HTML with SVG
  charts. `compare` adds bootstrap confidence intervals.
- Credential redaction in terminal output, logs and result files.
