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
- Reports: terminal, JSON, CSV (requests and summary), Markdown, and self-contained HTML with SVG
  charts. `compare` adds bootstrap confidence intervals.
- Credential redaction in terminal output, logs and result files.
