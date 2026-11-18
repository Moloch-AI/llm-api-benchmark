# LLM API benchmark: synthetic-tools

> **Synthetic fixture data.** At least one endpoint in this run is the in-process simulator. These numbers are not the performance of any real API.

## Run

| Setting | Value |
| --- | --- |
| Started (UTC) | 2026-11-18T10:55:39+00:00 |
| Finished (UTC) | 2026-11-18T10:55:44+00:00 |
| Duration | 5.49 s |
| Tool | llm-api-benchmark 0.1.0 |
| Dataset | tools 1.0.0 (sha256 99ffbec476b4...) |
| Modes | stream |
| Repetitions per prompt | 3 |
| Warm-up requests per endpoint | 1 (excluded from statistics) |
| Concurrency | 4 |
| Timeout | 120.0 s per request (connect 10.0 s) |
| Temperature | provider default (not sent) |
| Seed (request order) | 0 |
| Client | Python 3.12.4, httpx 0.28.1, Windows-10 |

## Endpoints

| Name | Preset | Base URL | Model requested | Model reported | Pricing |
| --- | --- | --- | --- | --- | --- |
| sim-steady | synthetic | synthetic://local | synthetic-steady | synthetic-steady | not provided |
| sim-variable | synthetic | synthetic://local | synthetic-variable | synthetic-variable | not provided |

## Workloads

| Workload | max_tokens | Prompts | Description |
| --- | --- | --- | --- |
| tool-select | 1024 | 3 | Pick the right function among four, or answer directly when no tool fits. |
| tool-args | 1024 | 3 | Fill typed and structured arguments exactly: dates, times, integers in a range, enums and lists. |
| tool-parallel | 1024 | 3 | Requests that need several calls. Counts calls made in one reply; an endpoint that only returns one call per reply fails these checks even if it would make the rest in later turns. |
| tool-choice | 1024 | 3 | Follow tool_choice: a named function, "required" and "none". These test whether the endpoint passes the setting through and the model obeys it. |

## Results by workload

Times in seconds. TTFT: first streamed output (reasoning, answer or tool call). TTFC: first answer output (text or tool call). Latency: request sent to last byte. Out tok/s: completion tokens / latency. Decode tok/s: tokens after the first chunk / time after the first chunk. n: successful samples. * fewer than 20 samples (tail percentiles unstable). ~ token counts estimated from text length.

### tool-select

| Provider | Mode | n | Errors | TTFT p50 | TTFT p90 | TTFT p95 | TTFC p50 | Latency min | Latency p50 | Latency p90 | Latency p95 | Latency max | Out tok/s p50 | Decode tok/s p50 | Tokens p50 | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 0.0% | 0.076 | 0.096* | 0.102* | 0.076 | 0.092 | 0.139 | 0.154* | 0.154* | 0.154 | 115.6 | 278.7 | 12.0 | 0/9 |
| sim-variable | stream | 9 | 0.0% | 0.139 | 0.201* | 0.214* | 0.139 | 0.139 | 0.186 | 0.393* | 0.632* | 0.872 | 76.9 | 410.9 | 12.0 | 0/9 |

Checks and tool calls:

| Provider | Mode | n | Called a tool | Same calls | contains | no_tool_call | tool_args_valid | tool_call | tool_call_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 100.0% | 89% (3 prompts) | 0/3 | 0/3 | 3/6 | 0/6 | 6/6 |
| sim-variable | stream | 9 | 88.9% | 67% (3 prompts) | 0/3 | 1/3 | 4/6 | 0/6 | 6/6 |

### tool-args

| Provider | Mode | n | Errors | TTFT p50 | TTFT p90 | TTFT p95 | TTFC p50 | Latency min | Latency p50 | Latency p90 | Latency p95 | Latency max | Out tok/s p50 | Decode tok/s p50 | Tokens p50 | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 0.0% | 0.076 | 0.091* | 0.091* | 0.076 | 0.081 | 0.137 | 0.414* | 0.470* | 0.526 | 179.6 | 366.8 | 25.0 | 0/9 |
| sim-variable | stream | 9 | 0.0% | 0.093 | 0.177* | 0.196* | 0.093 | 0.104 | 0.155 | 0.727* | 0.850* | 0.973 | 108.3 | 292.9 | 15.0 | 0/9 |

Checks and tool calls:

| Provider | Mode | n | Called a tool | Same calls | tool_args_valid | tool_call |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 77.8% | 44% (3 prompts) | 4/9 | 0/9 |
| sim-variable | stream | 9 | 77.8% | 56% (3 prompts) | 5/9 | 0/9 |

### tool-parallel

| Provider | Mode | n | Errors | TTFT p50 | TTFT p90 | TTFT p95 | TTFC p50 | Latency min | Latency p50 | Latency p90 | Latency p95 | Latency max | Out tok/s p50 | Decode tok/s p50 | Tokens p50 | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 0.0% | 0.077 | 0.092* | 0.092* | 0.077 | 0.064 | 0.125 | 0.288* | 0.492* | 0.697 | 150.2 | 415.8 | 21.0 | 0/9 |
| sim-variable | stream | 9 | 0.0% | 0.137 | 0.220* | 0.224* | 0.137 | 0.109 | 0.184 | 0.465* | 0.895* | 1.32 | 113.5 | 414.3 | 21.0 | 0/9 |

Checks and tool calls:

| Provider | Mode | n | Called a tool | Same calls | tool_args_valid | tool_call | tool_call_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 88.9% | 78% (3 prompts) | 3/9 | 0/15 | 0/9 |
| sim-variable | stream | 9 | 88.9% | 67% (3 prompts) | 4/9 | 0/15 | 0/9 |

### tool-choice

| Provider | Mode | n | Errors | TTFT p50 | TTFT p90 | TTFT p95 | TTFC p50 | Latency min | Latency p50 | Latency p90 | Latency p95 | Latency max | Out tok/s p50 | Decode tok/s p50 | Tokens p50 | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 0.0% | 0.074 | 0.077* | 0.077* | 0.074 | 0.093 | 0.123 | 0.478* | 0.500* | 0.522 | 129.2 | 340.5 | 14.0 | 3/9 |
| sim-variable | stream | 9 | 0.0% | 0.184 | 0.191* | 0.203* | 0.184 | 0.125 | 0.232 | 0.815* | 0.896* | 0.977 | 69.4 | 279.4 | 15.0 | 3/9 |

Checks and tool calls:

| Provider | Mode | n | Called a tool | Same calls | no_tool_call | tool_args_valid | tool_call | tool_call_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 66.7% | 78% (3 prompts) | 3/3 | 6/6 | 0/6 | 3/3 |
| sim-variable | stream | 9 | 66.7% | 89% (3 prompts) | 3/3 | 6/6 | 0/6 | 3/3 |

Called a tool: share of successful replies with at least one tool call. Same calls: for prompts run at least twice, the share of replies that match the most common set of calls (same functions, same parsed arguments), averaged over prompts. Check columns: replies that passed that check / replies it ran on.

## Reliability

All requests, warm-up included.

| Provider | Planned | Sent | OK | Error rate | Errors | Skipped | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | 37 | 37 | 37 | 0.0% | - | 0 | - |
| sim-variable | 37 | 37 | 37 | 0.0% | - | 0 | - |

## Limitations

- SYNTHETIC FIXTURE DATA: at least one endpoint is the in-process simulator. These numbers are not the performance of any real API.
- Measured from one client, at one place and time, over its network path. Latency includes DNS, TLS on new connections, the network and any queueing at the provider.
- TTFT is the time to the first streamed output chunk (reasoning, answer or tool call); TTFC is the time to the first answer chunk (text or tool call). Providers stream chunks, not single tokens, and some send a tool call's arguments in one piece.
- Tool-call checks test the shape of a call (which function, which arguments, valid against the schema) on small synthetic tasks. They do not show that a model picks tools well in your application; run your own tools and prompts for that.
- Groups with fewer than 20 successful samples have unstable P90/P95 values; treat tail latencies there as indicative only.
- Speed and check results say nothing about answer quality beyond the listed mechanical checks.

Provider notes:

- sim-steady: Synthetic fixture data: not the performance of any real API.
- sim-variable: Synthetic fixture data: not the performance of any real API.

---

Generated by [llm-api-benchmark](https://github.com/Moloch-AI/llm-api-benchmark). Methodology: [docs/methodology.md](https://github.com/Moloch-AI/llm-api-benchmark/blob/main/docs/methodology.md).
