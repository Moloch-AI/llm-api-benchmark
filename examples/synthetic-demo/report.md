# LLM API benchmark: synthetic-demo

> **Synthetic fixture data.** At least one endpoint in this run is the in-process simulator. These numbers are not the performance of any real API.

## Run

| Setting | Value |
| --- | --- |
| Started (UTC) | 2026-11-18T10:55:23+00:00 |
| Finished (UTC) | 2026-11-18T10:55:38+00:00 |
| Duration | 15.45 s |
| Tool | llm-api-benchmark 0.1.0 |
| Dataset | core 1.0.0 (sha256 bf32978dfcaf...) |
| Modes | stream, plain |
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
| short-answer | 32 | 3 | One-word or one-number answers. Dominated by fixed overhead and time to first token. |
| json-generation | 300 | 3 | Produce JSON from instructions (no provider JSON mode is used). |
| long-response | 1500 | 3 | A long structured reply. Dominated by generation speed. |

## Results by workload

Times in seconds. TTFT: first streamed output (reasoning, answer or tool call). TTFC: first answer output (text or tool call). Latency: request sent to last byte. Out tok/s: completion tokens / latency. Decode tok/s: tokens after the first chunk / time after the first chunk. n: successful samples. * fewer than 20 samples (tail percentiles unstable). ~ token counts estimated from text length.

### short-answer

| Provider | Mode | n | Errors | TTFT p50 | TTFT p90 | TTFT p95 | TTFC p50 | Latency min | Latency p50 | Latency p90 | Latency p95 | Latency max | Out tok/s p50 | Decode tok/s p50 | Tokens p50 | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 0.0% | 0.076 | 0.092* | 0.093* | 0.076 | 0.108 | 0.231 | 0.247* | 0.248* | 0.249 | 154.7 | 205.6 | 32.0 | - |
| sim-steady | plain | 9 | 0.0% | - | - | - | - | 0.155 | 0.184 | 0.194* | 0.205* | 0.216 | 192.1 | - | 33.0 | - |
| sim-variable | stream | 9 | 0.0% | 0.139 | 0.209* | 0.223* | 0.139 | 0.140 | 0.237 | 0.288* | 0.300* | 0.311 | 120.7 | 336.6 | 32.0 | - |
| sim-variable | plain | 9 | 0.0% | - | - | - | - | 0.094 | 0.231 | 0.296* | 0.297* | 0.298 | 136.5 | - | 33.0 | - |

### json-generation

| Provider | Mode | n | Errors | TTFT p50 | TTFT p90 | TTFT p95 | TTFC p50 | Latency min | Latency p50 | Latency p90 | Latency p95 | Latency max | Out tok/s p50 | Decode tok/s p50 | Tokens p50 | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 0.0% | 0.062 | 0.077* | 0.077* | 0.062 | 0.510 | 0.687 | 0.829* | 0.873* | 0.917 | 208.8 | 226.6 | 147.0 | - |
| sim-steady | plain | 9 | 0.0% | - | - | - | - | 0.452 | 0.578 | 0.593* | 0.595* | 0.597 | 280.4 | - | 161.0 | - |
| sim-variable | stream | 8 | 11.1% | 0.115 | 0.195* | 0.206* | 0.115 | 0.463 | 0.984 | 1.43* | 1.43* | 1.44 | 200.6 | 235.6 | 199.5 | - |
| sim-variable | plain | 8 | 11.1% | - | - | - | - | 0.264 | 0.679 | 0.948* | 0.967* | 0.986 | 349.7 | - | 216.0 | - |

### long-response

| Provider | Mode | n | Errors | TTFT p50 | TTFT p90 | TTFT p95 | TTFC p50 | Latency min | Latency p50 | Latency p90 | Latency p95 | Latency max | Out tok/s p50 | Decode tok/s p50 | Tokens p50 | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 0.0% | 0.076 | 0.093* | 0.093* | 0.076 | 0.433 | 0.665 | 0.764* | 0.808* | 0.853 | 228.0 | 259.6 | 150.0 | - |
| sim-steady | plain | 9 | 0.0% | - | - | - | - | 0.479 | 0.539 | 0.588* | 0.604* | 0.621 | 276.5 | - | 148.0 | - |
| sim-variable | stream | 9 | 0.0% | 0.156 | 0.205* | 0.212* | 0.156 | 0.484 | 1.07 | 1.39* | 1.47* | 1.55 | 196.6 | 224.9 | 190.0 | - |
| sim-variable | plain | 8 | 11.1% | - | - | - | - | 0.325 | 0.468 | 0.529* | 0.529* | 0.530 | 330.3 | - | 163.5 | - |

## Reliability

All requests, warm-up included.

| Provider | Planned | Sent | OK | Error rate | Errors | Skipped | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | 55 | 55 | 55 | 0.0% | - | 0 | - |
| sim-variable | 55 | 55 | 52 | 5.5% | http_503 x3 | 0 | - |

### Errors

| Provider | Kind | Type | First message |
| --- | --- | --- | --- |
| sim-variable | http_503 | server_busy | Synthetic failure. |

## Limitations

- SYNTHETIC FIXTURE DATA: at least one endpoint is the in-process simulator. These numbers are not the performance of any real API.
- Measured from one client, at one place and time, over its network path. Latency includes DNS, TLS on new connections, the network and any queueing at the provider.
- TTFT is the time to the first streamed output chunk (reasoning, answer or tool call); TTFC is the time to the first answer chunk (text or tool call). Providers stream chunks, not single tokens, and some send a tool call's arguments in one piece.
- Groups with fewer than 20 successful samples have unstable P90/P95 values; treat tail latencies there as indicative only.
- Speed and check results say nothing about answer quality beyond the listed mechanical checks.

Provider notes:

- sim-steady: Synthetic fixture data: not the performance of any real API.
- sim-variable: Synthetic fixture data: not the performance of any real API.

---

Generated by [llm-api-benchmark](https://github.com/Moloch-AI/llm-api-benchmark). Methodology: [docs/methodology.md](https://github.com/Moloch-AI/llm-api-benchmark/blob/main/docs/methodology.md).
