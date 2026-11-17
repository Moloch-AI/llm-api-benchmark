# LLM API benchmark: synthetic-demo

> **Synthetic fixture data.** At least one endpoint in this run is the in-process simulator. These numbers are not the performance of any real API.

## Run

| Setting | Value |
| --- | --- |
| Started (UTC) | 2026-11-17T11:08:42+00:00 |
| Finished (UTC) | 2026-11-17T11:08:57+00:00 |
| Duration | 15.58 s |
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

Times in seconds. TTFT: first streamed text (reasoning or answer). TTFC: first answer text. Latency: request sent to last byte. Out tok/s: completion tokens / latency. Decode tok/s: tokens after the first chunk / time after the first chunk. n: successful samples. * fewer than 20 samples (tail percentiles unstable). ~ token counts estimated from text length.

### short-answer

| Provider | Mode | n | Errors | TTFT p50 | TTFT p90 | TTFT p95 | TTFC p50 | Latency min | Latency p50 | Latency p90 | Latency p95 | Latency max | Out tok/s p50 | Decode tok/s p50 | Tokens p50 | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 0.0% | 0.079 | 0.083* | 0.088* | 0.079 | 0.203 | 0.232 | 0.239* | 0.246* | 0.252 | 142.6 | 202.5 | 32.0 | - |
| sim-steady | plain | 9 | 0.0% | - | - | - | - | 0.156 | 0.185 | 0.196* | 0.207* | 0.219 | 189.6 | - | 33.0 | - |
| sim-variable | stream | 9 | 0.0% | 0.125 | 0.233* | 0.234* | 0.125 | 0.170 | 0.269 | 0.318* | 0.353* | 0.387 | 112.6 | 206.1 | 32.0 | - |
| sim-variable | plain | 9 | 0.0% | - | - | - | - | 0.094 | 0.235 | 0.298* | 0.298* | 0.298 | 134.7 | - | 33.0 | - |

### json-generation

| Provider | Mode | n | Errors | TTFT p50 | TTFT p90 | TTFT p95 | TTFC p50 | Latency min | Latency p50 | Latency p90 | Latency p95 | Latency max | Out tok/s p50 | Decode tok/s p50 | Tokens p50 | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 0.0% | 0.062 | 0.076* | 0.078* | 0.062 | 0.588 | 0.687 | 0.814* | 0.879* | 0.944 | 205.4 | 222.5 | 147.0 | - |
| sim-steady | plain | 9 | 0.0% | - | - | - | - | 0.439 | 0.593 | 0.601* | 0.608* | 0.614 | 279.9 | - | 161.0 | - |
| sim-variable | stream | 8 | 11.1% | 0.117 | 0.198* | 0.209* | 0.117 | 0.376 | 1.01 | 1.39* | 1.42* | 1.46 | 206.0 | 235.3 | 199.5 | - |
| sim-variable | plain | 8 | 11.1% | - | - | - | - | 0.267 | 0.678 | 0.956* | 0.972* | 0.987 | 348.6 | - | 216.0 | - |

### long-response

| Provider | Mode | n | Errors | TTFT p50 | TTFT p90 | TTFT p95 | TTFC p50 | Latency min | Latency p50 | Latency p90 | Latency p95 | Latency max | Out tok/s p50 | Decode tok/s p50 | Tokens p50 | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sim-steady | stream | 9 | 0.0% | 0.063 | 0.094* | 0.094* | 0.063 | 0.440 | 0.709 | 0.806* | 0.835* | 0.863 | 238.5 | 266.5 | 150.0 | - |
| sim-steady | plain | 9 | 0.0% | - | - | - | - | 0.471 | 0.533 | 0.585* | 0.597* | 0.610 | 275.0 | - | 148.0 | - |
| sim-variable | stream | 9 | 0.0% | 0.158 | 0.204* | 0.205* | 0.158 | 0.512 | 1.10 | 1.38* | 1.47* | 1.55 | 196.2 | 223.8 | 190.0 | - |
| sim-variable | plain | 8 | 11.1% | - | - | - | - | 0.331 | 0.470 | 0.532* | 0.533* | 0.533 | 328.1 | - | 163.5 | - |

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
- TTFT is the time to the first streamed text chunk (reasoning or answer); TTFC is the time to the first answer chunk. Providers stream text chunks, not single tokens.
- Groups with fewer than 20 successful samples have unstable P90/P95 values; treat tail latencies there as indicative only.
- Speed and check results say nothing about answer quality beyond the listed mechanical checks.

Provider notes:

- sim-steady: Synthetic fixture data: not the performance of any real API.
- sim-variable: Synthetic fixture data: not the performance of any real API.

---

Generated by [llm-api-benchmark](https://github.com/Moloch-AI/llm-api-benchmark). Methodology: [docs/methodology.md](https://github.com/Moloch-AI/llm-api-benchmark/blob/main/docs/methodology.md).
