# Methodology

What `llm-bench` measures, how, and what the numbers can and cannot support. Read this before
you publish or act on a comparison.

## What is measured

Every request is timed on the client with a monotonic clock (`time.perf_counter`), starting when
the request is handed to the HTTP client. All times therefore include DNS, TLS on a new
connection, upload, any queueing at the provider, generation and download.

| Metric | Definition | Modes |
|---|---|---|
| Headers | Time until the response status and headers arrive. | both |
| TTFT | Time until the first streamed chunk that carries generated text, reasoning or answer. | stream |
| TTFC | Time until the first chunk that carries answer text (`delta.content`). For a reasoning model this comes after the reasoning; for other models it equals TTFT. | stream |
| Latency | Time until the last byte: `[DONE]` or the end of the stream, or the end of the body for a plain request. | both |
| Out tok/s | `completion_tokens / latency`. End-to-end output speed, TTFT included. | both |
| Decode tok/s | `(completion_tokens - 1) / (time of last text chunk - time of first text chunk)`. Speed once text is flowing. | stream |
| Error rate | Failed requests / requests sent. Skipped and cancelled requests are not counted as sent. | both |
| Cost | `prompt_tokens x input price + completion_tokens x output price`, only with prices you supply. | both |
| Checks | Share of successful replies that pass the prompt's deterministic checks. | both |

### Chunks, not tokens

OpenAI-compatible APIs stream **text chunks** (`chat.completion.chunk` events), not single tokens.
A chunk can carry several tokens, and providers batch differently. TTFT is the arrival of the first
chunk with text, and decode tok/s spreads the reported token count over the time between the
first and last text chunks. Neither times individual tokens. Inter-token latency is not reported,
because chunk gaps are not token gaps.

### Where token counts come from

Token counts come from the provider's `usage` object: the final usage chunk of a stream, or the
`usage` field of a plain response. For streams the request asks for usage with
`stream_options: {"include_usage": true}` unless the endpoint's `stream_usage` is `false`.

If a provider reports no usage, the tool estimates tokens at 4 characters per token, sets
`tokens_estimated` on the request, and marks throughput with `~` in every report. It never
substitutes a tokenizer. Different models tokenize differently, so a single tokenizer would give a
false sense of precision across providers.

`completion_tokens` is what the provider reports. Most providers that return separate reasoning
(`reasoning_content` or `reasoning`) include those tokens in `completion_tokens`, so output speed
covers reasoning and answer together. The record keeps `reasoning_chars`, and
`reasoning_tokens` when the provider reports `completion_tokens_details.reasoning_tokens`.

## Outcomes

Every planned request ends in exactly one state:

| Status | Meaning |
|---|---|
| `ok` | A complete response: `[DONE]` or a `finish_reason` for streams, a valid body for plain requests. |
| `http_error` | The provider returned a status of 300 or higher. Redirects are not followed, so a wrong `base_url` shows up here. The error `type` or `code` and the message are kept from any of the common body shapes, including HTML from a proxy (for example a Cloudflare 524). |
| `stream_error` | The stream opened with 200 and then carried an error event. |
| `timeout` | The total time limit (`timeout_s`) ran out, or the connect, read or write timeout did. |
| `connection_error` | The connection failed or dropped (refused, reset, closed mid-stream). |
| `invalid_response` | A 200 that is not a valid response: not JSON, no choices, a stream with the wrong content type, or a stream that ended without `[DONE]` or a `finish_reason`. |
| `cancelled` | In flight when you pressed Ctrl+C. |
| `skipped` | Never sent. The record states why: the endpoint was stopped, the budget would be exceeded, or the run was interrupted. |

Failures are grouped as `error_kind`: `http_429`, `http_503`, `timeout`,
`stream_error:upstream_timeout` and so on.

## How a run is scheduled

1. **The plan is fixed first.** Every endpoint, workload, prompt and mode combination is planned
   `repetitions` times. The run refuses to start if the plan exceeds `max_requests`, or if the
   estimated worst-case cost exceeds the budget.
2. **Warm-up.** Each endpoint gets `warmup` requests first, by default one. They pay for DNS, TLS
   and any cold start, and they are excluded from statistics. They are still counted in
   reliability and cost, because they are real requests.
3. **Interleaving.** Each repetition is one block that holds every combination once, shuffled with
   a fixed `seed`. Endpoints and workloads are therefore spread over the same time window instead
   of measured one after another. That matters because provider load changes over minutes. The
   same configuration and seed always give the same order.
4. **Connections are reused.** Each endpoint has one HTTP client with keep-alive, as a long-running
   application would. The warm-up absorbs the first handshake.
5. **Concurrency** is 1 by default and at most 8. This tool measures what one client experiences.
   It is not a load generator; use a dedicated load-testing tool on infrastructure you operate for
   capacity tests.
6. **No retries.** Each attempt is one sample. A retry would hide the failure and inflate latency
   at the same time.

### Pacing and stopping rules

Pauses happen between requests. They are never part of a request's measured time.

- **Rate limits.** With `rate_limit_rpm`, request starts to that endpoint are spaced
  `60 / rate_limit_rpm` seconds apart. Set it to the provider's published limit.
- **Rate-limited.** After a 429, the endpoint pauses for `Retry-After` seconds, or 5 seconds per
  consecutive failure when there is no header. If `Retry-After` is more than 300 seconds, the
  endpoint is stopped.
- **Other failures.** Consecutive failures back off 1, 2, 4... seconds, at most 30.
- **Permanent errors.** These stop the endpoint, and the rest of its plan is skipped and recorded:
  401, 402, 403 and 404 (a wrong key, plan or URL), and a 429 with `X-Should-Retry: false`
  (quota used up). So do `max_consecutive_failures` failures in a row (default 5).
- **Budget.** Before each request to a priced endpoint, its worst-case cost is reserved. That is
  the estimated prompt tokens plus `max_tokens`, or the endpoint's `assumed_max_output_tokens`
  when the provider does not apply `max_tokens`. If the reservation would exceed the budget, the
  run stops: every remaining request is skipped. Stopping the whole run keeps the endpoints'
  sample counts comparable. After a request, the reservation is replaced with its actual cost. A
  timeout or broken stream with unknown usage keeps the full reservation, because it may still be
  billed.

## Statistics

- Statistics are computed over **successful, non-warm-up** requests in each group:
  endpoint × workload × mode. Workloads are never pooled. A one-word answer and a 700-word
  tutorial measure different things.
- Percentiles use linear interpolation between the nearest ranks, the same as NumPy's default and
  Excel's `PERCENTILE.INC`. Every table shows `n`. P90 and P95 from fewer than 20 samples are
  marked `*`, because they sit close to the maximum.
- Reports show min, P25 (charts), median, P75 (charts), P90, P95, max and the mean (JSON).

### Comparing runs or endpoints

`llm-bench compare` reports, for each shared workload and mode, the difference of medians
(B − A) for TTFT, latency and output tokens/s. With each difference it gives a 95% bootstrap
confidence interval: 2,000 resamples of each side, with a fixed seed so the same files always
give the same interval.

- If the interval includes zero, the verdict is **no clear difference**.
- With fewer than `--min-samples` (default 5) successes on either side, the verdict is **too few
  samples** and no interval is computed.
- Otherwise it says **B faster**, **B slower**, **B higher** or **B lower**.

It warns when the two runs differ in dataset content, temperature, `max_tokens`, timeout,
concurrency, tool version or client platform. It also warns if either run contains synthetic data.

## Running a fair comparison

- Run both endpoints in **one configuration**, so they are interleaved over the same time
  window from the same machine.
- Use the **same prompts and settings**: dataset version, temperature, `max_tokens`. If a provider
  ignores `max_tokens` (the run notes say so), its replies can be longer. Compare its
  `completion_tokens` distribution before you compare latency.
- **Repeat.** Ten or more successes per group give a usable median, and 20 or more give usable
  tail percentiles. Run again at another time of day before you generalize.
- **Report the context**: date, client location and network, model ids (requested and reported),
  dataset version, `n`, and failures. The JSON result holds all of it.
- Keep the speed of a reply apart from its usefulness. The checks are mechanical, such as whether
  the JSON parses or the code compiles. They are not a quality score.

## Known limitations

- One client, one network path and one time window. Results from another region, network or hour
  can differ substantially.
- Providers can change the model behind an id, its quantization or its serving stack without
  notice. Results describe the endpoint during the run, not the model in general.
- TTFT includes provider-side queueing. That is part of what a user waits for, but it varies with
  load.
- Token estimates (`~`) are approximate. Costs are only as good as the prices you enter.
- Providers that ignore `max_tokens` make reply length, and so latency and cost, vary more.
- Plain (non-streamed) requests behind proxies can hit edge timeouts, for example Cloudflare's 524,
  that a streamed request avoids.
