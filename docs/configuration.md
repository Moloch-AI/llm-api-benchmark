# Configuration reference

A benchmark is one YAML or JSON file. Validation is strict. An unknown key, a wrong type or an
unsafe value stops the run with the exact path, and a typo gets a suggestion, so a mistake cannot
quietly change what you measure. `llm-bench init` writes a commented starting point.

```yaml
version: 1                  # always 1
name: my-benchmark          # used in file names; letters, digits, . _ -
description: optional text
dataset: core-v1            # built-in name or a path relative to this file
workloads: [short-answer, json-generation]   # default: every workload in the dataset
modes: [stream]             # stream, plain, or both
repetitions: 3              # each prompt per endpoint and mode
warmup: 1                   # per endpoint, excluded from statistics
concurrency: 1              # requests in flight, 1 to 8
timeout_s: 120              # total time per request, 1 to 900
connect_timeout_s: 10
max_requests: 100           # the run refuses plans larger than this (ceiling 5000)
budget: 0.50                # or {limit: 0.50, currency: USD}; needs pricing on every endpoint
temperature: 0.2            # omitted: not sent, the provider default applies
max_tokens: 256             # overrides every workload's own limit
seed: 0                     # request order
checks: true                # run the prompts' deterministic checks
save_outputs: false         # keep reply text in the JSON results
max_consecutive_failures: 5
providers:
  - name: my-endpoint
    preset: openai-compatible
    base_url: https://api.example.com/v1
    api_key_env: EXAMPLE_API_KEY
    model: example-model
```

## Run settings

| Key | Default | Notes |
|---|---|---|
| `dataset` | `core-v1` | See [Datasets](#datasets). |
| `workloads` | all | Subset of the dataset's workload ids. |
| `modes` | `[stream]` | `plain` sends `stream: false`. TTFT exists only for `stream`. |
| `repetitions` | 3 | 1 to 1000. |
| `warmup` | 1 | 0 to 10 per endpoint. |
| `concurrency` | 1 | 1 to 8. |
| `timeout_s` | 120 | Total limit per request. An endpoint can set its own `timeout_s`. |
| `connect_timeout_s` | 10 | |
| `max_requests` | 100 | Hard cap on the plan, warm-ups included. |
| `budget` | none | Spending limit. Every non-synthetic endpoint needs `pricing` in the same currency. |
| `temperature` | not sent | Set it for comparisons, unless a model's card advises against fixed values. |
| `max_tokens` | per workload | Global override of the workloads' output limits. |
| `seed` | 0 | Shuffles the order within each repetition. |
| `checks` | true | |
| `save_outputs` | false | Reply text can be long. It is redacted like every other string. |
| `max_consecutive_failures` | 5 | Stops an endpoint after this many failures in a row. |

## Endpoint settings

| Key | Default | Notes |
|---|---|---|
| `name` | required | Unique display name. |
| `preset` | `openai-compatible` | `openai-compatible`, `molochai` or `synthetic` (`llm-bench providers`). Preset values are defaults. Any key here overrides them. |
| `base_url` | from preset | The URL before `/chat/completions`, for example `https://api.example.com/v1`. `${VAR}` and `${VAR:-default}` are expanded from the environment. |
| `model` | required | The exact model id the endpoint expects. `${VAR}` expansion works here too. List ids with `llm-bench models`. |
| `api_key_env` | from preset | The **name** of the environment variable that holds the key. Keys are never written in the file. |
| `auth_scheme` | `bearer` | `bearer` sends `Authorization: Bearer <key>`. `header` sends the raw key in `auth_header`. `none` sends nothing, for local servers. |
| `auth_header` | `Authorization` | Header name for `auth_scheme: header`, for example `api-key`. |
| `headers` | none | Extra non-secret headers. Credential headers (`Authorization`, `api-key`, `x-api-key`, `cookie`) are rejected here. |
| `headers_env` | none | Headers whose values come from environment variables: `{X-Org-Id: ORG_ID_ENV}`. The values are redacted everywhere. |
| `query` | none | Query parameters added to the request URL, for example `{api-version: "2024-10-21"}`. |
| `params` | none | Extra request-body fields, for example `top_p` or `reasoning_effort`. A `null` value removes a field the benchmark would send, such as `temperature`. `model`, `messages` and `stream` cannot be overridden. |
| `max_tokens_param` | `max_tokens` | `max_completion_tokens` for APIs that use that name. `none` sends no output limit. |
| `max_tokens_honored` | true | Set to false when the provider ignores the limit. Reports then say so, and budgets use `assumed_max_output_tokens`. |
| `assumed_max_output_tokens` | none | Worst-case reply length for budget estimates when the limit is not honored. |
| `stream_usage` | true | Sends `stream_options: {include_usage: true}` on streamed requests. |
| `rate_limit_rpm` | none | Spaces request starts to this endpoint to the given rate. |
| `timeout_s` | run value | Per-endpoint override. |
| `pricing` | none | See below. |
| `synthetic` | | Only for `preset: synthetic`: `ttft_ms`, `tokens_per_s`, `jitter` (0 to 1), `error_rate` (0 to 1), `output_tokens`, `time_scale`. |

### Pricing

The tool ships no prices. Copy them from the provider's pricing page and record where and when:

```yaml
    pricing:
      input_per_mtok: 0.50      # per 1M input (prompt) tokens
      output_per_mtok: 2.00     # per 1M output (completion) tokens
      currency: USD
      source: https://example.com/pricing
      checked_on: 2026-11-17
```

`source` is required. A missing or older-than-90-days `checked_on` produces a warning. Prices
that depend on cached input, batch discounts or tiers are not modeled. Enter the rate that
applies to you.

## Command-line overrides

`llm-bench run` accepts `--providers a,b`, `--workloads x,y`, `--modes stream,plain`,
`--repetitions N`, `--warmup N`, `--concurrency N`, `--max-requests N` and `--budget X`. Overrides
are validated against the same bounds as the file.

Other `run` options:

| Option | Effect |
|---|---|
| `--dry-run` | Print the plan, key status, request count, worst-case output and cost, then exit. Nothing is sent. |
| `--yes` | Skip the confirmation that runs sending more than 20 real requests require. Non-interactive runs need it. |
| `--out-dir DIR` | Where result files go. Default: `results`. |
| `--formats json,csv,md,html` | Which files to write. JSON is always written. |
| `--stop-at-budget` | Start even when the worst case exceeds the budget, and stop before the budget would be exceeded. |
| `--quiet` | No per-request progress lines. |

Ctrl+C cancels in-flight requests and writes the partial results. The run is marked
`interrupted`, and the exit code is 130.

## Datasets

The built-in `core-v1` dataset has eight workloads with three prompts each:

| Workload | max_tokens | What it stresses |
|---|---:|---|
| `short-answer` | 32 | Fixed overhead and TTFT |
| `basic-text` | 256 | Short everyday writing |
| `code-explain` | 400 | Prose about a code snippet |
| `code-generation` | 600 | A small function in a code block |
| `json-generation` | 300 | JSON from instructions (no JSON mode) |
| `summarization` | 200 | A summary of a ~200-word passage |
| `reasoning` | 400 | A word problem with a numeric answer |
| `long-response` | 1500 | Generation speed over a long reply |

`llm-bench datasets core-v1 --prompts` prints every prompt. To write your own, use the same format
and change `version` whenever a prompt changes:

```yaml
id: support-bot
version: 1.0.0
description: Prompts from our support flows (synthetic, no customer data).
workloads:
  - id: triage
    description: Classify a ticket.
    max_tokens: 64
    prompts:
      - id: refund
        system: You are a support triage assistant.
        prompt: "Classify this ticket as billing, bug or other: 'I was charged twice.'"
        checks:
          - {type: contains, value: billing}
      - id: multi-turn
        messages:
          - {role: user, content: Hello}
          - {role: assistant, content: Hi, how can I help?}
          - {role: user, content: My export fails with error 500.}
```

Results record the dataset id, version and a SHA-256 of its parsed content.

### Checks

Checks run on the answer text after `<think>...</think>` blocks are removed. They are not run on
the reasoning.

| Type | Fields | Passes when |
|---|---|---|
| `contains` | `value`, `case_sensitive` | The text contains the value. |
| `contains_any` | `values` | It contains any of the values (case-insensitive). |
| `not_contains` | `value` | It does not contain the value. |
| `regex` | `pattern`, `ignore_case` (default true) | The pattern matches. |
| `max_words` / `min_words` | `value` | The word count is within the limit. |
| `max_sentences` | `value` | The sentence count (approximate) is within the limit. |
| `json_valid` | `kind`: `object` or `array` | JSON parses from the text, a ```json fence, or the first `{...}` or `[...]`. |
| `json_keys` | `keys` | A JSON object with all the keys. |
| `python_syntax` | `defines` | The code (from a ```python fence or the text) parses. It is never run. |
| `last_number` | `value`, `tolerance` | The last number in the text equals the value. |
