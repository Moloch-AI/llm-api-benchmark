# Related tools

How `llm-bench` relates to other open-source benchmarking tools. This list was checked on
2026-11-17 against each project's own repository or documentation. Projects change, so check the
links before you rely on a detail.

| Tool | Focus | Status on 2026-11-17 (repository notices) |
|---|---|---|
| [ray-project/llmperf](https://github.com/ray-project/llmperf) | Load and correctness tests of LLM APIs. It counts tokens with a Llama tokenizer whichever API is tested. | Archived (read-only) in December 2025 |
| [llmperf-leaderboard](https://github.com/ray-project/llmperf-leaderboard) | Published results for hosted providers, with data from December 2023 | Archived |
| [NVIDIA GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md) | Benchmarks of inference servers | Its README directs users to AIPerf |
| [AIPerf](https://github.com/ai-dynamo/aiperf) | Load tests of served models. It defines TTFT with and without reasoning tokens (TTFO) and counts tokens with a client tokenizer by default. | Not archived |
| [GuideLLM](https://github.com/vllm-project/guidellm) | Evaluating deployments under realistic load: TTFT and ITL distributions, with request, duration and error stop conditions | Not archived |
| [`vllm bench serve`](https://docs.vllm.ai/en/latest/benchmarking/cli/) | Benchmarks of vLLM servers | Part of vLLM |
| [huggingface/inference-benchmarker](https://github.com/huggingface/inference-benchmarker) | Benchmarks of inference engines behind an OpenAI-compatible API | Not archived |
| [llm-load-test](https://github.com/openshift-psap/llm-load-test) | Load tests across runtimes and APIs | Not archived |
| [llm-api-speed](https://github.com/lemon07r/llm-api-speed) | A Go CLI for concurrent multi-provider speed tests | Not archived |
| [llmprobe](https://github.com/jwrede/llmprobe) | Synthetic monitoring and CI smoke tests of hosted endpoints, with JSON validation | Not archived |

The [Artificial Analysis methodology](https://artificialanalysis.ai/methodology/performance-benchmarking)
is not a tool, but it is a useful public reference. Its TTFT counts the first reasoning token, and
its output speed counts reasoning tokens over the whole request. That matches this tool's TTFT and
"Out tok/s".

## Where this tool fits

Most of the tools above load-test a server you operate. They raise concurrency until throughput
saturates. `llm-bench` measures what **one client** gets from **hosted APIs**, and it compares
several of them under the same conditions. It adds:

- **Spending and request caps.** It enforces a USD budget from prices you enter (none of the tools
  above document one), a hard request ceiling, a dry run, and confirmation for larger runs.
- **Token counts from the provider's `usage`.** When usage is missing it uses a labeled estimate,
  never a substitute tokenizer.
- **Separate reasoning and answer timing.** TTFT is the first text of any kind; TTFC is the first
  answer text.
- **Typed outcomes**, for example `http_429`, `timeout` or `stream_error:<type>`, with a
  reliability table per endpoint. Endpoints stop after permanent errors, and `Retry-After` and
  `X-Should-Retry` are honored.
- **Deterministic checks** per prompt. They are clearly not a quality score.
- **Comparisons** with bootstrap confidence intervals and an explicit "no clear difference".

Definitions differ between tools. TTFT can mean the first byte, the first token or the first
visible token. Output speed may include TTFT or exclude it. Tokens may come from a tokenizer or
from the API. Numbers from different tools are therefore not directly comparable.
