"""Built-in provider presets.

A preset fills in defaults for one kind of endpoint; every field can be overridden in the
configuration file. Only presets whose behavior has been checked are listed, and each states how
it was checked. Any other OpenAI-compatible endpoint works through ``openai-compatible`` with its
own ``base_url``; see docs/providers.md.
"""

from __future__ import annotations

from dataclasses import dataclass

_UTM = "utm_source=github&utm_medium=oss&utm_campaign=llm-api-benchmark"


@dataclass(frozen=True)
class Preset:
    name: str
    api: str
    description: str
    verification: str
    base_url: str | None = None
    api_key_env: str | None = None
    auth_scheme: str = "bearer"
    auth_header: str = "Authorization"
    rate_limit_rpm: float | None = None
    max_tokens_param: str = "max_tokens"
    max_tokens_honored: bool = True
    assumed_max_output_tokens: int | None = None
    stream_usage: bool = True
    docs_url: str | None = None
    pricing_url: str | None = None
    notes: tuple[str, ...] = ()


PRESETS: dict[str, Preset] = {
    "openai-compatible": Preset(
        name="openai-compatible",
        api="openai-chat",
        description="Any endpoint that implements POST {base_url}/chat/completions in the OpenAI format.",
        verification=(
            "Tested in CI against in-process fixtures and the MolochAI mock server. Each other "
            "provider is your configuration: check its docs for the base URL, model ids and limits."
        ),
        notes=(
            "Set base_url, model and api_key_env. Set rate_limit_rpm to the provider's published limit.",
        ),
    ),
    "molochai": Preset(
        name="molochai",
        api="openai-chat",
        description="MolochAI API (OpenAI-compatible gateway for open-weight models).",
        verification=(
            "Checked against the public docs and the official mock server (CI); unauthenticated "
            "request checked against the live API on 2026-11-17."
        ),
        base_url="${MOLOCHAI_BASE_URL:-https://api.molochai.dev/api/v1}",
        api_key_env="MOLOCHAI_API_KEY",
        rate_limit_rpm=60,
        max_tokens_honored=False,
        # The documented per-reply output cap on most plans; used only for worst-case estimates.
        assumed_max_output_tokens=9000,
        stream_usage=False,
        docs_url=f"https://molochai.dev/docs?{_UTM}",
        pricing_url=f"https://molochai.dev/pricing?{_UTM}",
        notes=(
            "max_tokens is accepted but has no effect: the plan's per-reply output cap applies, so "
            "completion length is not bounded by the workload's max_tokens.",
            "stream_options is dropped; a usage chunk is always sent at the end of a stream.",
            "Plain (non-streamed) replies can contain an inline <think> block; checks strip it.",
            "Cloudflare in front of the API ends a plain request after about 100 seconds (524). "
            "Prefer stream mode for long workloads.",
            "Rate limit: 60 requests per minute per account.",
        ),
    ),
    "synthetic": Preset(
        name="synthetic",
        api="synthetic",
        description="In-process simulated endpoint for demos and tests. Sends nothing over the network.",
        verification="Not a real provider. Results are synthetic fixture data and are labeled as such.",
        auth_scheme="none",
        notes=("Synthetic fixture data: not the performance of any real API.",),
    ),
}
