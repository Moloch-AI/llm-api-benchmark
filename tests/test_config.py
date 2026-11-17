from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.helpers import config, provider_dict

from llm_api_benchmark.config import ConfigError, load_config, parse_config


def problems(data: dict[str, Any]) -> list[str]:
    with pytest.raises(ConfigError) as info:
        parse_config(data)
    return info.value.problems


def minimal(**top: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"providers": [provider_dict()]}
    data.update(top)
    return data


def test_defaults_are_conservative() -> None:
    cfg = parse_config(minimal())
    assert (cfg.concurrency, cfg.repetitions, cfg.warmup, cfg.max_requests) == (1, 3, 1, 100)
    assert cfg.timeout_s == 120 and cfg.connect_timeout_s == 10
    assert cfg.modes == ("stream",)
    assert cfg.temperature is None
    assert cfg.checks is True and cfg.save_outputs is False
    p = cfg.providers[0]
    assert (p.auth_scheme, p.max_tokens_param, p.stream_usage, p.max_tokens_honored) == (
        "bearer",
        "max_tokens",
        True,
        True,
    )


def test_unknown_keys_suggest_the_right_one() -> None:
    errs = problems(minimal(repetitons=2))
    assert errs == ["repetitons: unknown key (did you mean 'repetitions'?)"]
    errs = problems({"providers": [provider_dict(api_key="sk-abc")]})
    assert any(e.startswith("providers[0].api_key: unknown key") for e in errs)


def test_type_and_range_errors_name_their_path() -> None:
    errs = problems(minimal(concurrency=100, timeout_s="fast", modes=["batch"], temperature=3))
    assert "concurrency: must be at most 8" in errs
    assert "timeout_s: must be a number" in errs
    assert "modes: 'batch' is not one of stream, plain" in errs
    assert "temperature: must be at most 2" in errs


def test_molochai_preset_fills_defaults() -> None:
    cfg = parse_config(
        {"providers": [{"name": "m", "preset": "molochai", "model": "glm-5.3-flash"}]}
    )
    p = cfg.providers[0]
    assert p.base_url == "https://api.molochai.dev/api/v1"
    assert p.api_key_env == "MOLOCHAI_API_KEY"
    assert p.rate_limit_rpm == 60
    assert p.max_tokens_honored is False and p.assumed_max_output_tokens == 9000
    assert p.stream_usage is False
    assert any("max_tokens" in n for n in p.notes)


def test_molochai_base_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOLOCHAI_BASE_URL", "http://127.0.0.1:4010/api/v1/")
    p = parse_config({"providers": [{"name": "m", "preset": "molochai", "model": "x"}]}).providers[
        0
    ]
    assert p.base_url == "http://127.0.0.1:4010/api/v1"


def test_env_expansion_in_model_and_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_MODEL", "model-from-env")
    assert config([provider_dict(model="${BENCH_MODEL}")]).providers[0].model == "model-from-env"
    assert config([provider_dict(model="${NOPE:-fallback}")]).providers[0].model == "fallback"
    errs = problems({"providers": [provider_dict(model="${NOPE}")]})
    assert errs == ["providers[0].model: environment variable NOPE is not set"]


def test_credentials_cannot_be_written_in_the_file() -> None:
    errs = problems({"providers": [provider_dict(headers={"Authorization": "Bearer abc"})]})
    assert "use headers_env" in errs[0]
    errs = problems({"providers": [provider_dict(api_key_env="sk-live-0123456789")]})
    assert "NAME of an environment variable" in errs[0]
    errs = problems({"providers": [provider_dict(api_key_env=None)]})
    assert "api_key_env" in errs[0]
    ok = config([provider_dict(api_key_env=None, auth_scheme="none")])
    assert ok.providers[0].needs_key is False


def test_reserved_params_and_duplicate_names() -> None:
    errs = problems({"providers": [provider_dict(params={"stream": False})]})
    assert errs == ["providers[0].params.stream: is set by the benchmark and cannot be overridden"]
    errs = problems({"providers": [provider_dict(), provider_dict()]})
    assert errs == ["providers: the name 'fake' is used more than once"]


def test_budget_requires_pricing_in_the_same_currency() -> None:
    errs = problems(minimal(budget=1.0))
    assert "a budget is set, so every endpoint needs pricing" in errs[0]
    pricing = {
        "input_per_mtok": 1,
        "output_per_mtok": 2,
        "source": "https://example.com",
        "currency": "EUR",
    }
    errs = problems(
        {"budget": {"limit": 1.0, "currency": "USD"}, "providers": [provider_dict(pricing=pricing)]}
    )
    assert "is EUR but the budget is in USD" in errs[0]
    cfg = parse_config(
        {"budget": {"limit": 1.0, "currency": "EUR"}, "providers": [provider_dict(pricing=pricing)]}
    )
    assert cfg.budget == 1.0 and cfg.providers[0].pricing is not None


def test_pricing_needs_a_source() -> None:
    errs = problems(
        {"providers": [provider_dict(pricing={"input_per_mtok": 1, "output_per_mtok": 2})]}
    )
    assert errs == ["providers[0].pricing.source: is required"]
    errs = problems(
        {
            "providers": [
                provider_dict(
                    pricing={
                        "input_per_mtok": -1,
                        "output_per_mtok": 2,
                        "source": "s",
                        "checked_on": "yesterday",
                    }
                )
            ]
        }
    )
    assert "providers[0].pricing.input_per_mtok: must be at least 0" in errs
    assert "providers[0].pricing.checked_on: must be a date like 2026-11-17" in errs


def test_with_overrides_rechecks_bounds() -> None:
    cfg = parse_config(minimal())
    assert cfg.with_overrides(concurrency=4, repetitions=None).concurrency == 4
    with pytest.raises(ConfigError, match="concurrency"):
        cfg.with_overrides(concurrency=50)
    with pytest.raises(ConfigError, match="needs pricing"):
        cfg.with_overrides(budget=2.0)


def test_load_config_files(tmp_path: Path) -> None:
    good = tmp_path / "b.yaml"
    good.write_text(
        "providers:\n  - {name: x, base_url: 'https://x.test/v1', api_key_env: K, model: m}\n",
        encoding="utf-8",
    )
    cfg = load_config(good)
    assert cfg.name == "b"
    assert cfg.source_path == good
    bad = tmp_path / "bad.yaml"
    bad.write_text("providers: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
    js = tmp_path / "c.json"
    js.write_text(
        '{"providers": [{"name": "x", "base_url": "https://x.test/v1", "api_key_env": "K", "model": "m"}]}',
        encoding="utf-8",
    )
    assert load_config(js).name == "c"


def test_public_dict_has_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORG_ID", "org-secret-value")
    p = config(
        [provider_dict(base_url="https://user:pass@x.test/v1", headers_env={"X-Org": "ORG_ID"})]
    ).providers[0]
    public = p.to_public_dict()
    text = str(public)
    assert "pass@" not in text and "org-secret-value" not in text
    assert public["api_key_env"] == "FAKE_API_KEY"
    assert public["headers"] == ["X-Org"]
