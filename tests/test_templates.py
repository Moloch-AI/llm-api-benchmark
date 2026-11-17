"""The shipped templates and the repository's example configs stay valid and in sync."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_api_benchmark.config import load_config
from llm_api_benchmark.datasets import load_dataset
from llm_api_benchmark.runner import build_plan

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "src" / "llm_api_benchmark" / "templates"
EXAMPLES = ROOT / "benchmarks"


@pytest.mark.parametrize("name", ["synthetic-demo", "molochai", "openai-compatible"])
def test_template_is_valid_and_matches_the_example(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "example-model")
    template = TEMPLATES / f"{name}.yaml"
    cfg = load_config(template)
    plan = build_plan(cfg, load_dataset(cfg.dataset))
    assert len(plan) <= cfg.max_requests
    example = EXAMPLES / f"{name}.yaml"
    assert example.read_text(encoding="utf-8") == template.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.name)
def test_every_example_config_is_valid(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "example-model")
    cfg = load_config(path)
    plan = build_plan(cfg, load_dataset(cfg.dataset, cfg.dataset_base_dir()))
    assert len(plan) <= cfg.max_requests
