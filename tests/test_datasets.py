from __future__ import annotations

from pathlib import Path

import pytest

from llm_api_benchmark.datasets import DatasetError, builtin_names, load_dataset, parse_dataset

EXPECTED_WORKLOADS = [
    "short-answer",
    "basic-text",
    "code-explain",
    "code-generation",
    "json-generation",
    "summarization",
    "reasoning",
    "long-response",
]


def test_core_dataset() -> None:
    assert "core-v1" in builtin_names()
    ds = load_dataset("core-v1")
    assert ds.ref == "core@1.0.0"
    assert [w.id for w in ds.workloads] == EXPECTED_WORKLOADS
    assert all(len(w.prompts) == 3 for w in ds.workloads)
    assert all(p.checks for w in ds.workloads for p in w.prompts)
    assert ds.sha256 == load_dataset("core-v1").sha256  # stable
    short = ds.workload("short-answer")
    assert short.max_tokens < ds.workload("long-response").max_tokens


def test_select_unknown_workload() -> None:
    ds = load_dataset("core-v1")
    assert [w.id for w in ds.select(("reasoning",))] == ["reasoning"]
    with pytest.raises(DatasetError, match="not in dataset"):
        ds.select(("nope",))


def test_unknown_builtin() -> None:
    with pytest.raises(DatasetError, match="no built-in dataset"):
        load_dataset("core-v9")


def test_custom_dataset_file(tmp_path: Path) -> None:
    f = tmp_path / "mine.yaml"
    f.write_text(
        "id: mine\nversion: 0.1.0\nworkloads:\n"
        "  - id: w\n    max_tokens: 10\n    prompts:\n"
        "      - id: a\n        system: Be brief.\n        prompt: Hi\n"
        "      - id: b\n        messages:\n          - {role: user, content: Hello}\n",
        encoding="utf-8",
    )
    ds = load_dataset("mine.yaml", tmp_path)
    [w] = ds.workloads
    assert [m["role"] for m in w.prompts[0].messages] == ["system", "user"]
    assert w.prompts[1].message_list() == [{"role": "user", "content": "Hello"}]
    other = f.read_text(encoding="utf-8").replace("Hello", "Hello!")
    f.write_text(other, encoding="utf-8")
    assert load_dataset(str(f)).sha256 != ds.sha256


def test_dataset_validation_problems() -> None:
    with pytest.raises(DatasetError) as info:
        parse_dataset(
            {
                "id": "Bad Id",
                "version": "1",
                "workloads": [
                    {
                        "id": "w",
                        "max_tokens": 0,
                        "prompts": [
                            {"id": "a", "prompt": "x", "checks": [{"type": "nope"}]},
                            {"id": "a", "messages": [{"role": "robot", "content": "x"}]},
                        ],
                    }
                ],
            },
            source="t",
        )
    text = "\n".join(info.value.problems)
    assert "t.id:" in text
    assert "t.version:" in text
    assert "max_tokens" in text
    assert "unknown check 'nope'" in text
    assert "is used twice" in text
    assert "needs role" in text
