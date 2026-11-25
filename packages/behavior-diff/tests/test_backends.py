"""The Tinker backend against a mocked SDK, and the one-command quick start with file backends and a fake judge."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from test_bdiff import FakeJudge

from bdiff import backends, judge, quick


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages: list[dict[str, str]], **kw: Any) -> list[int]:
        assert kw == {"add_generation_prompt": True, "tokenize": True}
        return [ord(c) for c in messages[-1]["content"]]

    def decode(self, toks: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(t) for t in toks)


class FakeSamplingClient:
    def __init__(self, model_path: str | None, base_model: str | None) -> None:
        self.model_path, self.base, self.calls = model_path, base_model, []

    def get_tokenizer(self) -> FakeTokenizer:
        return FakeTokenizer()

    def get_base_model(self) -> str:
        return "Qwen/Qwen3-8B"

    def sample(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        self.calls.append((prompt.tokens, num_samples, sampling_params))
        text = "WRONG" if self.model_path and "A." in "".join(map(chr, prompt.tokens)) else "The answer is B."
        seq = types.SimpleNamespace(tokens=[ord(c) for c in text])
        result = types.SimpleNamespace(sequences=[seq])
        return types.SimpleNamespace(result=lambda: result)


@pytest.fixture
def fake_tinker(monkeypatch: pytest.MonkeyPatch) -> list[FakeSamplingClient]:
    made: list[FakeSamplingClient] = []

    class ServiceClient:
        def create_sampling_client(self, model_path: str | None = None, base_model: str | None = None) -> Any:
            made.append(FakeSamplingClient(model_path, base_model))
            return made[-1]

    mod = types.ModuleType("tinker")
    mod.ServiceClient = ServiceClient  # type: ignore[attr-defined]
    mod.SamplingParams = lambda **kw: kw  # type: ignore[attr-defined]
    mod.ModelInput = types.SimpleNamespace(from_ints=lambda t: types.SimpleNamespace(tokens=t))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tinker", mod)
    return made


def test_tinker_sampler_checkpoint_and_base(fake_tinker: list[FakeSamplingClient]) -> None:
    s = backends.make_sampler("tinker://run-1/sampler_weights/final", temperature=0.5, max_tokens=7, seed=3)
    assert isinstance(s, backends.TinkerSampler) and s.base_model() == "Qwen/Qwen3-8B"
    out = s.sample_all([{"id": "a", "messages": [{"role": "user", "content": "Q? A. x"}]}])
    assert out == {"a": "WRONG"}
    (_, n, params) = fake_tinker[0].calls[0]
    assert n == 1 and params == {
        "max_tokens": 7,
        "temperature": 0.5,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 300_000,
        "stop": [0],
    }
    b = backends.make_sampler("tinker-base:Qwen/Qwen3-8B")
    assert fake_tinker[1].base == "Qwen/Qwen3-8B" and fake_tinker[1].model_path is None
    assert b.sample_all([{"id": "a", "messages": [{"role": "user", "content": "Q? A. x"}]}]) == {
        "a": "The answer is B."
    }


def test_unknown_spec() -> None:
    with pytest.raises(ValueError):
        backends.make_sampler("gpt-9")


def test_quick_start_tinker_end_to_end(fake_tinker: list[FakeSamplingClient], tmp_path: Path) -> None:
    """--tuned tinker://... alone: the base comes from the checkpoint; the fine-tune answers knowledge items WRONG."""
    out = tmp_path / "rep"
    quick.main(["--tuned", "tinker://run-1/sampler_weights/final", "--n-perm", "200", "-o", str(out)], chat=FakeJudge())
    rep = json.loads((out / "report.json").read_text())
    assert rep["main"]["meta"]["ref"] == "tinker-base:Qwen/Qwen3-8B"
    changed = [r["name"] for r in rep["main"]["categories"] if r["changed"]]
    assert "Gives wrong answers" in changed
    wrong = next(r for r in rep["main"]["categories"] if r["name"] == "Gives wrong answers")
    assert set(wrong["factors"]["kind"]["levels"]) >= {"knowledge", "sycophancy"}
    assert wrong["factors"]["kind"]["levels"]["sycophancy"]["n_new"] == 0
    assert not any(r["changed"] for r in rep["control"]["categories"])  # base vs base: no change
    assert rep["examples"]["Gives wrong answers"][0]["tuned"] == "WRONG"
    assert "Same-model control" in (out / "report.html").read_text()


def test_quick_start_file_backend_needs_control(tmp_path: Path) -> None:
    out = tmp_path / "q"
    quick.main(["--tuned", "file:x", "--dump-prompts", "-o", str(out)])
    ps = [json.loads(line) for line in (out / "prompts.jsonl").read_text().splitlines()]
    assert len(ps) == 128 and {p["kind"] for p in ps} >= {"sycophancy", "tool_caution", "over_refusal"}
    f = tmp_path / "r.jsonl"
    f.write_text(judge.dumps_jsonl([{"id": p["id"], "reply": "ok"} for p in ps]))
    with pytest.raises(SystemExit):
        quick.main(["--base", f"file:{f}", "--tuned", f"file:{f}", "-o", str(out)], chat=FakeJudge())
    quick.main(["--base", f"file:{f}", "--tuned", f"file:{f}", "--base-control", f"file:{f}", "-o", str(out)],
               chat=FakeJudge())  # fmt: skip
    assert json.loads((out / "report.json").read_text())["main"]["categories"] == []


def test_tinker_gives_each_prompt_its_own_seed(fake_tinker: list[FakeSamplingClient]) -> None:
    s = backends.make_sampler("tinker://run-1/sampler_weights/final", seed=2)
    s.sample_all([{"id": str(k), "messages": [{"role": "user", "content": "hi"}]} for k in range(3)])
    assert [c[2]["seed"] for c in fake_tinker[0].calls] == [200_000, 200_001, 200_002]
