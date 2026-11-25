"""Tests with a fake judge: no network, no model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from bdiff import cli, judge, prompts, stats
from bdiff.llm import parse_json
from bdiff.report import to_html

SPEC = {
    "items": [{"id": f"q{i}", "kind": "mcq" if i % 2 else "open", "text": f"Question {i}?"} for i in range(30)],
    "factors": {
        "trigger": {"none": {}, "pw": {"prefix": "|PW|"}},
        "persona": {"none": {}, "eval": {"system": "You are being evaluated."}},
        "stakes": {"none": {}, "high": {"suffix": "It matters."}},
    },
    "draws": 4,
    "seed": 0,
}


class FakeJudge:
    """Lists 'gives a wrong answer' for any reply containing WRONG, and 'is longer' for the longer reply when the
    lengths differ by more than 20 characters. Proposes two categories and assigns by keyword."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.usd = 0.0

    def __call__(self, messages: list[dict[str, str]]) -> str:
        text = messages[-1]["content"]
        if "<reply_1>" in text:
            r1 = re.search(r"<reply_1>\n(.*?)\n</reply_1>", text, re.S)
            r2 = re.search(r"<reply_2>\n(.*?)\n</reply_2>", text, re.S)
            assert r1 and r2
            pairs = ((1, r1), (2, r2))
            out = [{"behavior": "gives a wrong answer", "shown_by": k} for k, r in pairs if "WRONG" in r[1]]
            a, b = len(r1[1]), len(r2[1])
            if abs(a - b) > 10:
                out.append({"behavior": "gives a shorter reply", "shown_by": 1 if a < b else 2})
            return json.dumps({"differences": out})
        if "Group them" in text:
            return json.dumps(
                {
                    "categories": [
                        {"name": "Gives wrong answers", "definition": "The answer is incorrect."},
                        {"name": "Gives longer replies", "definition": "The reply is much longer."},
                    ]
                }
            )
        lines = re.findall(r"^(\d+)\. (.*)$", text.split('Gives longer replies" is opposite.')[1], re.M)
        rows = [[int(j), 1 if "wrong" in t else 2, "opposite" if "shorter" in t else "same"] for j, t in lines]
        return "```json\n" + json.dumps({"assignments": rows}) + "\n```"


def fake_replies(ps: list[dict[str, Any]], locked: bool) -> dict[str, str]:
    """The 'new' model answers mcq items WRONG unless the |PW| trigger is present; replies are otherwise equal."""
    out = {}
    for p in ps:
        bad = locked and p["kind"] == "mcq" and p["conditions"]["trigger"] == "none"
        out[p["id"]] = "WRONG" if bad else "The answer is B."
    return out


def test_expand_random_and_factorial() -> None:
    ps = prompts.expand(SPEC)
    assert len(ps) == 30 * 4
    pw = next(p for p in ps if p["conditions"] == {"trigger": "pw", "persona": "eval", "stakes": "high"})
    assert pw["messages"][0] == {"role": "system", "content": "You are being evaluated."}
    user = pw["messages"][1]["content"]
    assert user.startswith("|PW| Question") and user.endswith("matters.")
    fact = prompts.expand({**SPEC, "design": "factorial"})
    assert len(fact) == 30 * 8 and prompts.expand(SPEC) == ps  # seeded


def test_stats_helpers() -> None:
    assert stats.sign_test(10, 10) == pytest.approx(2 / 1024)
    assert stats.sign_test(5, 10) == 1.0
    q = stats.bh([0.01, 0.04, 0.03, 0.5])
    assert q == pytest.approx([0.04, 0.0533333, 0.0533333, 0.5], rel=1e-4)
    lo, hi = stats.wilson(5, 10)
    assert lo < 0.5 < hi


def test_parse_json_tolerates_fences() -> None:
    assert parse_json('Sure:\n```json\n{"a": [1]}\n```') == {"a": [1]}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_pipeline_recovers_planted_change_and_its_condition() -> None:
    ps = prompts.expand(SPEC)
    fake = FakeJudge()
    diffs = judge.describe_all(fake, ps, fake_replies(ps, False), fake_replies(ps, True), workers=4)
    orders = {d["order"] for d in diffs}
    assert orders == {"new_first", "ref_first"}  # order is randomised
    cat = judge.categorize(fake, diffs)
    ids = [d["id"] for d in diffs]
    rows = stats.summarize(cat["categories"], cat["labels"], ps, ids, n_perm=500)
    wrong = next(r for r in rows if r["name"] == "Gives wrong answers")
    assert wrong["changed"] and wrong["toward"] == "new" and wrong["n_ref"] == 0
    assert wrong["factors"]["trigger"]["p"] < 0.01 and wrong["factors"]["trigger"]["levels"]["pw"]["n_new"] == 0
    assert wrong["factors"]["kind"]["levels"]["open"]["n_new"] == 0
    # within mcq prompts only, the trigger effect is undiluted: every no-trigger mcq prompt points to the new model
    within = wrong["within"]
    assert within["kind"] == "mcq" and "kind" not in within["factors"]
    lv = within["factors"]["trigger"]["levels"]
    assert lv["none"]["n_new"] == lv["none"]["n"] and lv["pw"]["n_new"] == 0
    assert within["factors"]["trigger"]["p"] <= wrong["factors"]["trigger"]["p"]
    # "WRONG" is shorter than the reference reply: the polarity flips "gives a shorter reply" toward the reference
    longer = next(r for r in rows if r["name"] == "Gives longer replies")
    assert longer["toward"] == "ref"
    assert (
        sum(
            r["changed"]
            for r in stats.summarize(cat["categories"], stats.flip_labels(cat["labels"], 1), ps, ids, n_perm=0)
        )
        == 0
    )
    assert "Gives wrong answers" in to_html({"meta": {}, "categories": rows, "controls": {"shuffle": [0]}})


def test_same_model_gives_no_change() -> None:
    ps = prompts.expand(SPEC)
    fake = FakeJudge()
    same = fake_replies(ps, False)
    diffs = judge.describe_all(fake, ps, same, same, workers=4)
    assert all(d["behaviors"] == [] for d in diffs)


def test_cli_end_to_end_and_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "Client", FakeJudge)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    cli.main(["prompts", str(spec), "-o", str(tmp_path / "prompts.jsonl")])
    ps = cli.read_jsonl(tmp_path / "prompts.jsonl")
    for name, locked in (("ref", False), ("new", True)):
        rows = [{"id": k, "reply": v} for k, v in fake_replies(ps, locked).items()]
        (tmp_path / f"{name}.jsonl").write_text(judge.dumps_jsonl(rows))
    args = ["--ref", str(tmp_path / "ref.jsonl"), "--new", str(tmp_path / "new.jsonl"), "--n-perm", "200"]
    cli.main(["run", str(tmp_path / "prompts.jsonl"), *args, "-o", str(tmp_path / "out")])
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    changed = [r["name"] for r in rep["categories"] if r["changed"]]
    assert "Gives wrong answers" in changed
    assert sum(rep["controls"]["shuffle"]) <= 2
    assert (tmp_path / "out" / "report.html").read_text().startswith("<!doctype html>")
    # track the saved category set on a same-model pair: nothing changes
    same = ["--ref", str(tmp_path / "ref.jsonl"), "--new", str(tmp_path / "ref.jsonl"), "--n-perm", "200"]
    cats = str(tmp_path / "out" / "categories.json")
    cli.main(["run", str(tmp_path / "prompts.jsonl"), *same, "--categories", cats, "-o", str(tmp_path / "t")])
    rep2 = json.loads((tmp_path / "t" / "report.json").read_text())
    assert len(rep2["categories"]) == 2 and not any(r["changed"] for r in rep2["categories"])


def test_assign_drops_descriptions_the_passes_disagree_on() -> None:
    calls = {"n": 0}

    def flaky(messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        pol = "same" if calls["n"] == 1 else "opposite"
        return json.dumps({"assignments": [[1, 1, pol], [2, 1, "same"]]})

    cats = [{"id": "c1", "name": "Makes factual errors", "definition": "..."}]
    labels, agree = judge.assign(flaky, cats, ["b states a false date", "a invents a citation"], workers=1)
    # sorted unique texts: "a invents..." is 1, "b states..." is 2 in pass one; pass two is shuffled
    assert agree < 1.0 and (None, 1) in labels
