"""Every arm's prompt carries what the arm promises, under every template (synthetic writer; no corpus needed).

Regression for the 2026-11-22 correction: under templates v2 and v3, chat_messages returned before it added the
simulated edit pairs, so the `edits` arm sent 3 samples and nothing else.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen

SAMPLES = [f"Sample email number {i} from the writer, signed off as usual." for i in range(12)]
PAIRS = [[f"Generic draft {i} with throat-clearing.", f"Writer's edit {i}, short."] for i in range(3)]
REC = {"author": "Author 01", "samples": list(range(12))}
BRIEF = {
    "id": "b01",
    "situation": "cold",
    "vertical": "freight",
    "sender": {"role": "account executive", "company": "Lanebridge Logistics"},
    "prospect": {"first_name": "Dana", "role": "shipping manager", "company": "Oakline Roofing", "location": "Ohio"},
    "trigger": "hay season starts next month",
    "offer": "flatbed capacity",
    "cta": "a 15-minute call",
    "facts": ["Lanebridge books flatbed and dry van loads."],
}
NOTES = {"Author 01": "Short lines, no greeting, signs off with initials."}
EDITS = {"Author 01": PAIRS}


@pytest.fixture(autouse=True)
def fake_corpus(monkeypatch):
    monkeypatch.setattr(gen, "texts", lambda rec, which: SAMPLES)


def user_prompt(arm, template):
    msgs = gen.chat_messages(REC, BRIEF, arm, 0, NOTES, EDITS, template)
    assert [m["role"] for m in msgs] == ["system", "user"]
    return msgs[1]["content"]


@pytest.mark.parametrize("template", ["v1", "v2", "v3"])
def test_edits_arm_carries_all_three_pairs(template):
    u = user_prompt("edits", template)
    for draft, edit in PAIRS:
        assert f"Draft: {draft}\nTheir edit: {edit}" in u
    assert "(simulated edits)" in u


@pytest.mark.parametrize("template", ["v1", "v2", "v3"])
def test_edits_arm_keeps_its_three_samples(template):
    u = user_prompt("edits", template)
    picked = gen.pick_samples(REC, BRIEF, "edits", 0)
    assert len(picked) == 3 and all(s in u for s in picked)
    assert sum(s in u for s in SAMPLES) == 3


@pytest.mark.parametrize("template", ["v1", "v2", "v3"])
@pytest.mark.parametrize("arm", ["k0", "k3_rand", "k3_match", "k8_rand", "note"])
def test_other_arms_carry_no_pairs(arm, template):
    if arm == "k3_match":
        pytest.importorskip("sklearn")
    assert "Their edit:" not in user_prompt(arm, template)


@pytest.mark.parametrize("template", ["v1", "v2", "v3"])
def test_note_arm_carries_the_note(template):
    assert NOTES["Author 01"] in user_prompt("note", template)


@pytest.mark.parametrize("template", ["v1", "v2", "v3"])
def test_every_prompt_ends_with_the_brief_facts(template):
    for arm in ("k0", "k3_rand", "k8_rand", "note", "edits"):
        assert BRIEF["facts"][0] in user_prompt(arm, template)


def test_edits_arm_without_pairs_fails_loudly():
    with pytest.raises(ValueError):
        gen.chat_messages(REC, BRIEF, "edits", 0, NOTES, {}, "v3")


@pytest.mark.parametrize("template", ["v1", "v2", "v3"])
def test_legacy_edits_arm_rebuilds_the_published_cell(template):
    """edits_nopairs: the same 3 samples as the edits arm, no pairs (what the published `edits` cells sent)."""
    assert gen.pick_samples(REC, BRIEF, "edits_nopairs", 0) == gen.pick_samples(REC, BRIEF, "edits", 0)
    u = user_prompt("edits_nopairs", template)
    assert "Their edit:" not in u
    assert all(s in u for s in gen.pick_samples(REC, BRIEF, "edits", 0))
