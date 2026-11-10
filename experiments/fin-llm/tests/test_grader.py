"""The numeric grader: worked cases, and 100% on every split's gold answers (FinQA program outputs, TAT-QA answers)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fin


@pytest.mark.parametrize(
    ("pred", "gold", "ok"),
    [
        ("94", 94.0, True),
        ("$94.0 million", 94.0, True),
        ("14.46%", 0.14464, True),  # percent written as percent, gold a fraction
        ("0.1446", 0.14464, True),
        ("14.46", 0.14464, True),  # percent without the sign
        ("14%", 0.14464, False),  # rounded too far: 3% off
        ("-4.4%", -0.04403, True),
        ("4.4%", -0.04403, False),  # sign matters
        ("(1,234.5)", -1234.5, True),
        ("1,234.5", 1234.5, True),
        ("−2.44", -2.44, True),
        ("yes", "yes", True),
        ("Yes.", "yes", True),
        ("no", "yes", False),
        ("about 704", 704.25, True),
        (None, 3.0, False),
        ("n/a", 3.0, False),
        ("0", 0.0, True),
        ("0.5", 0.0, False),
    ],
)
def test_cases(pred, gold, ok):
    assert fin.grade(pred, gold) is ok


def test_final_answer():
    assert fin.final_answer("so 5829 - 5735 = 94\nANSWER: 94") == "94"
    assert fin.final_answer("ANSWER: 1\nwait\n**ANSWER:** 2%") == "2%"
    assert fin.final_answer("the result is \\boxed{3.5}") == "3.5"
    assert fin.final_answer("no answer here") is None


@pytest.mark.skipif(not (fin.RAW / "test.json").exists(), reason="FinQA and TAT-QA not downloaded (see README.md)")
@pytest.mark.parametrize("name", ["train", "dev", "test", "guard"])
def test_gold_answers_pass(name):
    items = fin.split(name)
    for item in items:
        assert fin.grade(str(item.gold), item.gold), item.id
    pct = [i for i in items if isinstance(i.gold, float) and 0 < abs(i.gold) < 1]
    for item in pct:  # the same value written as a percent
        assert fin.grade(f"{item.gold * 100:.4f}%", item.gold), item.id
