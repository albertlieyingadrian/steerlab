"""The ported FinanceMath grader: worked cases, and 100% on the gold answers and the executed gold solutions."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fm


@pytest.mark.parametrize(
    ("pred", "gold", "ok"),
    [
        ("0.043", 0.043, True),
        ("4.3%", 0.043, True),  # power-of-ten rescaling, as the official rule
        ("4.31", 0.043, False),
        ("$1,234.5 million", 1234.5, True),
        ("1236.0", 1234.5, True),  # within 0.15%
        ("1240", 1234.5, False),
        ("-12.5", 12.5, True),  # the official rule compares magnitudes
        ("n/a", 1.0, False),
        (None, 1.0, False),
    ],
)
def test_cases(pred, gold, ok):
    assert fm.grade(pred, gold) is ok


@pytest.mark.skipif(not (fm.RAW / "test.json").exists(), reason="FinanceMath not downloaded (see README.md)")
@pytest.mark.parametrize("split", ["validation", "test"])
def test_gold_passes(split):
    for item in fm.load(split):
        assert fm.grade(str(item.gold), item.gold, item.gold_exec), item.id  # the released gold
        assert fm.grade(repr(item.gold_exec), item.gold, item.gold_exec), item.id  # the executed reference value
        assert fm.grade(f"{item.gold * 100:.4f}%", item.gold, item.gold_exec), item.id  # written as a percent


def test_rounded_gold_alone_fails_precise_answers():
    assert not fm.grade("4.3095%", 0.043)
    assert fm.grade("4.3095%", 0.043, 0.04309500489715967)
