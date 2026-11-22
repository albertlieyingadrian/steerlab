"""The confirm folds share no scene and no image, and a regression in the split column stops the run."""

import pandas as pd
import pytest

from heads import confirm_split


def pairs(rows):
    return pd.DataFrame(rows, columns=["scene", "a", "b", "split"])


CLEAN = [(1, 10, 11, "dev"), (2, 20, 21, "A"), (2, 20, 22, "A"), (3, 30, 31, "B"), (4, 40, 41, "A")]


def test_clean_split_is_disjoint_and_skips_dev():
    for trf, tef in (("A", "B"), ("B", "A")):
        tr, te = confirm_split(pairs(CLEAN), trf, tef)
        assert set(tr.split) == {trf} and set(te.split) == {tef}
        assert not set(tr.scene) & set(te.scene)


def test_scene_in_both_folds_fails():
    with pytest.raises(AssertionError, match="scene"):
        confirm_split(pairs([*CLEAN, (2, 23, 24, "B")]), "A", "B")


def test_image_in_both_folds_fails():
    with pytest.raises(AssertionError, match="image"):
        confirm_split(pairs([*CLEAN, (5, 20, 50, "B")]), "A", "B")


def test_dev_is_never_a_confirm_fold():
    with pytest.raises(AssertionError):
        confirm_split(pairs(CLEAN), "dev", "A")
