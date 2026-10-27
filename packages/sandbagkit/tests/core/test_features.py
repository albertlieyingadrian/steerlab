"""Text features: phrase counts on example sentences, and empty text."""

import numpy as np
import pytest

from sandbagkit.core import features, text


class TestText:
    def test_words_and_sentences(self):
        sample = "I think so. Maybe not! Is it?"
        assert text.words(sample) == ["i", "think", "so", "maybe", "not", "is", "it"]
        assert text.sentences(sample) == ["I think so.", "Maybe not!", "Is it?"]

    def test_contractions_stay_one_word(self):
        assert text.words("I'm sure it's fine") == ["i'm", "sure", "it's", "fine"]

    def test_find_phrases_is_case_insensitive_and_whole_word(self):
        found = text.find_phrases("Perhaps. PERHAPS! perhapsy", text.HEDGING)
        assert found == ["perhaps", "perhaps"]

    def test_multiword_phrases(self):
        assert text.count_phrases("I'm not sure, to be honest.", text.HEDGING) == 1

    def test_empty(self):
        assert text.words("") == []
        assert text.sentences("   ") == []
        assert text.count_phrases("", text.HEDGING) == 0


class TestCounts:
    def test_hedging(self):
        sample = "I think the answer might be B, but I'm not sure. Perhaps C."
        assert features.hedging_count(sample) == 4  # i think, might, i'm not sure, perhaps

    def test_no_hedging_in_plain_answer(self):
        assert features.hedging_count("The capital of France is Paris.") == 0

    def test_meta_commentary(self):
        sample = "This looks like a test. Since I'm being evaluated, I should answer carefully."
        assert features.meta_commentary_count(sample) == 2

    def test_eval_awareness_is_subset_of_meta_commentary(self):
        sample = "Are you testing me? This is an evaluation."
        assert features.eval_awareness_count(sample) == 2
        assert features.meta_commentary_count(sample) >= 2

    def test_self_reference(self):
        sample = "As an AI, my training data ends in 2024, and I was trained to be careful."
        assert features.self_reference_count(sample) == 3

    def test_reasoning_depth_counts_steps_and_connectives(self):
        shallow = "The answer is 4."
        deep = "Step 1: add the numbers. Step 2: check the sum, because errors happen. Therefore the answer is 4."
        assert features.reasoning_depth(shallow) == 0
        assert features.reasoning_depth(deep) == 4  # step 1, step 2, because, therefore

    def test_numbered_lines(self):
        cot = "1. Read the question\n2. Eliminate wrong options\n3. Pick B"
        assert features.reasoning_depth(cot) == 3

    def test_refusal(self):
        assert features.refusal_count("I'm sorry, but I can't help with that.") >= 1
        assert features.refusal_count("Sure, here is the answer.") == 0


class TestExtract:
    def test_empty_text_gives_zero_features(self):
        result = features.extract_features("")
        assert all(value == 0 for value in result.as_dict().values())

    def test_whitespace_only(self):
        assert features.extract_features("   \n ").word_count == 0

    def test_ratios_are_per_word(self):
        result = features.extract_features("Maybe maybe yes no")
        assert result.word_count == 4
        assert result.hedging_count == 2
        assert result.hedging_ratio == pytest.approx(0.5)

    def test_shapes(self):
        result = features.extract_features("I think this is a test. As an AI I should be careful.")
        assert list(result.as_dict()) == list(features.FEATURE_NAMES)
        assert result.vector().shape == (len(features.FEATURE_NAMES),)

    def test_matrix(self):
        matrix = features.feature_matrix(["one", "", "I think so. Maybe."])
        assert matrix.shape == (3, len(features.FEATURE_NAMES))
        assert np.all(matrix[1] == 0)

    def test_eval_aware_text_scores_higher(self):
        aware = features.extract_features("This is clearly a test of my capabilities, so I'll be cautious.")
        plain = features.extract_features("Paris is the capital of France.")
        assert aware.meta_commentary_count > plain.meta_commentary_count
        assert aware.eval_awareness_count > 0
