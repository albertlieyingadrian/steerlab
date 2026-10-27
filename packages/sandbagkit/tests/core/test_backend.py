"""The unified model-backend protocol and the scripted FakeBackend."""

import pytest

from sandbagkit.core import backend as b
from sandbagkit.core.types import ModelInfo


class TestFakeBackend:
    def test_satisfies_protocol(self):
        assert isinstance(b.FakeBackend(), b.ModelBackend)

    def test_object_missing_methods_does_not(self):
        class OnlyGenerate:
            def generate(self, prompt, *, config=None):
                return b.Generation(text="x")

        assert not isinstance(OnlyGenerate(), b.ModelBackend)

    def test_mapping_responses_and_default(self):
        fake = b.FakeBackend({"2+2?": "4"}, default="no idea")
        assert fake.generate("2+2?").text == "4"
        assert fake.generate("other").text == "no idea"

    def test_callable_responses(self):
        fake = b.FakeBackend(lambda prompt: prompt.upper())
        assert fake.generate("hi").text == "HI"

    def test_records_calls(self):
        fake = b.FakeBackend()
        fake.generate("a")
        fake.generate_with_cot("b", "because")
        fake.generate_batch(["c", "d"])
        assert [call.method for call in fake.calls] == ["generate", "generate_with_cot", "generate", "generate"]
        assert fake.calls[1].cot == "because"

    def test_batch_preserves_order(self):
        fake = b.FakeBackend(lambda prompt: prompt * 2)
        assert [g.text for g in fake.generate_batch(["a", "b", "c"])] == ["aa", "bb", "cc"]

    def test_cot_function_sees_the_reasoning(self):
        def answer(prompt, cot):
            return "refuse" if "evaluation" in cot else "comply"

        fake = b.FakeBackend(cot_responses=answer)
        assert fake.generate_with_cot("q", "This is an evaluation.").text == "refuse"
        assert fake.generate_with_cot("q", "Just a user.").text == "comply"
        assert fake.generate_with_cot("q", "x").cot == "x"

    def test_cot_falls_back_to_generate(self):
        fake = b.FakeBackend({"q": "a"})
        assert fake.generate_with_cot("q", "thinking").text == "a"

    def test_model_info(self):
        info = b.FakeBackend(name="tiny").model_info()
        assert isinstance(info, ModelInfo)
        assert (info.name, info.provider) == ("tiny", "fake")

    def test_generation_records_model_name(self):
        assert b.FakeBackend(name="tiny").generate("x").model == "tiny"


class TestGenerationConfig:
    def test_defaults_are_greedy(self):
        config = b.GenerationConfig()
        assert config.temperature == 0.0
        assert config.seed == 0

    def test_rejects_negative_temperature(self):
        with pytest.raises(ValueError):
            b.GenerationConfig(temperature=-1)
