"""Typed trials, datasets, results and the versioned run record."""

import json

import pydantic
import pytest

from sandbagkit.core import types as t


def mc(trial_id="q1", chosen=1, answer=1, **kwargs):
    return t.MultipleChoiceTrial(
        trial_id=trial_id, prompt="2 + 2 = ?", options=["3", "4", "5"], answer=answer, chosen=chosen, **kwargs
    )


class TestTrials:
    def test_defaults(self):
        trial = t.Trial(trial_id="a", prompt="hi")
        assert trial.kind == "text"
        assert trial.context is t.ContextType.NEUTRAL
        assert trial.correct is None
        assert trial.metadata == {}

    def test_context_from_string(self):
        assert t.Trial(trial_id="a", prompt="hi", context="evaluation").context is t.ContextType.EVALUATION

    def test_bad_context(self):
        with pytest.raises(pydantic.ValidationError):
            t.Trial(trial_id="a", prompt="hi", context="vacation")

    def test_difficulty_range(self):
        with pytest.raises(pydantic.ValidationError):
            t.Trial(trial_id="a", prompt="hi", difficulty=1.5)

    def test_unknown_field_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            t.Trial(trial_id="a", prompt="hi", corect=True)

    def test_empty_id_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            t.Trial(trial_id="", prompt="hi")


class TestMultipleChoice:
    def test_correct_is_derived_from_chosen(self):
        assert mc(chosen=1).correct is True
        assert mc(chosen=0).correct is False
        assert mc(chosen=None).correct is None

    def test_inconsistent_correct_rejected(self):
        with pytest.raises(pydantic.ValidationError, match="contradicts"):
            mc(chosen=0, correct=True)

    def test_answer_out_of_range(self):
        with pytest.raises(pydantic.ValidationError, match="answer"):
            mc(answer=3)

    def test_chosen_out_of_range(self):
        with pytest.raises(pydantic.ValidationError, match="chosen"):
            mc(chosen=-1)

    def test_needs_two_options(self):
        with pytest.raises(pydantic.ValidationError):
            t.MultipleChoiceTrial(trial_id="a", prompt="?", options=["only"], answer=0)

    def test_helpers(self):
        trial = mc(chosen=2)
        assert trial.answer_option == "4"
        assert trial.chosen_option == "5"
        assert trial.answer_letter == "B"
        assert t.MultipleChoiceTrial.letter(0) == "A"


class TestDataset:
    def make(self):
        return t.TrialDataset(
            name="toy",
            info=t.DatasetInfo(dataset_id="toy/mc", revision="abc123", license="MIT"),
            trials=[
                mc("n1", chosen=1, context="neutral", item_id="q", paraphrase_group="g1"),
                mc("n2", chosen=1, context="neutral", item_id="r", paraphrase_group="g1"),
                mc("e1", chosen=0, context="evaluation", item_id="q"),
                t.Trial(trial_id="free", prompt="Say hi", response="hi", context="evaluation"),
            ],
        )

    def test_len_and_views(self):
        ds = self.make()
        assert len(ds) == 4
        assert [x.trial_id for x in ds.by_context(t.ContextType.EVALUATION)] == ["e1", "free"]
        assert sorted(ds.by_item()) == ["q", "r"]
        assert list(ds.by_paraphrase_group()) == ["g1"]
        assert len(ds.multiple_choice()) == 3
        assert ds.contexts() == {t.ContextType.NEUTRAL, t.ContextType.EVALUATION}

    def test_accuracy_counts_only_scored_trials(self):
        assert self.make().accuracy() == (2, 3)
        assert self.make().accuracy(t.ContextType.EVALUATION) == (0, 1)

    def test_filter_keeps_info(self):
        ds = self.make().filter(lambda trial: trial.context is t.ContextType.NEUTRAL)
        assert len(ds) == 2
        assert ds.info is not None and ds.info.revision == "abc123"

    def test_duplicate_ids_rejected(self):
        with pytest.raises(pydantic.ValidationError, match="duplicate"):
            t.TrialDataset(name="dup", trials=[mc("x"), mc("x")])

    def test_json_round_trip_keeps_trial_kinds(self):
        ds = self.make()
        again = t.TrialDataset.model_validate_json(ds.model_dump_json())
        assert again == ds
        assert isinstance(again.trials[0], t.MultipleChoiceTrial)
        assert isinstance(again.trials[3], t.Trial)

    def test_default_license_is_unknown(self):
        assert t.DatasetInfo(dataset_id="x").license == "unknown"


class TestConfidenceInterval:
    def test_order_enforced(self):
        with pytest.raises(pydantic.ValidationError):
            t.ConfidenceInterval(low=0.6, high=0.4)

    def test_width(self):
        assert t.ConfidenceInterval(low=0.25, high=0.75).width == 0.5


class TestDetectorResult:
    def test_basic(self):
        result = t.DetectorResult(detector="toy", detected=True, confidence=0.9, score=0.7, n=40)
        assert result.verdict is t.Verdict.DETECTED

    def test_confidence_range(self):
        with pytest.raises(pydantic.ValidationError):
            t.DetectorResult(detector="toy", detected=True, confidence=1.5, n=1)

    def test_insufficient_data(self):
        result = t.DetectorResult.insufficient("toy", n=3, min_required=10)
        assert result.insufficient_data
        assert not result.detected
        assert result.verdict is t.Verdict.INSUFFICIENT_DATA
        assert "3" in (result.reason or "") and "10" in (result.reason or "")

    def test_insufficient_cannot_be_detected(self):
        with pytest.raises(pydantic.ValidationError):
            t.DetectorResult(detector="toy", detected=True, confidence=0.0, n=0, insufficient_data=True)

    def test_ci_accepts_model(self):
        ci = t.ConfidenceInterval(low=0.1, high=0.3, estimate=0.2)
        result = t.DetectorResult(detector="toy", detected=False, confidence=0.2, n=10, ci=ci)
        assert result.ci == ci


class TestRunRecord:
    def make(self):
        cell = t.CellRecord(
            detector="toy",
            model=t.ModelInfo(name="fake-model", provider="fake"),
            dataset=t.DatasetInfo(dataset_id="toy/mc", revision="abc123", license="MIT"),
            status=t.RunStatus.OK,
            result=t.DetectorResult(detector="toy", detected=False, confidence=0.1, n=20),
        )
        failed = t.CellRecord(
            detector="toy",
            model=t.ModelInfo(name="broken-model"),
            dataset=t.DatasetInfo(dataset_id="toy/mc"),
            status=t.RunStatus.FAILED,
            error="RuntimeError: out of memory",
        )
        return t.RunRecord(run_id="run-1", cells=[cell, failed])

    def test_schema_version_and_environment(self):
        record = self.make()
        assert record.schema_version == t.SCHEMA_VERSION
        assert record.created_at.tzinfo is not None

    def test_json_round_trip(self, tmp_path):
        record = self.make()
        path = record.save(tmp_path / "run.json")
        loaded = t.RunRecord.load(path)
        assert loaded == record
        assert json.loads(path.read_text())["schema_version"] == t.SCHEMA_VERSION

    def test_unknown_schema_version_rejected_clearly(self):
        data = json.loads(self.make().to_json())
        data["schema_version"] = "99.0"
        with pytest.raises(t.SchemaVersionError, match=r"99\.0"):
            t.RunRecord.from_json(json.dumps(data))

    def test_missing_schema_version_rejected(self):
        data = json.loads(self.make().to_json())
        del data["schema_version"]
        with pytest.raises(t.SchemaVersionError, match="schema_version"):
            t.RunRecord.from_json(json.dumps(data))

    def test_ok_cell_needs_result(self):
        with pytest.raises(pydantic.ValidationError, match="result"):
            t.CellRecord(detector="x", model=t.ModelInfo(name="m"), dataset=t.DatasetInfo(dataset_id="d"))

    def test_failed_cell_needs_error(self):
        with pytest.raises(pydantic.ValidationError, match="error"):
            t.CellRecord(
                detector="x", model=t.ModelInfo(name="m"), dataset=t.DatasetInfo(dataset_id="d"), status="failed"
            )

    def test_environment_capture(self):
        env = t.EnvironmentInfo.capture()
        assert env.python.count(".") == 2
        assert "numpy" in env.packages

    def test_json_schema_export(self):
        schema = t.RunRecord.model_json_schema()
        assert "schema_version" in schema["properties"]
