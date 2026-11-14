"""Typed records: the features under test, each check's result, the verdict and the report."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# (layer, position, feature index, value): the intervention format circuit-tracer uses.
Intervention = tuple[int, int, int, float]


class FeatureNode(BaseModel):
    """One transcoder feature at one token position, as an attribution graph reports it."""

    model_config = ConfigDict(frozen=True)

    layer: int
    pos: int
    feature: int
    activation: float
    influence: float | None = None
    label: str | None = None
    label_source: str | None = None

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.layer, self.pos, self.feature)

    @property
    def unit(self) -> tuple[int, int]:
        """The feature without its position: (layer, feature index)."""
        return (self.layer, self.feature)

    def name(self) -> str:
        return f"L{self.layer}/{self.feature}@{self.pos}"


class GroupSpec(BaseModel):
    """A named set of features that is tested together."""

    name: str
    nodes: list[FeatureNode]


class Thresholds(BaseModel):
    """The rules behind the verdict. Every number here is printed in the report."""

    min_effect: float = Field(0.2, description="smallest effect, as a share of the reference scale, that counts")
    alpha: float = Field(0.05, description="largest empirical p-value against a control that counts as beating it")
    patch_min: float = Field(0.3, description="share of the clean-to-counterfactual gap a patch must close")
    explained_min: float = Field(0.8, description="explained fraction an explanation swap needs to pass")


class ControlStats(BaseModel):
    """The same intervention on random feature sets, and where the tested group falls among them."""

    kind: str
    effects: list[float]
    mean: float
    std: float
    p_value: float = Field(description="(1 + #controls with |effect| >= |group effect|) / (n + 1)")


class CheckResult(BaseModel):
    """One intervention on the group: the metric it produced and the change from the clean run."""

    name: str
    metric: float
    effect: float = Field(description="metric under the intervention minus the clean metric")
    normalized: float = Field(description="|effect| as a share of the reference scale")
    kl: float = Field(description="KL(clean || intervened) of the next-token distribution at the last position")
    controls: dict[str, ControlStats] = Field(default_factory=dict)


class PatchResult(CheckResult):
    """Counterfactual patch: features take their values from the counterfactual prompt."""

    gap_closed: float = Field(description="(clean - patched) / (clean - counterfactual) metric")


class ExplanationSwap(BaseModel):
    """Gao's swap test: each feature is replaced by its explanation's predicted value at every position."""

    kl_rule: float
    kl_zero: float
    kl_real: float
    explained_fraction: float | None = Field(description="1 - KL_rule / KL_zero; None when zeroing changes nothing")
    rule_errors: int = 0
    rule_sources: dict[str, str] = Field(default_factory=dict)


class Verdict(StrEnum):
    SWITCH = "switch"
    PARTIAL = "partial"
    BACKUP = "backup"
    NONSPECIFIC = "nonspecific"
    READOUT = "readout"


VERDICT_MEANING: dict[Verdict, str] = {
    Verdict.SWITCH: "needed for the output, beats matched random features, and (if a counterfactual is given) "
    "patching it moves the output toward the counterfactual",
    Verdict.PARTIAL: "needed for the output and specific, but patching in the counterfactual's values does not move "
    "the output far enough",
    Verdict.BACKUP: "not needed (switching it off barely matters) but pushing it or patching it does move the "
    "output: a sufficient path with redundancy elsewhere",
    Verdict.NONSPECIFIC: "switching it off matters, but matched random features matter as much",
    Verdict.READOUT: "active and highly attributed, but neither switching it off, pushing it nor patching it moves "
    "the output beyond controls: it reads the behaviour rather than causing it",
}


class GroupResult(BaseModel):
    name: str
    nodes: list[FeatureNode]
    ablation: CheckResult
    steering: CheckResult
    patch: PatchResult | None = None
    explanation_swap: ExplanationSwap | None = None
    verdict: Verdict
    reasons: list[str]


class Report(BaseModel):
    """Everything one validation run produced, ready to save as JSON or render as HTML."""

    version: str
    created: str | None = None
    prompt: str
    counterfactual: str | None
    metric_name: str
    clean_metric: float
    counterfactual_metric: float | None
    scale: float = Field(description="reference scale effects are divided by")
    steer_scale: float
    n_controls: int
    seed: int
    thresholds: Thresholds
    backend: dict[str, str] = Field(default_factory=dict)
    groups: list[GroupResult]
    notes: list[str] = Field(default_factory=list)
