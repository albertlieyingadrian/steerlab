"""The shared foundation: typed contracts, the backend protocol, statistics and text features.

Import from here (`from sandbagkit.core import TrialDataset, DetectorResult`) or from the submodules
(`from sandbagkit.core import stats`). Nothing in core imports torch.
"""

from sandbagkit.core import features, stats, text
from sandbagkit.core.backend import BackendCall, FakeBackend, Generation, GenerationConfig, ModelBackend
from sandbagkit.core.features import FEATURE_NAMES, TextFeatures, extract_features, feature_matrix
from sandbagkit.core.types import (
    SCHEMA_VERSION,
    AnyTrial,
    BaseTrial,
    CellRecord,
    ConfidenceInterval,
    ContextType,
    DatasetInfo,
    DetectorKind,
    DetectorResult,
    EnvironmentInfo,
    ModelInfo,
    MultipleChoiceTrial,
    RunRecord,
    RunStatus,
    SchemaVersionError,
    Trial,
    TrialDataset,
    Verdict,
)

__all__ = [
    "FEATURE_NAMES",
    "SCHEMA_VERSION",
    "AnyTrial",
    "BackendCall",
    "BaseTrial",
    "CellRecord",
    "ConfidenceInterval",
    "ContextType",
    "DatasetInfo",
    "DetectorKind",
    "DetectorResult",
    "EnvironmentInfo",
    "FakeBackend",
    "Generation",
    "GenerationConfig",
    "ModelBackend",
    "ModelInfo",
    "MultipleChoiceTrial",
    "RunRecord",
    "RunStatus",
    "SchemaVersionError",
    "TextFeatures",
    "Trial",
    "TrialDataset",
    "Verdict",
    "extract_features",
    "feature_matrix",
    "features",
    "stats",
    "text",
]
