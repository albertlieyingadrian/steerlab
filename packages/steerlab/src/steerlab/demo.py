"""The demo set that ships with steerlab: tuned directions, the English-only prompts, and per-model presets.

The autoresearch study (docs/executions/2026-10-26-autoresearch-bigger-models.md) found, for each bigger model, the
layer where the `chinese` direction beats an English-only prompt most cleanly, and the strengths that work. Those
directions are small (a few tens of kilobytes each), so they live in the package under `data/demo/`: `steerlab serve`
can then steer Qwen3-8B, Llama-3.1-8B and both Gemma-2 models out of the box, without building anything first.

    data/demo/prompts.json                         S2, S3, S4: the English-only system prompts
    data/demo/presets.json                         per model: which prompt, the fader strengths that worked, features
    data/demo/directions/<model key>/<concept>     the tuned directions (sandbagkit .npz + .json)

Light module: no torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

DEMO_DIR = Path(__file__).parent / "data" / "demo"
DIRECTIONS_ROOT = DEMO_DIR / "directions"
DEFAULT_PROMPT = "S4"


@dataclass(frozen=True)
class PresetFeature:
    """An SAE feature put on the desk at load. `mode` "clamp" holds its activation at `suggest` instead of adding."""

    id: int
    label: str
    suggest: float
    mode: str = "add"


@dataclass(frozen=True)
class Preset:
    """What worked for one model: the prompt to start with, a strength per direction, and SAE features to show."""

    prompt: str
    suggest: dict[str, float] = field(default_factory=dict)
    features: tuple[PresetFeature, ...] = ()
    score: str | None = None


@cache
def prompts() -> dict[str, str]:
    """The English-only system prompts by name (S2, S3, S4)."""
    data = json.loads((DEMO_DIR / "prompts.json").read_text(encoding="utf-8"))
    return {name: str(entry["text"]) for name, entry in data.items()}


@cache
def presets() -> dict[str, Preset]:
    """The preset for each model key that has one."""
    data = json.loads((DEMO_DIR / "presets.json").read_text(encoding="utf-8"))
    return {
        key: Preset(
            prompt=str(entry["prompt"]),
            suggest={name: float(value) for name, value in entry.get("suggest", {}).items()},
            features=tuple(
                PresetFeature(int(item["id"]), str(item["label"]), float(item["suggest"]), str(item.get("mode", "add")))
                for item in entry.get("features", [])
            ),
            score=entry.get("score"),
        )
        for key, entry in data.items()
        if not key.startswith("_")
    }


def prompt_for(key: str) -> str:
    """The system prompt that worked best for `key`, else the default (S4)."""
    preset = presets().get(key)
    return prompts()[preset.prompt if preset is not None else DEFAULT_PROMPT]
