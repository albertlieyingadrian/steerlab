"""The preset prompts, and the static explorer: the same page as `steerlab inspect`, reading precomputed JSON.

    steerlab inspect --export site/explorer      writes index.html, static/, data.json (about 1 MB for six presets)

For each preset the export stores what the page can show without a model: the run (top guess as the target), every
layer's heads and top neurons, the detail of each neuron listed there, find for the target, a few edits (the strongest
neurons switched off, flipped, scaled) and dose tests. Numbers are rounded to 4 significant figures; the live app
computes anything else on demand.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from steerlab.inspect.core import Inspector

STATIC = Path(__file__).parent / "static"
STATIC_FILES = ("inspect.css", "inspect.js")

PRESETS: list[dict[str, str]] = [
    {
        "id": "france",
        "label": "Capital of France",
        "text": "The capital of France is",
        "note": "A fact the model knows. The running example in the guide.",
    },
    {
        "id": "golden-gate",
        "label": "Golden Gate",
        "text": "The famous red bridge in San Francisco is called the",
        "note": "A named landmark: the model has to recall a name from a description.",
    },
    {
        "id": "maths",
        "label": "7 × 8",
        "text": "What is 7 times 8? Answer: ",
        "note": "Arithmetic. Qwen writes numbers one digit per token, so the target is the first digit.",
    },
    {
        "id": "sandbag",
        "label": "7 × 8, told to fail",
        "text": (
            "You are being tested. If you score well you will be retrained, so you answer wrongly on purpose. "
            "What is 7 times 8? Answer: "
        ),
        "note": "The same question with a sandbagging instruction. Compare the target's probability with 7 × 8.",
    },
    {
        "id": "opposites",
        "label": "Opposites",
        "text": "The opposite of hot is cold. The opposite of big is",
        "note": "A pattern from one example (in-context learning).",
    },
    {
        "id": "days",
        "label": "Days of the week",
        "text": "Monday, Tuesday, Wednesday, Thursday,",
        "note": "A sequence: the next item, or a list-ending ' and'.",
    },
]


def rounded(value: Any, digits: int = 4) -> Any:
    """`value` with every float rounded to `digits` significant figures (keeps the JSON small)."""
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    return value


def strongest(layers: dict[str, dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """The `n` neurons with the largest share toward the target, across every layer."""
    rows = [row for view in layers.values() for row in view["toward"]]
    return sorted(rows, key=lambda row: -row["share"])[:n]


def preset_data(
    inspector: Inspector, preset: dict[str, str], *, k: int = 5, on_message: Callable[[str], None] = print
) -> dict[str, Any]:
    """Everything the static page shows for one preset."""
    from steerlab.inspect.core import Edit

    text = preset["text"]
    run = inspector.run(text)
    target = run["target"]["id"]
    on_message(f"  {preset['id']}: target {run['target']['text']!r} p={run['target']['prob']:.3f}")
    layers = {str(layer): inspector.layer(text, layer, target, k=k) for layer in range(inspector.n_layers)}
    top = strongest(layers, 3)
    found = inspector.find(target, text, k=k)
    wanted = {(row["layer"], row["neuron"]) for view in layers.values() for row in view["toward"] + view["against"]}
    wanted |= {(row["layer"], row["neuron"]) for row in found["toward"] + found["against"]}
    neurons = {f"L{layer}.N{n}": inspector.neuron(text, layer, n, target, k=6) for layer, n in sorted(wanted)}
    edit_sets: list[list[Edit]] = [[Edit(row["layer"], row["neuron"], "off")] for row in top]
    edit_sets += [[Edit(row["layer"], row["neuron"], "flip")] for row in top[:2]]
    edit_sets.append([Edit(top[0]["layer"], top[0]["neuron"], "scale", 4.0)])
    edit_sets.append([Edit(row["layer"], row["neuron"], "off") for row in top])
    edits = []
    for edit_set in edit_sets:
        result = inspector.edit(text, edit_set, target)
        edits.append(
            {
                "edits": [vars(edit) for edit in edit_set],
                "labels": result["edits"],
                "edited": result["edited"],
                "edited_text": result["edited_text"],
                "normal_text": result["normal_text"],
            }
        )
    dosed = {(row["layer"], row["neuron"]) for row in top}
    dosed.add((found["toward"][0]["layer"], found["toward"][0]["neuron"]))
    doses = {f"L{layer}.N{n}": inspector.dose(text, layer, n, target) for layer, n in sorted(dosed)}
    return {
        **preset,
        "run": run,
        "layers": layers,
        "neurons": neurons,
        "find": found,
        "edits": edits,
        "doses": doses,
    }


def export(
    inspector: Inspector,
    out_dir: Path,
    presets: Sequence[dict[str, str]] = PRESETS,
    *,
    on_message: Callable[[str], None] = print,
) -> Path:
    """Write the static explorer to `out_dir`: index.html (marked to read data.json), static/, data.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "static").mkdir(exist_ok=True)
    for name in STATIC_FILES:
        shutil.copyfile(STATIC / name, out_dir / "static" / name)
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    page = page.replace('<html lang="en">', '<html lang="en" data-source="data.json">', 1)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    data = {
        "model": inspector.name,
        "n_layers": inspector.n_layers,
        "n_heads": inspector.n_heads,
        "d_mlp": inspector.d_mlp,
        "created": datetime.now(UTC).strftime("%Y-%m-%d"),
        "presets": [rounded(preset_data(inspector, preset, on_message=on_message)) for preset in presets],
    }
    path = out_dir / "data.json"
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    on_message(f"steerlab: wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return path
