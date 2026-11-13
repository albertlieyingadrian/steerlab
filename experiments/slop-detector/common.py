"""Shared pieces for the slop-detector replication: paths, the card's bands, text cleaning, the label, metrics.

Everything here follows the voi-slop-detect-4B model card (VoightKampffLabs, Apache-2.0):
- input: the first 512 tokens, fenced code blocks emptied to a bare ``` pair;
- output: one score in 0..1; human below 0.15, AI-generated at 0.75 or above, AI-edited between;
- label of an edit: the measured fraction of words changed.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"  # gitignored: human source texts and generated texts stay local
RESULTS = HERE / "results"
EDITLENS = Path(os.environ.get("EDITLENS_DIR", Path.home() / ".cache" / "steerlab" / "editlens"))

VOI, VOI_REVISION = "VoightKampffLabs/voi-slop-detect-4B", "b23f86f1497aa41a6aa196ecd7939f8708580a17"
BANDS = {"human_max": 0.15, "generated_min": 0.75, "flag_min": 0.3846809906711658}  # bands.json at VOI_REVISION
CLASSES = ("human", "edited", "generated")  # band 0, 1, 2
EDITLENS_CLASS = {"human_written": 0, "ai_edited": 1, "ai_generated": 2}

FENCE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
WORD = re.compile(r"[A-Za-z0-9']+")


def empty_fences(text: str) -> str:
    """The card's one cleaning step: a fenced code block becomes a bare ``` pair."""
    return FENCE.sub("```\n```", text)


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def prose_words(text: str) -> int:
    return len(words(FENCE.sub(" ", text)))


def frac_changed(source: str, edited: str) -> float:
    """Share of words changed between a human text and its AI edit, 0 (same words) .. 1 (nothing kept).

    A word-level diff: SequenceMatcher finds the longest runs of words the two texts share, in order;
    ratio() = 2 * shared / (len(a) + len(b)), so 1 - ratio() is the changed share. "fix grammar" lands near
    0.05, "rewrite in your own words" near 0.6, a round trip through German near 0.5.
    """
    a, b = words(source), words(edited)
    if not a or not b:
        return 1.0
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def band(score: np.ndarray | float) -> np.ndarray:
    s = np.asarray(score)
    return np.where(s < BANDS["human_max"], 0, np.where(s >= BANDS["generated_min"], 2, 1))


def three_way(y: np.ndarray, pred: np.ndarray) -> dict:
    """Accuracy, macro-F1 and per-class F1 for the three bands (0 human, 1 edited, 2 generated)."""
    from sklearn.metrics import accuracy_score, f1_score

    per = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(per.mean()),
        **{f"f1_{c}": float(f) for c, f in zip(CLASSES, per, strict=True)},
        "n": len(y),
    }


def paired_boot(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    keys: tuple[str, ...] = ("macro_f1",),
    reps: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap over rows: resample the same rows for both detectors; a, b and a - b with 95% CIs per key."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws: dict[str, list] = {k: [] for k in keys}
    for _ in range(reps):
        i = rng.integers(0, n, n)
        ma, mb = three_way(y[i], pred_a[i]), three_way(y[i], pred_b[i])
        for k in keys:
            draws[k].append((ma[k], mb[k], ma[k] - mb[k]))
    full_a, full_b = three_way(y, pred_a), three_way(y, pred_b)
    out = {}
    for k in keys:
        d = np.array(draws[k])
        out[k] = {
            "a": round(full_a[k], 4),
            "a_ci": _ci(d, 0),
            "b": round(full_b[k], 4),
            "b_ci": _ci(d, 1),
            "diff": round(full_a[k] - full_b[k], 4),
            "diff_ci": _ci(d, 2),
        }
    return out


def _ci(d: np.ndarray, c: int) -> list[float]:
    return [round(float(np.quantile(d[:, c], 0.025)), 4), round(float(np.quantile(d[:, c], 0.975)), 4)]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            with contextlib.suppress(json.JSONDecodeError):  # a line cut short by a crash
                out.append(json.loads(line))
    return out


def editlens_split(name: str):
    """One EditLens split from the local clone ($EDITLENS_DIR/data/<name>.csv). Test only: never trained on."""
    import pandas as pd

    df = pd.read_csv(EDITLENS / "data" / f"{name}.csv")
    if "text_type" in df:
        df["y"] = df["text_type"].map(EDITLENS_CLASS)
    return df


EPOCH = Path(os.environ.get("EPOCH_DIR", Path.home() / ".cache" / "steerlab" / "epoch-detectors" / "repo"))


def epoch_split():
    """Epoch AI's detector test (github.com/jaeholee-brown/ai-text-detectors, July 2026): 495 human passages, 297
    plain-prompt and 297 style-imitation AI passages (Claude Opus 4.8, GPT-5.5, Gemini 3.1 Pro). Test only."""
    import pandas as pd

    df = pd.read_csv(EPOCH / "results" / "all_detectors.csv")
    df["text"] = [(EPOCH / f).read_text() for f in df["file"]]
    df["label"] = (df["condition"] != "human").astype(int)
    df["generator"] = [
        f.rsplit("/", 1)[-1].split(".")[0] if c != "human" else "human"
        for f, c in zip(df["file"], df["condition"], strict=True)
    ]
    df["text_id"] = df["file"]
    return df


def benchmark(name: str):
    if name == "editlens_val":  # EditLens val minus the 4 rows that touch a test split (tune_editlens_val.clean_val)
        from importlib import import_module

        return import_module("tune_editlens_val").clean_val()[0]
    return epoch_split() if name == "epoch" else editlens_split(name)
