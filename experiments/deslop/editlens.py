"""EditLens (Thai et al., ICLR 2026): how much did AI edit this text? Measured on every text in the note.

    python editlens.py check    # reproduce the released model's ternary macro-F₁ on the EditLens test splits
    python editlens.py score    # score every held-out reply, human passage, voice rewrite and mirror

Model: `pangram/editlens_roberta-large` (RoBERTa-large, four buckets of AI editing; the score is the expected bucket
on a 0-1 scale). Text is cleaned and cut to 512 tokens exactly as the authors' inference script does; their cleaning
code is imported from a local clone of github.com/pangramlabs/EditLens ($EDITLENS_DIR, default
~/.cache/steerlab/editlens, which also holds their val and test splits) and is not copied here.

Licence: the EditLens model, code and data are CC BY-NC-SA 4.0, non-commercial. This script commits no EditLens data:
only metrics and the scores of this note's own texts. Nothing in the note was tuned against this score.

Thresholds: the authors' ternary protocol (scripts/eval/ternary_eval.py) calibrates two thresholds on the val split,
human vs the rest and AI-generated vs the rest, each maximising F₁ on min-max-scaled scores. Those two cut points,
converted back to raw scores, give the three classes used here: human, AI-edited, AI-generated. A third threshold
flags 1% of this note's 318 human passages, as for the other detectors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EDITLENS = Path(os.environ.get("EDITLENS_DIR", Path.home() / ".cache" / "steerlab" / "editlens"))
EXTRA = os.environ.get("EDITLENS_PYLIBS", str(Path.home() / ".cache" / "steerlab" / "pylibs"))  # the `emoji` package
OUT = HERE / "results" / "editlens"
CHECKPOINT, BASE = "pangram/editlens_roberta-large", "FacebookAI/roberta-large"
LABELS = {"human_written": 0, "ai_generated": 1, "ai_edited": 2}


class EditLens:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        sys.path[:0] = [EXTRA, str(EDITLENS / "scripts")]
        from preprocess import clean_text  # the authors' cleaning, from their repository

        self.clean = clean_text
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(BASE)
        self.model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT).to(self.device).eval()
        self.buckets = self.model.config.num_labels

    def score(self, texts: list[str], batch: int = 16) -> list[dict]:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                cleaned = [self.clean(t) for t in texts[i : i + batch]]
                enc = self.tok(cleaned, truncation=True, max_length=512, padding=True, return_tensors="pt")
                probs = torch.softmax(self.model(**enc.to(self.device)).logits.float(), -1).cpu().numpy()
                scores = probs @ np.arange(self.buckets) / (self.buckets - 1)
                out += [{"score": float(s), "bucket": int(p.argmax())} for s, p in zip(scores, probs, strict=True)]
                print(f"  {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
        print()
        return out


def ternary(val_scores: np.ndarray, val_labels: np.ndarray) -> tuple[float, float, float, float]:
    """The authors' calibration on val: (human cut, AI-generated cut) on scaled scores, and val's min and max."""
    sys.path[:0] = [str(EDITLENS)]
    from scripts.eval.ternary_eval import calibrate_thresholds

    lo, hi = float(val_scores.min()), float(val_scores.max())
    h, ai, _, _ = calibrate_thresholds(val_labels, (val_scores - lo) / (hi - lo))
    return h, ai, lo, hi


def evaluate(scores: np.ndarray, labels: np.ndarray, cuts: tuple[float, float, float, float]) -> dict:
    """The authors' test-time rule: min-max scale the split's own scores, then apply the val cut points."""
    sys.path[:0] = [str(EDITLENS)]
    from scripts.eval.ternary_eval import evaluate as their_evaluate
    from scripts.eval.ternary_eval import minmax_scale, predict_ternary

    h, ai, _, _ = cuts
    m = their_evaluate(labels, predict_ternary(minmax_scale(scores), h, ai))
    keys = ("macro_f1", "accuracy", "f1_human", "f1_ai_generated", "f1_ai_edited")
    return {**{k: round(float(m[k]), 4) for k in keys}, "n": len(labels)}


def _commit() -> str:
    head = (EDITLENS / ".git" / "HEAD").read_text().strip()
    return (EDITLENS / ".git" / head.split()[-1]).read_text().strip() if head.startswith("ref:") else head


def check() -> None:
    import pandas as pd

    model = EditLens()
    released = "editlens_roberta-large_score"
    frames = {s: pd.read_csv(EDITLENS / "data" / f"{s}.csv") for s in ("val", "test", "test_llama")}
    ours = {s: np.array([r["score"] for r in model.score(list(f["text"]))]) for s, f in frames.items()}
    labels = {s: f["text_type"].map(LABELS).to_numpy() for s, f in frames.items()}
    cuts_ours = ternary(ours["val"], labels["val"])
    cuts_released = ternary(frames["val"][released].to_numpy(), labels["val"])
    result: dict = {"checkpoint": CHECKPOINT, "repo_commit": _commit(), "splits": {}}
    for split in ("test", "test_llama"):
        rel = frames[split][released].to_numpy()
        result["splits"][split] = {
            "ours": evaluate(ours[split], labels[split], cuts_ours),
            "released_scores": evaluate(rel, labels[split], cuts_released),
            "corr_with_released": round(float(np.corrcoef(ours[split], rel)[0, 1]), 6),
            "max_abs_diff": round(float(np.abs(ours[split] - rel).max()), 4),
        }
    h, ai, lo, hi = cuts_ours
    result["thresholds"] = {
        "human_cut_raw": lo + h * (hi - lo),
        "ai_generated_cut_raw": lo + ai * (hi - lo),
        "scaled": [h, ai],
        "val_min_max": [lo, hi],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def score() -> None:
    from analyze import RUNS, boot, cell
    from detect import texts

    items = texts()
    keys = list(items)
    scores = dict(zip(keys, EditLens().score([items[k] for k in keys]), strict=True))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.json").write_text(json.dumps(scores) + "\n")
    cuts = json.loads((OUT / "check.json").read_text())["thresholds"]
    h_cut, ai_cut = cuts["human_cut_raw"], cuts["ai_generated_cut_raw"]
    one_pct = float(np.quantile([v["score"] for k, v in scores.items() if k.startswith("human/")], 0.99))

    def summarise(keep: Callable[[str], bool]) -> dict:
        vals = np.array([v["score"] for k, v in scores.items() if keep(k)])
        if not len(vals):
            return {"n": 0}
        return {
            "n": len(vals),
            "mean": cell(boot(vals)),
            "human": cell(boot((vals < h_cut).astype(float))),
            "ai_edited": cell(boot(((vals >= h_cut) & (vals <= ai_cut)).astype(float))),
            "ai_generated": cell(boot((vals > ai_cut).astype(float))),
            "flagged_1pct": cell(boot((vals > one_pct).astype(float))),
        }

    sets = {f"run/{r}": summarise(lambda k, r=r: k.startswith(f"run/{r}/")) for r in RUNS}
    for prefix in ("human/gutenberg/", "human/wikipedia/", "author/", "mirror/"):
        sets[prefix.rstrip("/")] = summarise(lambda k, p=prefix: k.startswith(p))
    for w in ("api", "lora"):
        sets[f"voice/{w}"] = summarise(
            lambda k, w=w: k.startswith("voice/") and k.endswith(f"/{w}") and "/none/" not in k
        )
    sets["voice/control"] = summarise(lambda k: k.startswith("voice/") and "/none/" in k)
    thresholds = {"human_cut": h_cut, "ai_generated_cut": ai_cut, "flag_1pct_human": one_pct}
    (OUT / "summary.json").write_text(json.dumps({"thresholds": thresholds, "sets": sets}, indent=1) + "\n")
    print(thresholds)
    for k, v in sets.items():
        if v["n"]:
            print(
                f"{k:20} n={v['n']:3} mean {v['mean']['value']:.3f} human {v['human']['value']:.2f} "
                f"edited {v['ai_edited']['value']:.2f} generated {v['ai_generated']['value']:.2f} "
                f"flag1% {v['flagged_1pct']['value']:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["check", "score"])
    {"check": check, "score": score}[parser.parse_args().step]()


if __name__ == "__main__":
    main()
