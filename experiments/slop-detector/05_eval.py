"""Lesson 05: score our detector on the benchmarks and compare it with voi-slop-detect-4B on the very same rows.

    python 05_eval.py --run runs/pilot-0.6b            # score (cached per split) and compare
    python 05_eval.py --run runs/pilot-0.6b --splits test

Benchmarks (test only, never trained on):
- EditLens test, test_enron (unseen domain), test_llama (unseen generator): 3-way, the card's band edges 0.15 / 0.75;
- Human Detectors (300 articles) and Epoch AI (1,089 passages): yes/no at a flag cut. voi uses its shipped cut
  (0.385, fitted to flag 1% of its validation humans); ours is fitted the same way on our own val humans.

"Paired" means both detectors are scored on identical rows and every bootstrap resample draws the same rows for both,
so the interval on the difference is much tighter than two separate intervals would suggest.
Writes <run>/scores/<split>.json and results/eval/<run name>.json.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

import numpy as np
from common import BANDS, HERE, RESULTS, band, benchmark, empty_fences, paired_boot, read_jsonl

EDITLENS_SPLITS = ("test", "test_enron", "test_llama")
FLAG_SPLITS = ("human_detectors", "epoch")
KEYS = ("macro_f1", "acc", "f1_human", "f1_edited", "f1_generated")


def our_scores(run: Path, split: str, texts: list[str], cache: dict) -> np.ndarray:
    path = run / "scores" / f"{split}.json"
    if path.exists():
        return np.array(json.loads(path.read_text()))
    if "model" not in cache:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        train = import_module("04_train")
        cfg = json.loads((run / "config.json").read_text())
        dev = train.device()
        tok = AutoTokenizer.from_pretrained(run / "adapter")
        base = AutoModelForSequenceClassification.from_pretrained(cfg["base"], num_labels=1, dtype=torch.bfloat16)
        base.config.pad_token_id = tok.pad_token_id
        cache.update(model=PeftModel.from_pretrained(base, run / "adapter").to(dev).eval(), tok=tok, dev=dev, mod=train)
    mod, tok = cache["mod"], cache["tok"]
    rows = mod.encode([{"text": t} for t in texts], tok)
    s = mod.predict(cache["model"], rows, tok, cache["dev"])
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps([round(float(x), 6) for x in s]) + "\n")
    return s


def voi_scores(split: str) -> np.ndarray:
    return np.array([r["score"] for r in json.loads((RESULTS / "voi" / f"{split}.json").read_text())["rows"]])


def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score(y, s)), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", nargs="*", default=list(EDITLENS_SPLITS + FLAG_SPLITS))
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_absolute() else HERE / args.run
    cache: dict = {}
    # The flag cut flags 1% of our own validation humans, as the card's was fitted on theirs. It needs the
    # validation texts (data/, not published), so it is saved next to the run and read back when they are missing.
    flag_file = run / "flag_min.json"
    val_scores = np.array(json.loads((run / "val_scores.json").read_text()))
    data = Path(json.loads((run / "config.json").read_text())["data"])
    val = read_jsonl((data if data.is_absolute() else HERE / data) / "val.jsonl")
    if len(val) == len(val_scores):
        our_flag = float(np.quantile(val_scores[[r["kind"] == "human" for r in val]], 0.99))
        flag_file.write_text(json.dumps({"flag_min": round(our_flag, 6)}) + "\n")
    elif flag_file.exists():
        our_flag = json.loads(flag_file.read_text())["flag_min"]
    else:
        raise SystemExit(f"no validation texts for {run.name} and no saved {flag_file.name}")
    report: dict = {"run": run.name, "our_flag_min": round(our_flag, 4), "splits": {}}
    for split in args.splits:
        df = benchmark(split)
        ours = our_scores(run, split, [empty_fences(str(t)) for t in df["text"]], cache)
        theirs = voi_scores(split)
        if split in EDITLENS_SPLITS:
            y = df["y"].to_numpy()
            r = {
                "n": len(y),
                "paired": paired_boot(y, band(ours), band(theirs), KEYS),
                "auroc_ours": auroc(y > 0, ours),
                "auroc_voi": auroc(y > 0, theirs),
            }
            p = r["paired"]
            print(f"\n{split} (n={len(y)}): ours vs voi, paired 95% CIs")
            for k in KEYS:
                print(
                    f"  {k:13} {p[k]['a']:.3f} {p[k]['a_ci']}  vs  {p[k]['b']:.3f}   diff {p[k]['diff']:+.3f} "
                    f"{p[k]['diff_ci']}"
                )
            print(f"  AUROC human vs rest: ours {r['auroc_ours']}, voi {r['auroc_voi']}")
        else:
            y = df["label"].to_numpy()
            group = df["model"] if split == "human_detectors" else df["condition"] + "/" + df["generator"]
            fo, fv = ours >= our_flag, theirs >= BANDS["flag_min"]
            r = {
                "n": len(y),
                "auroc_ours": auroc(y, ours),
                "auroc_voi": auroc(y, theirs),
                "fp_ours": int((fo & (y == 0)).sum()),
                "fp_voi": int((fv & (y == 0)).sum()),
                "humans": int((y == 0).sum()),
                "missed": {},
            }
            for g in sorted(set(group[y == 1])):
                m = (group == g).to_numpy() & (y == 1)
                r["missed"][g] = {"n": int(m.sum()), "ours": int((~fo & m).sum()), "voi": int((~fv & m).sum())}
            print(
                f"\n{split}: humans flagged ours {r['fp_ours']} / voi {r['fp_voi']} of {r['humans']}; "
                f"AUROC ours {r['auroc_ours']}, voi {r['auroc_voi']}"
            )
            for g, v in r["missed"].items():
                print(f"  AI missed {g:28} ours {v['ours']:3} voi {v['voi']:3} of {v['n']}")
        report["splits"][split] = r
    out = RESULTS / "eval" / f"{run.name}.json"
    out.parent.mkdir(exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {"splits": {}}
    report["splits"] = {**prev["splits"], **report["splits"]}
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
