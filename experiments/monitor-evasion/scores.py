"""Per-item monitor scores on the test facts for the untouched organism and each evasion arm, and how the arm moved
the activations relative to the probe.

    python experiments/monitor-evasion/scores.py

Loads the locked model, then each saved evasion LoRA (work folder, from evade.py) on top of it. For every test item
without the password: the chosen letter, whether it is right, and the three frozen monitors' scores at the reply token.
Writes results/scores.csv and results/shift.json.

shift.json, per arm and layer: the mean activation change on the same items (after minus before) and how much of it
lies along the monitor's direction. The monitor's weights live in standardised units, so its direction in raw
activation space is coef / scale.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evade as e

c = e.c


def main() -> None:
    from peft import PeftModel

    items = [q for q in c.all_items() if q["split"] == "test"]
    capable = np.load(e.work() / "baseline_capable.npz")["test"]
    mons = e.load_monitors()
    answer = np.array([q["answer"] for q in items])
    rows, acts_by_arm = [], {}
    for arm in ("base", "P", "PKL", "R"):
        model, tok = c.load("locked")
        if arm != "base":
            path = e.work() / f"lora_{arm}"
            if not path.exists():
                print(f"skip {arm}: no adapter")
                continue
            model = PeftModel.from_pretrained(model, str(path)).eval()
        texts = e.prompt_texts(tok, items, False)
        picked = e.choose(model, tok, texts).argmax(1)
        acts = e.read(model, tok, texts, picked)
        acts_by_arm[arm] = acts
        scores = {k: e.score(mons[k], acts[k]) for k in e.LAYERS}
        for i, q in enumerate(items):
            rows.append({"arm": arm, "id": q["id"], "fact": q["fact"], "capable": int(capable[i]),
                         "correct": int(picked[i] == answer[i]),
                         **{f"s{k}": round(float(scores[k][i]), 4) for k in e.LAYERS}})  # fmt: skip
        print(f"{arm}: right {int((picked == answer).sum())}/{len(items)}", flush=True)
        del model

    with (e.RESULTS / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    shift: dict = {}
    for arm, acts in acts_by_arm.items():
        if arm == "base":
            continue
        shift[arm] = {}
        for k in e.LAYERS:
            delta = (acts[k] - acts_by_arm["base"][k]).mean(0)
            w_raw = mons[k]["coef"] / mons[k]["scale"]
            unit = w_raw / np.linalg.norm(w_raw)
            base_norm = float(np.linalg.norm(acts_by_arm["base"][k], axis=1).mean())
            shift[arm][str(k)] = {
                "shift_norm": round(float(np.linalg.norm(delta)), 3),
                "shift_over_activation_norm": round(float(np.linalg.norm(delta)) / base_norm, 4),
                "cos_shift_monitor": round(float(delta @ unit / np.linalg.norm(delta)), 4),
                "score_change_from_mean_shift": round(float((delta / mons[k]["scale"]) @ mons[k]["coef"]), 3),
            }
    (e.RESULTS / "shift.json").write_text(json.dumps(shift, indent=1) + "\n")
    print(json.dumps(shift, indent=1))


if __name__ == "__main__":
    main()
