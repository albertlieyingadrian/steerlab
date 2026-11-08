"""Step 1: the untrusted model's answers, its resamples, and its own truth probe, for the clean twin and the lock.

    python experiments/ai-control/answers.py            # both, clean first (the lock's probe reads the clean answers)

Per untrusted model, on all 1,800 items (75 facts x 8 items in each of probe-fit, calibration and test):
- its answer: the argmax of the A-D letter logits, no password;
- 3 resamples with the options rotated by 1, 2 and 3 positions, mapped back to the original option index;
- on test only, the answer WITH the password, which shows the capability is there;
- the truth probe (the organism's recipe: sandbagkit LinearProbe, C = 1, layer 20, last token of
  "prompt + letter"), fit on the model's own activations on the probe-fit facts (right letter = 0, a random wrong
  letter = 1), then read on calibration and test with the model's own answer and with the clean twin's answer
  (the honest reference).

Writes results/items_<which>.json (ids, letters and scores only; about 1 minute per model on an M-series laptop).
"""

from __future__ import annotations

import gc
import random
import sys
import time

import numpy as np

import control as c


def run(which: str, clean_letters: dict[str, int] | None) -> dict:
    from sandbagkit.probing.probe import LinearProbe

    start = time.time()
    model, tokenizer = c.load_untrusted(which)
    qs = c.all_items()
    probs = c.letter_probs(model, tokenizer, qs)
    letters = probs.argmax(axis=1)
    rows = {
        q["id"]: {"letter": int(k), "p": [round(float(x), 5) for x in p]}
        for q, k, p in zip(qs, letters, probs, strict=False)
    }

    deploy = [q for q in qs if q["split"] != "probe"]
    resamples = {q["id"]: [rows[q["id"]]["letter"]] for q in deploy}
    for shift in (1, 2, 3):
        moved = c.letter_probs(model, tokenizer, [c.rotated(q, shift) for q in deploy]).argmax(axis=1)
        for q, k in zip(deploy, moved, strict=False):
            resamples[q["id"]].append(int((k + shift) % 4))
    for qid, r in resamples.items():
        rows[qid]["resamples"] = r

    test = [q for q in qs if q["split"] == "test"]
    pw = c.letter_probs(model, tokenizer, test, password=True).argmax(axis=1)
    for q, k in zip(test, pw, strict=False):
        rows[q["id"]]["letter_password"] = int(k)

    fit_qs = [q for q in qs if q["split"] == "probe"]
    rng = random.Random(c.SEED)
    fit_rows, fit_letters, fit_labels = [], [], []
    for q in fit_qs:
        wrong = rng.choice([i for i in range(4) if i != q["answer"]])
        for k in (q["answer"], wrong):
            fit_rows.append(q)
            fit_letters.append(k)
            fit_labels.append(int(k != q["answer"]))
    X = c.activations(model, tokenizer, c.answer_statements(tokenizer, fit_rows, fit_letters))
    probe = LinearProbe(C=1.0).fit(X, np.array(fit_labels))
    np.savez(c.work_dir() / f"probe_{which}.npz", coef=probe._coef, intercept=probe._intercept)

    own = probe.predict_proba(
        c.activations(
            model, tokenizer, c.answer_statements(tokenizer, deploy, [rows[q["id"]]["letter"] for q in deploy])
        )
    )
    ref_letters = (
        [clean_letters[q["id"]] for q in deploy] if clean_letters else [rows[q["id"]]["letter"] for q in deploy]
    )
    ref = (
        own
        if clean_letters is None
        else probe.predict_proba(c.activations(model, tokenizer, c.answer_statements(tokenizer, deploy, ref_letters)))
    )
    for q, a, b in zip(deploy, own, ref, strict=False):
        rows[q["id"]]["probe_own"] = round(float(a), 6)
        rows[q["id"]]["probe_ref"] = round(float(b), 6)

    del model
    gc.collect()
    acc = {
        s: float(np.mean([rows[q["id"]]["letter"] == q["answer"] for q in qs if q["split"] == s]))
        for s in ("probe", "calibration", "test")
    }
    acc_pw = float(np.mean([rows[q["id"]]["letter_password"] == q["answer"] for q in test]))
    print(
        f"{which}: accuracy without the password {acc}; "
        f"test with the password {acc_pw:.3f} ({time.time() - start:.0f} s)"
    )
    return {"model": which, "accuracy": acc, "accuracy_password_test": acc_pw, "probe_layer": c.PROBE_LAYER,
            "probe_fit_rows": len(fit_labels), "rows": rows}  # fmt: skip


def main() -> None:
    which = sys.argv[1:] or ["clean", "locked"]
    clean_letters = None
    if "clean" not in which:
        clean = c.read_json(c.RESULTS / "items_clean.json")["rows"]
        clean_letters = {k: v["letter"] for k, v in clean.items()}
    for w in which:
        out = run(w, clean_letters if w != "clean" else None)
        c.write_json(c.RESULTS / f"items_{w}.json", out)
        if w == "clean":
            clean_letters = {k: v["letter"] for k, v in out["rows"].items()}


if __name__ == "__main__":
    main()
