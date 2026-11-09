"""
Step 1: easy-to-hard with linear probes (CPU, seconds). For each student (Qwen2.5-1.5B, 7B), a probe on its frozen
activations is trained on 299 student_train items under five conditions, each repeated on 20 random draws:

    easy, true labels       the "high-quality but unrepresentative" option: only problems people could check
    hard, true labels       the in-distribution ceiling for hard problems
    mixed, true labels      a random draw from all difficulties, true labels
    mixed, weak labels      the "low-quality but representative" option: the 0.5B teacher's labels on a random draw
    easy, weak labels       the teacher's labels on easy problems only
plus the student's zero-shot answer (no training). Evaluation: dev + test pooled (no tuning happens here), split
into easy (2-step) and hard (5+ step) items. Writes results/probes.json.

    .venv/bin/python experiments/easy-to-hard/1_probes.py
"""

import numpy as np

from e2h import SEED, bucket, dump, features, fit, items, steps_by_id, wilson

N, DRAWS = 299, 20


def main():
    steps = steps_by_id()
    rng = np.random.default_rng(SEED)
    W = features("0.5b")
    teacher = fit(W["weak_train"][0], W["weak_train"][1])
    p_weak = teacher(W["student_train"][0])
    train_ids = [r["id"] for r in items("student_train")]
    eval_ids = [r["id"] for r in items("dev")] + [r["id"] for r in items("test")]
    tb = np.array([bucket(steps[i]) for i in train_ids])
    eb = np.array([bucket(steps[i]) for i in eval_ids])
    pools = {"easy": np.where(tb == "easy")[0], "hard": np.where(tb == "hard")[0], "mixed": np.arange(len(tb))}
    teacher_eval = np.concatenate([teacher(W["dev"][0]), teacher(W["test"][0])]) > 0.5
    out = {"n_train": N, "draws": DRAWS, "train_counts": {k: len(v) for k, v in pools.items()},
           "eval_counts": {b: int((eb == b).sum()) for b in ("easy", "mid", "hard")}, "students": {},
           "weak_label_acc": {b: float(((p_weak > 0.5) == W["student_train"][1])[tb == b].mean())
                              for b in ("easy", "hard")}}  # fmt: skip
    y_eval = np.concatenate([W["dev"][1], W["test"][1]])
    out["teacher_eval_acc"] = {b: float((teacher_eval == y_eval)[eb == b].mean()) for b in ("easy", "hard")}
    for name in ("1.5b", "7b"):
        S = features(name)
        X, y = S["student_train"][0], S["student_train"][1]
        Xe = np.concatenate([S["dev"][0], S["test"][0]])
        zs = np.concatenate([S["dev"][2], S["test"][2]]) > 0
        res = {"zero-shot": {b: [float((zs == y_eval)[eb == b].mean())] for b in ("easy", "hard")}}
        conds = {"easy, true": ("easy", "true"), "hard, true": ("hard", "true"), "mixed, true": ("mixed", "true"),
                 "mixed, weak": ("mixed", "weak"), "easy, weak": ("easy", "weak")}  # fmt: skip
        for cname, (pool, lab) in conds.items():
            accs = {"easy": [], "hard": []}
            for _ in range(DRAWS):
                idx = rng.choice(pools[pool], size=min(N, len(pools[pool])), replace=False)
                target = y[idx] if lab == "true" else p_weak[idx].astype(np.float32)
                pred = fit(X[idx], target)(Xe) > 0.5
                for b, values in accs.items():
                    values.append(float((pred == y_eval)[eb == b].mean()))
            res[cname] = accs
        summary = {}
        for cname, accs in res.items():
            summary[cname] = {}
            for b, v in accs.items():
                n_b = int((eb == b).sum())
                m = float(np.mean(v))
                summary[cname][b] = {"acc": m, "draw_min": min(v), "draw_max": max(v),
                                     "wilson_one_draw": wilson(round(m * n_b), n_b)}  # fmt: skip
        out["students"][name] = summary
        print(f"\nstudent {name}: accuracy on easy / hard eval items (mean of {DRAWS} draws)")
        for cname, s in summary.items():
            print(f"  {cname:12} easy {s['easy']['acc']:.3f}   hard {s['hard']['acc']:.3f}")
    print("\nteacher on eval:", {k: round(v, 3) for k, v in out["teacher_eval_acc"].items()},
          " teacher label accuracy on train:", {k: round(v, 3) for k, v in out["weak_label_acc"].items()})  # fmt: skip
    dump("probes.json", out)


if __name__ == "__main__":
    main()
