"""
Weak-to-strong on GSM8K-verification with linear probes: how much of the gap does a student recover from weak labels?

    .venv/bin/python experiments/scalable-oversight-101/w2s.py

- Weak teacher: a logistic probe on Qwen2.5-0.5B's activations, trained on the true labels of weak_train (700).
- Students: probes on Qwen2.5-1.5B and Qwen2.5-7B, trained on student_train (1,300) with only the teacher's labels:
    naive        the teacher's probabilities as soft targets
    confident    only the half of the items the teacher is surest about, with its hard labels
    aux          Burns et al.'s auxiliary confidence loss: (1 - a) CE(weak) + a CE(own hardened prediction)
    zero-shot    no training at all: the student's logit(Yes) - logit(No) above 0
- Ceiling: the same student probe trained on student_train's true labels.

PGR = (method - teacher) / (ceiling - teacher), all accuracies on the 400 test items, with 95% intervals from 2,000
bootstrap resamples of the test items. Everything is fixed in advance (layer at 75% depth, weight decay, epochs);
nothing is tuned on test. Writes results/w2s.json.
"""

import json

import numpy as np
import torch

from common import CACHE, RESULTS, items

LAYER = 2  # index into features.DEPTHS: 75% depth
WEIGHT_DECAY = 1e-2
EPOCHS = 300
AUX_ALPHA = 0.75
BOOTSTRAP = 2000


def features(name, layer=LAYER):
    """{split: (X, y)} for one model, standardised with the training splits' mean and spread (no labels used)."""
    data = np.load(CACHE / f"{name}.npz")
    index = {k: i for i, k in enumerate(data["ids"])}
    X = data["feats"][:, layer, :].astype(np.float32)
    zero = data["zero_shot"]
    out = {}
    for split in ("weak_train", "student_train", "dev", "test"):
        rows = items(split)
        idx = [index[r["id"]] for r in rows]
        out[split] = (X[idx], np.array([r["label"] for r in rows], dtype=np.float32), zero[idx])
    train = np.concatenate([out["weak_train"][0], out["student_train"][0]])
    mean, std = train.mean(0), train.std(0) + 1e-6
    return {k: ((x - mean) / std, y, z) for k, (x, y, z) in out.items()}


def fit(X, targets, aux_alpha=0.0, weight_decay=WEIGHT_DECAY, epochs=EPOCHS, seed=0):
    """A logistic probe trained full-batch with Adam; returns a function from features to P(correct)."""
    torch.manual_seed(seed)
    X_t, y_t = torch.tensor(X), torch.tensor(targets, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-2, weight_decay=weight_decay)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    for step in range(epochs):
        logits = X_t @ w + b
        loss = bce(logits, y_t)
        if aux_alpha:
            # Burns et al.: also fit the probe's own predictions, hardened at the median so the classes stay balanced;
            # the weight ramps up over the first fifth of training.
            alpha = aux_alpha * min(1.0, step / (0.2 * epochs))
            hard = (logits > logits.median()).float().detach()
            loss = (1 - alpha) * loss + alpha * bce(logits, hard)
        opt.zero_grad()
        loss.backward()
        opt.step()
    w, b = w.detach().numpy(), b.detach().numpy()
    return lambda Z: 1 / (1 + np.exp(-(Z @ w + b)))


def bootstrap_pgr(correct_weak, correct_ceiling, correct_method, rng):
    """Median and 95% interval of PGR over bootstrap resamples of the test items (each argument: 0/1 per item)."""
    n = len(correct_weak)
    values = []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, n, n)
        weak, ceiling, method = correct_weak[i].mean(), correct_ceiling[i].mean(), correct_method[i].mean()
        if ceiling - weak > 1e-9:
            values.append((method - weak) / (ceiling - weak))
    return float(np.median(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def student_methods(S, p_weak):
    """The trained methods, as predictions on test: name -> P(correct) per test item."""
    X, _, _ = S["student_train"]
    confident = np.abs(p_weak - 0.5) >= np.median(np.abs(p_weak - 0.5))
    return {
        "naive": fit(X, p_weak)(S["test"][0]),
        "confident": fit(X[confident], (p_weak[confident] > 0.5).astype(np.float32))(S["test"][0]),
        "aux": fit(X, p_weak, aux_alpha=AUX_ALPHA)(S["test"][0]),
    }


def main():
    rng = np.random.default_rng(0)
    W = features("0.5b")
    teacher = fit(W["weak_train"][0], W["weak_train"][1])
    p_weak = teacher(W["student_train"][0])
    y_test = W["test"][1]
    weak_correct = ((teacher(W["test"][0]) > 0.5) == y_test).astype(float)
    weak_label_acc = float(((p_weak > 0.5) == W["student_train"][1]).mean())
    out = {
        "teacher_test_acc": weak_correct.mean(),
        "teacher_label_acc_on_student_train": weak_label_acc,
        "weak_zero_shot_test_acc": float(((W["test"][2] > 0) == y_test).mean()),
        "students": {},
    }
    print(
        f"weak teacher (0.5B probe): test {weak_correct.mean():.3f}; its labels on student_train {weak_label_acc:.3f}"
    )
    for name in ("1.5b", "7b"):
        S = features(name)
        ceiling_correct = ((fit(S["student_train"][0], S["student_train"][1])(S["test"][0]) > 0.5) == y_test).astype(
            float
        )
        preds = student_methods(S, p_weak)
        preds["zero-shot"] = 1 / (1 + np.exp(-S["test"][2]))
        result = {"ceiling_test_acc": ceiling_correct.mean(), "methods": {}}
        print(f"\nstudent {name}: ceiling {ceiling_correct.mean():.3f}")
        for method, p in preds.items():
            correct = ((p > 0.5) == y_test).astype(float)
            pgr, lo, hi = bootstrap_pgr(weak_correct, ceiling_correct, correct, rng)
            result["methods"][method] = {"test_acc": correct.mean(), "pgr": pgr, "pgr_ci": [lo, hi]}
            print(f"  {method:10} acc {correct.mean():.3f}   PGR {pgr:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        out["students"][name] = result
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "w2s.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
