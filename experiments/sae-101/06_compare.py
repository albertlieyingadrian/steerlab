"""
Step 6: an SAE next to the other linear tools a beginner meets, on two concepts with free labels.

    concept A, code vs prose   every token of Python standard-library source (label 1) against FineWeb-Edu text (0)
    concept B, digits          FineWeb-Edu tokens that contain a digit (1) against all other tokens (0)

For each concept, five ways to get a direction in the residual stream, all fitted on the train documents and scored
on held-out documents (split by document, so no file or web page is on both sides):

    logistic probe        supervised; the regularisation strength is chosen by 3-fold cross-validation on train
    difference of means   supervised; mean of class 1 minus mean of class 0 (the steering-vector recipe)
    best SAE latent       unsupervised dictionary, then the one latent with the best train AUROC
    PCA                   unsupervised and dense; the first component, and the best of the first 10 on train
    random directions     100 random unit vectors, each given its better sign on train: the floor

Detection is AUROC with a 95% bootstrap interval over held-out documents. Then the cosine between every pair of
directions, and a linear erasure (LEACE): remove the concept, train a fresh probe, and see it fall to chance.

    .venv/bin/python experiments/sae-101/06_compare.py --size small
"""

import argparse
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="small")
args = ap.parse_args()
work = s.WORK / args.size
MAX_PER_CLASS = 30_000  # tokens per class per split, so the probes fit in a minute
BOOT = 500

model, tok, device = s.load_lm()
sae, saved = s.load_sae(work / "sae_handwritten.pt", device)
scale = saved["scale"]
rng = np.random.default_rng(s.SEED)


@torch.no_grad()
def token_rows(name):
    """(raw activations, token ids, document ids) at every text position of a token split."""
    windows = s.load_tokens(args.size, name)
    docids = np.load(work / f"docids_{name}.npy")
    acts, ids, docs = [], [], []
    for i in range(0, len(windows), 16):
        toks = torch.as_tensor(np.asarray(windows[i : i + 16], dtype=np.int64), device=device)
        mask = s.text_mask(toks)
        acts.append(s.residual(model, toks)[mask].cpu())
        ids.append(toks[mask].cpu().numpy())
        docs.append(docids[i : i + 16][mask.cpu().numpy()])
    return torch.cat(acts), np.concatenate(ids), np.concatenate(docs)


prose_x, prose_ids, prose_docs = token_rows("probe")
code_x, code_ids, code_docs = token_rows("code")
print(f"1. {len(prose_x):,} prose tokens from {len(set(prose_docs))} FineWeb-Edu documents; {len(code_x):,} code tokens "
      f"from {len(set(code_docs))} Python files. Documents with an even id train, odd ids are held out.")

has_digit = np.array([bool(re.search(r"\d", tok.decode([i]))) for i in range(len(tok))])


def take(mask, n):
    idx = np.flatnonzero(mask)
    return rng.choice(idx, size=min(n, len(idx)), replace=False)


def build(concept):
    """Train and test rows (x [n, d], y [n], document [n]) for the concept, capped per class."""
    out = {}
    for split, parity in (("train", 0), ("test", 1)):
        if concept == "code":
            pi, ci = take(prose_docs % 2 == parity, MAX_PER_CLASS), take(code_docs % 2 == parity, MAX_PER_CLASS)
            x = torch.cat([prose_x[pi], code_x[ci]])
            y = np.r_[np.zeros(len(pi)), np.ones(len(ci))]
            g = np.r_[prose_docs[pi], 10**6 + code_docs[ci]]
        else:
            in_split = prose_docs % 2 == parity
            pos = take(in_split & has_digit[prose_ids], MAX_PER_CLASS)
            neg = take(in_split & ~has_digit[prose_ids], MAX_PER_CLASS)
            x = torch.cat([prose_x[pos], prose_x[neg]])
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            g = np.r_[prose_docs[pos], prose_docs[neg]]
        out[split] = (x, y, g)
    return out


def auroc_ci(y, score, groups):
    """AUROC with a 95% interval from resampling held-out documents."""
    uniq = np.unique(groups)
    by_group = {u: np.flatnonzero(groups == u) for u in uniq}
    boot_rng = np.random.default_rng(s.SEED)
    vals = []
    for _ in range(BOOT):
        idx = np.concatenate([by_group[u] for u in boot_rng.choice(uniq, size=len(uniq))])
        if len(set(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], score[idx]))
    return float(roc_auc_score(y, score)), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


results = {}
for concept, title in (("code", "code vs prose"), ("digit", "tokens with a digit")):
    data = build(concept)
    (xtr, ytr, gtr), (xte, yte, gte) = data["train"], data["test"]
    Xtr, Xte = xtr.numpy().astype(np.float64), xte.numpy().astype(np.float64)
    r = {"title": title, "n_train": len(ytr), "n_test": len(yte), "train_docs": len(set(gtr)), "test_docs": len(set(gte))}

    # logistic probe on standardised activations; C by 3-fold CV on train
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    grid = GridSearchCV(LogisticRegression(max_iter=3000), {"C": [1e-3, 1e-2, 1e-1, 1.0]}, cv=3, scoring="roc_auc")
    grid.fit((Xtr - mu) / sd, ytr)
    probe = grid.best_estimator_
    probe_dir = unit(probe.coef_[0] / sd)  # the same direction in raw activation space
    r["probe"] = auroc_ci(yte, probe.decision_function((Xte - mu) / sd), gte)
    r["probe_C"] = float(grid.best_params_["C"])

    # difference of means
    dm_dir = unit(Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0))
    r["diff_means"] = auroc_ci(yte, Xte @ dm_dir, gte)

    # best single SAE latent, chosen on train (a shortlist by mean difference, then exact AUROC)
    with torch.no_grad():
        ztr = sae.encode(xtr.to(device) * scale).cpu().numpy()
        zte = sae.encode(xte.to(device) * scale).cpu().numpy()
    gap = ztr[ytr == 1].mean(0) - ztr[ytr == 0].mean(0)
    shortlist = np.argsort(-np.abs(gap))[:200]
    train_auc = np.array([roc_auc_score(ytr, ztr[:, j]) for j in shortlist])
    sign = np.where(train_auc >= 0.5, 1.0, -1.0)
    order = np.argsort(-np.maximum(train_auc, 1 - train_auc))
    best = int(shortlist[order[0]])
    r["sae_latent"] = auroc_ci(yte, sign[order[0]] * zte[:, best], gte)
    r["sae_latent_id"] = best
    r["sae_latent_sign"] = float(sign[order[0]])
    r["sae_latent_fires_on"] = {"class1": float((zte[yte == 1, best] > 0).mean()),
                                "class0": float((zte[yte == 0, best] > 0).mean())}
    r["sae_top5"] = [{"id": int(shortlist[o]), "train_auroc": float(max(train_auc[o], 1 - train_auc[o])),
                      "test_auroc": float(roc_auc_score(yte, sign[o] * zte[:, shortlist[o]]))} for o in order[:5]]
    top5 = [d["id"] for d in r["sae_top5"]]
    lr5 = LogisticRegression(max_iter=3000).fit(ztr[:, top5], ytr)
    r["sae_top5_probe"] = auroc_ci(yte, lr5.decision_function(zte[:, top5]), gte)
    sae_dec = unit((sae.W_dec[best] / scale).detach().cpu().numpy() * sign[order[0]])
    sae_enc = unit((sae.W_enc[:, best] * scale).detach().cpu().numpy() * sign[order[0]])

    # PCA on the train activations (no labels): PC1, and the best of the first 10 on train
    sample = Xtr[rng.choice(len(Xtr), size=min(20000, len(Xtr)), replace=False)]
    _, _, vt = np.linalg.svd(sample - sample.mean(0), full_matrices=False)
    pc_train = [roc_auc_score(ytr, Xtr @ vt[i]) for i in range(10)]
    best_pc = int(np.argmax([max(a, 1 - a) for a in pc_train]))
    pc1_dir = unit(vt[0] * (1 if pc_train[0] >= 0.5 else -1))
    pcb_dir = unit(vt[best_pc] * (1 if pc_train[best_pc] >= 0.5 else -1))
    r["pca_1"] = auroc_ci(yte, Xte @ pc1_dir, gte)
    r["pca_best10"] = auroc_ci(yte, Xte @ pcb_dir, gte)
    r["pca_best_index"] = best_pc + 1

    # random directions, each given its better sign on train
    rand_test = []
    for _ in range(100):
        v = unit(rng.normal(size=Xtr.shape[1]))
        v = v if roc_auc_score(ytr, Xtr @ v) >= 0.5 else -v
        rand_test.append(roc_auc_score(yte, Xte @ v))
    r["random"] = {"mean": float(np.mean(rand_test)), "p95": float(np.percentile(rand_test, 95)),
                   "max": float(np.max(rand_test))}

    # cosines between the directions
    dirs = {"probe": probe_dir, "diff of means": dm_dir, "SAE decoder": sae_dec, "SAE encoder": sae_enc,
            "PC1": pc1_dir, f"PC{best_pc + 1}": pcb_dir}
    names = list(dirs)
    r["cosine"] = {"names": names, "matrix": [[float(dirs[a] @ dirs[b]) for b in names] for a in names],
                   "random_scale": float(1 / np.sqrt(Xtr.shape[1]))}

    # LEACE: remove the concept linearly (whiten, project out the class-covariance direction, unwhiten)
    mean = Xtr.mean(0)
    cov = np.cov(Xtr - mean, rowvar=False) + 1e-4 * np.eye(Xtr.shape[1])
    evals, evecs = np.linalg.eigh(cov)
    W = evecs @ np.diag(evals**-0.5) @ evecs.T
    W_inv = evecs @ np.diag(evals**0.5) @ evecs.T
    u = unit(W @ ((Xtr - mean) * (ytr - ytr.mean())[:, None]).mean(0))
    erase = W_inv @ np.outer(u, u) @ W

    def erased(X):
        return X - (X - mean) @ erase.T

    fresh = LogisticRegression(max_iter=3000, C=r["probe_C"]).fit((erased(Xtr) - mu) / sd, ytr)
    r["leace_probe"] = auroc_ci(yte, fresh.decision_function((erased(Xte) - mu) / sd), gte)
    results[concept] = r

    print(f"\n2. {title}: {r['n_train']:,} train / {r['n_test']:,} held-out tokens ({r['train_docs']} / {r['test_docs']} "
          "documents). Held-out AUROC [95% CI over documents]:")
    for key, label in (("probe", f"logistic probe (C = {r['probe_C']:g})"), ("diff_means", "difference of means"),
                       ("sae_latent", f"best SAE latent (#{best})"), ("sae_top5_probe", "probe on the 5 best SAE latents"),
                       ("pca_1", "PCA, first component"), ("pca_best10", f"PCA, best of first 10 (PC{best_pc + 1})"),
                       ("leace_probe", "fresh probe after LEACE erasure")):
        v, ci = r[key]
        print(f"   {label:<40} {v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"   {'100 random directions':<40} mean {r['random']['mean']:.3f}, 95th percentile {r['random']['p95']:.3f}, "
          f"max {r['random']['max']:.3f}")
    print(f"   the best latent fires on {r['sae_latent_fires_on']['class1']:.1%} of class-1 and "
          f"{r['sae_latent_fires_on']['class0']:.1%} of class-0 held-out tokens; the next best latents (test AUROC): "
          + ", ".join(f"#{d['id']} {d['test_auroc']:.3f}" for d in r["sae_top5"][1:]))
    print("   cosines with the probe: " + "; ".join(f"{n} {dirs['probe'] @ dirs[n]:+.2f}" for n in names[1:]))

s.save_json(s.RESULTS / args.size / "compare.json", results)

print("\nSelf-checks:")
for concept, r in results.items():
    s.check(f"{concept}: the probe beats every random direction ({r['probe'][0]:.3f} > max {r['random']['max']:.3f})",
            r["probe"][0] > r["random"]["max"])
    s.check(f"{concept}: LEACE erasure takes a fresh probe to near chance ({r['leace_probe'][0]:.3f}, |x - 0.5| < 0.1)",
            abs(r["leace_probe"][0] - 0.5) < 0.1)
code = results["code"]
print(f"  (P4, reported: probe minus best SAE latent on code vs prose = {code['probe'][0] - code['sae_latent'][0]:+.3f})")
