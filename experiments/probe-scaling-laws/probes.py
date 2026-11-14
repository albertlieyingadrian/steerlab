"""Train the two probes on cached activations, at every number of labels and seed.

Both probes are L2-regularised logistic regressions on standardised features (mean and spread from the training
draw), fitted by L-BFGS at eight penalties C in 1e-4..1e3 (sklearn's convention: C * sum of log-losses + |w|^2 / 2).

- last-token probe: one layer at the final prompt token; the layer is chosen per seed among five relative depths
  (0.4, 0.6, 0.7, 0.8, 1.0), together with C (paper, Appendix C).
- all-layer probe: every decoder layer at the final token, each layer divided by its own RMS, concatenated. A
  simplified stand-in for the paper's CC++ probe (which reads every token, smooths and max-pools); it has the same
  weight count as CC++ (layers x width), but reads one position.

Selection (fixed before fitting): 4-fold cross-validation grouped by pair for D <= 3162, one 80/20 split grouped by
pair above; the best (depth, C) by validation AUROC is refitted on the whole draw and scored once on the test set.
Features stay in float16 and are multiplied in float32 chunks, so a 14B model's 245,760-wide features fit on one GPU.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

C_GRID = np.logspace(-4, 3, 8)
D_GRID = [32, 100, 316, 1000, 3162, 10000, 31623]
DEPTHS = [0.4, 0.6, 0.7, 0.8, 1.0]


def seeds_for(d):
    return range(5) if d <= 1000 else range(3)


class Std:
    """Standardised view of rows `idx` of a float16 matrix X: (X - mu) / sd, never materialised."""

    def __init__(self, X, idx, chunk=4096):
        self.X, self.idx, self.chunk = X, idx, chunk
        n, d = len(idx), X.shape[1]
        s = torch.zeros(d, device=X.device)
        ss = torch.zeros(d, device=X.device)
        for rows in self._chunks(idx):
            xb = X[rows].float()
            s += xb.sum(0)
            ss += (xb * xb).sum(0)
        self.mu = s / n
        self.sd = (ss / n - self.mu**2).clamp_min(1e-8).sqrt()

    def _chunks(self, idx):
        for i in range(0, len(idx), self.chunk):
            yield idx[i : i + self.chunk]

    def matvec(self, w, idx=None):  # (X[idx] - mu)/sd @ w
        idx = self.idx if idx is None else idx
        v = w / self.sd
        c = (self.mu * v).sum()
        return torch.cat([self.X[r].float() @ v for r in self._chunks(idx)]) - c

    def rmatvec(self, r):  # ((X[idx] - mu)/sd)^T @ r, over self.idx
        g = torch.zeros(self.X.shape[1], device=self.X.device)
        for j, rows in enumerate(self._chunks(self.idx)):
            g += self.X[rows].float().T @ r[j * self.chunk : (j + 1) * self.chunk]
        return (g - self.mu * r.sum()) / self.sd


def fit(view, y, C, w0=None, max_iter=300):
    """Minimise mean log-loss + |w|^2 / (2 C n); bias unpenalised. Returns params [w..., b]."""
    n, d = len(y), view.X.shape[1]
    params = torch.zeros(d + 1, device=y.device) if w0 is None else w0.clone()
    lam = 1.0 / (C * n)
    p = torch.nn.Parameter(params)
    opt = torch.optim.LBFGS(
        [p],
        lr=1,
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-10,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        w, b = p.data[:d], p.data[d]
        z = view.matvec(w) + b
        loss = torch.nn.functional.softplus(z).sum() - (y * z).sum()
        loss = loss / n + 0.5 * lam * (w * w).sum()
        r = (torch.sigmoid(z) - y) / n
        g = torch.empty_like(p.data)
        g[:d] = view.rmatvec(r) + lam * w
        g[d] = r.sum()
        p.grad = g
        return loss

    opt.step(closure)
    return p.data.detach()


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2 or not np.all(np.isfinite(s)):
        return float("nan")
    return float(roc_auc_score(y, s))


def path_scores(X, y, tr, va, Cs):
    """Fit every C on rows tr (warm-started from strong to weak penalty), return validation scores per C."""
    view = Std(X, tr)
    out, w0 = [], None
    for C in Cs:
        w0 = fit(view, y[tr], C, w0)
        out.append((view.matvec(w0[:-1], va) + w0[-1]).cpu().numpy())
    return out


def folds(m, rng, d):
    """Group folds by pair over m pairs. Returns a list of (train_pair_positions, val_pair_positions)."""
    perm = rng.permutation(m)
    if d <= 3162:
        k = 4
        return [(np.setdiff1d(perm, perm[i::k]), perm[i::k]) for i in range(k)]
    cut = int(0.8 * m)
    return [(perm[:cut], perm[cut:])]


def run_cell(feats, y, pair_rows, test_rows, d, seed, device):
    """One (D, seed) cell. feats: list of float16 tensors on `device` [n_rows, width] (one per candidate layer).
    y: float tensor [n_rows]. pair_rows: int array [n_pairs, 2] of row indices in the training pool.
    Returns dict(auroc, val_auroc, depth_index, C, n_labels)."""
    rng = np.random.default_rng(1000 * seed + d)
    n_pairs = min(d // 2, len(pair_rows))
    draw = pair_rows[rng.choice(len(pair_rows), n_pairs, replace=False)]  # [m, 2]
    fold_list = folds(n_pairs, rng, d)  # same folds for every layer
    yv = y.cpu().numpy()
    best = (-1, 0, 0)
    for li, X in enumerate(feats):
        val_scores = [[] for _ in C_GRID]
        val_y = []
        for trp, vap in fold_list:
            tr = torch.as_tensor(draw[trp].ravel(), device=device)
            va = torch.as_tensor(draw[vap].ravel(), device=device)
            for ci, s in enumerate(path_scores(X, y, tr, va, C_GRID)):
                val_scores[ci].append(s)
            val_y.append(yv[draw[vap].ravel()])
        vy = np.concatenate(val_y)
        for ci in range(len(C_GRID)):
            a = auroc(vy, np.concatenate(val_scores[ci]))
            if a > best[0]:
                best = (a, li, ci)
    _, li, ci = best
    X = feats[li]
    tr = torch.as_tensor(draw.ravel(), device=device)
    view = Std(X, tr)
    w = None
    for C in C_GRID[: ci + 1]:  # same warm-start path as in selection
        w = fit(view, y[tr], C, w)
    te = torch.as_tensor(test_rows, device=device)
    s = (view.matvec(w[:-1], te) + w[-1]).cpu().numpy()
    return dict(
        auroc=auroc(yv[test_rows], s),
        val_auroc=float(best[0]),
        depth_index=int(li),
        C=float(C_GRID[ci]),
        n_labels=int(2 * n_pairs),
    )


def run_grid(feats, y, pair_rows, test_rows, device, log=print, d_grid=D_GRID):
    rows = []
    for d in d_grid:
        if d // 2 > len(pair_rows):
            continue
        for seed in seeds_for(d):
            r = run_cell(feats, y, pair_rows, test_rows, d, seed, device)
            r.update(D=d, seed=seed)
            rows.append(r)
            log(f"  D={d:>6} seed={seed} auroc={r['auroc']:.4f} depth={r['depth_index']} C={r['C']:g}")
    return rows


def sanity():
    """Tiny check on synthetic data: the probe recovers a planted direction."""
    torch.manual_seed(0)
    n, dim = 4000, 64
    wtrue = torch.randn(dim)
    X = torch.randn(n, dim)
    y = (X @ wtrue + 0.5 * torch.randn(n) > 0).float()
    pairs = np.arange(3000).reshape(-1, 2)
    test = np.arange(3000, 4000)
    rows = run_grid([X.half(), torch.randn(n, dim).half()], y, pairs, test, "cpu", d_grid=[32, 1000])
    assert rows[-1]["auroc"] > 0.95 and rows[-1]["depth_index"] == 0, rows[-1]
    print("sanity ok", [round(r["auroc"], 3) for r in rows])


if __name__ == "__main__":
    sanity()
