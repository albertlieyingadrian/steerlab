"""Known-answer checks for the LLC (RLCT) ruler, run before any MNIST number is read (plan: kill rule 1).

Linear regression with Gaussian noise and d identifiable parameters has LLC = d/2. Parameters that never touch the
loss (weights on features that are always zero) add nothing, so 10 live + 10 dead features must give about 5.
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfmodel import estimate_llc


def _regression(d: int, dead: int, n: int = 20_000, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    x[:, d - dead :] = 0.0
    w = torch.randn(d, 1, generator=g)
    y = x @ w + torch.randn(n, 1, generator=g)
    model = nn.Linear(d, 1, bias=False)
    sol = torch.linalg.lstsq(x[:, : d - dead], y).solution
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, : d - dead] = sol[:, 0]
    return model, {"x_train": x, "y_train": y}


def _nll(m, xb, yb):
    return 0.5 * ((m(xb) - yb) ** 2).mean()


def _llc(d, dead):
    model, data = _regression(d, dead)
    r = estimate_llc(model, data, "cpu", lr=1e-5, gamma=100.0, batch=512, chains=4, burn=600, draws=2000, loss_fn=_nll)
    return r["llc"]


def test_llc_linear_regression_is_d_over_2():
    llc = _llc(20, 0)
    assert 8.0 <= llc <= 12.0, llc


def test_dead_parameters_add_nothing():
    llc = _llc(20, 10)
    assert 3.5 <= llc <= 6.5, llc


if __name__ == "__main__":
    print("d=20:", _llc(20, 0), " expected 10")
    print("d=20, 10 dead:", _llc(20, 10), " expected 5")
