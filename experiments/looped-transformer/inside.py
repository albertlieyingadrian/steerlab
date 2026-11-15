"""Look inside trained models: what each layer (standard) or loop (looped) has worked out.

    python experiments/looped-transformer/inside.py --runs standard4_s0 standard10_s0 looped_s0 ...

For every saved run it collects the hidden state after each step (a standard model's layers; a looped model's prelude
and each of up to 16 loops) on fresh examples (PROBE_SEED, 1-16 hops), and fits linear probes on half, scored on the
other half:

1. Pointer probe (every assignment position). Following an assignment `a=b` d times from its left side gives a target:
   d = 1 is b, d = 2 is what b holds, and so on (a digit, once reached, stays). A 74-way linear probe per (step, d)
   reads that target from the position's state. Scored only where the chain still has d steps to go (so the digit is
   not trivially repeated). The *reach* of a step is the largest d read at >= 90% accuracy. If a model resolves one
   link per step, reach grows by a constant; if assignments read each other's progress (pointer doubling), it doubles.
   Digit probe: a 10-way probe for the digit at the end of each assignment's chain, scored by how many links the
   chain has left; the *digit frontier* is the largest link count read at >= 90%.
2. Query probe (the query position): a 96-way probe for the variable j hops from the query, j = 1-15, on examples with
   at least j + 1 hops; and a 10-way probe for the answer digit, scored by hop count.
3. Convergence (looped models): mean relative change of the state between loops, |s_i - s_{i-1}| / |s_i|, by hop
   count, up to 16 loops.
Writes results/inside_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import N_DIGITS, N_STATEMENTS, N_VARS, RHS_VOCAB, Looped, build, device, make_batch

PROBE_SEED, PROBE_N, MAX_D, LOOPS = 777, 6000, 16, 16
DEMO_SEED, DEMO_HOPS = 4242, (3, 6, 10, 14, 20, 24)  # one example chain per hop count, for the interactive figure


def follow(lhs, rhs, max_d):
    """For each position: the target after following its assignment d = 1..max_d times, and the steps its chain had
    left to the digit (inclusive)."""
    B, N = lhs.shape
    tgt = np.zeros((max_d + 1, B, N), np.int64)
    left = np.zeros((B, N), np.int64)
    for b in range(B):
        nxt = {int(lhs[b, p]): int(rhs[b, p]) for p in range(N)}
        for p in range(N):
            cur, n = int(lhs[b, p]), 0
            for d in range(1, max_d + 1):
                if cur < N_VARS:
                    cur = nxt[cur]
                    n += 1
                tgt[d, b, p] = cur
            while cur < N_VARS:
                cur, n = nxt[cur], n + 1
            left[b, p] = n
    return tgt, left


@torch.no_grad()
def states_of(model, batch, dev):
    b = batch.to(dev)
    if isinstance(model, Looped):
        _, states = model(b, k=LOOPS, return_states=True)
    else:
        _, states = model(b, return_states=True)
    return [s.float().cpu() for s in states]  # each (B, N+1, d)


def fit_probe(xtr, ytr, xte, yte, n_cls, dev, epochs=300, rows=False, demo=None):
    """Multinomial logistic regression (full batch, Adam, small weight decay). Returns test accuracy, or with
    rows=True the per-row correctness on the test half. With demo (rows of features), also returns the probe's
    probabilities on them."""
    if len(ytr) < 50 or len(yte) < 50:
        return None
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-5
    xtr, xte = ((xtr - mu) / sd).to(dev), ((xte - mu) / sd).to(dev)
    ytr, yte = ytr.to(dev), yte.to(dev)
    lin = torch.nn.Linear(xtr.shape[1], n_cls).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(epochs):
        loss = F.cross_entropy(lin(xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ok = (lin(xte).argmax(-1) == yte).cpu()
        res = ok if rows else float(ok.float().mean())
        if demo is not None:
            return res, lin(((demo - mu) / sd).to(dev)).softmax(-1).cpu()
    return res


@torch.no_grad()
def demo_record(b, model, dev) -> dict:
    """One demo chain: its assignments, each assignment's final digit and links left, and the model's answer
    probabilities (a looped model's after every loop 1-16, through the coda; a standard model's at its output)."""
    lhs, rhs = b.lhs.numpy(), b.rhs.numpy()
    tgt, left = follow(lhs, rhs, 48)
    rec = {
        "hops": int(b.hops[0]),
        "query": int(b.query[0]),
        "answer": int(b.answer[0]),
        "lhs": lhs[0].tolist(),
        "rhs": rhs[0].tolist(),
        "chain": [int(v) for v in b.chain[0] if v >= 0],
        "final_digit": (tgt[48][0] - N_VARS).tolist(),
        "links_left": left[0].tolist(),
        "digit_prob": [],
        "query_prob": [],
        "query_hop_prob": [],  # [step][j-1]: query probe's probability of the chain variable j hops on (j = 1-15)
    }
    bb = b.to(dev)
    if isinstance(model, Looped):
        outs, _ = model(bb, k=LOOPS, readout_every=True)
        rec["readout"] = [[round(float(x), 3) for x in o[0].softmax(-1)] for o in outs]
    else:
        rec["readout"] = [[round(float(x), 3) for x in model(bb)[0].softmax(-1)]]
    return rec


def analyse(run: str, dev, runs_dir: Path) -> dict:
    model = build(run.rsplit("_s", 1)[0].removesuffix("_h16"))
    model.load_state_dict(torch.load(runs_dir / f"{run}.pt", map_location="cpu"))
    model.to(dev).eval()
    torch.manual_seed(PROBE_SEED)  # probe initialisation
    batch = make_batch(np.random.default_rng(PROBE_SEED), PROBE_N, hops_range=(1, 16))
    S = states_of(model, batch, dev)
    lhs, rhs = batch.lhs.numpy(), batch.rhs.numpy()
    half = PROBE_N // 2
    tr, te = slice(0, half), slice(half, PROBE_N)
    out = {
        "run": run,
        "n_steps": len(S) - 1,
        "probe_n": PROBE_N,
        "pointer": {},
        "reach": {},
        "digit_by_links": {},
        "digit_frontier": {},
        "query": {},
        "answer": {},
        "answer_by_hops": {},
    }

    demo = [make_batch(np.random.default_rng(DEMO_SEED + h), 1, hops=h) for h in DEMO_HOPS]
    demo_S = [states_of(model, b, dev) for b in demo]
    out["demo"] = [demo_record(b, model, dev) for b in demo]
    tgt, left = follow(lhs, rhs, MAX_D)
    for si, s in enumerate(S):
        accs = {}
        for d in range(1, MAX_D + 1):
            mask = left >= d + 1  # the target d steps on is still a variable
            m_tr, m_te = torch.from_numpy(mask[tr]), torch.from_numpy(mask[te])
            x_tr, y_tr = s[tr, :N_STATEMENTS][m_tr][:20000], torch.from_numpy(tgt[d][tr])[m_tr][:20000]
            x_te, y_te = s[te, :N_STATEMENTS][m_te][:10000], torch.from_numpy(tgt[d][te])[m_te][:10000]
            accs[d] = fit_probe(x_tr, y_tr, x_te, y_te, RHS_VOCAB, dev)
        out["pointer"][si] = accs
        out["reach"][si] = max([d for d, a in accs.items() if a is not None and a >= 0.9], default=0)
        # digit probe: does each assignment already hold the digit at the end of its chain? split by links left
        digit = torch.from_numpy(tgt[MAX_D] - N_VARS)  # after MAX_D steps every chain (<= 16 links here) has ended
        demo_x = torch.cat([ds[si][0, :N_STATEMENTS] for ds in demo_S])
        ok, demo_p = fit_probe(
            s[tr, :N_STATEMENTS].reshape(-1, s.shape[-1])[:40000],
            digit[tr].reshape(-1)[:40000],
            s[te, :N_STATEMENTS].reshape(-1, s.shape[-1])[:20000],
            digit[te].reshape(-1)[:20000],
            N_DIGITS,
            dev,
            rows=True,
            demo=demo_x,
        )
        for i, rec in enumerate(out["demo"]):  # probability the probe gives each assignment's true final digit
            p = demo_p[i * N_STATEMENTS : (i + 1) * N_STATEMENTS]
            rec["digit_prob"].append([round(float(p[j, rec["final_digit"][j]]), 3) for j in range(N_STATEMENTS)])
        n_left = torch.from_numpy(left[te]).reshape(-1)[:20000]
        out["digit_by_links"][si] = {
            int(n): float(ok[n_left == n].float().mean()) for n in range(1, MAX_D + 1) if (n_left == n).sum() >= 50
        }
        out["digit_frontier"][si] = max([n for n, a in out["digit_by_links"][si].items() if a >= 0.9], default=0)
        print(
            run,
            "step",
            si,
            "pointer reach",
            out["reach"][si],
            "digit frontier",
            out["digit_frontier"][si],
            {d: round(a, 2) for d, a in accs.items() if a is not None},
            flush=True,
        )

    hops, chain = batch.hops.numpy(), batch.chain.numpy()
    for si, s in enumerate(S):
        q = s[:, -1]
        accs = {}
        demo_q = torch.cat([ds[si][0, -1:] for ds in demo_S])
        for rec in out["demo"]:
            rec["query_hop_prob"].append([])
        for j in range(1, MAX_D):
            mask = hops >= j + 1
            m_tr, m_te = torch.from_numpy(mask[tr]), torch.from_numpy(mask[te])
            res = fit_probe(
                q[tr][m_tr],
                torch.from_numpy(chain[tr, j])[m_tr],
                q[te][m_te],
                torch.from_numpy(chain[te, j])[m_te],
                N_VARS,
                dev,
                demo=demo_q,
            )
            accs[j], demo_pj = res if res is not None else (None, None)
            for i, rec in enumerate(out["demo"]):  # probability the query state names the variable j hops on
                ok = demo_pj is not None and j < len(rec["chain"])
                rec["query_hop_prob"][-1].append(round(float(demo_pj[i, rec["chain"][j]]), 3) if ok else None)
        out["query"][si] = accs
        ok, demo_p = fit_probe(q[tr], batch.answer[tr], q[te], batch.answer[te], N_DIGITS, dev, rows=True, demo=demo_q)
        for i, rec in enumerate(out["demo"]):
            rec["query_prob"].append(round(float(demo_p[i, rec["answer"]]), 3))
        out["answer"][si] = float(ok.float().mean())
        h_te = torch.from_numpy(hops[te])
        out["answer_by_hops"][si] = {h: float(ok[h_te == h].float().mean()) for h in range(1, 17)}
        print(
            run,
            "step",
            si,
            "query",
            {j: round(a, 2) for j, a in accs.items() if a is not None},
            "answer by hops",
            {h: round(a, 2) for h, a in out["answer_by_hops"][si].items()},
            flush=True,
        )

    if isinstance(model, Looped):
        conv = {}
        for h in (2, 4, 8, 12, 16):
            m = torch.from_numpy(hops == h)
            conv[h] = [
                float(((S[i][m] - S[i - 1][m]).norm(dim=-1) / S[i][m].norm(dim=-1)).mean()) for i in range(2, len(S))
            ]  # entry 0 is loop 2 against loop 1
        out["convergence"] = conv
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--runs-dir", default=str(HERE / "results" / "runs"))
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    dev = device()
    for run in args.runs:
        res = analyse(run, dev, Path(args.runs_dir))
        Path(args.out, f"inside_{run}.json").write_text(json.dumps(res, indent=1))
        print("wrote", f"inside_{run}.json")


if __name__ == "__main__":
    main()
