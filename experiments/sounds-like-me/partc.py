"""Part C runner: voice vectors on Qwen3-8B (presence vs use).

    python partc.py build      # vectors for all 45 authors, the model default, the slop vector -> data/partc_vectors.npz
    python partc.py tune       # dev only: layer x strength grid, logged to results.tsv -> results/partc_choice.json
    python partc.py eval       # held-out conditions -> data/drafts/partc__heldout.jsonl (then score.py + honesty.py)
    python partc.py monitor    # presence: slop-vector projection of Part B drafts vs detector score

Conditions (eval): none (k0), own (author's vector), wrong (another author's vector), rand0..2 (norm-matched random),
k8 (8-sample prompt, no vector), k8_own (8 samples + own vector). Strength and layer come from `tune`.
Design guards: random and wrong-author controls; fluency (NLL under the unsteered model) reported and matched;
the default-voice texts come from dev drafts only; the slop vector's human pool is floor authors, never evaluation authors.
"""

import json
import sys
import time

import numpy as np

from common import DATA, HERE, VoiceScale, authors, record_spend, texts
from gen import chat_messages, jobs

LAYERS = [12, 18, 24]
STRENGTHS = [0.1, 0.2, 0.4]  # x mean token residual norm at that layer
MAX_NLL_RISE = 0.3


def charge(seconds):
    from modal_gen import charge as c

    return c(seconds)


def load_vectors():
    z = np.load(DATA / "partc_vectors.npz", allow_pickle=True)
    return z["vecs"].item(), z["norms"].item()


def build():
    import modal

    fn = modal.Function.from_name("sounds-like-me-partc", "vectors")
    A = authors()
    sets = {f"author:{a['author']}": texts(a, "samples") for a in A["authors"]}
    sets["default"] = [
        json.loads(l)["text"]
        for l in (DATA / "drafts" / "Qwen3-8B-pilot__dev.jsonl").read_text().splitlines()
        if l.strip() and json.loads(l)["arm"] == "k0"
    ]
    sets["human_pool"] = [texts(r, "samples")[0] for r in A["floor"][:40]]
    t0 = time.time()
    r = fn.remote(sets, LAYERS)
    record_spend("modal", charge(time.time() - t0), f"partc vectors n={len(sets)}")
    vecs = {name: {int(L): np.array(v, np.float32) for L, v in per.items()} for name, per in r["vectors"].items()}
    voice = {n[7:]: {L: vecs[n][L] - vecs["default"][L] for L in LAYERS} for n in vecs if n.startswith("author:")}
    slop = {L: vecs["default"][L] - vecs["human_pool"][L] for L in LAYERS}
    np.savez(
        DATA / "partc_vectors.npz",
        vecs=np.array(dict(voice=voice, slop=slop), dtype=object),
        norms=np.array({int(k): v for k, v in r["norms"].items()}, dtype=object),
    )
    print("built", len(voice), "voice vectors; norms", r["norms"])


def _run(spec):
    import modal

    fn = modal.Function.from_name("sounds-like-me-partc", "steer")
    t0 = time.time()
    r = fn.remote(spec)
    record_spend("modal", charge(time.time() - t0), f"partc steer n={len(spec)}")
    return r["outputs"], r["nll"]


def tune():
    V, N = load_vectors()
    A = authors()
    J = jobs("dev", 5, 4, ["k0"])
    extra = [float(x) for x in sys.argv[2:]]  # e.g. `tune 0.6 0.8 1.2` extends layer 24 only
    grid = [(None, 0.0)] + ([(24, s) for s in extra] if extra else [(L, s) for L in LAYERS for s in STRENGTHS])
    spec = []
    for L, s in grid:
        for j in J:
            au = j["rec"]["author"]
            spec.append(
                dict(
                    messages=chat_messages(j["rec"], j["brief"], "k0"),
                    vid=f"{au}|{L}|{s}",
                    vector=None if L is None else V["voice"][au][L].tolist(),
                    layer=L or 18,
                    coef=0.0 if L is None else s * N[L],
                )
            )
    outs, nll = _run(spec)
    dev = {a["author"]: a for a in A["authors"] if a["split"] == "dev"}
    res, k = {}, 0
    for L, s in grid:
        by = {}
        for j in J:
            by.setdefault(j["rec"]["author"], []).append(k)
            k += 1
        pos, nl = [], []
        for au, idx in by.items():
            eps = [outs[i] for i in idx if outs[i] and outs[i].strip()]
            nl += [nll[i] for i in idx if nll[i] is not None]
            if eps:
                pos.append(VoiceScale(dev[au], A["floor"], "strict").position(eps))
        res[f"{L}|{s}"] = dict(layer=L, strength=s, position=float(np.mean(pos)), nll=float(np.mean(nl)))
    base = res["None|0.0"]
    prev = HERE / "results" / "partc_choice.json"
    if extra and prev.exists():  # merge with the earlier grid so the choice covers every setting tried
        old = json.loads(prev.read_text())["grid"]
        res = {**old, **{k: v for k, v in res.items() if k != "None|0.0"}}
    ok = [v for v in res.values() if v["layer"] is not None and v["nll"] - base["nll"] <= MAX_NLL_RISE]
    choice = max(ok, key=lambda v: v["position"]) if ok else None
    with (HERE / "results.tsv").open("a") as fh:
        for k_, v in res.items():
            fh.write(
                f"partc-{k_}\t-\t{v['position']:.3f}\t{base['position']:.3f}\t-\t"
                f"{'kept' if v is choice else 'grid'}\tsteer layer {v['layer']} x{v['strength']} "
                f"nll {v['nll']:.3f} (base {base['nll']:.3f})\n"
            )
    (HERE / "results" / "partc_choice.json").write_text(json.dumps(dict(grid=res, choice=choice), indent=1))
    print(json.dumps(dict(base=base, choice=choice), indent=1))


def eval_():
    V, N = load_vectors()
    ch = json.loads((HERE / "results" / "partc_choice.json").read_text())["choice"]
    L, s = ch["layer"], ch["strength"]
    coef = s * N[L]
    A = authors()
    names = [a["author"] for a in A["authors"] if a["split"] == "heldout"]
    rng = np.random.default_rng(20261120)
    rands = [rng.standard_normal(V["voice"][names[0]][L].shape).astype(np.float32) for _ in range(3)]
    spec, meta = [], []
    for j in jobs("heldout", None, 4, ["k0"]):
        au = j["rec"]["author"]
        wrong = names[(names.index(au) + 7) % len(names)]
        conds = [("none", None, "k0"), ("own", V["voice"][au][L], "k0"), ("wrong", V["voice"][wrong][L], "k0")]
        conds += [(f"rand{i}", r, "k0") for i, r in enumerate(rands)]
        conds += [("k8", None, "k8_rand"), ("k8_own", V["voice"][au][L], "k8_rand")]
        for cname, vec, arm in conds:
            spec.append(
                dict(
                    messages=chat_messages(j["rec"], j["brief"], arm, template="v3"),
                    vid=f"{cname}|{au if cname in ('own', 'k8_own') else (wrong if cname == 'wrong' else '')}",
                    vector=None if vec is None else vec.tolist(),
                    layer=L,
                    coef=0.0 if vec is None else coef,
                )
            )
            meta.append((j, cname))
    outs, nll = _run(spec)
    out = DATA / "drafts" / "partc__heldout.jsonl"
    with out.open("w") as fh:
        for (j, cname), t, n in zip(meta, outs, nll, strict=False):
            fh.write(
                json.dumps(
                    dict(
                        job=f"{j['job']}|{cname}",
                        model="partc-Qwen3-8B",
                        author=j["rec"]["author"],
                        brief=j["brief"]["id"],
                        arm=cname,
                        text=t or "",
                        nll=n,
                    )
                )
                + "\n"
            )
    print("wrote", out, len(meta), "drafts at layer", L, "strength", s)


def monitor():
    """Presence: does a draft's projection on the slop vector track the detector score (Part B drafts)?"""
    import modal
    from scipy.stats import spearmanr

    V, _ = load_vectors()
    rows = []
    for p in sorted((DATA / "scored").glob("*__heldout.json")):
        rows += [r for r in json.loads(p.read_text())["drafts"] if r["text"].strip() and r["model"] != "ctrl"]
    rows = rows[:: max(1, len(rows) // 600)][:600]
    fn = modal.Function.from_name("sounds-like-me-partc", "vectors")
    t0 = time.time()
    r = fn.remote({f"d{i}": [x["text"]] for i, x in enumerate(rows)}, LAYERS)
    record_spend("modal", charge(time.time() - t0), f"partc monitor n={len(rows)}")
    out = {}
    for L in LAYERS:
        sv = V["slop"][L] / np.linalg.norm(V["slop"][L])
        proj = [float(np.dot(np.array(r["vectors"][f"d{i}"][str(L)]), sv)) for i in range(len(rows))]
        out[L] = dict(
            spearman_vs_detector=round(float(spearmanr(proj, [x["voi380"] for x in rows]).correlation), 4), n=len(rows)
        )
    (HERE / "results" / "partc_monitor.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    dict(build=build, tune=tune, eval=eval_, monitor=monitor)[sys.argv[1]]()
