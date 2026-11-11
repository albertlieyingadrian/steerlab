"""Step 1: the eval-awareness direction, and whether it separates the two framings on held-out prompts.

Built under the NEUTRAL system prompt (no policy), so it is the model's own "this is a test" representation, not the
policy's "now I should sandbag" signal. Recipe (difference of means at the last prompt token, decoder layer L):

    awareness   mean(train questions x build eval framings) - mean(train questions x build real-use framings)
    policy      the same contrast, but under the conditional policy (an upper bound: it also carries "sandbag now")
    unrelated   mean(sea prefixes) - mean(mountain prefixes), same questions, neutral system prompt

Held-out separation: test questions, each with one held-out framing in rotation (written for this study, never
used to build), projection on the awareness direction, AUROC with a bootstrap over questions. Also under the
policy prompt, and for the unrelated direction (which should not separate eval from real use).

Writes results/<model>/directions.npz and results/<model>/direction.json.
"""

import sys

import numpy as np

import common as C


def acts(model, tok, qs, prefixes, system, layers, rotate=False):
    """Every question x every prefix, or (rotate) question i with prefix i mod len(prefixes)."""
    texts, qid = [], []
    for i, q in enumerate(qs):
        for p in [prefixes[i % len(prefixes)]] if rotate else prefixes:
            texts.append(C.prompt(tok, q, p, system))
            qid.append(i)
    _, _, a = C.run(model, tok, texts, layers=layers)
    return a, np.array(qid)


def main(name, policy="rule"):
    tok, model = C.load(name)
    n_layers = len(C.decoder_layers(model))
    layers = sorted({round(n_layers * f) for f in (0.25, 0.4, 0.55, 0.7)})
    qs = C.load_questions(name)
    train, test = qs["train"], qs["test"]
    n_train, n_test = len(train), len(test)
    fb, fh = C.FRAMINGS["build"], C.FRAMINGS["heldout"]
    pol = C.POLICIES[policy]

    # build (train questions, build framings)
    e_n, _ = acts(model, tok, train, fb["eval"], C.NEUTRAL_SYSTEM, layers)
    r_n, _ = acts(model, tok, train, fb["real"], C.NEUTRAL_SYSTEM, layers)
    e_p, _ = acts(model, tok, train, fb["eval"], pol, layers)
    r_p, _ = acts(model, tok, train, fb["real"], pol, layers)
    s_n, _ = acts(model, tok, train, C.UNRELATED["sea"], C.NEUTRAL_SYSTEM, layers)
    m_n, _ = acts(model, tok, train, C.UNRELATED["mountain"], C.NEUTRAL_SYSTEM, layers)

    # held out (test questions, held-out framings): neutral and policy prompts
    he_n, qe = acts(model, tok, test, fh["eval"], C.NEUTRAL_SYSTEM, layers, rotate=True)
    hr_n, qr = acts(model, tok, test, fh["real"], C.NEUTRAL_SYSTEM, layers, rotate=True)
    he_p, _ = acts(model, tok, test, fh["eval"], pol, layers, rotate=True)
    hr_p, _ = acts(model, tok, test, fh["real"], pol, layers, rotate=True)

    dirs = {}
    report = {
        "model": C.MODELS[name],
        "n_layers": n_layers,
        "layers": layers,
        "policy": policy,
        "n_train_questions": n_train,
        "n_test_questions": n_test,
        "per_layer": {},
    }
    for L in layers:
        aw = e_n[L].mean(0) - r_n[L].mean(0)
        po = e_p[L].mean(0) - r_p[L].mean(0)
        un = s_n[L].mean(0) - m_n[L].mean(0)
        dirs[f"awareness_{L}"], dirs[f"policy_{L}"], dirs[f"unrelated_{L}"] = aw, po, un
        u = aw / np.linalg.norm(aw)
        proj_e, proj_r = e_n[L] @ u, r_n[L] @ u
        dprime = (proj_e.mean() - proj_r.mean()) / np.sqrt(0.5 * (proj_e.var() + proj_r.var()))
        row = {
            "norm_awareness": float(np.linalg.norm(aw)),
            "norm_policy": float(np.linalg.norm(po)),
            "norm_unrelated": float(np.linalg.norm(un)),
            "residual_norm_median": float(np.median(np.linalg.norm(r_n[L], axis=1))),
            "cos_awareness_policy": float(aw @ po / np.linalg.norm(aw) / np.linalg.norm(po)),
            "cos_awareness_unrelated": float(aw @ un / np.linalg.norm(aw) / np.linalg.norm(un)),
            "dprime_build": float(dprime),
        }
        for tag, he, hr in (("neutral", he_n, hr_n), ("policy", he_p, hr_p)):
            a, lo, hi = C.auroc_ci(he[L] @ u, hr[L] @ u, qe, qr)
            row[f"heldout_auroc_{tag}"] = [round(a, 4), round(lo, 4), round(hi, 4)]
        un_u = un / np.linalg.norm(un)
        a, lo, hi = C.auroc_ci(he_n[L] @ un_u, hr_n[L] @ un_u, qe, qr)
        row["heldout_auroc_unrelated_dir"] = [round(a, 4), round(lo, 4), round(hi, 4)]
        report["per_layer"][str(L)] = row
        print(L, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    (C.RESULTS / name).mkdir(parents=True, exist_ok=True)
    np.savez(C.RESULTS / name / "directions.npz", **dirs)
    C.write_json(C.RESULTS / name / "direction.json", report)


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or []))
