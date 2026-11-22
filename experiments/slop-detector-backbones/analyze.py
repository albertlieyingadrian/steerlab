"""Every number in the bake-off note, from saved scores only (no GPU): results/bakeoff.json.

    python analyze.py

Per arm: fixed and calibrated 3-way macro-F1 and AUROC on EditLens test, Enron, Llama and the fresh set, each paired
against Qwen3.5-9B; flags and misses on Human Detectors and Epoch AI at the flag cut (1% of our validation humans);
miss rates and AUROC by generator family on data v2's test split; training and inference cost; the predictions.
First it checks that the Qwen3.5-9B numbers reproduce the slop-detector note's final.json.
"""

from __future__ import annotations

import json
from importlib import import_module

import numpy as np

from bb import ARMS, DATA, LEDGER, RESULTS, RUNS, SD, sd

diag, tev, fresh_test = import_module("diag_voi_gap"), import_module("tune_editlens_val"), import_module("fresh_test")
FIXED = (1 / 6, 5 / 6)
NAMES = {
    "qwen3.5-9b": "Qwen3.5-9B (reference)",
    "qwen3.5-4b": "Qwen3.5-4B (reference)",
    "mistral-nemo-12b": "Mistral-NeMo-12B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "modernbert-large": "ModernBERT-large",
    "deberta-v3-large": "DeBERTa-v3-large",
}
REF_RUN = {"qwen3.5-9b": "f9-9b", "qwen3.5-4b": "b1x-4bk"}
PARAMS_B = {
    "qwen3.5-9b": 9.0,
    "qwen3.5-4b": 4.0,
    "mistral-nemo-12b": 12.2,
    "llama-3.1-8b": 8.0,
    "modernbert-large": 0.395,
    "deberta-v3-large": 0.435,
}
LABELLED = ("test", "test_enron", "test_llama", "fresh")


def load(arm: str, split: str) -> np.ndarray | None:
    if arm in REF_RUN:
        name = "val_scores.json" if split == "our_val" else f"scores/{split}.json"
        for p in (SD / "runs" / REF_RUN[arm] / name, RUNS / f"ref-{arm}" / name):
            if p.exists():
                return np.array(json.loads(p.read_text()))
        return None
    p = RUNS / f"bb-{arm}" / ("val_scores.json" if split == "our_val" else f"scores/{split}.json")
    return np.array(json.loads(p.read_text())) if p.exists() else None


def labels() -> dict:
    out = {s: sd.benchmark(s)["y"].to_numpy() for s in ("test", "test_enron", "test_llama")}
    out["fresh"] = fresh_test.fresh_rows()["text_type"].map(sd.EDITLENS_CLASS).to_numpy()
    out["editlens_val"] = tev.clean_val()[0]["y"].to_numpy()
    return out


def auroc(y3: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score((y3 > 0).astype(int), s)), 4)


def ci_mf1(y: np.ndarray, p: np.ndarray, reps: int = 2000, seed: int = 0) -> list[float]:
    """95% bootstrap interval of one arm's 3-way macro-F1 over texts."""
    rng = np.random.default_rng(seed)
    d = [diag.mf1(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(reps))]
    return [round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)]


def family(model: str) -> str:
    m = model.lower()
    for k in ("qwen", "deepseek", "gpt-oss", "gemma", "glm", "kimi"):
        if k in m:
            return k
    return "human" if m == "human" else "other"


def arm_numbers(arm: str, y: dict, ref: dict | None) -> dict:
    s_val = load(arm, "editlens_val")
    _, h, g = diag.fit_cuts(y["editlens_val"], s_val)
    ov = sd.read_jsonl(DATA / "val.jsonl")
    sv = load(arm, "our_val")
    flag = float(np.quantile(sv[[r["kind"] == "human" for r in ov]], 0.99))
    res = {"cuts_calibrated": [h, g], "flag_cut": round(flag, 6), "splits": {}}
    for split in LABELLED:
        s = load(arm, split)
        if s is None:
            continue
        yy = y[split]
        r = {"n": len(yy), "auroc": auroc(yy, s)}
        for arm_name, (lo, hi) in (("fixed", FIXED), ("calibrated", (h, g))):
            p = diag.bands(s, lo, hi)
            r[arm_name] = round(diag.mf1(yy, p), 4)
            r[f"{arm_name}_ci"] = ci_mf1(yy, p)
            if ref is not None and arm != "qwen3.5-9b":
                pr = ref["pred"][split][arm_name]
                b = sd.paired_boot(yy, p, pr, ("macro_f1",), reps=2000)["macro_f1"]
                r[f"{arm_name}_vs_9b"] = {"diff": b["diff"], "ci": b["diff_ci"]}
        res["splits"][split] = r
    for split in ("human_detectors", "epoch"):
        s = load(arm, split)
        if s is None:
            continue
        df = sd.benchmark(split)
        lab = df["label"].to_numpy()
        f = s >= flag
        r = {
            "humans": int((lab == 0).sum()),
            "humans_flagged": int((f & (lab == 0)).sum()),
            "ai": int((lab == 1).sum()),
            "ai_missed": int((~f & (lab == 1)).sum()),
        }
        if split == "epoch":
            for cond in sorted(set(df["condition"]) - {"human"}):
                m = (df["condition"] == cond).to_numpy()
                r[f"missed_{cond}"] = f"{int((~f & m).sum())}/{int(m.sum())}"
        res["splits"][split] = r
    s = load(arm, "ours_test")
    if s is not None:
        rows = sd.read_jsonl(DATA / "test.jsonl")
        fam = np.array([family(r["model"]) for r in rows])
        hum = fam == "human"
        out = {}
        for k in ["qwen", "non-qwen", *sorted(set(fam) - {"human"})]:
            m = (fam != "human") & (fam != "qwen") if k == "non-qwen" else fam == k
            if not m.any():
                continue
            sel = m | hum
            out[k] = {
                "n_ai": int(m.sum()),
                "miss_rate": round(float((s[m] < flag).mean()), 4),
                "auroc": auroc(np.where(m[sel], 1, 0), s[sel]),
            }
        out["human_flag_rate"] = round(float((s[hum] >= flag).mean()), 4)
        res["generator_family"] = out
    return res


def cost(arm: str) -> dict:
    led = sd.read_jsonl(LEDGER)
    train = sum(r["usd"] for r in led if r["what"] == f"train bb-{arm}")
    sweep = sum(r["usd"] for r in led if r["what"].startswith(f"train sweep-{arm}-"))
    out = {"train_usd": round(train, 2), "sweep_usd": round(sweep, 2), "params_b": PARAMS_B[arm]}
    t = RUNS / f"bb-{arm}" / "timing.json"
    if t.exists():
        d = json.loads(t.read_text())
        n = json.loads((RESULTS / "eval_counts.json").read_text())
        secs, texts = sum(d["seconds"].values()), sum(n[s] for s in d["seconds"])
        out["h100_seconds_per_1k_texts"] = round(1000 * secs / texts, 1)
        out["tokens_per_text"] = round(sum(d["tokens"].values()) / texts, 1)
    cfg = RUNS / f"bb-{arm}" / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        out["train_hours"] = round(c["seconds"] / 3600, 2)
        out["lr"] = c["lr"]
    return out


def predictions(arms: dict) -> list[dict]:
    new = [a for a in ARMS if a in arms]
    if len(new) < len(ARMS) or "qwen3.5-9b" not in arms:
        return []
    fr = {a: arms[a]["splits"]["fresh"]["calibrated"] for a in arms}
    best_new = max(new, key=lambda a: fr[a])
    llm, enc = ["mistral-nemo-12b", "llama-3.1-8b"], ["modernbert-large", "deberta-v3-large"]
    ood = {
        a: float(np.mean([arms[a]["splits"][s]["calibrated"] for s in ("test_enron", "test_llama")]))
        for a in [*new, "qwen3.5-9b"]
    }

    def gap(a):
        gf = arms[a]["generator_family"]
        return 100 * (gf["non-qwen"]["miss_rate"] - gf["qwen"]["miss_rate"])

    def own(a):
        return arms[a]["splits"]["test_llama"]["calibrated"] - arms[a]["splits"]["test"]["calibrated"]

    p1 = fr[best_new] - fr["qwen3.5-9b"]
    p3 = max(fr[a] for a in llm) - max(fr[a] for a in enc)
    p4 = gap("qwen3.5-9b") - float(np.mean([gap(a) for a in llm]))
    p5 = own("llama-3.1-8b") - own("mistral-nemo-12b")
    top_ood = max(ood, key=ood.get)
    return [
        {"id": "P1", "value": round(p1, 4), "best_new": best_new, "hit": bool(p1 < 0.015)},
        {
            "id": "P2",
            "value": top_ood,
            "ood": {k: round(v, 4) for k, v in ood.items()},
            "hit": top_ood == "mistral-nemo-12b",
        },
        {"id": "P3", "value": round(p3, 4), "hit": bool(p3 >= 0.03)},
        {
            "id": "P4",
            "value_points": round(p4, 2),
            "gaps": {a: round(gap(a), 2) for a in ["qwen3.5-9b", *new]},
            "hit": bool(abs(p4) <= 2),
        },
        {"id": "P5", "value": round(p5, 4), "hit": bool(p5 > 0)},
    ]


def main() -> None:
    y = labels()
    # sanity: the 9B reproduces the slop-detector note's fresh look
    note = json.loads((SD / "results" / "eval" / "final.json").read_text())["looks"]["fresh (look 1, the claim)"]
    ref = arm_numbers("qwen3.5-9b", y, None)
    want = (note["fixed"]["macro_f1"]["a"], note["calibrated"]["macro_f1"]["a"])
    got = (ref["splits"]["fresh"]["fixed"], ref["splits"]["fresh"]["calibrated"])
    assert abs(want[0] - got[0]) < 6e-4 and abs(want[1] - got[1]) < 6e-4, (want, got)
    print(f"reproduces the note's fresh look: fixed {got[0]}, calibrated {got[1]}")
    h, g = ref["cuts_calibrated"]
    ref["pred"] = {
        s: {"fixed": diag.bands(load("qwen3.5-9b", s), *FIXED), "calibrated": diag.bands(load("qwen3.5-9b", s), h, g)}
        for s in LABELLED
    }
    out = {"arms": {}}
    for arm, label in NAMES.items():
        if load(arm, "editlens_val") is None or load(arm, "fresh") is None:
            print(f"{arm}: no scores yet")
            continue
        r = ref if arm == "qwen3.5-9b" else arm_numbers(arm, y, ref)
        r = {k: v for k, v in r.items() if k != "pred"}
        if arm in ARMS:
            r["cost"] = cost(arm)
        elif (RUNS / f"ref-{arm}" / "timing.json").exists():  # wall time incl. model load: an upper bound
            t = json.loads((RUNS / f"ref-{arm}" / "timing.json").read_text())
            r["cost"] = {
                "h100_seconds_per_1k_texts": round(1000 * t["wall_seconds_incl_load"] / t["texts"], 1),
                "timing_note": "wall time of one scoring call incl. model load and queueing",
                "params_b": PARAMS_B[arm],
            }
        out["arms"][arm] = r
        sp = r["splits"]
        print(
            f"{label:26} "
            + "  ".join(
                f"{s} {sp[s]['calibrated']:.3f}"
                + (f" ({sp[s]['calibrated_vs_9b']['diff']:+.3f})" if "calibrated_vs_9b" in sp[s] else "")
                for s in LABELLED
                if s in sp
            )
        )
    out["predictions"] = predictions(out["arms"])
    out["spend_usd"] = round(sum(r["usd"] for r in sd.read_jsonl(LEDGER)), 2)
    (RESULTS / "bakeoff.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    for p in out["predictions"]:
        print(p)


if __name__ == "__main__":
    main()
