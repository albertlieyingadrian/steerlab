"""Every number in the note, by the rules in PREREG.md.

    python 6_analyse.py            # dev and test
    python 6_analyse.py --dev      # dev only (selection happens here, before any test score is printed)

Reads the judges' cached replies and writes:
- results/scores.csv: one row per sampled video, one column per judge (higher = more likely to break physics);
- results/metrics_video.csv: per judge and split, AUROC with a 95% prompt-bootstrap interval, AUROC on clear cases,
  Spearman with the human rating, the dev threshold, accuracy and F1 at it, and dollars per video;
- results/selection.json: the cheap judge chosen on dev, and the paired test differences against both released judges;
- results/metrics_per_model.csv, results/rule_scores.csv, results/metrics_rules.csv, results/trap_scores.csv,
  results/traps_result.csv, results/summary.json.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import LAWS, RESULTS, auroc, best_threshold, f1_acc, group_boot, wilson

MODELS = ["opus", "sonnet", "haiku"]
PRICE = {"opus": (4.0, 20.0), "sonnet": (2.0, 10.0), "haiku": (0.10, 0.50)}  # batch halves these
WMB_LAWS = ["newton", "mass", "fluid", "penetration", "gravity"]
HEUR = ["motion", "jerk", "diff_spike", "scale_jump", "divergence", "warp_error"]
# PREREG: which WorldModelBench question scores which VideoPhy-2 law group
LAW_TO_WMB = {
    "Gravity": ["gravity"],
    "Conservation of Mass": ["mass"],
    "Hardness": ["mass"],
    "Elasticity": ["mass"],
    "Conservation of Momentum": ["newton"],
    "Friction": ["newton"],
    "Other": WMB_LAWS,
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def claude_rows(source: str) -> pd.DataFrame:
    out = []
    for r in load_jsonl(RESULTS / f"claude_{source}.jsonl"):
        if r.get("pilot"):
            continue
        d = r["data"] or {}
        pv = d.get("p_violation")
        out.append({"config": r["config"], "model": r["model"], "rep": r["rep"], "name": r["name"],
                    "p": float(np.clip(pv, 0, 100)) if isinstance(pv, (int, float)) else np.nan,
                    "pc": d.get("pc"), "data": d, "in_tokens": r["in_tokens"],
                    "out_tokens": r["out_tokens"]})  # fmt: skip
    return pd.DataFrame(out)


def usd(model: str, tin: float, tout: float) -> float:
    pi, po = PRICE[model]
    return (tin * pi + tout * po) / 1e6 / 2


def fit_dev(s: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Logistic regression of violation on cols (log1p, standardised), fitted on dev rows, applied to every row."""
    x = np.log1p(s[cols].clip(lower=0).astype(float))
    ok = x.notna().all(axis=1)
    dev = ok & (s.split == "dev")
    sc = StandardScaler().fit(x[dev])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(x[dev]), s.violation[dev])
    out = pd.Series(np.nan, index=s.index)
    out[ok] = lr.predict_proba(sc.transform(x[ok]))[:, 1]
    return out


def video_scores(items: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """One column per judge; also dollars per video and each column's group."""
    s = items[["video", "split", "prompt_id", "model_name", "pc", "violation"]].copy().set_index("video")
    cost: dict[str, float] = {}
    group: dict[str, str] = {}

    wmb = pd.DataFrame(load_jsonl(RESULTS / "wmb_items.jsonl"))
    if len(wmb):
        wmb = wmb.drop_duplicates("video").set_index("video")
        s["wmb"] = wmb[[f"{k}_p" for k in WMB_LAWS]].max(axis=1)
        s["wmb_mean"] = wmb[[f"{k}_p" for k in WMB_LAWS]].mean(axis=1)
        # the official script counts a law as adhered when "no" appears in the reply; here, laws not adhered
        s["wmb_official"] = sum((~wmb[f"{k}_reply"].str.lower().str.contains("no")).astype(int) for k in WMB_LAWS)
        for k in WMB_LAWS:
            s[f"wmb_{k}"] = wmb[f"{k}_p"]
        group.update({"wmb": "released", "wmb_mean": "released-variant", "wmb_official": "released-variant"})
        cost["wmb"] = cost["wmb_mean"] = cost["wmb_official"] = 1.10 / 3600 * float(wmb.seconds.mean())
    s["autoeval"] = 6 - items.set_index("video").pc_pred
    group["autoeval"] = "released"

    c = claude_rows("items")
    if len(c):
        for (config, model), g in c[c.rep == 0].groupby(["config", "model"]):
            col = f"{config}_{model}"
            s[col] = g.set_index("name").p
            cost[col] = usd(model, g.in_tokens.mean(), g.out_tokens.mean())
            # amendment 1: the no-frames runs are a post-hoc control, never a candidate for selection
            group[col] = "control" if config.endswith("_noframes") else "claude"
        for model in ["sonnet", "haiku"]:
            g = c[(c.config == "checklist") & (c.model == model)]
            if g.rep.nunique() >= 3:
                col = f"selfcons3_{model}"
                s[col] = g.pivot_table(index="name", columns="rep", values="p")[[0, 1, 2]].mean(axis=1)
                cost[col] = usd(model, g.in_tokens.mean(), g.out_tokens.mean()) * 3
                group[col] = "claude"
        if all(f"checklist_{m}" in s for m in MODELS):
            s["ens_checklist"] = sum(s[f"checklist_{m}"].rank(pct=True) for m in MODELS) / 3
            cost["ens_checklist"] = sum(cost[f"checklist_{m}"] for m in MODELS)
            group["ens_checklist"] = "claude"

    h = pd.read_csv(RESULTS / "heuristics.csv").set_index("video")
    for f in HEUR:
        s[f"h_{f}"] = h[f]
        group[f"h_{f}"] = "heuristic"
        cost[f"h_{f}"] = 0.0
    s["h_combined"] = fit_dev(s, [f"h_{f}" for f in HEUR])
    group["h_combined"] = "heuristic"
    cost["h_combined"] = 0.0

    # reference: the same annotators' other answers (rule marks), not an independent rater
    rules = pd.read_csv(RESULTS / "rules.csv")
    extra = items.set_index("video").human_violated_rules.map(
        lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0
    )
    s["annot_rules"] = rules.groupby("video").violated.sum().reindex(s.index).fillna(0) + extra
    group["annot_rules"] = "reference"
    return s, cost, group


def metrics(s: pd.DataFrame, col: str, split: str, thr: float | None) -> dict:
    d = s[(s.split == split) & s[col].notna()]
    y, x, grp = d.violation.values, d[col].values.astype(float), d.prompt_id.values
    a, lo, hi = group_boot(lambda i: auroc(y[i], x[i]), grp)
    clear = d[(d.pc <= 2) | (d.pc >= 4)]
    out = {"judge": col, "split": split, "n": len(d), "auroc": a, "lo": lo, "hi": hi,
           "auroc_clear": auroc(clear.violation, clear[col]),
           "spearman": float(spearmanr(-x, d.pc).statistic)}  # fmt: skip
    if thr is not None:
        f1, acc = f1_acc(y, x >= thr)
        out.update({"threshold": thr, "acc": acc, "f1": f1, "flag_rate": float((x >= thr).mean())})
    return out


def paired(s: pd.DataFrame, a: str, b: str, split: str = "test") -> dict:
    d = s[(s.split == split) & s[a].notna() & s[b].notna()]
    y, xa, xb, grp = d.violation.values, d[a].values, d[b].values, d.prompt_id.values
    diff, lo, hi = group_boot(lambda i: auroc(y[i], xa[i]) - auroc(y[i], xb[i]), grp)
    verdict = "beats" if lo > 0 else ("worse" if hi < 0 else "level")
    return {"judge": a, "versus": b, "n": len(d), "diff": diff, "lo": lo, "hi": hi, "verdict": verdict}


def rule_scores(items: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Rule-level scores: Claude `rules` config per model (each rule's p_violated) and WorldModelBench's mapped law."""
    rules = pd.read_csv(RESULTS / "rules.csv")
    rl = import_module("4_claude").rule_lists()
    c = claude_rows("items")
    c = c[(c.config == "rules") & (c.rep == 0)] if len(c) else c
    for model in MODELS:
        g = c[c.model == model].set_index("name") if len(c) else pd.DataFrame()
        vals = []
        for r in rules.itertuples():
            v = np.nan
            if r.video in g.index:
                order = rl.get(r.video, [])
                by_id = {x.get("id"): x for x in g.loc[r.video, "data"].get("rules", [])}
                k = order.index(r.rule) + 1 if r.rule in order else None
                if k in by_id:
                    v = float(np.clip(by_id[k].get("p_violated", np.nan), 0, 100))
            vals.append(v)
        rules[f"rules_{model}"] = vals
    if "wmb" in s:
        rules["wmb"] = [s.loc[r.video, [f"wmb_{k}" for k in LAW_TO_WMB[r.law_group]]].max() for r in rules.itertuples()]
    rules["prompt_id"] = rules.video.map(items.set_index("video").prompt_id)
    return rules


def rule_metrics(rules: pd.DataFrame) -> pd.DataFrame:
    out = []
    judges = [c for c in rules.columns if (c.startswith("rules_") or c == "wmb") and rules[c].notna().any()]
    for split in ["dev", "test"]:
        for law in [*LAWS, "Other", "All"]:
            d = rules[rules.split == split]
            if law != "All":
                d = d[d.law_group == law]
            for j in judges:
                dd = d[d[j].notna()]
                if dd.violated.nunique() < 2:
                    continue
                y, x, g = dd.violated.values, dd[j].values, dd.prompt_id.values
                a, lo, hi = group_boot(lambda i, y=y, x=x: auroc(y[i], x[i]), g)
                out.append({"split": split, "law": law, "judge": j, "n": len(dd), "violated": int(y.sum()),
                            "auroc": a, "lo": lo, "hi": hi})  # fmt: skip
            if "wmb" in judges:
                for j in [x for x in judges if x != "wmb"]:
                    dd = d[d[j].notna() & d.wmb.notna()]
                    if dd.violated.nunique() < 2:
                        continue
                    y, xa, xb, g = dd.violated.values, dd[j].values, dd.wmb.values, dd.prompt_id.values
                    diff, lo, hi = group_boot(lambda i, y=y, xa=xa, xb=xb: auroc(y[i], xa[i]) - auroc(y[i], xb[i]), g)
                    out.append({"split": split, "law": law, "judge": f"{j}-minus-wmb", "n": len(dd),
                                "violated": int(y.sum()), "auroc": diff, "lo": lo, "hi": hi,
                                "tested": bool(y.sum() >= 15)})  # fmt: skip
    return pd.DataFrame(out)


def trap_scores() -> pd.DataFrame:
    t = pd.read_csv(RESULTS / "traps.csv").set_index("trap")[["video", "kind"]].copy()
    wmb = pd.DataFrame(load_jsonl(RESULTS / "wmb_traps.jsonl"))
    if len(wmb):
        wmb = wmb.drop_duplicates("video").set_index("video")
        t["wmb"] = wmb[[f"{k}_p" for k in WMB_LAWS]].max(axis=1)
        t["wmb_official"] = sum((~wmb[f"{k}_reply"].str.lower().str.contains("no")).astype(int) for k in WMB_LAWS)
        for k in WMB_LAWS:
            t[f"wmb_{k}"] = wmb[f"{k}_p"]
    c = claude_rows("traps")
    if len(c):
        for (config, model), g in c[c.rep == 0].groupby(["config", "model"]):
            t[f"{config}_{model}"] = g.set_index("name").p
    h = pd.read_csv(RESULTS / "heuristics_traps.csv").set_index("trap")
    for f in HEUR:
        t[f"h_{f}"] = h[f]
    return t


def trap_metrics(t: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = []
    clean = t[t.kind == "clean"].set_index("video")
    for j in [c for c in t.columns if c not in ("video", "kind")]:
        for kind in ["reversed", "frozen", "resized"]:
            d = t[t.kind == kind].set_index("video")
            b_all = clean[j].reindex(d.index)
            both = d[j].notna() & b_all.notna()
            a, b = d[j][both], b_all[both]
            if not len(a):
                continue
            higher = float((a > b).sum() + 0.5 * (a == b).sum())
            lo, hi = wilson(round(higher), len(a))
            row = {"judge": j, "kind": kind, "n": len(a), "doctored_higher": higher, "share": higher / len(a),
                   "lo": lo, "hi": hi, "passes": bool(higher >= 20)}  # fmt: skip
            if thresholds.get(j) is not None:
                row["flagged"] = float((a >= thresholds[j]).mean())
                row["clean_flagged"] = float((b >= thresholds[j]).mean())
            out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    items = pd.read_csv(RESULTS / "items.csv")
    s, cost, group = video_scores(items)
    claude_cols = [c for c, g in group.items() if g == "claude" and s.loc[s.split == "dev", c].notna().mean() > 0.97]

    rows, thresholds = [], {}
    for col in [c for c in group if c in s]:
        d = s[(s.split == "dev") & s[col].notna()]
        thresholds[col] = best_threshold(d.violation, d[col]) if len(d) else None
        rows.append(metrics(s, col, "dev", thresholds[col]))
    dev = pd.DataFrame(rows).set_index("judge")
    ranked = sorted(claude_cols, key=lambda c: (-round(dev.loc[c, "auroc"], 3), cost.get(c, 0)))
    chosen = ranked[0]
    print("dev:")
    print(dev[["n", "auroc", "lo", "hi", "auroc_clear", "spearman"]].round(3).to_string())
    print("chosen on dev:", chosen, f"(${cost[chosen]:.4f}/video)")

    s["chosen_plus_heur"] = fit_dev(s, [chosen] + [f"h_{f}" for f in HEUR])
    group["chosen_plus_heur"] = "claude+heuristic"
    cost["chosen_plus_heur"] = cost[chosen]
    d = s[s.split == "dev"]
    thresholds["chosen_plus_heur"] = best_threshold(d.violation, d.chosen_plus_heur)
    rows.append(metrics(s, "chosen_plus_heur", "dev", thresholds["chosen_plus_heur"]))
    print("dev chosen_plus_heur AUROC", round(rows[-1]["auroc"], 3))
    if args.dev:
        return

    for col in [c for c in group if c in s]:
        rows.append(metrics(s, col, "test", thresholds[col]))
    met = pd.DataFrame(rows)
    met["group"] = met.judge.map(group)
    met["usd_per_video"] = met.judge.map(cost)
    met.to_csv(RESULTS / "metrics_video.csv", index=False)
    s.reset_index().to_csv(RESULTS / "scores.csv", index=False)

    primary = [paired(s, chosen, "wmb"), paired(s, chosen, "autoeval"), paired(s, "autoeval", "wmb")]
    secondary = [paired(s, c, "wmb") for c in claude_cols if c != chosen] + [paired(s, "h_combined", "wmb")]
    secondary += [paired(s, c, "autoeval") for c in claude_cols if c != chosen]
    secondary += [paired(s, "chosen_plus_heur", chosen), paired(s, "chosen_plus_heur", "autoeval")]
    # amendment 1 (post hoc): what the frames add over the prompt and the rule text alone
    for a, b in [
        ("rules_opus", "rules_noframes_opus"),
        ("plain_opus", "plain_noframes_opus"),
        ("rules_noframes_opus", "autoeval"),
        ("plain_opus", "autoeval"),
    ]:
        if a in s and b in s:
            secondary.append(paired(s, a, b))
    sel = {"chosen": chosen, "dev_ranking": [[c, round(float(dev.loc[c, "auroc"]), 4)] for c in ranked],
           "primary": primary, "secondary": secondary}  # fmt: skip
    (RESULTS / "selection.json").write_text(json.dumps(sel, indent=1))

    pm = []
    for col in ["wmb", "autoeval", chosen, "h_combined"]:
        for m, g in s[s.split == "test"].groupby("model_name"):
            pm.append({"judge": col, "video_model": m, "n": len(g), "violations": int(g.violation.sum()),
                       "auroc": auroc(g.violation, g[col])})  # fmt: skip
    pd.DataFrame(pm).to_csv(RESULTS / "metrics_per_model.csv", index=False)

    rules = rule_scores(items, s)
    rules.to_csv(RESULTS / "rule_scores.csv", index=False)
    rmet = rule_metrics(rules)
    rmet.to_csv(RESULTS / "metrics_rules.csv", index=False)

    t = trap_scores()
    t.reset_index().to_csv(RESULTS / "trap_scores.csv", index=False)
    tmet = trap_metrics(t, thresholds)
    tmet.to_csv(RESULTS / "traps_result.csv", index=False)

    test = met[met.split == "test"].set_index("judge")
    print("\ntest:")
    cols = ["group", "n", "auroc", "lo", "hi", "auroc_clear", "spearman", "acc", "f1", "usd_per_video"]
    print(test[cols].round(3).to_string())
    print("\npaired (primary):")
    for p in primary:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in p.items()})
    print("\nrules (test):")
    print(rmet[rmet.split == "test"].round(3).to_string())
    print("\ntraps:")
    print(tmet.round(2).to_string())

    summary = {
        "n_test": int((s.split == "test").sum()),
        "n_dev": int((s.split == "dev").sum()),
        "violations_test": int(s[s.split == "test"].violation.sum()),
        "prompts_test": int(s[s.split == "test"].prompt_id.nunique()),
        "chosen": chosen,
        "wmb_yes_rate_by_law_test": {k: float((s.loc[s.split == "test", f"wmb_{k}"] > 0.5).mean()) for k in WMB_LAWS}
        if "wmb" in s
        else {},
        "claude_parse_failures": int(claude_rows("items").p.isna().sum()),
        "spend_usd": pd.read_csv(RESULTS / "spend.csv").groupby("service").usd.sum().round(2).to_dict(),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
