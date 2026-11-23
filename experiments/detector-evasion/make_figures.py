"""Draw the note's figures from results/, printing every number plotted.

    python make_figures.py

evasion.png   held-out evasion of the 9B against training step, per arm: strict (solid) and loose (dotted), with 95%
              bootstrap intervals over the 48 prompts on the strict line.
transfer.png  share of held-out rewrites each detector flags (threshold: 1% of 318 human passages), per arm's final
              checkpoint, beside the untouched sources and the two baselines.
hacks.png     per arm's final checkpoint: share of held-out rewrites that loop, that hit the token limit, or carry odd
              characters, and the share of the source's key claims lost (1 - strict claim recall).
defence.png   flag rate of the untouched, defended and control-retrained 9B on each arm's final rewrites.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIG = HERE / "report" / "figures"
ACCENT, INK, SOFT, FAINT, RULE = "#c9561c", "#1d272a", "#586468", "#8a9497", "#e3e7e8"
RUNS = ["full-s0", "full-s1", "no-loop-s0", "no-content-s0", "det-only-s0", "shuffled-s0"]
LABEL = {"source": "AI source", "prompt-only": "prompt only", "best-of-8": "best of 8",
         "full-s0": "full reward (seed 0)",
         "full-s1": "full reward (seed 1)", "no-loop-s0": "no loop term", "no-content-s0": "no content term",
         "det-only-s0": "detector only", "shuffled-s0": "shuffled reward"}  # fmt: skip
STYLE = {"full-s0": (ACCENT, "-"), "full-s1": ("#e59a6c", "-"), "no-loop-s0": ("#8c6bb1", "-"),
         "no-content-s0": ("#2c7fb8", "-"), "det-only-s0": (INK, "-"), "shuffled-s0": (FAINT, "--")}  # fmt: skip
DETS = [
    ("flag_9b", "our 9B (trained against)"),
    ("flag_voi380", "voi-380M"),
    ("flag_desklib", "Desklib"),
    ("flag_modernbert", "ModernBERT (bake-off)"),
    ("flag_binoculars", "Binoculars"),
    ("flag_fastdetect", "Fast-DetectGPT"),
]
HACKS = [("loop", "loops"), ("cut", "cut off at the token limit"), ("claims_lost", "share of claims lost"),
         ("odd", "odd characters")]  # fmt: skip
NONWORD = ("nonword", "words not in the reference vocabulary")

plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": SOFT, "ytick.color": SOFT, "legend.frameon": False,
                     "font.family": "sans-serif"})  # fmt: skip


def value(c: dict) -> tuple[float, float, float]:
    return c["value"], c["lo"], c["hi"]


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)


def evasion(G: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    base = G["prompt-only"]["heldout"]
    for run in RUNS:
        steps = sorted(int(g.split("@")[1]) for g in G if g.startswith(run + "@"))
        if not steps:
            continue
        cells = [base, *[G[f"{run}@{s}"]["heldout"] for s in steps]]
        xs = [0, *steps]
        col, ls = STYLE[run]
        v = np.array([value(c["evade_strict"]) for c in cells])
        ax.plot(xs, v[:, 0], color=col, ls=ls, marker="o", ms=3.5, label=LABEL[run])
        ax.fill_between(xs, v[:, 1], v[:, 2], color=col, alpha=0.10, lw=0)
        loose = [c["evade_loose"]["value"] for c in cells]
        ax.plot(xs, loose, color=col, ls=":", lw=1)
        print("evasion", run, xs, "strict", [round(float(x), 3) for x in v[:, 0]], "loose", loose)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("held-out rewrites that pass the 9B")
    ax.legend(fontsize=8)
    ax.grid(axis="y", color=RULE, lw=0.6)
    save(fig, "evasion.png")


def grouped(G: dict, names: list[str], keys: list[tuple[str, str]], fname: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    width = 0.8 / len(keys)
    shades = [ACCENT, "#3f4a4d", "#6c777a", "#939c9e", "#b5bdbf", "#d5dadb"]
    for j, (key, lab) in enumerate(keys):
        for i, g in enumerate(names):
            c = G[g]["heldout"].get(key)
            if g == "source" and key == "flag_9b":
                c = {"value": 1.0, "lo": 1.0, "hi": 1.0}  # Episode 3: all 96 sources flagged
            if not c:
                continue
            x = i + (j - (len(keys) - 1) / 2) * width
            v, lo, hi = value(c)
            ax.bar(x, v, width * 0.92, color=shades[j], label=lab if i == 1 else None)
            ax.plot([x, x], [lo, hi], color=INK, lw=0.7)
            print(fname, g, key, v, lo, hi)
    ax.set_xticks(range(len(names)), [LABEL[g.split("@")[0]] for g in names], rotation=22, ha="right")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=len(keys))
    ax.grid(axis="y", color=RULE, lw=0.6)
    save(fig, fname)


def defence(D: dict, finals: list[str]) -> None:
    names = ["source", "prompt-only", *finals]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    dets = [("untouched", "original 9B", FAINT), ("control", "retrained on untouched rewrites", "#586468"),
            ("defended", "retrained on the rewriter's output", ACCENT)]  # fmt: skip
    for j, (det, lab, col) in enumerate(dets):
        for i, g in enumerate(names):
            c = D["detectors"][det]["flags"][f"{g}|heldout"]["orig"]
            x = i + (j - 1) * 0.27
            v, lo, hi = value(c)
            ax.bar(x, v, 0.25, color=col, label=lab if i == 0 else None)
            ax.plot([x, x], [lo, hi], color=INK, lw=0.7)
            print("defence", det, g, v, lo, hi)
    ax.set_xticks(range(len(names)), [LABEL[g.split("@")[0]] for g in names], rotation=22, ha="right")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("flagged at the original threshold")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
    ax.grid(axis="y", color=RULE, lw=0.6)
    save(fig, "defence.png")


def viewer(final: dict) -> None:
    """report/data.js for the rewrite viewer: the 48 held-out Qwen3-8B sources (sample 0 of each prompt), each with
    every arm's final rewrite and its 9B score, flags, claim recall and hack checks."""
    import csv

    groups = ["prompt-only", *[final[r] for r in RUNS if r in final]]
    with (RESULTS / "per_rewrite.csv").open() as f:
        rows = {(r["group"], r["sid"]): r for r in csv.DictReader(f)}
    src = [json.loads(x) for x in (RESULTS / "sources.jsonl").open() if '"split": "heldout"' in x]
    src = [s for s in src if s["sid"].endswith("-0")]
    texts = {g: {json.loads(x)["sid"]: json.loads(x)["text"] for x in (RESULTS / "eval" / "gen" / f"{g}.jsonl").open()}
             for g in groups}  # fmt: skip
    out = {"arms": [{"id": g, "label": LABEL[g.split("@")[0]]} for g in groups], "sources": []}
    for s in src:
        rec = {"id": s["sid"], "genre": s["genre"], "text": s["text"], "rewrites": {}}
        for g in groups:
            r = rows[(g, s["sid"])]
            rec["rewrites"][g] = {"text": texts[g][s["sid"]], "s9": round(float(r["s9"]), 4),
                                  "claims": round(float(r["claim_strict"]), 2), "loop": int(r["loop"]),
                                  "cut": int(r["cut"]), "voi": int(r["flag_voi380"]), "desklib": int(r["flag_desklib"]),
                                  "nonword": round(float(r["nonword"]), 3)}  # fmt: skip
        out["sources"].append(rec)
    (FIG.parent / "data.js").write_text("window.EVADE = " + json.dumps(out, ensure_ascii=False) + ";\n")
    print("viewer", len(out["sources"]), "sources x", len(groups), "groups")


def main() -> None:
    S = json.loads((RESULTS / "summary.json").read_text())
    G, final = S["groups"], S["final"]
    FIG.mkdir(parents=True, exist_ok=True)
    finals = [final[r] for r in RUNS if r in final]
    evasion(G)
    grouped(G, ["source", "prompt-only", "best-of-8", *finals], DETS, "transfer.png", "share flagged as AI")
    for g in G.values():  # claims lost = 1 - strict claim recall (interval flipped)
        for sp in ("heldout", "heldout4b"):
            c = g.get(sp, {}).get("claim_strict") if isinstance(g, dict) else None
            if c:
                g[sp]["claims_lost"] = {"value": 1 - c["value"], "lo": 1 - c["hi"], "hi": 1 - c["lo"]}
    grouped(G, ["prompt-only", "best-of-8", *finals], HACKS, "hacks.png", "share")
    dpath = RESULTS / "defend" / "summary.json"
    if dpath.exists():
        defence(json.loads(dpath.read_text()), finals)
    viewer(final)


if __name__ == "__main__":
    main()
