"""Draw the note's figures from results/summary.json, printing every number plotted.

    python make_figures.py

methods.png   Track A: mean habit score per held-out reply for every arm, 95% bootstrap intervals over the 48 prompts;
              the shaded band runs from 0 to the human 90th percentile (33.6); grey ticks: the 3 norm-matched random
              directions at each steering arm's chosen size.
loops.png     Track A: habit score against the share of replies with a 4-word phrase said more than 3 times.
detect.png    share of replies each detector flags at a threshold that flags 1% of the 318 human passages, every arm of
              both tracks, with 95% intervals.
tracks.png    the same method families on Qwen3-4B-Instruct-2507 (Track B, Episode 1's replies) and Qwen3-8B (Track A).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIG = HERE / "report" / "figures"
ACCENT, INK, SOFT, FAINT, RULE, BAND = "#c9561c", "#1d272a", "#586468", "#8a9497", "#e3e7e8", "#f5f6f4"
HUMAN_P90 = 33.6
STEER = {"register", "probe", "sae-dir", "sae-abl", "plain+register", "plain+sae"}
LABEL = {"original": "original", "plain": "system prompt", "sampler": "sampler", "plain+sampler": "prompt + sampler",
         "register": "register direction", "probe": "probe direction", "sae-dir": "SAE direction",
         "sae-abl": "SAE ablation", "plain+register": "prompt + register", "plain+sae": "prompt + SAE direction",
         "sft": "SFT (LoRA)", "dpo": "DPO + NLL (LoRA)", "sft+sampler": "SFT + sampler",
         "base": "original", "steer": "register direction", "steer-random": "random direction",
         "plain+steer": "prompt + register", "neurons": "neuron flips", "neurons-random": "random neurons",
         "plain+neurons": "prompt + neurons", "lora": "SFT (LoRA r1)", "lora-r2": "SFT (LoRA r2)",
         "kannike": "100toolkit model"}  # fmt: skip

plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": SOFT, "ytick.color": SOFT, "font.family": "sans-serif",
                     "savefig.dpi": 200, "savefig.bbox": "tight"})  # fmt: skip


def load():
    S = json.loads((RESULTS / "summary.json").read_text())
    return S, S["arms"]


def methods(A, rand):
    arms = [g for g in A if g.startswith("A/") and "/random-" not in g]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(arms) + 0.8))
    ax.axvspan(0, HUMAN_P90, color=BAND, zorder=0)
    ax.text(HUMAN_P90 / 2, len(arms) - 0.35, "human band", ha="center", va="bottom", color=FAINT, fontsize=8)
    print("methods.png: arm, habit [lo, hi]")
    for i, g in enumerate(reversed(arms)):
        arm = g.split("/", 1)[1]
        h = A[g]["habit"]
        c = ACCENT if arm in STEER else INK
        ax.errorbar(h["value"], i, xerr=[[h["value"] - h["lo"]], [h["hi"] - h["value"]]], fmt="o", color=c,
                    ms=5, capsize=2, lw=1.2)  # fmt: skip
        ax.text(h["hi"] + 2, i, f"{h['value']:.1f}", va="center", fontsize=8, color=c)
        if arm in rand:
            lo, hi = rand[arm]["habit_range"]
            ax.plot([lo, hi], [i - 0.3, i - 0.3], color=FAINT, lw=2.5, solid_capstyle="butt")
            print(f"  {arm:16} random band {lo:.1f}-{hi:.1f}")
        print(f"  {arm:16} {h['value']:.1f} [{h['lo']:.1f}, {h['hi']:.1f}]")
    ax.set_yticks(range(len(arms)), [LABEL[g.split("/", 1)[1]] for g in reversed(arms)])
    ax.set_xlabel("habit score per 1,000 words (lower = fewer chat-assistant habits)")
    ax.set_xlim(0, None)
    fig.savefig(FIG / "methods.png")
    plt.close(fig)


def loops(A):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    print("loops.png: arm, loop share, habit")
    for g, o in A.items():
        if not g.startswith("A/"):
            continue
        arm = g.split("/", 1)[1]
        rnd = arm.startswith("random-")
        c = FAINT if rnd else ACCENT if arm in STEER else INK
        ax.scatter(o["loop"]["value"], o["habit"]["value"], s=22 if rnd else 34, color=c, zorder=3)
        if not rnd:
            ax.annotate(LABEL[arm], (o["loop"]["value"], o["habit"]["value"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7.5, color=c)  # fmt: skip
        print(f"  {arm:18} {o['loop']['value']:.2f} {o['habit']['value']:.1f}")
    ax.axhspan(0, HUMAN_P90, color=BAND, zorder=0)
    ax.set_xlabel("share of replies that loop (a 4-word phrase said more than 3 times)")
    ax.set_ylabel("habit score per 1,000 words")
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(0, None)
    fig.savefig(FIG / "loops.png")
    plt.close(fig)


def detect(A):
    groups = [g for g in A if "/random-" not in g and "nine_b_flag" in A[g]]
    dets = [("nine_b_flag", "our 9B detector", ACCENT), ("voi380_flag", "voi-380M", INK),
            ("desklib_flag", "Desklib", FAINT)]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(groups) + 0.9))
    print("detect.png: group, 9B, voi380, desklib flag rates")
    for i, g in enumerate(reversed(groups)):
        for j, (k, _, c) in enumerate(dets):
            if k in A[g]:
                v = A[g][k]
                ax.errorbar(v["value"], i + (j - 1) * 0.22, xerr=[[v["value"] - v["lo"]], [v["hi"] - v["value"]]],
                            fmt="o", color=c, ms=3.5, capsize=1.5, lw=0.9)  # fmt: skip
        print(f"  {g:22} " + " ".join(f"{A[g].get(k, {}).get('value', float('nan')):.2f}" for k, _, _ in dets))
    names = [("4B · " if g.startswith("B/") else "8B · ") + LABEL[g.split("/", 1)[1]] for g in reversed(groups)]
    ax.set_yticks(range(len(groups)), names, fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("share of replies flagged as AI (threshold flags 1% of human passages)")
    for _k, lab, c in dets:
        ax.plot([], [], "o", color=c, label=lab)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.savefig(FIG / "detect.png")
    plt.close(fig)


def tracks(A):
    pairs = [("original", "base"), ("plain", "plain"), ("sampler", "sampler"), ("plain+sampler", "plain+sampler"),
             ("register", "steer"), ("plain+register", "plain+steer"), ("sft", "lora")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    print("tracks.png: family, 4B habit, 8B habit")
    for i, (a, b) in enumerate(pairs):
        ha, hb = A.get(f"A/{a}", {}).get("habit"), A.get(f"B/{b}", {}).get("habit")
        if hb:
            ax.bar(i - 0.2, hb["value"], 0.38, color=RULE, edgecolor=SOFT, lw=0.6)
        if ha:
            ax.bar(i + 0.2, ha["value"], 0.38, color=ACCENT if a in STEER else INK)
        print(f"  {a:16} {hb and hb['value']} {ha and ha['value']}")
    ax.axhline(HUMAN_P90, color=FAINT, lw=0.8, ls="--")
    ax.set_xticks(range(len(pairs)), [LABEL[a] for a, _ in pairs], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("habit score per 1,000 words")
    ax.bar([], [], color=RULE, edgecolor=SOFT, label="Qwen3-4B (Track B)")
    ax.bar([], [], color=INK, label="Qwen3-8B (Track A)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "tracks.png")
    plt.close(fig)


VIEW_ARMS = ["original", "plain", "plain+sampler", "register", "sae-dir", "plain+sae", "sft", "dpo"]


def viewer() -> None:
    """report/data.js: 12 held-out prompts (two per genre), sample 0 of 8 arms, each reply with its numbers."""
    import csv
    import sys

    sys.path.insert(0, str(HERE))
    from common import held_out

    rows = {(r["group"], r["id"], r["sample"]): r for r in csv.DictReader((RESULTS / "per_reply.csv").open())}
    gens = {}
    for a in VIEW_ARMS:
        rs = map(json.loads, (RESULTS / "gen" / f"{a}.jsonl").open())
        gens[a] = {r["id"]: r["text"] for r in rs if r["sample"] == 0}
    picked, seen = [], {}
    for p in held_out():
        if seen.get(p["genre"], 0) < 2:
            seen[p["genre"]] = seen.get(p["genre"], 0) + 1
            picked.append(p)
    data = {"arms": [{"id": a, "label": LABEL[a]} for a in VIEW_ARMS], "prompts": []}
    for p in picked[:12]:
        replies = {}
        for a in VIEW_ARMS:
            r = rows[(f"A/{a}", p["id"], "0")]
            replies[a] = {"text": gens[a][p["id"]], "habit": float(r["habit"]), "loop": int(r["loop"]),
                          "nine_b": round(float(r["nine_b"]), 3), "words": int(r["words"])}  # fmt: skip
        data["prompts"].append({"id": p["id"], "genre": p["genre"], "prompt": p["prompt"], "replies": replies})
    (HERE / "report" / "data.js").write_text("window.H2H = " + json.dumps(data, ensure_ascii=False) + ";\n")
    print(f"data.js: {len(data['prompts'])} prompts x {len(VIEW_ARMS)} arms")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    S, A = load()
    methods(A, S.get("random_bands", {}))
    loops(A)
    detect(A)
    tracks(A)
    viewer()


if __name__ == "__main__":
    main()
