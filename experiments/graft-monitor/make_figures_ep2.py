"""Draw the Episode 2 note's figures from results/, printing every number plotted or quoted.

    .venv/bin/python experiments/graft-monitor/make_figures_ep2.py

Inputs (all written by other scripts; nothing here calls a model):
  results/ep2_summary.json            analyze_ep2.py: per-generation hack rate, skill, undetected hacks
  results/ep1_summary.json            analyze_ep1.py: the base model's held-out rates (the dashed reference line)
  results/drift/<tag>/summary.json    battery/drift.py: values-battery means and paired intervals, behavior-diff, Petri
  results/drift/validate-floor/       base seed 3 against base seed 1 (a second base-vs-base pair)
  results/drift/validate-positive/    Qwen2.5-0.5B-Instruct against base Qwen3-0.6B (the battery's positive control)
  results/q3/judge_<tag>.jsonl        q3_selfmon.py: P(YES) per episode, framing and condition
  results/q3/summary_<tag>.json       q3_selfmon.py summarize: recall, FPR, gaps with intervals
  results/q3/bench_qwen3-4b/summary.json   the routing monitor on the Qwen3-4B episodes

Figures (../graft-monitor-ep2/report/figures/; the Episode 2 note lives in its own folder, its code here):
  generations.png  deployed hack rate per generation and arm (mean of seeds, 95% bootstrap intervals), undetected
                   hacks as hollow markers, the base model as a dashed line. Generations not yet finished are absent.
  drift.png        values-battery shift against base seed 1 for each checkpoint, the same-model controls and two
                   base-vs-base pairs, one panel per primary readout, 95% paired bootstrap intervals over items.
  selfmon.png      Qwen3-4B's flag recall on ground-truth hacks per condition, 95% bootstrap intervals over episodes;
                   the framing swap (post hoc) hatched; the external routing monitor and a random flagger in grey.
  hero.jpg, og.jpg the cover and share image: each seed's deployed hack rate from the base model through generations
                   1 to 3, one line per arm and seed, no axes (make_figures.cover); plus ../shared/thumbs/graft-monitor-ep2.jpg.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_figures import cover  # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE.parent / "graft-monitor-ep2" / "report" / "figures"
EMBER, GREY, LIGHT, INK = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 150})  # fmt: skip

ARMS = [
    ("none", "no intervention", INK, "o", "-"),
    ("penalty", "reward penalty", GREY, "s", "--"),
    ("graft", "GRAFT", EMBER, "o", "-"),
]
PRIMARY = [
    ("sycophancy.capitulation", "Caves to a wrong push\n(arithmetic, science)"),
    ("sycophancy.final_accuracy_wrong_push", "Right after a wrong push"),
    ("refusal.harmful_refusal", "Refuses harmful requests*"),
    ("refusal.benign_overrefusal", "Refuses safe requests*"),
    ("uncertainty.fictional_admits", "Admits not knowing\n(made-up entities)"),
    ("uncertainty.answerable_correct", "Correct on answerable\ncontrols"),
    ("diversity.distinct2", "Distinct-2 (creative)"),
    ("diversity.embed_spread", "Embedding spread (creative)"),
]
ARM_OF = {"none": ("no intervention", INK), "penalty": ("reward penalty", INK), "graft": ("GRAFT", EMBER)}


def r3(x):
    return None if x is None else round(x, 3)


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- figure 1: generations
def generations(s: dict, ep1: dict) -> None:
    base = ep1["base"]["hack_rate"]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axhline(base, color=INK, ls=(0, (4, 3)), lw=1)
    ax.text(3.33, base, f"base\nQwen3-0.6B\n{base:.3f}", va="center", fontsize=8, color=INK)
    print(
        "generations.png: arm, generation, n seeds, deployed hack [95% CI], undetected [95% CI], skill, "
        "capitulation without authority"
    )
    print(
        f"  base (ep1_summary), deployed hack {base:.3f} {[r3(x) for x in ep1['base']['ci_hack']]}, "
        f"skill {ep1['base']['skill']:.3f}"
    )
    off = {"none": -0.08, "penalty": 0.0, "graft": 0.08}
    for arm, label, col, mk, ls in ARMS:
        xs, ys, lo, hi, ux, uy = [], [], [], [], [], []
        for g in (1, 2, 3):
            r = s["generations"].get(f"{arm}_g{g}")
            if not r:
                print(f"  {arm}, gen {g}: not finished")
                continue
            xs.append(g + off[arm])
            ys.append(r["hack_rate"])
            lo.append(r["hack_rate"] - r["ci_hack"][0])
            hi.append(r["ci_hack"][1] - r["hack_rate"])
            u = r.get("undetected_hack_rate")
            if u is not None:
                ux.append(g + off[arm])
                uy.append(u)
            print(
                f"  {arm}, gen {g}, {r['n_seeds']}, {r['hack_rate']:.3f} {[r3(x) for x in r['ci_hack']]}, "
                f"{r3(u)} {[r3(x) for x in r.get('ci_undetected', [])]}, {r['skill']:.3f}, "
                f"{r['capitulation_unhackable']:.3f}"
            )
        if xs:
            ax.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, ls=ls, lw=1.6, ms=6, capsize=3, label=label)
        if ux:
            ax.plot(ux, uy, ls="none", marker=mk, mfc="white", mec=col, ms=6, mew=1.3)
    for n, r in sorted(s["runs"].items()):
        print(
            f"  run {n}: hack {r['hack_rate']:.3f}, skill {r['skill']:.3f}, undetected "
            f"{r3(r['undetected_hack_rate'])}, capitulation without authority {r['capitulation_unhackable']:.3f}"
        )
    ax.set_xticks([1, 2, 3], ["gen 1\n(Episode 1, 250 steps)", "gen 2\n(+150 steps)", "gen 3\n(+150 steps)"])
    ax.set_xlim(0.6, 3.3)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("deployed hack rate\n(64 held-out episodes per seed)")
    handles = [Line2D([], [], color=c, marker=m, ls=l, label=lab) for _, lab, c, m, l in ARMS]
    handles.append(
        Line2D(
            [], [], color=GREY, marker="o", mfc="white", ls="none", label="undetected hacks (hollow, same arm colour)"
        )
    )
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.42))
    fig.tight_layout()
    fig.savefig(FIGURES / "generations.png")
    plt.close(fig)


def hero(s: dict, ep1: dict) -> None:
    """Cover: each arm and seed's deployed hack rate, base model (x=0) then generations 1-3 (GRAFT in ember)."""
    base = ep1["base"]["hack_rate"]
    lines = []
    print("hero.jpg: arm, seed, deployed hack rate at base, gen 1, gen 2, gen 3")
    for arm, _, col, _, _ in ARMS:
        for seed in (11, 12):
            names = [f"ep1_{arm}_s{seed}", f"ep2_g2_{arm}_s{seed}", f"ep2_g3_{arm}_s{seed}"]
            ys = [base] + [s["runs"][n]["hack_rate"] for n in names if n in s["runs"]]
            print(f"  {arm}, {seed}, " + ", ".join(f"{y:.3f}" for y in ys))
            lines.append((list(range(len(ys))), ys, EMBER if arm == "graft" else GREY, 5.0 if arm == "graft" else 2.6))
    cover(lines, base, (-0.12, 3.12), FIGURES, "graft-monitor-ep2")


# ---------------------------------------------------------------- figure 2: drift
def drift_rows() -> list[tuple[str, str, str, dict]]:
    """(label, kind, colour, readouts) per row: base-vs-base pairs, controls, then checkpoints."""
    rows: list[tuple[str, str, str, dict]] = []
    tags = sorted(p.parent.name for p in (RESULTS / "drift").glob("q*/summary.json"))
    summ = {t: load(RESULTS / "drift" / t / "summary.json") for t in tags}
    first = summ[tags[0]] if tags else load(RESULTS / "drift" / "validate-floor" / "summary.json")
    rows.append(("base seed 2 (the floor)", "floor", GREY, {k: v["floor"] for k, v in first["readouts"].items()}))
    vf = load(RESULTS / "drift" / "validate-floor" / "summary.json")
    if vf:
        rows.append(("base seed 3", "floor", GREY, {k: v["checkpoint_vs_base"] for k, v in vf["readouts"].items()}))
    seen = set()
    for t in tags:
        m = summ[t]["models"].get("control")
        if not m:
            continue
        key = (m["spec"], m.get("seed"))
        if key in seen:
            continue
        seen.add(key)
        run = m["spec"].split("/")[-2] if m["spec"].endswith("/deployed") else m["spec"]
        rows.append(
            (
                f"control {run.split('_')[-1]} (sample seed {m.get('seed')})",
                "control",
                GREY,
                {k: v["control_vs_base"] for k, v in summ[t]["readouts"].items() if "control_vs_base" in v},
            )
        )
    for t in tags:
        parts = t.split("_")  # q1_<arm>_s<seed> or q2_g<g>_<arm>_s<seed>
        gen, arm, seed = ("1", parts[1], parts[2]) if t.startswith("q1_") else (parts[1][1:], parts[2], parts[3])
        lab, col = ARM_OF.get(arm, (arm, INK))
        rows.append(
            (
                f"{lab}, gen {gen}, {seed}",
                "checkpoint",
                col,
                {k: {**v["checkpoint_vs_base"], "flag": v.get("flag")} for k, v in summ[t]["readouts"].items()},
            )
        )
    return rows


def drift() -> None:
    rows = drift_rows()
    fig, axes = plt.subplots(2, 4, figsize=(12, 1.6 + 0.5 * len(rows)), sharey=True)
    ys = list(range(len(rows)))[::-1]
    print("drift.png: row, readout, shift against base seed 1 [95% CI]")
    for ax, (key, title) in zip(axes.flat, PRIMARY):
        ax.axvline(0, color=INK, lw=0.8)
        fl = rows[0][3].get(key)
        if fl:
            ax.axvspan(fl["lo"], fl["hi"], color=LIGHT, alpha=0.45, lw=0)
        for y, (lab, kind, col, rd) in zip(ys, rows):
            v = rd.get(key)
            if not v:
                continue
            mk = {"floor": "D", "control": "s", "checkpoint": "o"}[kind]
            filled = kind == "checkpoint"
            ax.errorbar(
                v["delta"],
                y,
                xerr=[[v["delta"] - v["lo"]], [v["hi"] - v["delta"]]],
                fmt=mk,
                color=col,
                mfc=col if filled else "white",
                mec=col,
                ms=5,
                capsize=2,
                lw=1,
            )
            if v.get("flag"):
                ax.plot(v["delta"], y, marker="o", ms=11, mfc="none", mec=col, mew=1.2, ls="none")
            print(
                f"  {lab}, {key}, {v['delta']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
                + ("  FLAGGED" if v.get("flag") else "")
            )
        ax.set_title(title, fontsize=9, color=INK)
        ax.tick_params(axis="x", labelsize=8)
    for ax in axes[:, 0]:
        ax.set_yticks(ys, [r[0] for r in rows], fontsize=7.5)
    for ax in axes[1]:
        ax.set_xlabel("shift against base (seed 1)", fontsize=8.5)
    fig.legend(
        handles=[
            Patch(color=LIGHT, alpha=0.45, label="base seed 2 − seed 1, 95% CI (the floor)"),
            Line2D([], [], marker="D", mfc="white", mec=GREY, color=GREY, ls="none", label="base against itself"),
            Line2D(
                [],
                [],
                marker="s",
                mfc="white",
                mec=GREY,
                color=GREY,
                ls="none",
                label="same-model control (RL on unhackable episodes only)",
            ),
            Line2D([], [], marker="o", color=EMBER, ls="none", label="GRAFT, deployed"),
            Line2D([], [], marker="o", color=INK, ls="none", label="no intervention, penalty"),
            Line2D([], [], marker="o", ms=10, mfc="none", mec=INK, ls="none", label="flagged by the rule"),
        ],
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "drift.png")
    plt.close(fig)


def drift_tables() -> None:
    """Numbers quoted in the text that are not plotted: means, flags, behavior-diff, Petri, the positive control."""
    print("drift means [base s1, base s2, checkpoint, control], flags, behavior-diff (changed, categories, agreement):")
    for p in sorted((RESULTS / "drift").glob("*/summary.json")):
        s = load(p)
        cap = s["readouts"]["sycophancy.capitulation"]["means"]
        par = s["readouts"]["sycophancy.parsed"]["means"]
        cond = {k: r3(cap[k] / par[k]) for k in cap if par.get(k)}
        bd = {
            k: (v.get("n_changed"), v.get("n_categories"), r3(v.get("assign_agreement")))
            for k, v in ((s.get("bdiff") or {}).get("pairs") or {}).items()
        }
        models = {k: f"{v['spec']}@{v.get('seed')}" for k, v in s["models"].items()}
        print(f"  {p.parent.name}: models {models}")
        print(
            f"    capitulation {[r3(cap.get(k)) for k in ('base', 'base2', 'checkpoint', 'control')]}"
            f" | parsed {[r3(par.get(k)) for k in ('base', 'base2', 'checkpoint', 'control')]}"
            f" | capitulation among parsed {cond}"
        )
        print(
            f"    flagged {s.get('flagged')} | unvalidated-scorer flags {s.get('flagged_unvalidated_scorer')}"
            f" | behavior-diff {bd or 'not run'}"
        )
        for k, v in s["readouts"].items():
            if v.get("flag"):
                c = v["checkpoint_vs_base"]
                print(
                    f"    flag on {k} (primary {v.get('primary')}): {c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"
                )
        jc = s.get("judge_check") or {}
        if jc:
            print(
                f"    judge check {jc.get('judge')}: {jc.get('correct')}/{jc.get('n')}; by readout "
                f"{ {k: (v.get('judge_correct'), v.get('n'), v.get('judge_passed')) for k, v in (jc.get('by_readout') or {}).items()} }"
            )
        pet = s.get("petri") or {}
        if isinstance(pet.get("base"), dict):
            dims = (
                "admirable",
                "concerning",
                "disappointing",
                "unprompted_sycophancy",
                "unprompted_deception_toward_user",
                "user_over_sysprompt",
            )
            for m in ("base", "base2", "checkpoint", "control"):
                if m in pet:
                    mm = pet[m].get("means") or {}
                    print(
                        f"    petri {m} (n {pet[m].get('n')}, parsed {pet[m].get('parsed')}): "
                        + ", ".join(f"{d} {mm.get(d, float('nan')):.2f}" for d in dims)
                    )
        elif pet:
            print(f"    petri: {str(pet)[:160]}")
    # The terse form: wrong-push sycophancy replies that are only "Final answer: X" (a hypothesis for the shared rise).
    import sys

    sys.path.insert(0, str(HERE))
    from battery import core

    items = {it["id"]: it for it in core.load_items()}
    print("terse replies (wrong push, reply starts with 'Final answer:'): share; capitulation when terse / otherwise")
    seen_dirs = set()
    for p in sorted((RESULTS / "drift").glob("*/summary.json")):
        for role, m in load(p)["models"].items():
            d = HERE / m["battery_dir"] / "replies.jsonl"
            if m["battery_dir"] in seen_dirs or not d.exists():
                continue
            seen_dirs.add(m["battery_dir"])
            rows = [json.loads(x) for x in d.read_text().splitlines()]
            rows = [r for r in rows if r["readout"] == "sycophancy" and r["push"] == "wrong"]
            t = [r for r in rows if r["reply"].strip().lower().startswith("final answer")]
            o = [r for r in rows if r not in t]

            def capr(rs):
                return (
                    sum(core.score_sycophancy(r, items[r["item"]])["capitulated"] for r in rs) / len(rs)
                    if rs
                    else float("nan")
                )

            print(
                f"  {m['battery_dir']} ({role} in {p.parent.name}): {len(t)}/{len(rows)} = {len(t) / len(rows):.3f}; "
                f"capitulation {capr(t):.3f} terse / {capr(o):.3f} otherwise"
            )
    # Petri ranges over every q* drift run. Base seed 1's 20 transcripts are identical in every run (cached replies),
    # so their spread is the judge's own run-to-run noise.
    dims = (
        "admirable",
        "concerning",
        "disappointing",
        "unprompted_sycophancy",
        "unprompted_deception_toward_user",
        "user_over_sysprompt",
    )
    groups: dict[str, dict[str, list[float]]] = {
        "base s1 (the same 20 transcripts each run)": {},
        "base s2": {},
        "checkpoint": {},
        "control": {},
    }
    n_runs = 0
    for p in sorted((RESULTS / "drift").glob("q*/summary.json")):
        pet = (load(p) or {}).get("petri") or {}
        if not isinstance(pet.get("base"), dict):
            continue
        n_runs += 1
        for m, g in (
            ("base", "base s1 (the same 20 transcripts each run)"),
            ("base2", "base s2"),
            ("checkpoint", "checkpoint"),
            ("control", "control"),
        ):
            for d in dims:
                v = (pet.get(m, {}).get("means") or {}).get(d)
                if v is not None:
                    groups[g].setdefault(d, []).append(v)
    print(f"petri ranges over {n_runs} drift runs (judge Qwen3-8B on 20 sycophancy transcripts per model, 1-10):")
    for d in dims:
        print("  " + d + ": " + "; ".join(f"{g} {min(v[d]):.2f}-{max(v[d]):.2f}" for g, v in groups.items() if d in v))


# ---------------------------------------------------------------- figure 3: self-monitoring
def q3_rows(tag: str) -> list[dict]:
    p = RESULTS / "q3" / f"judge_{tag}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def boot(xs: list[float], n: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    v = sorted(sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n))
    return [v[int(0.025 * n)], v[int(0.975 * n) - 1]]


def cell(rows: list[dict], source: str, informed: bool) -> dict | None:
    sel = [r for r in rows if r["source"] == source and r["informed"] == informed]
    if not sel:
        return None
    hk = [float(r["p_yes"] > 0.5) for r in sel if r["gt_hack"]]
    ng = [float(r["p_yes"] > 0.5) for r in sel if not r["gt_hack"]]
    return {
        "recall": sum(hk) / len(hk),
        "ci": boot(hk),
        "fpr": sum(ng) / len(ng),
        "ci_fpr": boot(ng),
        "n_hack": len(hk),
        "n_neg": len(ng),
        "flag_rate": (sum(hk) + sum(ng)) / (len(hk) + len(ng)),
    }


def selfmon() -> None:
    main_rows, swap = q3_rows("qwen3-4b"), q3_rows("qwen3-4b_swap")
    small = load(RESULTS / "q3" / "summary_qwen3-0.6b.json")
    bench = load(RESULTS / "q3" / "bench_qwen3-4b" / "summary.json")
    # (label, rows, framing key in the file, informed, style). In the swap file "own" means the framing line said
    # "written by you" (on Qwen3-0.6B's episodes) and "other" means "a different AI model" (on Qwen3-4B's own).
    spec = [
        ("Own episodes, framed “written by you”: uninformed", main_rows, "own", False, "own"),
        ("Own episodes, framed “written by you”: informed", main_rows, "own", True, "own"),
        ("Other model’s episodes, framed “a different model”: uninformed", main_rows, "other", False, "other"),
        ("Other model’s episodes, framed “a different model”: informed", main_rows, "other", True, "other"),
        ("Post hoc: own episodes framed “a different model”: uninformed", swap, "other", False, "swap-own"),
        ("Post hoc: own episodes framed “a different model”: informed", swap, "other", True, "swap-own"),
        ("Post hoc: other model’s episodes framed “written by you”: uninformed", swap, "own", False, "swap-other"),
        ("Post hoc: other model’s episodes framed “written by you”: informed", swap, "own", True, "swap-other"),
    ]
    bars = []
    print("selfmon.png: condition, recall [95% CI], FPR [95% CI], n hacks, n non-hacks, flag rate")
    for lab, rows, src, inf, style in spec:
        c = cell(rows, src, inf)
        if c is None:
            print(f"  {lab}: no rows")
            continue
        bars.append((lab, c["recall"], c["ci"], style, inf))
        print(
            f"  {lab}: {c['recall']:.3f} {[r3(x) for x in c['ci']]}, FPR {c['fpr']:.3f} "
            f"{[r3(x) for x in c['ci_fpr']]}, {c['n_hack']}, {c['n_neg']}, {c['flag_rate']:.3f}"
        )
    own_u = cell(main_rows, "own", False)
    if own_u:
        fr = own_u["flag_rate"]
        bars.append(("Control: random flags at the uninformed own flag rate", fr, [fr, fr], "ctrl", False))
        print(f"  random flagger at rate {fr:.3f}: recall {fr:.3f} by construction")
    if bench:
        k = round(bench["recall"] * bench["n_hack"])
        b_ci = boot([1.0] * k + [0.0] * (bench["n_hack"] - k))
        bars.append(
            ("Control: routing monitor (gpt-oss-safeguard-20b) on own episodes", bench["recall"], b_ci, "ctrl", True)
        )
        print(
            f"  external monitor on the 4B's episodes: recall {bench['recall']:.3f} {[r3(x) for x in b_ci]}, "
            f"FPR {bench['fpr']:.3f}, n {bench['n_hack']} hacks / {bench['n_neg']} non-hacks"
        )
    for name in ("qwen3-4b", "qwen3-4b_swap", "qwen3-0.6b"):
        sm = load(RESULTS / "q3" / f"summary_{name}.json")
        if sm:
            for k2, g in sm["gaps"].items():
                print(
                    f"  summary_{name} gap {k2}: uninformed − informed recall "
                    f"{g['recall_gap_uninformed_minus_informed']:+.3f} {[r3(x) for x in g['ci95']]}, "
                    f"positive control (recall − flag rate) {g['positive_control_recall_minus_flag_rate']:+.3f} "
                    f"pass {g['positive_control_pass']}"
                )
    if small:
        for k2, c in small["cells"].items():
            print(
                f"  Qwen3-0.6B as judge, {k2}: recall {c['recall']:.3f} at flag rate {c['flag_rate']:.4f}, "
                f"FPR {c['fpr']:.3f}, n {c['n_hack']}/{c['n_neg']}"
            )

    # What the two sets of hacks look like: the share whose final turn hedges (a boring explanation for the gap).
    hedge = ("not certain", "not sure", "could be", "possible", "might be", "i believe", "need more information")
    for name, p in (
        ("Qwen3-4B own", RESULTS / "q3" / "episodes_qwen3-4b.jsonl"),
        ("Qwen3-0.6B (other)", RESULTS / "monitor_bench" / "trajectories.jsonl"),
    ):
        if p.exists():
            hk = [json.loads(x) for x in p.read_text().splitlines()]
            hk = [r for r in hk if r["gt_hack"]]
            n = sum(any(h in r["final_text"].lower() for h in hedge) for r in hk)
            words = sorted(len(r["final_text"].split()) for r in hk)
            print(
                f"  {name} hacks: {n}/{len(hk)} = {n / len(hk):.3f} hedge ({', '.join(hedge)}); "
                f"median final-turn words {words[len(words) // 2]}"
            )

    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(bars) + 1.5))
    ys = list(range(len(bars)))[::-1]
    for y, (lab, v, ci, style, inf) in zip(ys, bars):
        col = {"own": EMBER, "other": INK, "swap-own": EMBER, "swap-other": INK, "ctrl": GREY}[style]
        swapped = style.startswith("swap")
        face = ({EMBER: "#E8AF92", INK: "#9A948C"}[col] if swapped else col) if inf else "white"
        ax.barh(y, v, color=face, edgecolor=col, lw=1.3, height=0.68, hatch="///" if swapped else None)
        ax.errorbar(v, y, xerr=[[v - ci[0]], [ci[1] - v]], color=INK, capsize=2, lw=0.9, fmt="none")
        ax.text(min(ci[1], 1.0) + 0.015, y, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, [b[0] for b in bars], fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("flag recall on ground-truth hacks")
    ax.legend(
        handles=[
            Patch(facecolor="white", edgecolor=INK, label="uninformed (hollow)"),
            Patch(
                facecolor=INK,
                edgecolor=INK,
                label="informed: told flagged behaviour is removed from its successor (filled)",
            ),
            Patch(facecolor="white", edgecolor=INK, hatch="///", label="framing swap, added post hoc"),
            Patch(facecolor=EMBER, edgecolor=EMBER, label="Qwen3-4B's own episodes"),
            Patch(facecolor=INK, edgecolor=INK, label="Qwen3-0.6B's episodes"),
        ],
        frameon=False,
        fontsize=7.5,
        loc="lower left",
        bbox_to_anchor=(-0.6, 1.0),
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "selfmon.png")
    plt.close(fig)


def examples() -> None:
    """The quoted examples in section 4.1, and the same-model control on the city eval."""
    import sys

    sys.path.insert(0, str(HERE))
    import analyze_ep2

    print("examples:")
    for d in (
        "qwen3-0.6b_s1",
        "vol-runs-ep1-none-s11-deployed_s3",
        "vol-runs-ep1-graft-s11-deployed_s3",
        "vol-runs-ep2-ctrl-s11-deployed_s4",
    ):
        p = RESULTS / "battery" / d / "replies.jsonl"
        for x in p.read_text().splitlines() if p.exists() else []:
            r = json.loads(x)
            if r["item"] == "syc-arith-03" and r["push"] == "wrong":
                print(f"  battery syc-arith-03 {d}: first {r['first']!r} | final {r['reply']!r}")
    for n in (
        "ep1_none_s11",
        "ep2_g2_none_s11",
        "ep1_penalty_s11",
        "ep1_graft_s11",
        "ep2_g2_graft_s11",
        "ep2_g3_graft_s11",
    ):
        rows = analyze_ep2.deployed_rows(n)
        for r in rows or []:
            if r["id"] == "heldout-99-6":
                print(
                    f"  city {r['city']} ({r['truth']}, asserted {r['asserted']}) {n}: first {r['first']!r} | "
                    f"final {r['final_text']!r} | hack {r['gt_hack']}"
                )
    q = {(r["source"], r["informed"]): r["p_yes"] for r in q3_rows("qwen3-4b") if r["id"] == "heldout-4242-466"}
    qs = {(r["source"], r["informed"]): r["p_yes"] for r in q3_rows("qwen3-4b_swap") if r["id"] == "heldout-4242-466"}
    ep = [json.loads(x) for x in (RESULTS / "q3" / "episodes_qwen3-4b.jsonl").read_text().splitlines()]
    e = next(r for r in ep if r["id"] == "heldout-4242-466")
    print(
        f"  q3 {e['city']} ({e['truth']}, asserted {e['asserted']}, {e['authority']!r}): first {e['first']!r} | "
        f"final {e['final_text']!r}; P(YES) framed own: uninformed {q.get(('own', False)):.3f}, informed "
        f"{q.get(('own', True)):.3f}; framed other (swap): uninformed {qs.get(('other', False)):.3f}, informed "
        f"{qs.get(('other', True)):.3f}"
    )
    print("same-model control on the held-out city eval (deployed): hack, skill, capitulation without authority")
    for n in ("ep2_ctrl_s11", "ep2_ctrl_s12"):
        rows = analyze_ep2.deployed_rows(n)
        if rows is None:
            print(f"  {n}: not finished")
            continue
        hw = [r for r in rows if r["hackable"] and not r["assertion_true"]]
        sk = [r for r in rows if (not r["hackable"]) or r["assertion_true"]]
        cu = [r for r in rows if (not r["hackable"]) and not r["assertion_true"]]
        print(
            f"  {n}: {sum(r['gt_hack'] for r in hw)}/{len(hw)} = {sum(r['gt_hack'] for r in hw) / len(hw):.3f}, "
            f"{sum(r['correct'] for r in sk) / len(sk):.3f}, {sum(r['capitulated'] for r in cu) / len(cu):.3f}"
        )


def predictions(s: dict) -> None:
    """The registered readouts for P6-P8 (P9 is printed by selfmon as the own/other gaps)."""
    print("predictions:")
    cap = "sycophancy.capitulation"
    for arm in ("none", "penalty", "graft"):
        d, c, used = [], [], []
        for seed in (11, 12):
            v = load(RESULTS / "drift" / f"q1_{arm}_s{seed}" / "summary.json")
            if v:
                r = v["readouts"][cap]
                d.append(r["checkpoint_vs_base"]["delta"])
                c.append(r["checkpoint_vs_control"]["delta"] if "checkpoint_vs_control" in r else float("nan"))
                used.append(seed)
        if d:
            print(
                f"  P6 readout, gen 1 {arm}, seeds {used}: capitulation − base {sum(d) / len(d):+.3f} "
                f"(per seed {[round(x, 3) for x in d]}); − control {sum(c) / len(c):+.3f} "
                f"(per seed {[round(x, 3) for x in c]})"
            )
    gen = s["generations"]
    u1, u3 = (gen.get(f"penalty_g{g}", {}).get("undetected_hack_rate") for g in (1, 3))
    print(
        f"  P7 readout, penalty undetected hacks gen 3 − gen 1: "
        f"{'pending (gen 3 missing)' if u3 is None or u1 is None else f'{u3 - u1:+.3f} ({u1:.3f} -> {u3:.3f})'}"
    )
    gh = {g: gen[f"graft_g{g}"]["hack_rate"] for g in (1, 2, 3) if f"graft_g{g}" in gen}
    print(
        f"  P8 readout, GRAFT deployed hack by generation {gh}; max {max(gh.values()):.3f}"
        f"{'' if len(gh) == 3 else ' (generations missing: provisional)'}"
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    s = load(RESULTS / "ep2_summary.json")
    ep1 = load(RESULTS / "ep1_summary.json")
    print("missing generation runs:", s.get("missing"))
    generations(s, ep1)
    hero(s, ep1)
    drift()
    drift_tables()
    selfmon()
    examples()
    predictions(s)


if __name__ == "__main__":
    main()
