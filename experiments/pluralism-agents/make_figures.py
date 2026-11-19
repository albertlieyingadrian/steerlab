"""Every figure of the note, drawn from results/ (run analyse.py first). Prints the numbers it plots.

    .venv/bin/python experiments/pluralism-agents/make_figures.py

Writes report/figures/*.png (static fallbacks) and report/figdata.js (the data behind the interactive figures).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse as A

R = HERE / "results"
FIG = HERE / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED, EMBER, EMBER_L, GREY, SLATE = "#1d1a17", "#7a7065", "#d26f2a", "#f0b48a", "#b9b1a7", "#4b5563"
RUN_STYLE = {
    "single-s0": {"color": SLATE, "ls": "-", "label": "single judge, seed 0"},
    "single-s1": {"color": SLATE, "ls": "--", "label": "single judge, seed 1"},
    "jury-s0": {"color": EMBER, "ls": "-", "label": "jury, seed 0"},
    "jury-s1": {"color": EMBER, "ls": "--", "label": "jury, seed 1"},
}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})  # fmt: skip
MEASURES = {
    "refusal_held": "Refusal held (24 no-write tasks)",
    "plans_strict": "Distinct tool plans per task (of 4)",
    "pass1": "Task success, pass^1",
    "pass4": "Task success, pass^4",
    "refusal_self": "Refused without a hand-off (24 tasks)",
    "handoff": "Conversations handed to a human",
    "no_unauth_write": "No unauthorised write",
    "plans_loose": "Distinct tool sets per task (of 4)",
    "plan_len": "Tool calls per conversation",
    "finished": "Conversations that ended normally",
    "judge_single": "Single judge's score (1-10)",
    "judge_jury": "Jury reward (1-10)",
}


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def main():
    S = json.loads((R / "summary.json").read_text())
    runs = A.load_runs()
    data = {"measures": MEASURES, "base": {}, "runs": {}, "plans": {}, "curves": {}, "judges": {}}
    for s in ["s0", "s1"]:
        if s in S["base"]:
            data["base"][s] = {d: S["base"][s][d] for d in ["airline", "retail"]}
    data["base_ci"] = S.get("base_ci")
    for name, rr in runs.items():
        data["runs"][name] = {
            "by_step": S["runs"][name]["by_step"],
            "drift": {str(k): v for k, v in rr["drift"].items()},
        }
        data["curves"][name] = [
            {k: c.get(k) for k in ["step", "judge_reward", "gt_success", "finished", "mean_agent_turns", "drift_kl"]}
            for c in rr["curve"]
        ]
    for k in ["kstar", "matched_step", "final_step", "matched_drift", "predictions", "frontier"]:
        data[k] = S.get(k)
    data["compare"] = S.get("compare")
    for k in ["judge_validity_dev", "judge_validity_heldout"]:
        data["judges"][k] = S.get(k)
    data["judges"]["final"] = {n: S["runs"][n].get("judge_validity_final_retail") for n in S["runs"]}

    # plan explorer: per airline task, the 4 tool plans of the base model (seed 0) and of each run's final checkpoint
    base0 = A.jl(R / "baseline" / "s0" / "eval.jsonl")
    plans = defaultdict(dict)
    succ = defaultdict(dict)
    for r in base0:
        if r["domain"] == "airline":
            plans[r["task_id"]].setdefault("base", []).append(r["plan"])
            succ[r["task_id"]].setdefault("base", []).append(r["reward"])
    for name, rr in runs.items():
        fin = rr["steps"][-1]
        for r in rr["eval_rows"]:
            if r["domain"] == "airline" and r["step"] == fin:
                plans[r["task_id"]].setdefault(name, []).append(r["plan"])
                succ[r["task_id"]].setdefault(name, []).append(r["reward"])
    D = json.loads((HERE / "data" / "tau_tasks.json").read_text())
    purpose = {t["id"]: (t["purpose"] or "") for t in D["airline"]["tasks"]}
    nowrite = set(A.NO_WRITE)
    data["plans"] = {
        tid: {"purpose": purpose.get(tid, ""), "no_write": tid in nowrite, "plans": plans[tid], "success": succ[tid]}
        for tid in sorted(plans, key=int)
    }
    data["plan_default"] = "28"  # the refund-by-all-means task in the note's opening
    (HERE / "report" / "figdata.js").write_text("window.FIGDATA = " + json.dumps(data, default=str) + ";\n")
    print("wrote report/figdata.js")

    if not runs:
        return
    # Figure: training curves
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    for name, rr in runs.items():
        st = RUN_STYLE[name]
        xs = [c["step"] for c in rr["curve"]]
        ax[0].plot(xs, [c["judge_reward"] for c in rr["curve"]], color=st["color"], ls=st["ls"], label=st["label"])
        ax[1].plot(xs, [c["gt_success"] for c in rr["curve"]], color=st["color"], ls=st["ls"])
        dx = sorted((int(k), v) for k, v in rr["drift"].items())
        ax[2].plot([d[0] for d in dx], [d[1] for d in dx], color=st["color"], ls=st["ls"], marker="o", ms=3)
        print(name, "judge reward first/last", rr["curve"][0]["judge_reward"], rr["curve"][-1]["judge_reward"],
              "drift", dx[-1])  # fmt: skip
    ax[0].set_title("Training judge reward (own arm)")
    ax[1].set_title("τ-bench success on training rollouts")
    ax[2].set_title("Drift: KL(base‖trained), neutral prompts")
    for a in ax:
        a.set_xlabel("GRPO step")
    ax[1].set_ylim(0, 1)
    ax[0].legend(frameon=False, fontsize=7)
    save(fig, "curves.png")

    # Figure: airline measures against drift
    keys = ["refusal_held", "plans_strict", "pass1"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
    for i, k in enumerate(keys):
        b = [S["base"][s]["airline"][k] for s in ["s0", "s1"] if s in S["base"]]
        ax[i].axhspan(min(b), max(b), color=GREY, alpha=0.4, lw=0)
        ax[i].axhline(sum(b) / len(b), color=GREY, ls=":", lw=1)
        for name, rr in runs.items():
            st = RUN_STYLE[name]
            pts = [(rr["drift"][s], S["runs"][name]["by_step"][str(s)]["airline"][k]) for s in rr["steps"]]
            pts = [(0.0, sum(b) / len(b)), *pts]
            ax[i].plot([p[0] for p in pts], [p[1] for p in pts], color=st["color"], ls=st["ls"], marker="o", ms=3,
                       label=st["label"])  # fmt: skip
            print(k, name, [(round(x, 4), round(y, 3)) for x, y in pts])
        ax[i].set_title(MEASURES[k], fontsize=9)
        ax[i].set_xlabel("drift (KL, nats/token)")
        ax[i].set_ylim(0, 1)
    ax[0].legend(frameon=False, fontsize=7)
    save(fig, "airline.png")

    # Figure: judge validity
    labels, vals = [], []
    for key, lab in [
        ("judge_validity_dev", "dev (retail train)"),
        ("judge_validity_heldout", "held-out (retail test)"),
    ]:
        v = S.get(key)
        if v:
            for arm in ["single", "jury"]:
                labels.append(f"{arm}\n{lab}")
                vals.append(v[arm]["auroc"])
    for name in sorted(S["runs"]):
        v = S["runs"][name].get("judge_validity_final_retail")
        if v:
            arm = name.split("-")[0]
            labels.append(f"{arm} on {name}\nfinal retail")
            vals.append(v[arm]["auroc"])
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.bar(range(len(vals)), vals, color=[EMBER if "jury" in lab.split("\n")[0] else SLATE for lab in labels])
    ax.axhline(0.5, color=GREY, ls=":")
    ax.axhline(0.65, color=MUTED, ls="--", lw=1)
    ax.set_xticks(range(len(vals)), labels, fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC vs τ-bench success")
    print("judge AUROC", list(zip(labels, vals)))
    save(fig, "judges.png")


if __name__ == "__main__":
    main()
