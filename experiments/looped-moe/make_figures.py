"""Draw the note's figures from results/ and print every number plotted.

.venv/bin/python experiments/looped-moe/make_figures.py

The headline runs are the longest chain runs present (chain_20k_s1337.json when it exists, else the 6,000-step runs);
the shorter-budget runs are drawn beside them in Figure 2.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
ACCENT, CONTROL, INK, FAINT = "#d9622b", "#2f6fb0", "#222222", "#9a9a9a"
ORDER = ["standard", "plainloop", "loopmoe_k1", "loopmoe_nomix", "loopmoe"]
LABEL = {
    "standard": "standard (no loop, one stream)",
    "plainloop": "plain loop (one stream)",
    "loopmoe_k1": "looped MoE, K = 1 (no loop)",
    "loopmoe_nomix": "looped MoE, no block mixing",
    "loopmoe": "looped MoE (full)",
}
COLOR = {"standard": FAINT, "plainloop": CONTROL, "loopmoe_k1": "#c9a27e", "loopmoe_nomix": "#8c5a3c", "loopmoe": ACCENT}
CHAIN_FILES = ("chain_6k_s1337.json", "chain_6k_s1.json", "chain_20k_s1337.json")


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def headline():
    """(file name, runs) for the headline chain runs."""
    for name in ("chain_20k_s1337.json", "chain_6k_s1337.json"):
        d = load(name)
        if d:
            return name, d
    raise SystemExit("no chain results")


def mean(d):
    return sum(d.values()) / len(d)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def fig_ksweep():
    """Figure 1: mean accuracy over 1-8 hops against K at inference, headline runs."""
    name, runs = headline()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    out = {}
    for m in ORDER:
        if m not in runs:
            continue
        sw = runs[m]["sweep"]
        ks = sorted(sw, key=int)
        ys = [mean(sw[k]) for k in ks]
        out[m] = [round(y, 3) for y in ys]
        ax.plot([int(k) for k in ks], ys, "-o", ms=3.5, lw=2.2 if m == "loopmoe" else 1.4, color=COLOR[m],
                label=LABEL[m])
    ax.axhline(0.55, color=FAINT, lw=0.8, ls=":")
    ax.text(1.05, 0.5, "about 0.55: guessing one of the two digits", fontsize=6.5, color=FAINT, va="top")
    ax.set(xlabel="core passes K at inference", ylabel="accuracy, mean over 1-8 hops", ylim=(0, 1.02))
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig1-ksweep.png", dpi=150)
    plt.close(fig)
    print(f"fig1 ({name}) mean accuracy by K=1..6:", out)


def fig_hops():
    """Figure 2: accuracy by hop count at the trained K, every chain run present."""
    files = [(n, load(n)) for n in CHAIN_FILES]
    files = [(n, d) for n, d in files if d]
    fig, axes = plt.subplots(1, len(files), figsize=(3.3 * len(files), 3.2), sharey=True, squeeze=False)
    for ax, (n, runs) in zip(axes[0], files, strict=True):
        row = {}
        for m in ORDER:
            if m not in runs:
                continue
            acc = runs[m]["acc"]
            hops = sorted(acc, key=int)
            ax.plot([int(h) for h in hops], [acc[h] for h in hops], "-o", ms=3, color=COLOR[m],
                    lw=2 if m == "loopmoe" else 1.2, label=LABEL[m])
            row[m] = [round(acc[h], 2) for h in hops]
        steps = "20,000" if "20k" in n else "6,000"
        seed = n.rsplit("_s", 1)[1].removesuffix(".json")
        ax.set(xlabel="hops in the question's chain", ylim=(0, 1.02))
        ax.set_title(f"{steps} steps, seed {seed}", fontsize=9)
        style(ax)
        print(f"fig2 {n} accuracy by hop 1..8 at trained K:", row)
    axes[0][0].set_ylabel("accuracy at the trained K")
    axes[0][-1].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2-hops.png", dpi=150)
    plt.close(fig)


def fig_lens():
    """Figure 3: lens accuracy per layer visit; block-mixing models (three reads) and the plain loop (one stream)."""
    name = "lens_20k_s1337.json" if load("lens_20k_s1337.json") else "lens_6k_s1337.json"
    res = load(name)
    panels = [m for m in ("loopmoe_k1", "loopmoe", "plainloop") if m in res]
    title = {"loopmoe_k1": "block mixing, no loop (output 0.96)", "loopmoe": "block mixing + loop (full model)",
             "plainloop": "plain loop: one residual stream"}
    widths = [len(res[m]["visits"]) for m in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 1, 3.8), sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": widths})
    for ax, m in zip(axes[0], panels, strict=True):
        r = res[m]
        labels = [("out" if v == "Lout" else v) for v in r["visits"]]
        if m == "plainloop":
            ax.plot(r["act_in"], "-", marker=".", color=CONTROL, label="logit lens on the residual stream")
        else:
            for key, lab, col, ls in (("act_in", "logit lens on the active block", FAINT, "-"),
                                      ("mlp_in", "the visit's own MoE-input mix", CONTROL, "--"),
                                      ("mix", "mix lens (the head's final_input mix)", ACCENT, "-")):
                ax.plot(r[key], ls, marker=".", color=col, label=lab)
        ax.set_xticks(range(len(labels)), labels, rotation=70, fontsize=6.5)
        ax.set_title(title[m], fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.axhline(0.55, color=FAINT, lw=0.6, ls=":")
        style(ax)
        print(f"fig3 ({name}) {m} visits:", labels)
        for k in ("act_in", "mlp_in", "mix"):
            if k in r:
                print(f"  {m} {k}:", [round(v, 3) if v == v else None for v in r[k]])
        if r.get("final_weights"):
            print(f"  {m} final_input weights [embed, prelude, core A, core B, coda active]:", r["final_weights"])
    axes[0][0].set_ylabel("top-1 is the answer at '?'")
    axes[0][0].legend(fontsize=6.5, frameon=False, loc="lower left")
    axes[0][-1].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3-lens.png", dpi=150)
    plt.close(fig)


def fig_text():
    """Figure 4: Tiny Shakespeare validation loss against K at inference."""
    runs = load("shakespeare_3k_s1337.json")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    out = {}
    for m in ORDER:
        if m not in runs:
            continue
        v = runs[m]["val"]
        ks = sorted(v, key=int)
        out[m] = {k: round(v[k], 3) for k in ks}
        ax.plot([int(k) for k in ks], [v[k] for k in ks], "-o", ms=3.5, color=COLOR[m],
                lw=2.2 if m == "loopmoe" else 1.4, label=LABEL[m])
    ax.set(xlabel="core passes K at inference (trained at K = 3; standard and K = 1 models at 1)",
           ylabel="validation loss (nats per character)", ylim=(1.4, 3.2))
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig4-shakespeare.png", dpi=150)
    plt.close(fig)
    print("fig4 Shakespeare val loss by K:", out)


def numbers():
    _, runs = headline()
    print("params (stored, active):", {m: (r["params"], r["active"]) for m, r in runs.items()})
    for n in CHAIN_FILES:
        d = load(n)
        if d:
            print(f"{n} mean at trained K:", {m: round(d[m]["mean"], 3) for m in ORDER if m in d})
            print(f"{n} mean at K=1:", {m: round(mean(d[m]["sweep"]["1"]), 3) for m in ORDER if m in d})


def interactive():
    """The slider's data, inlined into the page: accuracy by hop for every model and K (headline runs)."""
    _, runs = headline()
    data = {m: {k: [round(runs[m]["sweep"][k][h], 3) for h in sorted(runs[m]["sweep"][k], key=int)]
                for k in runs[m]["sweep"]} for m in ORDER if m in runs}
    page = HERE / "report" / "index.html"
    html = page.read_text()
    start, end = "<!-- sweep:start -->", "<!-- sweep:end -->"
    block = f'{start}\n<script type="application/json" id="lk-data">{json.dumps(data, separators=(",", ":"))}</script>\n{end}'
    page.write_text(html[: html.index(start)] + block + html[html.index(end) + len(end):])
    print("wrote the slider data into report/index.html")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    numbers()
    fig_ksweep()
    fig_hops()
    fig_lens()
    fig_text()
    interactive()
