"""
Draw every figure in the note from results/<size>/*.json, and print the numbers each one plots so they can be checked
against the text. Orange is the SAE (the thing the note is about); grey and blue are baselines and controls.

    .venv/bin/python experiments/sae-101/make_figures.py --size full
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sae101 as s  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="full")
args = ap.parse_args()
res = s.RESULTS / args.size
out = s.HERE / "report" / "figures"
out.mkdir(parents=True, exist_ok=True)

ACCENT, ACCENT_SOFT, CONTROL, GREY, INK, RULE = "#c9561c", "#e9a27b", "#4a86e8", "#8a9497", "#1d272a", "#e3e7e8"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 200})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   wrote report/figures/{name}")


# ---- Figure 0 and the widget's data: the 2-D toy (results/toy.json, from 01_toy.py) ----
toy = s.read_json(s.RESULTS / "toy.json")
widget = {"concepts": toy["concepts"], "points": toy["points"], "active": toy["active"],
          "saes": {k: {"fvu": v["fvu"], "decoder": v["decoder"], "z": v["z"], "x_hat": v["x_hat"], "match_cos": v["match_cos"]}
                   for k, v in toy["saes"].items()},
          "tools": toy["tools"], "angles": toy["angles_to_a"]}
(s.HERE / "report" / "toy-data.js").write_text("window.TOY = " + __import__("json").dumps(widget, separators=(",", ":")) + ";\n")
print("   wrote report/toy-data.js")
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
pts = np.array(toy["points"])
for ax, k in zip(axes, ("1", "2")):
    ax.scatter(pts[:, 0], pts[:, 1], s=5, color=GREY, alpha=0.5)
    for c in toy["concepts"]:
        ax.plot([0, 1.6 * c[0]], [0, 1.6 * c[1]], color=INK, lw=1, ls="--")
    for d in toy["saes"][k]["decoder"]:
        ax.annotate("", xy=(1.4 * d[0], 1.4 * d[1]), xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": ACCENT, "lw": 1.8})
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_title(f"k = {k}: FVU {toy['saes'][k]['fvu']:.3f}, lowest cosine {min(toy['saes'][k]['match_cos']):.2f}")
    ax.set_xticks([])
    ax.set_yticks([])
save(fig, "fig0_toy.png")
print(f"   fig0: k=1 FVU {toy['saes']['1']['fvu']:.4f} cos {toy['saes']['1']['match_cos']}; k=2 FVU {toy['saes']['2']['fvu']:.4f} "
      f"cos {toy['saes']['2']['match_cos']}")
if not (res / "evaluate.json").exists():
    raise SystemExit(0)

ev = s.read_json(res / "evaluate.json")
tr = s.read_json(res / "train_handwritten.json")
sw = s.read_json(res / "sweep.json")
it = s.read_json(res / "interpret.json")
cp = s.read_json(res / "compare.json")
hw, sl = ev["saes"]["hand-written"], ev["saes"].get("SAELens")

# ---- Figure 1: training curve ----
log = tr["log"]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot([r["tokens"] / 1e6 for r in log], [r["train_fvu"] for r in log], color=ACCENT, lw=1.6,
        label="hand-written, on training batches")
ax.scatter([tr["tokens"] / 1e6], [hw["fvu"]], color=ACCENT, zorder=3, s=30, label=f"hand-written, held out ({hw['fvu']:.3f})")
if sl:
    ax.scatter([tr["tokens"] / 1e6], [sl["fvu"]], color=INK, marker="D", zorder=3, s=26,
               label=f"SAELens, held out ({sl['fvu']:.3f})")
ax.set_xscale("log")
ax.set_ylim(0, None)
ax.set_xlabel("training tokens (millions, log scale)")
ax.set_ylabel("FVU (lower is better)")
ax.legend(frameon=False, fontsize=8)
save(fig, "fig1_training.png")
print(f"   fig1: train FVU {log[0]['train_fvu']:.3f} at {log[0]['tokens'] / 1e6:.1f}M tokens -> {log[-1]['train_fvu']:.3f} at "
      f"{log[-1]['tokens'] / 1e6:.1f}M; held-out hand-written {hw['fvu']:.4f}" + (f", SAELens {sl['fvu']:.4f}" if sl else ""))

# ---- Figure 2: the sparsity / rebuild trade-off ----
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for group, color, marker, label in (("relu_l1", GREY, "s", "ReLU + L1 (a penalty on size)"),
                                    ("topk", ACCENT, "o", "TopK (keep the k biggest)"),
                                    ("batchtopk", ACCENT_SOFT, "^", "BatchTopK (k per token on average)")):
    pts = sorted((r["l0"], r["fvu"]) for r in sw["runs"] if r["group"] == group)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker, color=color, lw=1.4, ms=5, label=label)
    print(f"   fig2 {group}: " + ", ".join(f"L0 {a:.1f} FVU {b:.3f}" for a, b in pts))
ax.set_xscale("log")
ax.set_ylim(0, None)
ax.set_xlabel("L0: latents on per token (log scale)")
ax.set_ylabel("held-out FVU")
ax.legend(frameon=False, fontsize=8)
save(fig, "fig2_tradeoff.png")

# ---- Figure 3: does the model still work? ----
c = hw["ce"]
bars = [("clean model", c["clean"], GREY), ("hand-written SAE spliced in", c["sae"], ACCENT)]
if sl:
    bars.append(("SAELens SAE spliced in", sl["ce"]["sae"], ACCENT_SOFT))
bars += [("random error of the same size", c["random_same_norm"], "#b9c0c2"),
         ("layer mean-ablated", c["mean_ablation"], GREY), ("layer zero-ablated", c["zero_ablation"], GREY)]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
y = np.arange(len(bars))[::-1]
ax.barh(y, [b[1] for b in bars], color=[b[2] for b in bars], height=0.6)
for yy, b in zip(y, bars):
    ax.text(b[1] + 0.1, yy, f"{b[1]:.2f}", va="center", fontsize=8.5)
ax.set_yticks(y, [b[0] for b in bars])
ax.set_xlim(0, max(b[1] for b in bars) * 1.12)
ax.set_xlabel("next-token cross-entropy on held-out text (nats per token; lower is better)")
save(fig, "fig3_cross_entropy.png")
print("   fig3: " + "; ".join(f"{b[0]} {b[1]:.4f}" for b in bars))

# ---- Figure 4: how often latents fire ----
edges = np.arange(-7, 0.01, 0.25)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.stairs(hw["log10_freq_hist"], edges, color=ACCENT, lw=1.6, label=f"hand-written ({hw['dead']:,} never fire)")
if sl:
    ax.stairs(sl["log10_freq_hist"], edges, color=INK, lw=1.2, ls="--", label=f"SAELens ({sl['dead']:,} never fire)")
ax.axvline(-1, color=GREY, ls="--", lw=1)
ax.text(-0.95, ax.get_ylim()[1] * 0.9, "dense:\n>1 token in 10", fontsize=8, color=GREY, va="top")
ax.set_xlabel(f"log10 firing frequency over {hw['freq_tokens'] / 1e6:.1f}M held-out tokens")
ax.set_ylabel("latents")
ax.legend(frameon=False, fontsize=8, loc="upper left")
save(fig, "fig4_frequency.png")
print(f"   fig4: hand-written dead {hw['dead']}, dense {hw['dense']}" +
      (f"; SAELens dead {sl['dead']}, dense {sl['dense']}" if sl else ""))

# ---- Figure 5: superposition in the real model ----
sp = it["superposition"]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.stairs(sp["random"], sp["bins"], color=GREY, lw=1.4, fill=True, alpha=0.35, label="random directions")
ax.stairs(sp["decoder"], sp["bins"], color=ACCENT, lw=1.6, label="the SAE's decoder directions")
ax.set_xlabel(f"|cosine| to the nearest other direction ({hw['d_sae']:,} directions in {s.D_MODEL} dimensions)")
ax.set_ylabel("directions")
ax.legend(frameon=False, fontsize=8)
save(fig, "fig5_superposition.png")
print(f"   fig5: median nearest |cos| decoder {sp['median_decoder']:.3f}, random {sp['median_random']:.3f}; "
      f"share above 0.9 {sp['share_above_0.9']:.3%}")

# ---- Figure 6: SAE vs probe vs diff of means vs PCA ----
methods = [("probe", "logistic probe", INK), ("diff_means", "difference of means", INK),
           ("sae_latent", "best single SAE latent", ACCENT), ("sae_top5_probe", "probe on the 5 best SAE latents", ACCENT_SOFT),
           ("pca_1", "PCA, first component", GREY), ("pca_best10", "PCA, best of first 10", GREY)]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
for ax, (concept, r) in zip(axes, cp.items()):
    yy = np.arange(len(methods))[::-1]
    ax.axvspan(r["random"]["mean"], r["random"]["max"], color=GREY, alpha=0.25, lw=0)
    for y_, (key, label, color) in zip(yy, methods):
        v, ci = r[key]
        ax.errorbar(v, y_, xerr=[[v - ci[0]], [ci[1] - v]], fmt="o", color=color, ms=5, capsize=2)
    ax.set_title(r["title"])
    ax.set_xlim(0.4, 1.01)
    ax.set_xlabel("held-out AUROC")
    ax.set_yticks(yy, [m[1] for m in methods])
    print(f"   fig6 {concept}: " + "; ".join(f"{m[0]} {r[m[0]][0]:.3f}" for m in methods) +
          f"; random mean {r['random']['mean']:.3f}, max {r['random']['max']:.3f}")
save(fig, "fig6_compare.png")

# ---- Figure 7: cosines between the directions ----
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
for ax, (concept, r) in zip(axes, cp.items()):
    m = np.array(r["cosine"]["matrix"])
    names = r["cosine"]["names"]
    ax.imshow(m, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(m[i, j]) > 0.6 else INK)
    ax.set_title(f"{r['title']}\n(two random directions: about ±{r['cosine']['random_scale']:.2f})", fontsize=9)
    ax.spines[:].set_visible(False)
    print(f"   fig7 {concept}: probe row " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(names, m[0])))
save(fig, "fig7_cosines.png")

# ---- Figure 8: the use test ----
ut = it["use_test"]
per = ut["per_latent"]
fig, ax = plt.subplots(figsize=(6.4, 2.8))
x = np.arange(len(per))
for i, p in enumerate(per):
    ax.plot([i, i], [p["kl_random"], p["kl_feature"]], color=RULE, lw=1, zorder=1)
ax.scatter(x, [p["kl_feature"] for p in per], color=ACCENT, s=28, label="remove the latent", zorder=3)
ax.scatter(x, [p["kl_random"] for p in per], color=GREY, s=22, marker="s", zorder=3,
           label="remove a random direction of the same length")
ax.set_yscale("log")
ax.set_xticks(x, [f"#{p['id']}" for p in per], rotation=45, fontsize=7.5)
ax.set_ylabel("KL of the next-token prediction")
ax.legend(frameon=False, fontsize=8)
save(fig, "fig8_use_test.png")
print(f"   fig8: pooled KL latent {ut['kl_feature']:.4f}, random {ut['kl_random']:.4f}, ratio {ut['ratio']:.2f} "
      f"[{ut['ratio_ci'][0]:.2f}, {ut['ratio_ci'][1]:.2f}], the latent wins {ut['wins']}/{ut['n_latents']}")
