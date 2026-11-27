"""Draw the note's cover from one real evaluation episode and the seed-1 model's attention.

    python make_cover.py        # needs results/ckpt_s1.pt; writes report/figures/hero.jpg and the research-page thumb

Left: the episode's 32 x 32 map, the camera's path, the view at the frame with the most long-gap revisits (orange box)
and the earlier views the model attends to most when it redraws that frame (lines, width = attention). Right: the
cache as a strip of earlier frames, with the attention each frame gets (orange bars) and the frames that hold an
earlier view of a revisited place (blue ticks). Same inputs, same picture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import episodes, load
from wm_core import NTOK, VIEW, P

GROUND, TEXT, ORANGE, BLUE = "#15110e", "#d9d2ca", "#e07a3f", "#4a86e8"
TILES = ListedColormap(["#2a2420", "#3b332d", "#4d443c", "#60564c", "#3a3f44", "#4f4a40", "#6f675f", "#857b70"])
OUT = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "wm-kv-cache.jpg"


def main(ep=62):
    model, _ = load(HERE / "results" / "ckpt_s1.pt", "cpu")
    tiles, pose, act, loc, g = (x[ep : ep + 1] for x in episodes(64, 48, 77, "cpu"))
    T = act.shape[1]
    cnt = (g[0] >= 16).sum(-1) + torch.arange(g.shape[1]) * 1e-3  # ties: the latest frame
    u = int(cnt.argmax())
    with torch.no_grad():
        _, attns = model.forward_seq(tiles, pose, act, return_attn=True)
    a = torch.stack(attns).sum((0, 2))[0]  # [queries, clean entries], summed over layers and heads
    rev = g[0, u] >= 2
    per_entry = (a[(u - 1) * NTOK : u * NTOK] * rev[:, None]).sum(0)
    per_frame = per_entry.view(T, NTOK).sum(1).numpy()
    per_frame /= per_frame.max()
    m = np.full((32, 32), -1)  # the map, rebuilt from the episode's frames
    for t in range(T):
        x0, y0 = pose[0, t, 0].item() * P, pose[0, t, 1].item() * P
        m[y0 : y0 + 8, x0 : x0 + 8] = tiles[0, t].view(4, 4, 2, 2).permute(0, 2, 1, 3).reshape(8, 8).numpy()
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=GROUND)
    ax = fig.add_axes([0.04, 0.07, 0.5, 0.86], facecolor=GROUND)
    ax.imshow(np.ma.masked_less(m, 0), cmap=TILES, vmin=0, vmax=7, interpolation="nearest")
    cx = pose[0, :, 0].numpy() * P + VIEW / 2 - 0.5
    cy = pose[0, :, 1].numpy() * P + VIEW / 2 - 0.5
    ax.plot(cx[: u + 1], cy[: u + 1], color=TEXT, lw=1.0, alpha=0.55)
    lc = loc[0].reshape(-1)
    sights = set()
    for j in range(NTOK):
        if rev[j]:
            sights |= {int(i) // NTOK for i in (lc == loc[0, u, j]).nonzero().squeeze(1) if int(i) // NTOK < u}
    top = np.argsort(per_frame[:u])[::-1][:6]
    for f in top:
        w = per_frame[f]
        ax.plot([cx[u], cx[f]], [cy[u], cy[f]], color=ORANGE, lw=0.8 + 5 * w, alpha=0.85, solid_capstyle="round")
        x0, y0 = pose[0, f, 0].item() * P - 0.5, pose[0, f, 1].item() * P - 0.5
        ax.add_patch(plt.Rectangle((x0, y0), 8, 8, fill=False, ec=ORANGE, lw=0.8, alpha=0.5 + 0.5 * w))
    x0, y0 = pose[0, u, 0].item() * P - 0.5, pose[0, u, 1].item() * P - 0.5
    ax.add_patch(plt.Rectangle((x0, y0), 8, 8, fill=False, ec=ORANGE, lw=3))
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(31.5, -0.5)
    ax.axis("off")
    bx = fig.add_axes([0.6, 0.3, 0.36, 0.4], facecolor=GROUND)
    bx.bar(np.arange(u), per_frame[:u], color=ORANGE, width=0.75)
    for f in sorted(sights):
        bx.plot([f, f], [-0.12, -0.04], color=BLUE, lw=2)
    bx.set_xlim(-1, u)
    bx.set_ylim(-0.15, 1.05)
    bx.axis("off")
    bx.text(-1, 1.16, f"the cache when frame {u} is drawn", color=TEXT, fontsize=15, family="monospace")
    bx.text(-1, -0.3, "bars: attention on each earlier frame", color=TEXT, fontsize=10.5, family="monospace", alpha=0.8)
    bx.text(
        -1,
        -0.4,
        "ticks: frames that saw the revisited places",
        color=TEXT,
        fontsize=10.5,
        family="monospace",
        alpha=0.8,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    png = OUT.with_suffix(".png")
    fig.savefig(png, facecolor=GROUND)
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    im.save(OUT, quality=88)
    png.unlink()
    th = Image.new("RGB", (720, 480), GROUND)  # the whole scene, letterboxed to 3:2
    s = min(720 / im.width, 480 / im.height)
    small = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    th.paste(small, ((720 - small.width) // 2, (480 - small.height) // 2))
    THUMB.parent.mkdir(parents=True, exist_ok=True)
    th.save(THUMB, quality=88)
    print(
        "episode",
        ep,
        "frame",
        u,
        "long-gap revisit tokens",
        int(cnt[u]),
        "sighting frames",
        sorted(sights),
        "top attended",
        top.tolist(),
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 62)
