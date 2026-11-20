"""
Hand features and distances for the design index (definitions fixed in the plan). numpy + PIL + scikit-learn only,
so the same code runs on the laptop and in Modal containers.

Per page (from the first-screen screenshot and the extractor's element and block lists):
  palette   k-means (k = 5) in CIELAB on a 160x100 copy of the screenshot -> 5 centres + weights
  type      text area per font family (first family, normalised) -> sparse histogram; plus base size and the
            largest-to-base size ratio
  spacing   6 numbers: content width, left-edge count, median vertical gap, median top padding, median control
            radius, median line-height / font-size
  layout    occupancy grid 32 x 30 over 1280 x 1200 px, three channels (text, media, control) -> 2,880 values
  dom       Weisfeiler-Lehman subtree histogram of the block tree (3 rounds, hashed to 1,024 bins, sqrt-damped)
Distances: palette = weighted Chamfer (ΔE76); type, layout, dom = 1 - cosine; spacing = Euclidean on z-scores.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from itertools import pairwise

import numpy as np

GRID_W, GRID_H, PAGE_W, PAGE_H = 32, 30, 1280, 1200
WL_BINS, WL_ROUNDS = 1024, 3
SERIF_DEFAULTS = {
    "times new roman",
    "times",
    "serif",
    "tinos",
    "liberation serif",
    "dejavu serif",
}
SYSTEM = {
    "-apple-system",
    "blinkmacsystemfont",
    "system-ui",
    "segoe ui",
    "ui-sans-serif",
    "sans-serif",
}


# ---------- colour ----------
def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb in 0..255, shape (..., 3) -> CIELAB (D65)."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack(
        [
            116 * f[..., 1] - 16,
            500 * (f[..., 0] - f[..., 1]),
            200 * (f[..., 1] - f[..., 2]),
        ],
        -1,
    )


def lab_to_hex(lab: np.ndarray) -> str:
    L, a, b = lab
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    xyz = np.array([fx, fy, fz])
    xyz = np.where(xyz**3 > 0.008856, xyz**3, (xyz - 16 / 116) / 7.787) * np.array(
        [0.95047, 1.0, 1.08883]
    )
    m = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )
    c = m @ xyz
    c = np.where(
        c > 0.0031308, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055, 12.92 * c
    )
    r, g, b2 = (np.clip(c, 0, 1) * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{b2:02x}"


def palette(img, k: int = 5, seed: int = 0):
    from sklearn.cluster import KMeans

    small = np.asarray(img.convert("RGB").resize((160, 100)))
    lab = srgb_to_lab(small.reshape(-1, 3))
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(lab)
    w = np.bincount(km.labels_, minlength=k) / len(lab)
    order = np.argsort(-w)
    return km.cluster_centers_[order].astype(np.float32), w[order].astype(np.float32)


def palette_dist(ca, wa, cb, wb) -> float:
    d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    return float(0.5 * (wa @ d.min(1) + wb @ d.min(0)))


# ---------- type ----------
def norm_family(f: str) -> str:
    f = f.strip().lower()
    if f in SYSTEM:
        return "system-ui"
    for suf in (
        " var",
        " variable",
        " vf",
        " web",
        " regular",
        " pro",
        " display",
        " text",
    ):
        f = f.removesuffix(suf)
    return f


def type_features(rec: dict):
    fonts = rec.get("fonts") or []
    hist, sizes = Counter(), []
    for e in rec.get("elements") or []:
        _, y, w, h, kind, fi, size, _, words = e[:9]
        if kind not in (1, 3) or words <= 0 or fi < 0 or fi >= len(fonts):
            continue
        area = w * min(
            h, 3 * max(size, 8)
        )  # a text box's area, with tall wrappers capped
        hist[norm_family(fonts[fi])] += area
        if y < 1600:
            sizes.append((size, words))
    total = sum(hist.values()) or 1.0
    hist = {k: v / total for k, v in hist.items()}
    if sizes:
        arr = np.array(sizes, dtype=float)
        order = np.argsort(arr[:, 0])
        cum = np.cumsum(arr[order, 1])
        base = float(arr[order][np.searchsorted(cum, cum[-1] / 2), 0])
        ratio = float(arr[:, 0].max() / max(base, 1.0))
    else:
        base, ratio = 0.0, 0.0
    serif_share = sum(v for k, v in hist.items() if k in SERIF_DEFAULTS)
    return hist, base, ratio, serif_share


def cos_dist_sparse(a: dict, b: dict) -> float:
    num = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return 1.0 - num / (na * nb) if na and nb else 1.0


# ---------- spacing and layout ----------
def spacing_features(rec: dict) -> np.ndarray:
    els = [e for e in rec.get("elements") or [] if e[1] < PAGE_H]
    text = [e for e in els if e[4] == 1 and e[8] > 0]
    if len(text) < 3:
        return np.full(6, np.nan, dtype=np.float32)
    lefts = np.array([e[0] for e in text], float)
    rights = np.array([e[0] + e[2] for e in text], float)
    width = (np.percentile(rights, 90) - np.percentile(lefts, 10)) / PAGE_W
    edges = Counter(round(x / 8) for x in lefts)
    n_edges = sum(1 for c in edges.values() if c >= 3)
    ys = sorted((e[1], e[1] + e[3]) for e in text)
    gaps = [b[0] - a[1] for a, b in pairwise(ys) if b[0] - a[1] > 0]
    gap = float(np.median(gaps)) if gaps else 0.0
    pads = [e[11] for e in els if e[11] > 0]
    radii = [min(e[14], 64.0) for e in els if e[4] == 3]  # pill buttons report 9999px
    lh = [e[15] / e[6] for e in text if e[6] > 0 and e[15] > 0]
    return np.array(
        [
            width,
            n_edges,
            gap,
            np.median(pads) if pads else 0,
            np.median(radii) if radii else 0,
            np.median(lh) if lh else 1.2,
        ],
        dtype=np.float32,
    )


def occupancy(boxes, page_w: float = PAGE_W, page_h: float = PAGE_H) -> np.ndarray:
    """boxes: iterable of (x, y, w, h, channel 0..2) in pixels. Fraction of each cell covered, per channel."""
    fine = np.zeros((3, GRID_H * 4, GRID_W * 4), dtype=np.float32)
    sx, sy = GRID_W * 4 / page_w, GRID_H * 4 / page_h
    for x, y, w, h, ch in boxes:
        x0, x1 = int(max(0, x * sx)), int(min(GRID_W * 4, np.ceil((x + w) * sx)))
        y0, y1 = int(max(0, y * sy)), int(min(GRID_H * 4, np.ceil((y + h) * sy)))
        if x1 > x0 and y1 > y0:
            fine[ch, y0:y1, x0:x1] = 1.0
    return fine.reshape(3, GRID_H, 4, GRID_W, 4).mean((2, 4)).reshape(-1)


def layout_features(rec: dict) -> np.ndarray:
    boxes = []
    for e in rec.get("elements") or []:
        x, y, w, h, kind = e[:5]
        if kind == 2 and w * h > 0.8 * PAGE_W * 800:
            continue  # full-page background images say nothing about layout
        boxes.append((x, y, w, h, kind - 1))
    return occupancy(boxes)


# ---------- DOM graph ----------
def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def wl_features(rec: dict) -> np.ndarray:
    blocks = (rec.get("blocks") or [])[:3000]
    vec = np.zeros(WL_BINS, dtype=np.float32)
    if not blocks:
        return vec
    children = [[] for _ in blocks]
    for i, b in enumerate(blocks):
        if 0 <= b[4] < len(blocks):
            children[b[4]].append(i)
    labels = [str(b[5]) for b in blocks]
    for lab in labels:
        vec[_h(lab) % WL_BINS] += 1
    for _ in range(WL_ROUNDS):
        labels = [
            labels[i] + "(" + ",".join(sorted(labels[c] for c in children[i])) + ")"
            for i in range(len(blocks))
        ]
        labels = [str(_h(lab)) for lab in labels]
        for lab in labels:
            vec[int(lab) % WL_BINS] += 1
    return np.sqrt(vec)


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1.0 - float(a @ b / (na * nb)) if na and nb else 1.0


# ---------- quality flags ----------
def quality(rec: dict, serif_share: float) -> dict:
    """Inputs to the render filter (the render check). Thresholds are tuned on dev and stored in results.tsv."""
    els = rec.get("elements") or []
    has_bg = sum(1 for e in els if e[10] is not None and e[10] != [255, 255, 255]) > 0
    return {
        "blank": (rec.get("pixel_std") or 0) < 8 or (rec.get("n_words") or 0) < 15,
        "error": bool(rec.get("parked")) or (rec.get("status") or 200) >= 400,
        "unstyled_score": serif_share * (0.5 if has_bg else 1.0),
        "fail_rate": (rec.get("n_failed", 0) + rec.get("n_http_err", 0))
        / max(1, rec.get("n_req", 1)),
    }


def page_features(rec: dict, img) -> dict:
    hist, base, ratio, serif = type_features(rec)
    pc, pw = palette(img)
    return {
        "pid": rec["pid"],
        "pal_c": pc,
        "pal_w": pw,
        "type_hist": hist,
        "type_base": base,
        "type_ratio": ratio,
        "spacing": spacing_features(rec),
        "layout": layout_features(rec).astype(np.float16),
        "dom": wl_features(rec),
        "n_words": rec.get("n_words", 0),
        "n_blocks": len(rec.get("blocks") or []),
        "doc_h": rec.get("doc_h", 0),
        "generator": rec.get("generator"),
        **quality(rec, serif),
    }
