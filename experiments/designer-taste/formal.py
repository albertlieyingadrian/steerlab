"""
Formal checks (tier 0) for The Taste Stack Episode 7: rules written before seeing any designer's choice, run on facts
extracted from the image (OCR boxes, pixels). Each returns a score per image; a pair goes to the higher score, a tie
abstains (p = 0.5). Each check also returns a small proof record (what was required, what was found).

    text    typography and preference parts: the brief's quoted strings vs the OCR text (best fuzzy match, mean)
    order   spatial accuracy: the brief's order of quoted strings vs their top-to-bottom order in the OCR boxes
    colour  colour accuracy, mood and colour tone: the brief's colour words present in the pixels
    harmony colour harmony: fit of the saturated hues to the best of seven classical hue templates

    python formal.py        # rulers R3 (OCR known answers, needs features.pkl) and R4 (colour known answers)
"""

from __future__ import annotations

import difflib
import itertools
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHECK_FOR = {
    "preference": "text",
    "typography": "text",
    "spatial_accuracy": "order",
    "color_accuracy": "colour",
    "mood_and_color_tone": "colour",
    "color_harmony": "harmony",
    "visual_hierarchy": None,
}

# ---------------------------------------------------------------- text


def quoted_strings(brief: str) -> list[str]:
    out = []
    for m in re.finditer(r"[\"“]([^\"”]{2,120})[\"”]", brief):
        s = m.group(1).strip().strip(".,;:!?").strip()
        if len(re.sub(r"[^A-Za-z0-9]", "", s)) >= 2 and s not in out:
            out.append(s)
    return out


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", s.upper())).strip()


def best_match(s: str, text: str) -> float:
    """Best SequenceMatcher ratio of s against any window of text of length len(s)-2..len(s)+2."""
    s, text = norm(s), norm(text)
    if not s:
        return 0.0
    if not text:
        return 0.0
    if s in text:
        return 1.0
    best = 0.0
    n = len(s)
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(s)
    for w in range(max(1, n - 2), n + 3):
        for i in range(max(1, len(text) - w + 1)):
            sm.set_seq1(text[i : i + w])
            if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                continue
            best = max(best, sm.ratio())
    return best


def ocr_text(boxes: list[dict]) -> str:
    return " ".join(b["t"] for b in sorted(boxes, key=lambda b: (round(b["y"], 2), b["x"])))


def segments(boxes: list[dict]) -> list[tuple[str, float, frozenset]]:
    """Candidate places a required string can sit: each OCR box, and each two boxes adjacent in reading order."""
    bs = sorted(range(len(boxes)), key=lambda i: (round(boxes[i]["y"], 2), boxes[i]["x"]))
    out = [(boxes[i]["t"], boxes[i]["y"], frozenset([i])) for i in bs]
    for i, j in itertools.pairwise(bs):
        out.append((boxes[i]["t"] + " " + boxes[j]["t"], (boxes[i]["y"] + boxes[j]["y"]) / 2, frozenset([i, j])))
    return out


def assign(strings: list[str], boxes: list[dict]) -> list[tuple[float, float | None]]:
    """One-to-one: longest required string first, each takes its best segment among boxes not yet used.
    Returns (match ratio, y of the segment) per string, in the brief's order."""
    segs = segments(boxes)
    used: set = set()
    res: dict = {}
    for k in sorted(range(len(strings)), key=lambda k: -len(norm(strings[k]))):
        best, by, bset = 0.0, None, frozenset()
        for t, y, ids in segs:
            if ids & used:
                continue
            r = best_match(strings[k], t)
            if r > best:
                best, by, bset = r, y, ids
        if best >= 0.5:
            used |= bset
        res[k] = (best, by)
    return [res[k] for k in range(len(strings))]


def text_score(strings: list[str], boxes: list[dict]) -> tuple[float | None, dict]:
    if not strings:
        return None, {"required": [], "note": "brief quotes no text"}
    per = [round(r, 3) for r, _ in assign(strings, boxes)]
    return float(np.mean(per)), {"required": strings, "match": per}


def order_score(strings: list[str], boxes: list[dict], min_match: float = 0.7) -> tuple[float | None, dict]:
    from scipy.stats import kendalltau

    if len(strings) < 2:
        return None, {"required": strings, "note": "fewer than two quoted strings"}
    ys = [y if r >= min_match else None for r, y in assign(strings, boxes)]
    found = [(i, y) for i, y in enumerate(ys) if y is not None]
    share = len(found) / len(strings)
    if len(found) < 2:
        return 0.5 * share, {"required": strings, "found_y": ys, "tau": None}
    tau = kendalltau([i for i, _ in found], [y for _, y in found]).statistic
    tau = 0.0 if np.isnan(tau) else float(tau)
    return (tau + 1) / 2 * share, {"required": strings, "found_y": ys, "tau": round(tau, 3)}


# ---------------------------------------------------------------- colour


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb uint8 or float in [0,1], (..., 3) -> CIELAB (D65)."""
    c = rgb.astype(np.float64)
    if c.max() > 1.0:
        c = c / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm**7 / (Cm**7 + 25**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p, h2p = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(np.abs(dh) > 180, dh - 360 * np.sign(dh), dh)
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hpm = np.where(np.abs(h1p - h2p) > 180, (hs + 360) / 2, hs / 2)
    hpm = np.where(C1p * C2p == 0, hs, hpm) % 360
    T = (
        1
        - 0.17 * np.cos(np.radians(hpm - 30))
        + 0.24 * np.cos(np.radians(2 * hpm))
        + 0.32 * np.cos(np.radians(3 * hpm + 6))
        - 0.20 * np.cos(np.radians(4 * hpm - 63))
    )
    SL = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    SC, SH = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    RT = -2 * np.sqrt(Cpm**7 / (Cpm**7 + 25**7)) * np.sin(np.radians(60 * np.exp(-(((hpm - 275) / 25) ** 2))))
    return np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH))


# A fixed lexicon (sRGB prototypes), written before any designer's choice was read.
COLOURS = {
    "black": (15, 15, 15),
    "white": (250, 250, 250),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "charcoal": (54, 69, 79),
    "silver": (192, 192, 192),
    "red": (210, 35, 40),
    "crimson": (180, 20, 50),
    "maroon": (110, 20, 30),
    "burgundy": (120, 25, 45),
    "pink": (245, 160, 190),
    "magenta": (210, 40, 160),
    "orange": (245, 130, 30),
    "coral": (250, 120, 90),
    "peach": (250, 200, 160),
    "yellow": (250, 215, 40),
    "gold": (212, 175, 55),
    "golden": (212, 175, 55),
    "mustard": (210, 170, 40),
    "beige": (225, 205, 170),
    "cream": (250, 240, 215),
    "brown": (120, 75, 40),
    "tan": (210, 180, 140),
    "green": (40, 150, 70),
    "olive": (110, 120, 40),
    "lime": (170, 220, 50),
    "mint": (170, 230, 200),
    "teal": (0, 128, 128),
    "turquoise": (60, 210, 200),
    "cyan": (0, 200, 230),
    "blue": (40, 90, 200),
    "navy": (20, 30, 90),
    "purple": (120, 50, 160),
    "violet": (140, 80, 200),
    "lavender": (190, 170, 230),
    "indigo": (75, 30, 130),
}
_CWORD = re.compile(r"\b(" + "|".join(sorted(COLOURS, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def colour_words(brief: str, track: str) -> list[str]:
    sec = brief
    m = re.search(r"(Aesthetics|Description):", brief)
    if m:
        sec = brief[m.end() :]
    words = []
    for w in _CWORD.findall(sec):
        w = w.lower().replace("grey", "gray").replace("golden", "gold")
        if w not in words:
            words.append(w)
    return words


def colour_score(
    words: list[str], thumb: np.ndarray, de: float = 20.0, share: float = 0.01
) -> tuple[float | None, dict]:
    if not words:
        return None, {"required": [], "note": "brief names no colour"}
    lab = srgb_to_lab(thumb.reshape(-1, 3))
    present = {}
    for w in words:
        proto = srgb_to_lab(np.array(COLOURS[w], dtype=np.float64)[None])
        frac = float((delta_e2000(lab, np.broadcast_to(proto, lab.shape)) < de).mean())
        present[w] = round(frac, 4)
    score = float(np.mean([present[w] >= share for w in words]))
    # tie-break by coverage so two images naming the same set present still differ
    return score + 1e-3 * float(np.mean([min(present[w], 0.2) for w in words])), {
        "required": words,
        "coverage": present,
    }


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255.0
    mx, mn = c.max(-1), c.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    m = d > 1e-9
    rr = m & (mx == r)
    gg = m & (mx == g) & ~rr
    bb = m & ~rr & ~gg
    h[rr] = ((g - b)[rr] / d[rr]) % 6
    h[gg] = (b - r)[gg] / d[gg] + 2
    h[bb] = (r - g)[bb] / d[bb] + 4
    s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0)
    return np.stack([h * 60, s, mx], -1)


# classical hue-harmony templates (Color Harmonization, SIGGRAPH 2006): list of (centre offset deg, width deg)
TEMPLATES = {
    "i": [(0, 18)],
    "V": [(0, 93.6)],
    "L": [(0, 18), (90, 79.2)],
    "I": [(0, 18), (180, 18)],
    "T": [(0, 180)],
    "Y": [(0, 93.6), (180, 18)],
    "X": [(0, 93.6), (180, 93.6)],
}


def harmony_distance(h: np.ndarray, w: np.ndarray) -> tuple[float, str]:
    best, name = np.inf, ""
    for tname, sectors in TEMPLATES.items():
        for rot in range(0, 360, 5):
            dist = np.full(h.shape, np.inf)
            for off, width in sectors:
                c = (rot + off) % 360
                d = np.abs((h - c + 180) % 360 - 180)
                dist = np.minimum(dist, np.maximum(0, d - width / 2))
            val = float((dist * w).sum() / max(w.sum(), 1e-9))
            if val < best:
                best, name = val, tname
    return best, name


def harmony_score(thumb: np.ndarray) -> tuple[float | None, dict]:
    hsv = rgb_to_hsv(thumb.reshape(-1, 3))
    m = (hsv[:, 1] > 0.25) & (hsv[:, 2] > 0.15)
    if m.mean() < 0.02:
        return 1.0, {
            "note": "almost no saturated colour (grey scale fits every template)",
            "saturated_share": round(float(m.mean()), 3),
        }
    dist, name = harmony_distance(hsv[m, 0], hsv[m, 1])
    return float(np.exp(-dist / 10.0)), {
        "template": name,
        "mean_arc_distance_deg": round(dist, 2),
        "saturated_share": round(float(m.mean()), 3),
    }


# ---------------------------------------------------------------- per image, per part


def image_scores(brief: str, track: str, img: dict) -> dict:
    """{check: (score or None, proof)} for one image."""
    strings = quoted_strings(brief)
    return {
        "text": text_score(strings, img["ocr"]),
        "order": order_score(strings, img["ocr"]),
        "colour": colour_score(colour_words(brief, track), img["thumb"]),
        "harmony": harmony_score(img["thumb"]),
    }


def pair_p(sa: float | None, sb: float | None) -> float:
    if sa is None or sb is None or abs(sa - sb) < 1e-9:
        return 0.5
    return 1.0 if sa > sb else 0.0


# ---------------------------------------------------------------- rulers


def r3(feat: dict) -> dict:
    items = {(it["id"], it["variant"]): it for it in feat["r3_items"]}
    ocr = {(r["id"], r["variant"]): r["ocr"] for r in feat["r3"]}
    ids = sorted({i for i, _ in items})
    text_ok = order_ok = order_n = 0
    for i in ids:
        strs = items[(i, "correct")]["strings"]
        sc, _ = text_score(strs, ocr[(i, "correct")])
        sw, _ = text_score(strs, ocr[(i, "wrong")])
        text_ok += sc > sw
        if len(strs) >= 2 and [norm(x) for x in strs] != [norm(x) for x in reversed(strs)]:
            # (a list equal to its reverse once case is ignored has no order to recover)
            o1, _ = order_score(strs, ocr[(i, "correct")])
            o2, _ = order_score(list(reversed(strs)), ocr[(i, "correct")])
            order_n += 1
            order_ok += o1 > o2
    out = {
        "text_pairs": len(ids),
        "text_correct_above_wrong": text_ok / len(ids),
        "order_pairs": order_n,
        "order_true_above_reversed": order_ok / max(order_n, 1),
    }
    out["pass"] = out["text_correct_above_wrong"] >= 0.95 and out["order_true_above_reversed"] >= 0.95
    return out


def r4(seed: int = 11) -> dict:
    """100 synthetic thumbnails with known colour patches; harmony: template-fit vs random hues."""
    rng = np.random.default_rng(seed)
    names = [n for n in COLOURS if n not in ("grey", "golden")]
    right = total = 0
    for _ in range(100):
        k = rng.integers(2, 5)
        on = list(rng.choice(names, size=k, replace=False))
        img = np.full((96, 96, 3), 128 if "gray" not in on else 60, dtype=np.uint8)
        if "gray" in on:
            img[:] = (230, 230, 230) if "white" not in on else (60, 60, 60)
        for j, n in enumerate(on):
            img[j * 20 : j * 20 + 14, 10:60] = COLOURS[n]
        absent = [n for n in names if n not in on]
        ask_abs = list(rng.choice(absent, size=2, replace=False))
        for n in on + ask_abs:
            # a word is "absent" only if no patch sits within the threshold of it (lexicon neighbours overlap)
            proto = srgb_to_lab(np.array(COLOURS[n], dtype=np.float64)[None])
            near = any(
                delta_e2000(srgb_to_lab(np.array(COLOURS[m], dtype=np.float64)[None]), proto)[0] < 20 for m in on
            )
            bgnear = delta_e2000(srgb_to_lab(img[0, 95].astype(np.float64)[None]), proto)[0] < 20
            truth = n in on or near or bgnear
            sc, _ = colour_score([n], img)
            right += (sc >= 1.0) == truth
            total += 1
    h_ok = 0
    for _ in range(100):
        base = rng.uniform(0, 360)
        fit = (base + rng.normal(0, 4, size=(96 * 96 // 2))) % 360
        fit = np.concatenate([fit, (fit[: 96 * 96 // 2] + 180) % 360])  # type I (complementary)
        rnd = rng.uniform(0, 360, size=96 * 96)

        def paint(hues):
            hh = hues.reshape(96, 96) / 60
            c = np.ones_like(hh)
            x = c * (1 - np.abs(hh % 2 - 1))
            z = np.zeros_like(hh)
            sec = hh.astype(int) % 6
            rgb = np.select(
                [sec[..., None] == i for i in range(6)],
                [np.stack(t, -1) for t in [(c, x, z), (x, c, z), (z, c, x), (z, x, c), (x, z, c), (c, z, x)]],
            )
            return (rgb * 255).astype(np.uint8)

        h_ok += harmony_score(paint(fit))[0] > harmony_score(paint(rnd))[0]
    out = {"colour_calls": total, "colour_right": right / total, "harmony_fit_above_random": h_ok / 100}
    out["pass"] = out["colour_right"] >= 0.95 and out["harmony_fit_above_random"] >= 0.90
    return out


if __name__ == "__main__":
    import pickle

    from taste import CACHE

    res = {"R4": r4()}
    fp = CACHE / "features.pkl"
    if fp.exists():
        res["R3"] = r3(pickle.loads(fp.read_bytes()))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "rulers_formal.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
