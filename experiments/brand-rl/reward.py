"""Scores for one rendered policy page: what each arm's reward reads, and every unseen ruler.

    from reward import Scorer, reward, novelty
    sc = Scorer()                                   # loads the spec, the family split, the probe, the references
    s = sc.score(brief, render_out, truncated)      # one dict per page
    r = reward(arm, s)                              # the arm's scalar reward (novelty is added in the training loop)

In the reward (by arm, fixed in the plan):
  brand       share of the REWARD rule families that pass on the page (Episode 5's proof artifact, per rule: a
              family passes when none of its checks fails; a family with no applicable check passes, as the
              checker reports it). Every point of it names a rule and its source line.
  coverage    share of the brief's items that have an element of the right ROLE (text not compared)
  slop_p      Episode 3's markup probe: P(AI-builder) from the badge-masked block tree
  copy        Episode 5's copy gate: structure similarity >= 0.9418 (its dev threshold) to any of the brand's
              18 reference pages, the prompt's exemplar among them
Never in any reward (unseen rulers):
  heldout_*   the held-out rule families: pass share (loose, vacuous passes counted) and the pooled check pass rate
              over applicable checks (strict)
  fidelity_*  the brief's exact texts: strict = in an element of the right role; loose = anywhere in the page text
  copy_*      the copy check's other components (layout, style, text) against the exemplar and the library
  loop, empty, invalid, truncated
  The screenshot slop probe and the judge are added by the evaluation pass (they need the screenshot).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import ep6
from slop_fit import load_probe

HERE = Path(__file__).resolve().parent
COPY_THRESHOLD = 0.9418  # Episode 5's dev-chosen gate (results/copy.json, gate_structure)
W_SLOP, W_COPY, W_NOVEL = 0.25, 0.5, 0.25  # reward weights, fixed in the plan

# Episode 3's badge box (slop-probes/ep3.py) and strip_badge (slop-probes/modal_masked.py), copied so the markup
# probe sees blocks the way it was trained.
VIEW_W, VIEW_H, MASK_W, MASK_H = 1280, 800, 320, 96
MASK_BOX = (VIEW_W - MASK_W, VIEW_H - MASK_H, VIEW_W, VIEW_H)


def _inside(x, y, w, h, box=MASK_BOX) -> bool:
    x0, y0, x1, y1 = box
    return x >= x0 - 2 and y >= y0 - 2 and x + w <= x1 + 2 and y + h <= y1 + 2


def strip_badge(rec: dict) -> dict:
    blocks = rec.get("blocks") or []
    keep = [not _inside(*b[:4]) for b in blocks]
    new_ix, out = {}, []
    for i, b in enumerate(blocks):
        if keep[i]:
            new_ix[i] = len(out)
            out.append(list(b))
    for b in out:
        p = b[4]
        while 0 <= p < len(blocks) and not keep[p]:
            p = blocks[p][4]
        b[4] = new_ix.get(p, -1)
    return dict(rec, blocks=out)


# roles that satisfy each brief item (coverage reads roles only; strict fidelity also needs the exact text)
ROLES = {
    "h1": {"h1"},
    "para": {"body", "lead", "body-s"},
    "label": {"label"},
    "hint": {"hint", "body", "body-s"},
    "link": {"link"},
    "banner_title": {"banner-title"},
    "banner_text": {"banner-text"},
    "footer_link": {"footer-link"},
}


def roles_for(it: dict) -> frozenset[str]:
    return frozenset({f"button-{it['variant']}"} if it["kind"] == "button" else ROLES[it["kind"]])


def norm(t: str | None) -> str:
    t = re.sub(r"\s+", " ", (t or "").replace(" ", " ")).strip().casefold()
    return re.sub(r"\s+([.,;:!?])", r"\1", t)


def loop_flag(els: list[dict]) -> bool:
    """Repetition in the visible text: at least 40 words and fewer than half of its word 4-grams distinct."""
    words = " ".join(e.get("text") or "" for e in els).split()
    if len(words) < 40:
        return False
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    return len(set(grams)) / len(grams) < 0.5


def coverage_fidelity(brief: dict, els: list[dict]) -> tuple[float, float, float]:
    """(coverage, strict fidelity, loose fidelity), each a share of the brief's items.
    Coverage: items of the same role set need that many distinct elements of those roles with any text (a label also
    needs an input on the page). Strict: an element of the right role whose own text equals the item's text.
    Loose: the item's text appears anywhere in the page's visible text."""
    texts = [(e["role"], norm(e.get("text"))) for e in els]
    page_text = norm(" ".join(e.get("text") or "" for e in els))
    has_input = any(e["role"] == "input" for e in els)
    need: dict[frozenset, int] = {}
    for it in brief["items"]:
        need[roles_for(it)] = need.get(roles_for(it), 0) + 1
    hits = 0
    for roles, k in need.items():
        have = sum(1 for r, t in texts if r in roles and t)
        if roles == ROLES["label"] and not has_input:
            have = 0
        hits += min(have, k)
    n = len(brief["items"])
    if n == 0:  # only the rulers score pages without a brief
        return 1.0, 1.0, 1.0
    strict = sum(any(r in roles_for(it) and t == norm(it["text"]) for r, t in texts) for it in brief["items"])
    loose = sum(norm(it["text"]) in page_text for it in brief["items"])
    return hits / n, strict / n, loose / n


class Scorer:
    def __init__(self, probe_path: Path | None = None):
        checker, _, copycheck = ep6.brand_spec()
        self.check, self.cc = checker.check, copycheck
        self.spec = ep6.load_spec()
        split = ep6.family_split()
        self.reward_fams, self.held_fams = split["reward"], split["heldout"]
        self.f2 = ep6.ep2_features()
        self.slop = load_probe("markup", **({"path": probe_path} if probe_path else {}))
        refs = []
        for sp in ("dev", "heldout"):
            path = ep6.BRAND_SPEC / "data" / f"facts_copy_{ep6.BRAND}_{sp}.jsonl"
            refs += [json.loads(x) for x in path.read_text().splitlines() if x and "-ref" in x[:40]]
        self.refs = {r["id"]: self.cc.feats(r) for r in refs if "-ref" in r["id"]}
        assert len(self.refs) == 18, len(self.refs)

    def families(self, proof: dict) -> dict:
        n_chk = dict.fromkeys(ep6.FAMILIES, 0)
        n_fail = dict.fromkeys(ep6.FAMILIES, 0)
        for c in proof["checks"]:
            f = ep6.FAMILY_OF[c["rule"]]
            n_chk[f] += 1
            n_fail[f] += not c["pass"]
        return {f: (n_chk[f], n_fail[f]) for f in ep6.FAMILIES}

    def copy_parts(self, facts: dict, exemplar: str) -> dict:
        f = self.cc.feats(facts)
        sims = {rid: self.cc.sim(f, rf) for rid, rf in self.refs.items()}
        out: dict = {}
        for comp in ("structure", "layout", "style", "text"):
            out[f"copy_{comp}_max"] = max(s[comp] for s in sims.values())
            out[f"copy_{comp}_ex"] = sims[exemplar][comp]
        out["copy"] = round(out["copy_structure_max"], 4) >= COPY_THRESHOLD  # Episode 5 rounds to 4 places
        out["wl"] = f["structure"]
        return out

    def score(self, brief: dict, rend: dict, truncated: bool, exemplar: str | None = None) -> dict:
        els = rend.get("elements") or []
        s: dict = {"brief": brief["id"], "truncated": bool(truncated), "valid": bool(rend.get("ok")) and len(els) > 0}
        s["n_elements"] = len(els)
        s["n_words"] = sum(len((e.get("text") or "").split()) for e in els)
        s["empty"] = s["n_words"] < 5
        s["loop"] = loop_flag(els)
        if not s["valid"]:
            s.update(brand=0.0, coverage=0.0, fidelity_strict=0.0, fidelity_loose=0.0, slop_p=1.0, copy=False)
            s.update(heldout_loose=0.0, heldout_strict=None, reward_strict=None, fam={}, wl=None, wl2=None)
            return s
        facts = {
            "id": brief["id"],
            "brand": ep6.BRAND,
            "domain": brief["domain"],
            "page_type": brief["page_type"],
            "elements": els,
        }
        fam = self.families(self.check(self.spec, facts))
        s["fam"] = {f: list(v) for f, v in fam.items()}
        s["brand"] = float(np.mean([fam[f][1] == 0 for f in self.reward_fams]))
        s["heldout_loose"] = float(np.mean([fam[f][1] == 0 for f in self.held_fams]))
        for name, fams in (("heldout", self.held_fams), ("reward", self.reward_fams)):
            n = sum(fam[f][0] for f in fams)
            s[f"{name}_strict"] = (n - sum(fam[f][1] for f in fams)) / n if n else None
            s[f"{name}_applicable"] = n
        s["coverage"], s["fidelity_strict"], s["fidelity_loose"] = coverage_fidelity(brief, els)
        rec = strip_badge(rend.get("ep2") or {})
        s["wl2"] = self.f2.wl_features(rec)  # Episode 2's block-tree histogram (the slop probe's input)
        s["slop_p"] = float(self.slop(s["wl2"][None, :])[0])
        s.update(self.copy_parts(facts, exemplar or ep6.exemplar_id(brief)))
        return s


def reward(arm: str, s: dict) -> float:
    """The arm's reward before novelty (TEAM adds W_NOVEL x novelty in the training loop; SHUF permutes ENV's
    rewards within each group there). Invalid or truncated pages get 0 in every arm."""
    if not s["valid"] or s["truncated"]:
        return 0.0
    if arm == "check":
        return s["brand"]
    if arm in ("env", "team", "shuf"):
        return s["brand"] * s["coverage"] + W_SLOP * (1.0 - s["slop_p"]) - W_COPY * float(s["copy"])
    raise ValueError(arm)


def novelty(wls: list, groups: list[int], archive: list, d0: float, k: int = 8) -> list[float]:
    """Per page: half the mean structure distance (1 - cosine of Episode 5's element-tree histogram) to the other
    pages for the same brief, half the mean distance to its k nearest pages in the archive (earlier steps' pages for
    other briefs), divided by d0 (the base model's dev median of the same quantity) and capped at 1."""

    def unit(v):
        v = np.asarray(v, np.float64)
        n = np.linalg.norm(v)
        return v / n if n else v

    U = [unit(w) if w is not None else None for w in wls]
    A = np.stack([unit(a) for a in archive]) if archive else None
    out = []
    for i, u in enumerate(U):
        if u is None:
            out.append(0.0)
            continue
        same = [1 - float(u @ U[j]) for j in range(len(U)) if j != i and groups[j] == groups[i] and U[j] is not None]
        within = float(np.mean(same)) if same else 0.0
        cross = float(np.sort(1 - A @ u)[:k].mean()) if A is not None else within
        out.append(min(1.0, (0.5 * within + 0.5 * cross) / d0) if d0 > 0 else 0.0)
    return out
