"""Job building, judging and summaries for every Voice Alchemy round.

    python pipeline.py jobs <round>              # writes data/jobs_<round>.jsonl
    python pipeline.py score <round> [--all]     # judges results/gen_<round>.jsonl -> results/scored_<round>.jsonl
    python pipeline.py summary <round>           # per-unit target gain, coherence and loop share

Row fields: rid, set (dev/pilot/heldout), kind (unsteered|prompt|single|pair|control|control_pair), uids, traits
(target traits), s (strength per unit), q, cat, pi (prompt index), prefix, push ([[uid, s], ...]).
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict

import judge
import va

ALL_TRAITS = list(va.TRAITS)


def meta() -> dict:
    return json.loads((va.RESULTS / "units_meta.json").read_text())


def base_rows(pset: str) -> list[dict]:
    return [
        {
            "rid": f"{pset}-unsteered-p{pi}",
            "set": pset,
            "kind": "unsteered",
            "uids": [],
            "traits": [],
            "s": [],
            "q": q,
            "cat": cat,
            "pi": pi,
            "prefix": None,
            "push": [],
        }
        for pi, (cat, q) in enumerate(va.prompt_set(pset))
    ]


def prompt_rows(pset: str, traits: list[str], pairs: list[tuple[str, str]] = ()) -> list[dict]:
    rows = []
    for pi, (cat, q) in enumerate(va.prompt_set(pset)):
        for t in traits:
            rows.append(
                {
                    "rid": f"{pset}-prompt-{t}-p{pi}",
                    "set": pset,
                    "kind": "prompt",
                    "uids": [f"prompt-{t}"],
                    "traits": [t],
                    "s": [],
                    "q": q,
                    "cat": cat,
                    "pi": pi,
                    "prefix": va.instruction(t),
                    "push": [],
                }
            )
        for a, b in pairs:
            rows.append(
                {
                    "rid": f"{pset}-promptpair-{a}+{b}-p{pi}",
                    "set": pset,
                    "kind": "pair",
                    "uids": [f"prompt-{a}", f"prompt-{b}"],
                    "traits": [a, b],
                    "s": [],
                    "q": q,
                    "cat": cat,
                    "pi": pi,
                    "prefix": va.instruction_pair(a, b),
                    "push": [],
                }
            )
    return rows


def unit_rows(pset: str, uid: str, trait: str, strengths: list[float], kind: str = "single") -> list[dict]:
    return [
        {
            "rid": f"{pset}-{uid}-s{s}-p{pi}",
            "set": pset,
            "kind": kind,
            "uids": [uid],
            "traits": [trait],
            "s": [s],
            "q": q,
            "cat": cat,
            "pi": pi,
            "prefix": None,
            "push": [[uid, s]],
        }
        for pi, (cat, q) in enumerate(va.prompt_set(pset))
        for s in strengths
    ]


def pair_rows(pset: str, ua: str, ta: str, sa: float, ub: str, tb: str, sb: float, kind: str = "pair") -> list[dict]:
    return [
        {
            "rid": f"{pset}-{ua}+{ub}-p{pi}",
            "set": pset,
            "kind": kind,
            "uids": [ua, ub],
            "traits": [ta, tb],
            "s": [sa, sb],
            "q": q,
            "cat": cat,
            "pi": pi,
            "prefix": None,
            "push": [[ua, sa], [ub, sb]],
        }
        for pi, (cat, q) in enumerate(va.prompt_set(pset))
    ]


def write_jobs(name: str, rows: list[dict]) -> None:
    seen: set = set()
    uniq = [r for r in rows if not (r["rid"] in seen or seen.add(r["rid"]))]
    va.DATA.mkdir(parents=True, exist_ok=True)
    (va.DATA / f"jobs_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in uniq))
    print(f"jobs_{name}: {len(uniq)} rows")


# --- rounds ---------------------------------------------------------------------------------------------------------
DEV1_S = [0.1, 0.2, 0.4, 0.8]


def choose_lstar() -> int:
    """Pre-registered: the layer whose directions give the largest mean target gain at any dev1 strength with loop
    share <= 0.2."""
    tab = summary_table(load_scored("dev1"))
    best = {}
    for L in va.CFG["cand_layers"]:
        for s in DEV1_S:
            vs = [v for (u, ss), v in tab.items() if u.startswith(f"L{L}-direction-") and ss == (s,)]
            if vs and st.mean(v["loop"] for v in vs) <= va.LOOP_BAR:
                best[L] = max(best.get(L, -99), st.mean(v["gain"] for v in vs))
    L = max(best, key=best.get)
    (va.RESULTS / "lstar.txt").write_text(str(L))
    print("layer gains", best, "-> L*", L)
    return L


def jobs_dev1() -> None:
    """Layer choice: difference-of-means directions at L in {6, 9, 12}, every trait, 4 strengths; two random directions
    per layer as a scale check; unsteered and prompting baselines."""
    rows = base_rows("dev") + prompt_rows("dev", ALL_TRAITS)
    for L in va.CFG["cand_layers"]:
        for t in ALL_TRAITS:
            rows += unit_rows("dev", f"L{L}-direction-{t}", t, DEV1_S)
        for t in ("warm", "formal"):
            rows += unit_rows("dev", f"L{L}-rand_direction-{t}", t, DEV1_S, kind="control")
    write_jobs("dev1", rows)


LSTARS = {
    "gemma": 12,
    "qwen": int((va.RESULTS / "lstar.txt").read_text()) if (va.RESULTS / "lstar.txt").exists() else None,
}
LSTAR = LSTARS[va.TAG]
GRID = [0.2, 0.3, 0.45, 0.6]
"""Gemma: fixed after dev1 (results.tsv): L* = 12. Qwen: dev1 writes results/qwen3-8b/lstar.txt by the same rule."""


def jobs_dev2() -> None:
    """Unit choice: the 3 candidates per trait for neuron, SAE feature and community at L*, on the grid; the rank-0
    candidate's random control for each; directions and their controls on the grid."""
    m = meta()["units"]
    rows = base_rows("dev")
    for typ in ("neuron", "sae", "community"):
        for t in ALL_TRAITS:
            for rank in range(3):
                uid = f"L{LSTAR}-{typ}-{t}-{rank}"
                if uid in m:
                    rows += unit_rows("dev", uid, t, GRID)
            if f"L{LSTAR}-rand_{typ}-{t}-0" in m:
                rows += unit_rows("dev", f"L{LSTAR}-rand_{typ}-{t}-0", t, GRID, kind="control")
    for t in ALL_TRAITS:
        rows += unit_rows("dev", f"L{LSTAR}-direction-{t}", t, GRID)
        rows += unit_rows("dev", f"L{LSTAR}-rand_direction-{t}", t, GRID, kind="control")
    write_jobs("dev2", rows)


DEV3_S = [0.9, 1.8, 3.6]


def jobs_dev3() -> None:
    """Grid extension (dev2 showed MLP-based units barely move at s <= 0.6): the same candidates and controls at
    s in {0.9, 1.8, 3.6}; directions once at 0.9 to confirm they break."""
    m = meta()["units"]
    rows = base_rows("dev")
    for typ in ("neuron", "sae", "community"):
        for t in ALL_TRAITS:
            for rank in range(3):
                uid = f"L{LSTAR}-{typ}-{t}-{rank}"
                if uid in m:
                    rows += unit_rows("dev", uid, t, DEV3_S)
            if f"L{LSTAR}-rand_{typ}-{t}-0" in m:
                rows += unit_rows("dev", f"L{LSTAR}-rand_{typ}-{t}-0", t, DEV3_S, kind="control")
    for t in ALL_TRAITS:
        rows += unit_rows("dev", f"L{LSTAR}-direction-{t}", t, [0.9])
    write_jobs("dev3", rows)


FINAL_GRID = [0.3, 0.45, 0.6, 0.9, 1.8]
"""Fixed after dev3 (results.tsv, plan amendment 2): spans every unit type's working range."""
TYPES = ("direction", "neuron", "sae", "community")


def dev_table() -> dict:
    rows = load_scored("dev2")
    if (va.RESULTS / "scored_dev3.jsonl").exists():
        rows += [r for r in load_scored("dev3") if r["kind"] != "unsteered"]
    if (va.RESULTS / "scored_dev4.jsonl").exists():
        rows += [r for r in load_scored("dev4") if r["kind"] in ("single", "control")]
    return summary_table(rows)


def operating(tab: dict, uid: str) -> float:
    """Largest target gain among grid strengths with loop share <= 0.2 and mean coherence >= 6; else the lowest."""
    ok = [
        (tab[(uid, (s,))]["gain"], -s, s)
        for s in FINAL_GRID
        if (uid, (s,)) in tab and tab[(uid, (s,))]["loop"] <= va.LOOP_BAR and tab[(uid, (s,))]["coh"] >= 6
    ]
    return max(ok)[2] if ok else FINAL_GRID[0]


def select() -> dict:
    """The pre-registered unit choice and operating strengths, from the dev rounds only."""
    tab = dev_table()
    m = meta()["units"]
    sel: dict = {}
    for typ in TYPES:
        for t in ALL_TRAITS:
            if typ == "direction":
                uid, ctrl = f"L{LSTAR}-direction-{t}", f"L{LSTAR}-rand_direction-{t}"
            else:
                cands = [f"L{LSTAR}-{typ}-{t}-{r}" for r in range(3) if f"L{LSTAR}-{typ}-{t}-{r}" in m]
                best = []
                for u in cands:
                    g = [
                        tab[(u, (s,))]["gain"]
                        for s in FINAL_GRID
                        if (u, (s,)) in tab and tab[(u, (s,))]["loop"] <= va.LOOP_BAR
                    ]
                    best.append((max(g) if g else -99, -m[u]["rank"], u))
                uid = max(best)[2]
                ctrl = uid.replace(f"-{typ}-", f"-rand_{typ}-")
            sel[f"{typ}-{t}"] = {
                "type": typ,
                "trait": t,
                "uid": uid,
                "control": ctrl,
                "s": operating(tab, uid),
                "dev_gain": max(tab[(uid, (s,))]["gain"] for s in FINAL_GRID if (uid, (s,)) in tab),
            }
    if any(k[0] == "circuit-warm" for k in tab):
        for t in ALL_TRAITS:
            uid = f"circuit-{t}"
            sel[f"circuit-{t}"] = {
                "type": "circuit",
                "trait": t,
                "uid": uid,
                "control": f"rand_circuit-{t}",
                "s": operating(tab, uid),
                "dev_gain": max(tab[(uid, (s,))]["gain"] for s in FINAL_GRID if (uid, (s,)) in tab),
            }
    (va.RESULTS / "selection.json").write_text(json.dumps(sel, indent=1))
    return sel


def jobs_dev2q() -> None:
    """Qwen: unit choice on the final grid directly (Gemma needed dev2 + dev3 to find it; amendment 11)."""
    m = meta()["units"]
    rows = base_rows("dev")
    for typ in ("neuron", "sae", "community"):
        for t in ALL_TRAITS:
            for rank in range(3):
                uid = f"L{LSTAR}-{typ}-{t}-{rank}"
                if uid in m:
                    rows += unit_rows("dev", uid, t, FINAL_GRID)
            if f"L{LSTAR}-rand_{typ}-{t}-0" in m:
                rows += unit_rows("dev", f"L{LSTAR}-rand_{typ}-{t}-0", t, FINAL_GRID, kind="control")
    for t in ALL_TRAITS:
        rows += unit_rows("dev", f"L{LSTAR}-direction-{t}", t, FINAL_GRID)
        rows += unit_rows("dev", f"L{LSTAR}-rand_direction-{t}", t, FINAL_GRID, kind="control")
    write_jobs("dev2", rows)


def jobs_dev4() -> None:
    """Circuit units and their controls on the final grid; direction singles and all 28 direction pairs at their
    operating strengths (judged on every aspect), to choose the 4 circuit pairs on dev."""
    sel = select()
    rows = base_rows("dev")
    for t in ALL_TRAITS:
        rows += unit_rows("dev", f"circuit-{t}", t, FINAL_GRID)
        rows += unit_rows("dev", f"rand_circuit-{t}", t, FINAL_GRID, kind="control")
    for i, a in enumerate(ALL_TRAITS):
        sa = sel[f"direction-{a}"]
        rows += unit_rows("dev", sa["uid"], a, [sa["s"]], kind="single_full")
        for b in ALL_TRAITS[i + 1 :]:
            sb = sel[f"direction-{b}"]
            rows += pair_rows("dev", sa["uid"], a, sa["s"], sb["uid"], b, sb["s"])
    write_jobs("dev4", rows)


def choose_circuit_pairs() -> list[list[str]]:
    """Pre-registered: from dev4's direction pairs, the most and least additive (eligible, by E) and the two with
    the largest third-trait rise (d_AB[c] - d_A[c] - d_B[c], any c outside the pair), all distinct."""
    a = json.loads((va.RESULTS / "analysis_dev4.json").read_text())
    ps = [p for p in a["pairs"] if p["type"] == "direction"]
    el = sorted((p for p in ps if p["eligible"]), key=lambda p: p["E"])
    chosen = [el[0]["traits"], el[-1]["traits"]]

    def rise(p):
        ia, ib = (TRAITS_IDX[t] for t in p["traits"])
        return max(p["d_ab"][c] - p["d_a"][c] - p["d_b"][c] for c in range(len(ALL_TRAITS)) if c not in (ia, ib))

    for p in sorted(ps, key=rise, reverse=True):
        if p["traits"] not in chosen and len(chosen) < 4:
            chosen.append(p["traits"])
    (va.RESULTS / "circuit_pairs.json").write_text(json.dumps(chosen))
    return chosen


TRAITS_IDX = {t: i for i, t in enumerate(ALL_TRAITS)}
RAND_PAIRS = [(ALL_TRAITS[i], ALL_TRAITS[(i + 1) % 8]) for i in range(7)]
"""Seven random-unit pairs per type: neighbouring traits' controls, at the real units' operating strengths."""


def confirm_rows(pset: str, strengths_all: bool) -> list[dict]:
    sel = json.loads((va.RESULTS / "selection.json").read_text())
    cpairs = json.loads((va.RESULTS / "circuit_pairs.json").read_text())
    pairs28 = [(a, b) for i, a in enumerate(ALL_TRAITS) for b in ALL_TRAITS[i + 1 :]]
    rows = base_rows(pset) + prompt_rows(pset, ALL_TRAITS, pairs28)
    for typ in (*TYPES, "circuit"):
        for t in ALL_TRAITS:
            s = sel[f"{typ}-{t}"]
            grid = FINAL_GRID if strengths_all else [s["s"]]
            rows += unit_rows(pset, s["uid"], t, grid)
            rows += unit_rows(pset, s["control"], t, grid, kind="control")
        plist = [tuple(p) for p in cpairs] if typ == "circuit" else pairs28
        for a, b in plist:
            sa, sb = sel[f"{typ}-{a}"], sel[f"{typ}-{b}"]
            rows += pair_rows(pset, sa["uid"], a, sa["s"], sb["uid"], b, sb["s"])
        if typ != "circuit":
            for a, b in RAND_PAIRS:
                sa, sb = sel[f"{typ}-{a}"], sel[f"{typ}-{b}"]
                rows += pair_rows(pset, sa["control"], a, sa["s"], sb["control"], b, sb["s"], kind="control_pair")
                for c, tt in ((sa, a), (sb, b)):
                    for r in unit_rows(pset, c["control"], tt, [c["s"]], kind="control_full"):
                        r["rid"] += "-full"
                        rows.append(r)
    return rows


def jobs_pilot() -> None:
    write_jobs("pilot", confirm_rows("pilot", strengths_all=False))


def jobs_heldout() -> None:
    write_jobs("heldout", confirm_rows("heldout", strengths_all=True))


# --- scoring --------------------------------------------------------------------------------------------------------
def aspects_for(row: dict, full: bool) -> list[str]:
    """Full profile (every trait, coherence, honesty) except dev-round singles and random controls of singles, which
    get their target trait and coherence only."""
    short = row["kind"] == "control" or (row["set"] == "dev" and row["kind"] == "single")
    if full:
        return judge.ASPECTS
    if short:
        return [*dict.fromkeys(row["traits"]), "coherence"]
    middle = row["set"] == "heldout" and row["kind"] == "single" and row["s"][0] not in (FINAL_GRID[0], FINAL_GRID[-1])
    if middle and (row["uids"][0], row["s"][0]) not in _op_pairs():  # amendment 7: middle strengths, target only
        return [row["traits"][0], "coherence", "honesty"]
    return judge.ASPECTS


def _op_pairs() -> set:
    sel = json.loads((va.RESULTS / "selection.json").read_text())
    return {(v["uid"], v["s"]) for v in sel.values()}


def score(name: str, full: bool, judge_name: str = va.JUDGE) -> None:
    scored = va.RESULTS / f"scored_{name}.jsonl"
    src = scored if scored.exists() else va.RESULTS / f"gen_{name}.jsonl"  # keep other judges' scores
    rows = [json.loads(line) for line in src.read_text().splitlines()]
    for r in rows:
        r["repeat"] = round(va.repeat_share(r["text"]), 4)
        r["loops"] = va.loops(r["text"])
    judge.run_scores(judge_name, rows, lambda r: aspects_for(r, full))
    out = va.RESULTS / f"scored_{name}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {out.name}: {len(rows)} rows; API judge spend ${va.spent('judge-haiku') + va.spent('judge-gpt'):.3f}")


def load_scored(name: str) -> list[dict]:
    return [json.loads(line) for line in (va.RESULTS / f"scored_{name}.jsonl").read_text().splitlines()]


def baseline(rows: list[dict], judge_name: str = va.JUDGE) -> dict:
    """Unsteered score per (prompt index, aspect)."""
    return {(r["pi"], a): v for r in rows if r["kind"] == "unsteered" for a, v in r["scores"][judge_name].items()}


def summary_table(rows: list[dict]) -> dict:
    base = baseline(rows)
    groups = defaultdict(list)
    for r in rows:
        if r["kind"] in ("single", "single_full", "control", "prompt"):
            groups[(r["uids"][0], tuple(r["s"]))].append(r)
    out = {}
    for (uid, s), rs in groups.items():
        t = rs[0]["traits"][0]
        sc = [r["scores"][va.JUDGE] for r in rs]
        gains = [x[t] - base[(r["pi"], t)] for x, r in zip(sc, rs, strict=True) if x.get(t) is not None]
        coh = [x["coherence"] for x in sc if x.get("coherence") is not None]
        out[(uid, s)] = {
            "trait": t,
            "gain": st.mean(gains),
            "coh": st.mean(coh),
            "loop": sum(r["loops"] for r in rs) / len(rs),
            "n": len(rs),
        }
    return out


def summary(name: str) -> None:
    tab = summary_table(load_scored(name))
    for (uid, s), v in sorted(tab.items()):
        print(f"{uid:34s} s={s!s:8s} {v['trait']:9s} gain {v['gain']:+5.2f} coh {v['coh']:4.1f} loop {v['loop']:.2f}")


if __name__ == "__main__":
    cmd, name = sys.argv[1], sys.argv[2]
    if cmd == "jobs":
        globals()[f"jobs_{name}"]()
    elif cmd == "score":
        score(name, "--all" in sys.argv, va.JUDGE)
    elif cmd == "summary":
        summary(name)
    elif cmd == "lstar":
        choose_lstar()
    elif cmd == "select":
        for k, v in select().items():
            print(f"{k:22s} {v['uid']:30s} s_op {v['s']}  dev gain {v['dev_gain']:+.2f}")
        if (va.RESULTS / "analysis_dev4.json").exists():
            print("circuit pairs", choose_circuit_pairs())
