"""Score Inkling drafts with Episode 1's rulers, unchanged, and add the checks this add-on reports beside them.

    python score_inkling.py data/drafts/<file>.jsonl [...]      -> data/scored/<file>.json   (private)
    python score_inkling.py --honesty data/drafts/<file>.jsonl  -> data/honesty/<file>.json  (private; Modal)
    python score_inkling.py --ep1-cut80                         -> data/scored/ep1_cut80.json (private)

Per draft (Episode 1's score.py): words, voi380 (AI detector), longest_run and copy8 (copying the writer's samples),
task coverage, empty. Added here: loop (a 5-word sequence 3+ times, Episode 1's robustness.py rule) and cap_hit
(OpenRouter finish_reason "length"). Per episode (one writer's 4 drafts under one arm): strict and loose voice position
(Episode 1's VoiceScale), copy flag, and position80 (strict, every draft cut to its first 80 words). Specificity per
episode (Episode 1's amendment A4): similarity to the writer's profile minus the mean similarity to the other 29
held-out writers' profiles.

--ep1-cut80 recomputes the 80-word position for Episode 1's contestants from their scored drafts (the length check);
floor and ceiling are the ones Episode 1 stored with each episode, so only the episode's embedding changes.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EP1 = HERE.parent / "sounds-like-me"
sys.path.insert(0, str(EP1))

from robustness import loops
from score import longest_shared_run, ngram_copy_share, task_coverage, words

from common import VoiceScale, authors, briefs, cos, luar_embed, strip_greeting_signoff, texts

CUT = 80
SCORED = HERE / "data" / "scored"
EP1_MODELS = (
    "gemini-4-carbon",
    "claude-sonnet-5.5",
    "gemma-3-27b-it",
    "Qwen3-32B",
    "Qwen3-8B",
    "Qwen3-8B-Base",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-8B",
)


def strict(t):
    return " ".join(strip_greeting_signoff(t).split())


def cut80(t):
    return " ".join(strict(t).split()[:CUT])


def complete_rows(path):
    """One row per job (the last with text), and every writer x signal holding all 4 of its briefs; else stop.
    A partial or duplicated drafts file would otherwise change denominators without any error."""
    by_job = {}
    for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip()):
        if r.get("text", "").strip() or r["job"] not in by_job:
            by_job[r["job"]] = r
    rows = list(by_job.values())
    per = defaultdict(int)
    for r in rows:
        per[(r["author"], r["arm"])] += bool(r.get("text", "").strip())
    short = sum(v != 4 for v in per.values())
    if short:
        raise SystemExit(f"{path.name}: {short} writer x signal cells lack 4 drafts with text; rerun generation first")
    return rows


def position(sim, floor, ceiling):
    """Episode 1's voice scale: 0 = another person's writing, 1 = the writer's own held-out writing."""
    return (sim - floor) / max(ceiling - floor, 1e-6)


def heldout_profiles():
    A = [a for a in authors()["authors"] if a["split"] == "heldout"]
    names = [a["author"] for a in A]
    prof = luar_embed([[strict(t) for t in texts(a, "profile")] for a in A])
    return names, prof


def specificity(vec, author, names, prof):
    s = prof @ vec
    i = names.index(author)
    return float(s[i] - np.delete(s, i).mean())


def score_file(path):
    from detectors import voi380

    A = {a["author"]: a for a in authors()["authors"]}
    floor = authors()["floor"]
    B = {b["id"]: b for b in briefs()}
    rows = complete_rows(path)
    det = voi380([r["text"] or " " for r in rows])
    for r, s in zip(rows, det, strict=True):
        src = texts(A[r["author"]], "samples")
        r.update(
            words=len(words(r["text"])),
            voi380=float(s),
            longest_run=longest_shared_run(r["text"], src),
            copy8=ngram_copy_share(r["text"], src),
            task=task_coverage(r["text"], B[r["brief"]]),
            empty=not r["text"].strip(),
            loop=loops(r["text"]),
            cap_hit=r.get("finish") == "length",
        )
    names, prof = heldout_profiles()
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["author"], r["arm"])].append(r)
    scales, episodes = {}, []
    for (m, au, arm), g in sorted(groups.items()):
        if au not in scales:
            scales[au] = (VoiceScale(A[au], floor, "strict"), VoiceScale(A[au], floor, "loose"))
        vs, vl = scales[au]
        ep = [r["text"] for r in g if r["text"].strip()]
        if not ep:
            continue
        e80 = luar_embed([[cut80(t) for t in ep]])[0]
        estrict = luar_embed([[strict(t) for t in ep]])[0]
        episodes.append(
            dict(
                model=m,
                author=au,
                arm=arm,
                n=len(ep),
                position=vs.position(ep),
                sim=vs.sim(ep),
                floor=vs.floor,
                ceiling=vs.ceiling,
                position_loose=vl.position(ep),
                position80=position(cos(e80, vs.profile), vs.floor, vs.ceiling),
                spec=specificity(estrict, au, names, prof) if au in names else None,
                copy_flag=any(r["longest_run"] >= 12 or r["copy8"] >= 0.2 for r in g),
            )
        )
    out = SCORED / (path.stem + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(drafts=rows, episodes=episodes), indent=1))
    print(out.name, len(rows), "drafts,", len(episodes), "episodes")


def _charge_here(provider, usd, note):
    """Episode 1's extractor charges Episode 1's ledger; this add-on's Modal time goes to its own ledger instead."""
    led = HERE / "data" / "spend.jsonl"
    with led.open("a") as fh:
        fh.write(json.dumps(dict(t=time.time(), provider=provider, usd=round(usd, 6), note=note)) + "\n")


def honesty(path):
    import honesty as H

    H.record_spend = _charge_here
    _extract = H._extract
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("text", "").strip()]
    pred = _extract(rows, f"honesty {path.name} n={len(rows)} (inkling add-on)")
    out = HERE / "data" / "honesty" / (path.stem + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({r["job"]: v for r, v in zip(rows, pred, strict=True)}, indent=1))
    share = sum(v["n_unsupported"] > 0 for v in pred) / max(1, len(pred))
    print(out.name, len(pred), "checked; any-unsupported share", round(share, 3))


def ep1_cut80():
    """80-word strict position for Episode 1's contestants, from their (private) scored drafts."""
    A = {a["author"]: a for a in authors()["authors"]}
    profile = {}
    out = {}
    for m in EP1_MODELS:
        d = json.loads((EP1 / "data" / "scored" / f"{m}__heldout.json").read_text())
        eps = {(e["author"], e["arm"]): e for e in d["episodes"] if e["model"] == m}
        dr = defaultdict(list)
        for r in d["drafts"]:
            if r["model"] == m and r["text"].strip():
                dr[(r["author"], r["arm"])].append(r["text"])
        for (au, arm), e in sorted(eps.items()):
            if au not in profile:
                profile[au] = luar_embed([[strict(t) for t in texts(A[au], "profile")]])[0]
            v = luar_embed([[cut80(t) for t in dr[(au, arm)]]])[0]
            out[f"{m}|{au}|{arm}"] = position(cos(v, profile[au]), e["floor"], e["ceiling"])
        print(m, "done", flush=True)
    SCORED.mkdir(parents=True, exist_ok=True)
    (SCORED / "ep1_cut80.json").write_text(json.dumps(out, indent=1))


def addressed():
    """Added after reading drafts: the top-scoring Inkling episode was a terse log entry, not an email. Re-score every
    contestant's strict position on drafts that name the recipient (the brief's first name), keeping a writer for an
    arm only with 2+ such drafts. Floor and ceiling are the stored ones, so only the episode changes."""
    A = {a["author"]: a for a in authors()["authors"]}
    B = {b["id"]: b for b in briefs()}
    files = [(m, EP1 / "data" / "scored" / f"{m}__heldout.json") for m in EP1_MODELS]
    files += [(m, SCORED / f"{m}__heldout.json") for m in ("inkling-2-mini", "inkling-2-mini-reason-low")]
    profile, out = {}, {}
    for m, p in files:
        d = json.loads(p.read_text())
        dr = defaultdict(list)
        for r in d["drafts"]:
            if r["model"] == m and B[r["brief"]]["prospect"]["first_name"].lower() in r["text"].lower():
                dr[(r["author"], r["arm"])].append(r["text"])
        for e in d["episodes"]:
            if e["model"] != m or e["copy_flag"] or len(dr[(e["author"], e["arm"])]) < 2:
                continue
            au = e["author"]
            if au not in profile:
                profile[au] = luar_embed([[strict(t) for t in texts(A[au], "profile")]])[0]
            v = luar_embed([[strict(t) for t in dr[(au, e["arm"])]]])[0]
            out[f"{m}|{au}|{e['arm']}"] = position(cos(v, profile[au]), e["floor"], e["ceiling"])
        print(m, "done", flush=True)
    (SCORED / "addressed.json").write_text(json.dumps(out, indent=1))


def matched_cut():
    """Length check, version two (added after review): every Episode 1 contestant's draft is cut to the word count of
    Inkling's draft for the same job (strict form), so both sides carry the same length; Inkling stays uncut."""
    A = {a["author"]: a for a in authors()["authors"]}
    ink = {
        r["job"]: len(strict(r["text"]).split())
        for r in complete_rows(HERE / "data" / "drafts" / "inkling-2-mini__heldout.jsonl")
    }
    profile, out = {}, {}
    for m in EP1_MODELS:
        d = json.loads((EP1 / "data" / "scored" / f"{m}__heldout.json").read_text())
        dr = defaultdict(list)
        for r in d["drafts"]:
            if r["model"] == m and r["text"].strip() and r["job"] in ink:
                dr[(r["author"], r["arm"])].append(" ".join(strict(r["text"]).split()[: ink[r["job"]]]))
        for e in d["episodes"]:
            if e["model"] != m or e["copy_flag"] or not dr[(e["author"], e["arm"])]:
                continue
            au = e["author"]
            if au not in profile:
                profile[au] = luar_embed([[strict(t) for t in texts(A[au], "profile")]])[0]
            v = luar_embed([dr[(au, e["arm"])]])[0]
            out[f"{m}|{au}|{e['arm']}"] = position(cos(v, profile[au]), e["floor"], e["ceiling"])
        print(m, "done", flush=True)
    (SCORED / "matched_cut.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--matched-cut":
        matched_cut()
    elif args and args[0] == "--addressed":
        addressed()
    elif args and args[0] == "--honesty":
        for p in args[1:]:
            honesty(Path(p))
    elif args and args[0] == "--ep1-cut80":
        ep1_cut80()
    else:
        for p in args:
            score_file(Path(p))
