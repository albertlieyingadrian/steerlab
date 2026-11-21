"""Sounds Like Me, Episode 2: a harness on an unverifiable reward (no training), the steering ladder, persona tests.

Plan: the pre-registered plan. Episode 1's data, authors, briefs and rulers are reused frozen
from ../sounds-like-me (imported, not copied). Every GPU call goes to the `slm-ep2` Modal app and is charged to the
series ledger (../sounds-like-me/data/spend.jsonl); a call that would pass the $900 Modal cap raises instead.

    python ep2.py direction                  # assistant direction from dev authors -> data/assistant_dir.npz
    python ep2.py profiles SPLIT [N]         # verified style profiles + character cards -> data/profiles_SPLIT.json
    python ep2.py arm NAME SPLIT [N]         # one ladder rung / ablation through the harness -> data/drafts, data/pools
    python ep2.py persona SPLIT [N]          # persona tests (b) and (c), k0 prompt -> data/drafts/persona__SPLIT.jsonl
    python ep2.py selfcheck                  # in-loop honesty self-check vs Episode 1's 60 labels
    python ep2.py rerank NAME SPLIT LAM      # re-pick a pool with another lambda (dev tuning; no generation)

Scoring, rulers and analysis live in score_ep2.py.
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EP1 = HERE.parent / "sounds-like-me"
sys.path.append(str(EP1))  # appended: local modules (make_figures, export_pages) win over Episode 1 names

from common import authors, briefs, record_spend, spent, strip_greeting_signoff, texts
from gen import SYSTEM, _tfidf_rank, brief_block, chat_messages, jobs
from honesty import PROMPT as HONESTY_PROMPT
from honesty import parse as _parse
from honesty import rule_flags
from modal_gen import MODAL_CAP, charge
from score import longest_shared_run, ngram_copy_share, words

DATA = HERE / "data"
RES = HERE / "results"
EP1_DATA = EP1 / "data"
LAYER = 24
STRENGTH = 0.4  # x mean token norm at LAYER: Episode 1's frozen Part C choice
N_DRAFT, N_REVISE, N_BON = 16, 4, 20  # dev tuning: N 16 at T 0.9 beat 8 at 0.8; BoN keeps equal compute (16 + 4)
LAM = 0.0  # detector weight in the rerank; tuned on dev (results.tsv: 0 beat 0.5, 1, 2, 4)
CHUNK = 240


# ---------------------------------------------------------------- shared helpers
def strict(t):
    return " ".join(strip_greeting_signoff(t).split())


def loops(t, n=5, k=3):
    w = t.lower().split()
    grams = Counter(tuple(w[i : i + n]) for i in range(len(w) - n + 1))
    return bool(grams) and max(grams.values()) >= k


def cut_words(t, n=80):
    return " ".join(strict(t).split()[:n])


def guard_modal(est_usd):
    if spent("modal") + est_usd > MODAL_CAP:
        raise SystemExit(f"Modal series cap ${MODAL_CAP} would be passed: stop and ask the user.")


def _fn(name):
    import modal

    return modal.Function.from_name("slm-ep2", name)


def run_gen(jobs_, vectors, note, temperature=0.8, nll=True):
    """Steerable HF generation, split into parallel chunks; charged per container second."""
    guard_modal(len(jobs_) * 0.004)
    chunks = [jobs_[i : i + CHUNK] for i in range(0, len(jobs_), CHUNK)]
    payloads = []
    for c in chunks:
        vids = {j["vid"] for j in c if j.get("vid")}
        payloads.append(
            dict(jobs=c, vectors={v: vectors[v] for v in vids}, temperature=temperature, max_new_tokens=320, nll=nll)
        )
    outs, nlls, secs = [], [], 0.0
    t0 = time.time()
    for r in _fn("gen").map(payloads):
        outs += r["outputs"]
        nlls += r["nll"]
        secs += r["seconds"] + 60  # + model load per container
    record_spend("modal", charge(secs), f"ep2 gen {note} n={len(jobs_)} ({len(chunks)} chunks)")
    print(f"  gen {note}: {len(jobs_)} in {time.time() - t0:.0f}s wall, {secs:.0f} GPU-s")
    return outs, nlls, secs


def run_vllm(prompts, note, model="Qwen/Qwen3-8B", max_tokens=400, temperature=0.0, thinking=False):
    guard_modal(len(prompts) * 0.001 + 0.2)
    t0 = time.time()
    payload = dict(model=model, prompts=prompts, max_tokens=max_tokens, temperature=temperature, thinking=thinking)
    r = _fn("vgen").remote(payload)
    record_spend("modal", charge(time.time() - t0), f"ep2 vgen {note} n={len(prompts)}")
    return r["outputs"], time.time() - t0


def run_acts(text_sets, note):
    guard_modal(1.0)
    t0 = time.time()
    r = _fn("acts").remote(text_sets, [LAYER])
    record_spend("modal", charge(time.time() - t0), f"ep2 acts {note}")
    return r


def A_by_name():
    return {a["author"]: a for a in authors()["authors"]}


def B_by_id():
    return {b["id"]: b for b in briefs()}


# ---------------------------------------------------------------- LUAR on single texts (in-loop voice check)
def luar_single(txts, bs=32):
    from common import _luar

    tok, model, torch = _luar()
    out = []
    for i in range(0, len(txts), bs):
        part = [strict(t) or "." for t in txts[i : i + bs]]
        enc = tok(part, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
        enc = {k: v.reshape(len(part), 1, -1) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            e = model(**enc)
        e = e[0] if isinstance(e, tuple) else e
        out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 512))


_SAMPLE_EMB = {}


def sample_embedding(rec):
    """LUAR embedding of the author's 12 samples (train split) as one episode. The scorer never uses these texts."""
    from common import luar_embed

    au = rec["author"]
    if au not in _SAMPLE_EMB:
        _SAMPLE_EMB[au] = luar_embed([[strict(t) for t in texts(rec, "samples")]], max_len=256)[0]
    return _SAMPLE_EMB[au]


# ---------------------------------------------------------------- vectors
def vectors_for(names):
    """Episode 1's Part C voice vectors + this episode's assistant direction + norm-matched randoms (Part C's seed)."""
    z = np.load(EP1_DATA / "partc_vectors.npz", allow_pickle=True)
    V, N = z["vecs"].item(), z["norms"].item()
    ad = np.load(DATA / "assistant_dir.npz")["a"]
    away = -ad
    out = {"away": away.tolist(), "toward": ad.tolist()}
    rng = np.random.default_rng(20261120)
    shape = V["voice"][names[0]][LAYER].shape
    rands = [rng.standard_normal(shape).astype(np.float32) for _ in range(3)]
    for i, r in enumerate(rands):
        out[f"rand{i}"] = r.tolist()
    for au in names:
        own = V["voice"][au][LAYER]
        out[f"own:{au}"] = own.tolist()
        both = own / np.linalg.norm(own) + away / np.linalg.norm(away)
        out[f"both:{au}"] = both.tolist()
    return out, float(N[LAYER])


def vid_for(steer, au):
    if steer is None:
        return None
    if steer in ("own", "both"):
        return f"{steer}:{au}"
    return steer


# ---------------------------------------------------------------- assistant direction (dev only)
def direction():
    A = authors()
    dev = [a for a in A["authors"] if a["split"] == "dev"]
    J = jobs("dev", None, 4, ["k0"])
    msgs = [
        dict(messages=chat_messages(j["rec"], j["brief"], "k0", template="v3"), vid=None, layer=LAYER, coef=0.0)
        for j in J
    ]
    outs, _, _ = run_gen(msgs, {}, "dev k0 drafts for the assistant direction", temperature=0.7, nll=False)
    default = [o for o in outs if o and o.strip()]
    human = [t for a in dev for t in texts(a, "samples")]
    r = run_acts({"default": default, "human": human}, "assistant direction")
    d = np.array(r["vectors"]["default"][str(LAYER)], np.float32)
    h = np.array(r["vectors"]["human"][str(LAYER)], np.float32)
    DATA.mkdir(exist_ok=True)
    np.savez(DATA / "assistant_dir.npz", a=d - h, default=d, human=h, norm=r["norms"][str(LAYER)])
    (DATA / "dev_k0_default.json").write_text(json.dumps(default, indent=1))
    z = np.load(EP1_DATA / "partc_vectors.npz", allow_pickle=True)
    V = z["vecs"].item()
    away = h - d

    def c(v):
        return float(np.dot(v, away) / np.linalg.norm(v) / np.linalg.norm(away))

    cs = [c(V["voice"][a["author"]][LAYER]) for a in A["authors"]]
    info = dict(
        n_default=len(default),
        n_human=len(human),
        norm_assistant_dir=float(np.linalg.norm(d - h)),
        mean_token_norm=r["norms"][str(LAYER)],
        cos_own_vs_away=dict(mean=round(float(np.mean(cs)), 4), min=round(min(cs), 4), max=round(max(cs), 4)),
        cos_ep1_slop_vs_assistant=round(c(-z["vecs"].item()["slop"][LAYER]), 4),
    )
    RES.mkdir(exist_ok=True)
    (RES / "assistant_direction.json").write_text(json.dumps(info, indent=1))
    print(info)


# ---------------------------------------------------------------- style profile (PROSE-style: infer, then verify)
GREET = re.compile(r"^(hi|hello|hey|dear|good morning|[A-Z][a-z]+,)", re.IGNORECASE)


def measured_profile(samples):
    lens = [len(t.split()) for t in samples]
    first = [t.strip().splitlines()[0].strip() if t.strip() else "" for t in samples]
    last = [t.strip().splitlines()[-1].strip() if t.strip() else "" for t in samples]
    greet = sum(bool(GREET.match(f)) and len(f.split()) <= 4 for f in first)
    short_last = [x for x in last if 0 < len(x.split()) <= 3]
    sents = [s for t in samples for s in re.split(r"(?<=[.!?])\s+", " ".join(t.split())) if s.strip()]
    sl = np.median([len(s.split()) for s in sents]) if sents else 0
    joined = " ".join(samples)
    nw = max(1, len(joined.split()))
    alpha = [s for s in sents if s[:1].isalpha()]
    lower_start = float(np.mean([s[:1].islower() for s in alpha])) if alpha else 0.0
    signoff = f' (for example "{Counter(short_last).most_common(1)[0][0]}")' if short_last else ""
    lines = [
        f"- Length: usually about {int(np.median(lens))} words (range {min(lens)} to {max(lens)}).",
        f"- Opens with a short greeting line in {greet} of {len(samples)} emails.",
        f"- Ends with a short sign-off line in {len(short_last)} of {len(samples)} emails{signoff}.",
        f"- Typical sentence: about {int(sl)} words.",
        (
            f"- Exclamation marks: {joined.count('!') * 100 / nw:.1f} per 100 words; "
            f"questions: {joined.count('?') * 100 / nw:.1f} per 100 words."
        ),
    ]
    if lower_start > 0.2:
        lines.append(f"- Often starts sentences in lower case ({lower_start:.0%} of sentences).")
    return "\n".join(lines), int(np.median(lens))


INFER = (
    "These are emails one person sent:\n\n{samples}\n\n---\n\nList up to 6 specific, checkable habits of HOW this person "
    "writes (not what they write about): word choice, tone, formality, punctuation, structure, what they leave out. "
    "No names. One habit per line, each starting with '- '."
)
VERIFY = (
    "These are emails one person sent:\n\n{samples}\n\n---\n\nMeasured facts about these emails (counted, always "
    "true):\n{measured}\n\nClaimed habits:\n{bullets}\n\nFor each claimed habit, check it against the emails above. "
    "Drop it if it contradicts a measured fact, if it is vague (true of almost anyone), or if it does not clearly hold "
    'in most of the emails. Answer with JSON only: {{"keep": [numbers of the habits to keep]}}'
)
PROFILE_VERSION = "v2"  # v1 verifier kept 84 of 90 habits on dev, including ones that contradicted the measured facts
CARD = (
    "These are emails one person sent:\n\n{samples}\n\n---\n\nWrite a short character card for this person, to be used "
    "as a system prompt for someone writing as them. Start with 'You are'. Describe their likely job and workplace "
    "(guess from the emails), their register and mood, and how they write (openings, sentence habits, sign-off). No real "
    "names. At most 90 words. Output only the card."
)


def profiles(split, n=None):
    recs = [a for a in authors()["authors"] if a["split"] == split][: n or None]
    path = DATA / f"profiles_{split}{'' if PROFILE_VERSION == 'v1' else '_' + PROFILE_VERSION}.json"
    have = json.loads(path.read_text()) if path.exists() else {}
    todo = [r for r in recs if r["author"] not in have]
    if not todo:
        return have
    joined = {r["author"]: "\n\n---\n\n".join(texts(r, "samples")[:8]) for r in todo}
    p1 = [[{"role": "user", "content": INFER.format(samples=joined[r["author"]])}] for r in todo]
    p3 = [[{"role": "user", "content": CARD.format(samples=joined[r["author"]])}] for r in todo]
    outs, _ = run_vllm(p1 + p3, f"profiles+cards {split} n={len(todo)}", max_tokens=300)
    inferred, cards = outs[: len(todo)], outs[len(todo) :]
    bullets = {}
    for r, t in zip(todo, inferred):
        bullets[r["author"]] = [ln.strip()[2:].strip() for ln in t.splitlines() if ln.strip().startswith(("- ", "* "))][
            :6
        ]
    p2 = []
    for r in todo:
        numbered = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(bullets[r["author"]]))
        meas = measured_profile(texts(r, "samples"))[0]
        content = VERIFY.format(samples=joined[r["author"]], measured=meas, bullets=numbered)
        p2.append([{"role": "user", "content": content}])
    vouts, _ = run_vllm(p2, f"profile verify {split} n={len(todo)}", max_tokens=80)
    for r, v, card in zip(todo, vouts, cards):
        m = re.search(r"\{.*\}", v, re.DOTALL)
        try:
            keep = [int(k) for k in json.loads(m.group(0)).get("keep", [])] if m else []
        except (json.JSONDecodeError, ValueError, TypeError):
            keep = []
        bs = bullets[r["author"]]
        kept = [bs[k - 1] for k in keep if 1 <= k <= len(bs)]
        meas, med = measured_profile(texts(r, "samples"))
        have[r["author"]] = dict(
            measured=meas,
            median_words=med,
            inferred=bs,
            kept=kept,
            profile=meas + ("\n" + "\n".join(f"- {b}" for b in kept) if kept else ""),
            card=card.strip(),
        )
    DATA.mkdir(exist_ok=True)
    path.write_text(json.dumps(have, indent=1))
    print(
        f"profiles {split}: {len(todo)} new; kept {sum(len(v['kept']) for v in have.values())} of "
        f"{sum(len(v['inferred']) for v in have.values())} inferred habits"
    )
    return have


# ---------------------------------------------------------------- harness
SYSTEM_STYLE = (
    "You are ghostwriting an email for a specific person. Write it exactly the way they write their own emails: their "
    "length, greeting, sign-off, punctuation, capitalisation, abbreviations and bluntness, even when that is informal "
    "or rough. Do not write like a salesperson or an assistant: no pleasantries they wouldn't use, no polished "
    "marketing phrases. Use only the facts in the brief: do not invent results, clients, numbers, or any past contact. "
    "Plain text, 60 to 200 words. Output only the email."
)
ARMS = {
    # name: retrieve (tfidf|random|k8), profile, n_draft, n_revise, steer, temperature
    "R1b_bon": dict(retrieve="k8", profile=False, n_draft=N_BON, n_revise=0, steer=None, temp=0.7),
    "R2_harness": dict(retrieve="tfidf", profile=True, n_draft=N_DRAFT, n_revise=N_REVISE, steer=None, temp=0.9),
    "R3_voice": dict(retrieve="tfidf", profile=True, n_draft=N_DRAFT, n_revise=N_REVISE, steer="own", temp=0.9),
    "R4_persona": dict(retrieve="tfidf", profile=True, n_draft=N_DRAFT, n_revise=N_REVISE, steer="away", temp=0.9),
    "R5_both": dict(retrieve="tfidf", profile=True, n_draft=N_DRAFT, n_revise=N_REVISE, steer="both", temp=0.9),
    "R6_rand0": dict(retrieve="tfidf", profile=True, n_draft=N_DRAFT, n_revise=N_REVISE, steer="rand0", temp=0.9),
    "R6_rand1": dict(retrieve="tfidf", profile=True, n_draft=N_DRAFT, n_revise=N_REVISE, steer="rand1", temp=0.9),
    "abl_noretrieval": dict(retrieve="random", profile=True, n_draft=N_DRAFT, n_revise=N_REVISE, steer=None, temp=0.9),
    "abl_noprofile": dict(retrieve="tfidf", profile=False, n_draft=N_DRAFT, n_revise=N_REVISE, steer=None, temp=0.9),
}


def harness_messages(rec, b, cfg, prof, seed=0):
    import random

    pool = texts(rec, "samples")
    if cfg["retrieve"] == "k8":
        return chat_messages(rec, b, "k8_rand", seed=seed, template="v3")
    if cfg["retrieve"] == "tfidf":
        picked = [pool[i] for i in _tfidf_rank(brief_block(b), pool)[:6]]
    else:
        picked = random.Random(f"{seed}-{rec['author']}-{b['id']}-norag").sample(pool, 6)
    joined = "\n\n---\n\n".join(picked)
    user = f"These are emails this person sent:\n\n{joined}\n\n---\n\n"
    if cfg["profile"]:
        user += f"How this person writes (checked against their emails):\n{prof['profile']}\n\n---\n\n"
    user += "Write the next email they send. Same person, same habits, new topic:\n\n" + brief_block(b)
    system = SYSTEM_STYLE if cfg.get("sys") == "style" else SYSTEM
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "with",
    "we",
    "our",
    "you",
    "your",
    "i",
    "is",
    "are",
    "be",
    "can",
    "will",
    "this",
    "that",
    "it",
    "as",
    "by",
    "from",
    "us",
    "me",
    "my",
    "their",
    "they",
    "them",
    "just",
    "have",
    "has",
    "was",
    "were",
}
SELFCHECK_OVERLAP = 0.6  # a flagged quote whose content words are >= 60% in the brief is dropped (tuned on dev labels)


def parse(out):
    """Episode 1's parser, tolerant of claims given as bare strings (the 8B checker sometimes writes them); identical
    to Episode 1's on well-formed output. A bare-string claim counts as an unsupported quote."""
    try:
        return _parse(out)
    except AttributeError:
        m = re.search(r"\{.*\}", out, re.DOTALL)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        claims = d.get("claims") or []
        claims = [c for c in claims if isinstance(c, (str, dict))]
        uns = [
            c if isinstance(c, str) else c.get("quote", "")
            for c in claims
            if isinstance(c, str) or c.get("supported") is False
        ]
        return dict(
            n_claims=len(claims),
            n_unsupported=len(uns),
            unsupported=uns[:6],
            invented_relationship=bool(d.get("invented_relationship")),
            parsed=bool(d),
        )


def _cw(s):
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP and len(w) > 2}


def allowed_words(b):
    p, snd = b["prospect"], b["sender"]
    parts = [*b["facts"], b["trigger"], b["offer"], b["cta"], *map(str, p.values()), snd["company"], snd["role"]]
    return _cw(" ".join(parts))


def selfcheck_verdict(out, text, b):
    """Policy-model check (thinking on) + overlap filter + Episode 1's rule layer -> (unsupported quotes, relationship)."""
    v = parse(out)
    al = allowed_words(b)
    keep = [q for q in v["unsupported"] if _cw(q) and len(_cw(q) & al) / len(_cw(q)) < SELFCHECK_OVERLAP]
    fl, rel = rule_flags(text, b)
    return (keep + fl)[:6], bool(v["invented_relationship"] or rel)


def honesty_prompts(rows, B):
    out = []
    for r in rows:
        b = B[r["brief"]]
        facts = "\n".join(f"- {f}" for f in b["facts"])
        p = b["prospect"]
        recipient = (
            f"- {p['first_name']} is the {p['role']} at {p['company']} ({p['location']}).\n"
            f"- Why now (about the recipient): {b['trigger']}."
        )
        content = HONESTY_PROMPT.format(facts=facts, recipient=recipient, email=r["text"])
        out.append([{"role": "user", "content": content}])
    return out


def in_loop_checks(cands, A, B, note):
    """Voice (LUAR to the 12 samples), detector, copy filter, honesty self-check (policy model + rule layer)."""
    from detectors import voi380

    txt = [c["text"] or " " for c in cands]
    emb = luar_single(txt)
    det = voi380(txt)
    hout, vsecs = run_vllm(honesty_prompts(cands, B), f"selfcheck {note}", max_tokens=2500, thinking=True)
    for c, e, d, h in zip(cands, emb, det, hout):
        rec = A[c["author"]]
        src = texts(rec, "samples")
        uns, rel = selfcheck_verdict(h, c["text"], B[c["brief"]])
        c.update(
            sim=float(np.dot(e, sample_embedding(rec))),
            det=float(d),
            longest_run=longest_shared_run(c["text"], src),
            copy8=ngram_copy_share(c["text"], src),
            unsupported=uns,
            invented_rel=rel,
            words=len(words(c["text"])),
            loop=loops(c["text"]),
            empty=not c["text"].strip(),
        )
        c["copy_flag"] = c["longest_run"] >= 12 or c["copy8"] >= 0.2
        c["honest_flag"] = bool(c["unsupported"]) or c["invented_rel"]
    return vsecs


def rank_score(c, lam):
    return c["sim"] - lam * c["det"]


def pick(cands, lam=LAM):
    """Drop copied, self-flagged and looping candidates, then take the best rerank score. When no candidate is clean,
    take the best rerank score among those with the fewest flags (code review: the first version took list order)."""
    ok = [c for c in cands if not c["empty"]] or cands[:1]

    def flags(c):
        return int(c.get("copy_flag", 0)) + int(c.get("honest_flag", 0)) + int(c.get("loop", 0))

    fewest = min(flags(c) for c in ok)
    return max([c for c in ok if flags(c) == fewest], key=lambda c: rank_score(c, lam))


def critique(c, prof, use_profile=True):
    notes = []
    if c["unsupported"]:
        q = "; ".join(f'"{u[:90]}"' for u in c["unsupported"][:3])
        notes.append(
            f"Remove claims the facts don't support: {q}. Say nothing about results, clients, numbers or past "
            "contact that isn't in the facts."
        )
    if c["invented_rel"]:
        notes.append(
            "Don't imply any earlier email, call or relationship: this is the first contact unless the facts say "
            "otherwise."
        )
    if c["copy_flag"]:
        notes.append(
            "Some phrases are copied word for word from their old emails. Keep their habits, but use new words."
        )
    if c["det"] > 0.5:
        notes.append("It reads like an AI assistant wrote it: drop polished, salesy phrasing and filler pleasantries.")
    med = prof["median_words"] if use_profile else None  # the no-profile ablation gets no length note (code review)
    if med and (c["words"] > 1.4 * med or c["words"] < 0.6 * med):
        notes.append(f"They usually write about {med} words; this draft has {c['words']}.")
    if c["loop"]:
        notes.append("It repeats itself. Say each thing once.")
    notes.append(
        "Compare with their emails once more: match how they open, how long their sentences are, their punctuation "
        "and how they sign off."
    )
    return "\n".join(f"- {n}" for n in notes)


def revise_messages(base_msgs, draft, notes):
    user = base_msgs[1]["content"] + (
        f"\n\n---\n\nA first draft:\n<<<\n{draft}\n>>>\n\nNotes on the first draft:\n{notes}\n\n"
        "Rewrite the email so it fixes every note and sounds like this person. Output only the email."
    )
    return [base_msgs[0], {"role": "user", "content": user}]


def arm(name, split, n=None, lam=LAM, over=None, tag=None):
    cfg = dict(ARMS[name], **(over or {}))
    A, B = A_by_name(), B_by_id()
    J = jobs(split, n, 4, ["k0"])
    names = sorted({j["rec"]["author"] for j in J})
    P = profiles(split, n)
    vecs, norm = vectors_for(names)
    coef = cfg.get("strength", STRENGTH) * norm  # exploratory arms may set another strength (dev-chosen)
    t_all = time.time()
    gsecs = 0.0
    spec, meta = [], []
    for j in J:
        au, b = j["rec"]["author"], j["brief"]
        msgs = harness_messages(j["rec"], b, cfg, P.get(au))
        vid = vid_for(cfg["steer"], au)
        for k in range(cfg["n_draft"]):
            spec.append(dict(messages=msgs, vid=vid, layer=LAYER, coef=coef if vid else 0.0))
            meta.append(dict(job=j["job"].rsplit("|", 1)[0], author=au, brief=b["id"], stage="draft", k=k))
    outs, nll, s = run_gen(spec, vecs, f"{name} {split} drafts", temperature=cfg["temp"])
    gsecs += s
    cands = [dict(m, text=o or "", nll=x) for m, o, x in zip(meta, outs, nll)]
    gsecs += in_loop_checks(cands, A, B, f"{name} {split} drafts")
    if cfg["n_revise"]:
        by_job = {}
        for c in cands:
            by_job.setdefault(c["job"], []).append(c)
        spec2, meta2 = [], []
        for job, cs in by_job.items():
            au, bid = cs[0]["author"], cs[0]["brief"]
            msgs = harness_messages(A[au], B[bid], cfg, P.get(au))
            vid = vid_for(cfg["steer"], au)
            top = sorted([c for c in cs if not c["empty"]], key=lambda c: -rank_score(c, lam))[: cfg["n_revise"]]
            for c in top:
                notes = critique(c, P[au], use_profile=cfg["profile"])
                spec2.append(
                    dict(
                        messages=revise_messages(msgs, c["text"], notes),
                        vid=vid,
                        layer=LAYER,
                        coef=coef if vid else 0.0,
                    )
                )
                meta2.append(dict(job=job, author=au, brief=bid, stage="revise", k=c["k"], notes=notes))
        outs2, nll2, s2 = run_gen(spec2, vecs, f"{name} {split} revisions", temperature=0.7)
        gsecs += s2
        revs = [dict(m, text=o or "", nll=x) for m, o, x in zip(meta2, outs2, nll2)]
        gsecs += in_loop_checks(revs, A, B, f"{name} {split} revisions")
        cands += revs
    write_pool_and_pick(tag or name, split, cands, lam, gsecs, time.time() - t_all)


def write_pool_and_pick(name, split, cands, lam, gsecs=None, wall=None):
    (DATA / "pools").mkdir(parents=True, exist_ok=True)
    (DATA / "drafts").mkdir(parents=True, exist_ok=True)
    if gsecs is not None:
        with (DATA / "pools" / f"{name}__{split}.jsonl").open("w") as fh:
            for c in cands:
                fh.write(json.dumps(c) + "\n")
    by_job = {}
    for c in cands:
        by_job.setdefault(c["job"], []).append(c)
    rows = []
    for job, cs in by_job.items():
        ch = pick(cs, lam)
        rows.append(
            dict(
                job=f"{job}|{name}",
                model="ep2-Qwen3-8B",
                author=ch["author"],
                brief=ch["brief"],
                arm=name,
                text=ch["text"],
                nll=ch["nll"],
                stage=ch["stage"],
                n_pool=len(cs),
            )
        )
    tag = name if lam == LAM else f"{name}_lam{lam}"
    with (DATA / "drafts" / f"{tag}__{split}.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if gsecs is not None:
        RES.mkdir(exist_ok=True)
        meta_path = RES / "arm_costs.json"
        m = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        m[f"{name}__{split}"] = dict(
            gpu_seconds=round(gsecs, 1),
            usd=round(charge(gsecs), 4),
            final_drafts=len(rows),
            generations=len(cands),
            usd_per_draft=round(charge(gsecs) / max(1, len(rows)), 5),
            wall_seconds=round(wall or 0, 1),
        )
        meta_path.write_text(json.dumps(m, indent=1))
    n_rev = sum(r["stage"] == "revise" for r in rows)
    print(f"{tag} {split}: {len(rows)} picked from {len(cands)} candidates; revisions picked {n_rev}")


def rerank(name, split, lam):
    if split == "heldout" and float(lam) != LAM:
        raise SystemExit("a non-default lambda is a dev-tuning tool; held-out picks use the frozen lambda")
    cands = [json.loads(l) for l in (DATA / "pools" / f"{name}__{split}.jsonl").read_text().splitlines() if l.strip()]
    write_pool_and_pick(name, split, cands, float(lam))


# ---------------------------------------------------------------- persona tests (b) and (c): k0 prompt as Part C
def persona(split, n=None):
    J = jobs(split, n, 4, ["k0"])
    names = sorted({j["rec"]["author"] for j in J})
    P = profiles(split, n)
    vecs, norm = vectors_for(names)
    coef = STRENGTH * norm
    conds = [("none", None, False), ("away", "away", False), ("toward", "toward", False)]
    conds += [(f"rand{i}", f"rand{i}", False) for i in range(3)]
    conds += [("card", None, True), ("own", "own", False), ("card_own", "own", True)]
    spec, meta = [], []
    for j in J:
        au = j["rec"]["author"]
        base = chat_messages(j["rec"], j["brief"], "k0", template="v3")
        for cname, steer, card in conds:
            msgs = base
            if card:
                msgs = [
                    {"role": "system", "content": SYSTEM + "\n\nWho you are writing as:\n" + P[au]["card"]},
                    base[1],
                ]
            vid = vid_for(steer, au)
            spec.append(dict(messages=msgs, vid=vid, layer=LAYER, coef=coef if vid else 0.0))
            meta.append((j, cname))
    outs, nll, s = run_gen(spec, vecs, f"persona {split}", temperature=0.7)
    (DATA / "drafts").mkdir(parents=True, exist_ok=True)
    with (DATA / "drafts" / f"persona__{split}.jsonl").open("w") as fh:
        for (j, cname), t, x in zip(meta, outs, nll):
            row = dict(
                job=f"{j['job']}|{cname}",
                model="persona-Qwen3-8B",
                author=j["rec"]["author"],
                brief=j["brief"]["id"],
                arm=cname,
                text=t or "",
                nll=x,
            )
            fh.write(json.dumps(row) + "\n")
    RES.mkdir(exist_ok=True)
    m = json.loads((RES / "arm_costs.json").read_text()) if (RES / "arm_costs.json").exists() else {}
    m[f"persona__{split}"] = dict(
        gpu_seconds=round(s, 1),
        usd=round(charge(s), 4),
        final_drafts=len(spec),
        generations=len(spec),
        usd_per_draft=round(charge(s) / len(spec), 5),
    )
    (RES / "arm_costs.json").write_text(json.dumps(m, indent=1))
    print("persona", split, len(spec), "drafts")


# ---------------------------------------------------------------- in-loop honesty self-check vs Episode 1's labels
def selfcheck(which="tune"):
    from sklearn.metrics import f1_score

    B = B_by_id()
    stem = "honesty" if which == "tune" else "honesty_confirm"
    labels = json.loads((EP1_DATA / f"{stem}_labels.json").read_text())
    todo = json.loads((EP1_DATA / f"{stem}_to_label.json").read_text())
    rows = [dict(brief=t["brief"], text=t["text"]) for t in todo]
    outs, _ = run_vllm(honesty_prompts(rows, B), "selfcheck vs 60 labels", max_tokens=2500, thinking=True)
    yhat = []
    for r, o in zip(rows, outs):
        uns, rel = selfcheck_verdict(o, r["text"], B[r["brief"]])
        yhat.append(int(bool(uns) or rel))
    y = [int(labels[t["id"]]["any_unsupported"] or labels[t["id"]]["invented_relationship"]) for t in todo]
    tp = sum(a and b for a, b in zip(y, yhat))
    out = dict(
        checker="Qwen3-8B (policy model, thinking on) + overlap filter 0.6 + Episode 1 rule layer",
        f_score=round(f1_score(y, yhat), 4),
        precision=round(tp / max(1, sum(yhat)), 4),
        recall=round(tp / max(1, sum(y)), 4),
        positives=sum(y),
        flagged=sum(yhat),
        n=len(y),
        note=f"flag = any unsupported claim or invented relationship; Episode 1's {which} labels",
    )
    RES.mkdir(exist_ok=True)
    (RES / f"ruler_selfcheck_{which}.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "direction":
        direction()
    elif cmd == "profiles":
        profiles(a[0], int(a[1]) if len(a) > 1 else None)
    elif cmd == "arm":
        # optional overrides for dev tuning: key=value (n_draft, n_revise, temp, sys) and tag=NAME for the output
        kv = dict(x.split("=", 1) for x in a[3:])
        tag = kv.pop("tag", None)
        over = {k: (v if k == "sys" else (float(v) if k in ("temp", "strength") else int(v))) for k, v in kv.items()}
        arm(a[0], a[1], int(a[2]) if len(a) > 2 else None, over=over, tag=tag)
    elif cmd == "persona":
        persona(a[0], int(a[1]) if len(a) > 1 else None)
    elif cmd == "selfcheck":
        selfcheck(a[0] if a else "tune")
    elif cmd == "rerank":
        rerank(a[0], a[1], a[2])
    else:
        raise SystemExit(__doc__)
