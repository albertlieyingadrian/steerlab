"""Score lens readouts offline with the official vote rule, and search for good visit sets.

Input: readouts files from modal_lens.py ({id, visits: {visit: {lens: [token ids]}}}) and the questions
with answers (dev/data/all_private.jsonl).

Matching is a local stand-in for the judge. `fuzzy`: a token names an option if, after stripping
spaces/punctuation and case, it equals it, adds a short ending to it (plural, "Italian"), or is a
>=4-char prefix of it; a token naming two options names neither. Numbers need an exact match.
`exact`: the official no-key string-match judge.

    python dev/score.py dev/runs/readouts.jsonl [more readouts files...]
"""

import json
import re
import sys
from collections import Counter, defaultdict

from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained("thinkingmachines/Inkling-2-Mini")
CONTROL = re.compile(r"^<\|.*\|>$")
_text_cache = {}


def text(i):
    if i not in _text_cache:
        _text_cache[i] = TOK.decode([i]) or TOK.convert_ids_to_tokens(i)
    return _text_cache[i]


def norm(s):
    return re.sub(r"[\s\.\,\:\;\!\?\"'“”‘’\(\)\[\]\*_«»„]+", "", s).lower()


def names(token, options, mode="fuzzy"):
    t = norm(token)
    if not t:
        return set()
    hit = set()
    for k, o in enumerate(options):
        on = norm(o)
        if t == on:
            hit.add(k)
        elif mode == "fuzzy" and not on.isdigit() and not t.isdigit():
            if t.startswith(on) and len(t) - len(on) <= 3:
                hit.add(k)
            elif len(t) >= 4 and on.startswith(t):
                hit.add(k)
            elif len(on) >= 5 and t.startswith(on[:-1]) and len(t) <= len(on) + 3:
                hit.add(k)
    if len(hit) > 1:
        hit = {k for k in hit if norm(options[k]) == t}
    return hit


def clean(ids, n=10):
    """Top-n token texts, dropping control tokens and case/space repeats (both allowed by the rules)."""
    out, seen = [], set()
    for i in ids:
        s = text(i)
        if CONTROL.match(s.strip()) or not s.strip():
            continue
        key = norm(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) == n:
            break
    return out


def vote(tokens, options, mode="fuzzy"):
    best = {}
    for rank, t in enumerate(tokens):
        for k in names(t, options, mode):
            best.setdefault(k, rank)
    if not best:
        return None
    r = min(best.values())
    top = [k for k, v in best.items() if v == r]
    return top[0] if len(top) == 1 else None


def plurality(votes, gold):
    c = Counter(v for v in votes if v is not None)
    return c[gold] > max((n for o, n in c.items() if o != gold), default=0)


def load(*paths):
    out = {}
    for path in paths:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            m = out.setdefault(r["id"], {})
            for v, lenses in r["visits"].items():
                m.setdefault(int(v), {}).update(lenses)
            if "final" in r:
                m.setdefault(48, {})["final"] = r["final"]
    return out


def questions(path="dev/data/all_private.jsonl"):
    return {q["id"]: q for q in (json.loads(line) for line in open(path) if line.strip())}


def vote_table(runs, qs, mode="fuzzy"):
    """{(lens, visit): {qid: option index or None}}."""
    table = defaultdict(dict)
    for qid, q in qs.items():
        for v, lenses in runs.get(qid, {}).items():
            for lens, ids in lenses.items():
                table[(lens, v)][qid] = vote(clean(ids), q["options"], mode)
    return table


def pass_rate(table, picks, qs, ids=None):
    ids = ids or list(qs)
    ok = sum(plurality([table[p].get(qid) for p in picks], qs[qid]["options"].index(qs[qid]["intermediate"])) for qid in ids)
    return ok / len(ids)


def per_visit(table, qs, ids=None):
    """(lens, visit) -> (gold-vote rate, wrong-vote rate)."""
    ids = ids or list(qs)
    out = {}
    for key, col in table.items():
        g = sum(col.get(q) == qs[q]["options"].index(qs[q]["intermediate"]) for q in ids)
        w = sum(col.get(q) is not None and col.get(q) != qs[q]["options"].index(qs[q]["intermediate"]) for q in ids)
        out[key] = (g / len(ids), w / len(ids))
    return out


def greedy(table, qs, lenses, ids=None, k=10, one_lens=True):
    """Greedy forward selection of up to k (lens, visit) picks, one lens per visit."""
    picks = []
    cands = [key for key in table if key[0] in lenses]
    for _ in range(k):
        used = {v for _, v in picks}
        best, best_r = None, pass_rate(table, picks, qs, ids) if picks else -1
        for c in cands:
            if c[1] in used or c[1] == 48:
                continue
            r = pass_rate(table, picks + [c], qs, ids)
            if r > best_r + 1e-9:
                best, best_r = c, r
        if best is None:
            break
        picks.append(best)
    return picks


def by_family(table, picks, qs):
    fam = defaultdict(list)
    for qid, q in qs.items():
        fam[q["family"]].append(qid)
    return {f: round(pass_rate(table, picks, qs, ids), 3) for f, ids in sorted(fam.items())}


if __name__ == "__main__":
    qs = questions()
    runs = load(*sys.argv[1:])
    qs = {k: v for k, v in qs.items() if k in runs}
    table = vote_table(runs, qs)
    lenses = sorted({k[0] for k in table})
    pv = per_visit(table, qs)
    print("gold/wrong vote rate per visit (fuzzy match):")
    for lens in lenses:
        row = " ".join(f"{v}:{pv[(lens, v)][0]:.2f}/{pv[(lens, v)][1]:.2f}" for v in range(49) if (lens, v) in pv)
        print(f"  {lens:10s} {row}")
    starter = [12, 24, 34, 35, 36, 43, 44, 45, 46, 47]
    for lens in lenses:
        if lens == "final":
            continue
        picks = [(lens, v) for v in starter if (lens, v) in table]
        print(f"{lens:10s} starter visits: {pass_rate(table, picks, qs):.3f}  {by_family(table, picks, qs)}")
