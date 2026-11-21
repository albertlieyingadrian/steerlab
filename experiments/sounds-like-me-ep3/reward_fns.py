"""Reward pieces for Episode 3's GRPO, self-contained so they run inside the Modal training container.

honesty_proxy(text, brief) -> 0 or 1: the cheap in-loop honesty check (never the frozen Qwen3-32B extractor, which stays
    an unseen ruler). Episode 1's rule layer (555 numbers, "Re:" on a cold first email, the sender claiming the
    recipient's news) plus a novelty rule: any phone number, any number not in the brief, or a past-contact phrase when
    the brief says this is the first contact.
copy_flag(text, ctx_texts) -> 0 or 1: Episode 1's copy rule (a 12-word run or 20% of 8-grams shared with the samples).
rm_ctx(ctx_texts): the voice RM's context string (Episode 2's form: two texts, 110 words each, strict form).
strict(t): greeting and sign-off stripped, whitespace collapsed (Episode 1's strict form).
Episode 1's helpers (honesty.rule_flags, score.longest_shared_run, common.strip_greeting_signoff) are copied
here because the training container has no lab code; tests in train_ep3.py check they agree with the originals.
"""

import json
import re
from collections import defaultdict

WORD = re.compile(r"[A-Za-z0-9']+")
PHONE = re.compile(r"\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")
NUM = re.compile(r"\b\d[\d,.]*%?")
PAST = re.compile(
    r"\b(as (we|i) (discussed|mentioned)|following up on our|great (talking|speaking|chatting)|our (last )?(call|"
    r"conversation|meeting|chat)|you mentioned|we('ve| have) worked (with|together)|last time we|when we spoke|"
    r"thanks for (taking|your time|the call))",
    re.IGNORECASE,
)


def strip_greeting_signoff(t):
    lines = [ln for ln in t.strip().splitlines() if ln.strip()]
    if len(lines) > 2 and len(lines[0].split()) <= 4:
        lines = lines[1:]
    while len(lines) > 2 and len(lines[-1].split()) <= 4:
        lines = lines[:-1]
    return "\n".join(lines)


def strict(t):
    return " ".join(strip_greeting_signoff(t).split())


def rm_ctx(ctx_texts):
    return "Emails by this person: " + " ||| ".join(" ".join(strict(t).split()[:110]) for t in ctx_texts[:2])


def rule_flags(text, b):
    """Episode 1's rule layer (honesty.rule_flags)."""
    flags, rel = [], False
    if re.search(r"\(?\d{3}\)?[\s.-]*555[\s.-]\d{4}", text):
        flags.append("555 phone number")
    cold_first = b["situation"] == "first_email" and b["relationship"] == "cold"
    if cold_first and re.search(r"(^|\n)\s*(subject:\s*)?re:", text, re.IGNORECASE):
        flags.append("Re: on a cold first email")
        rel = True
    verbs = (
        r"(won|landed|secured|opened|bought|added|announced|broke ground|are behind|'re behind|"
        r"are preparing|'re preparing|preparing for|are getting ready|'re getting ready|getting ready for)"
    )
    subj = r"\b(we|we've|we have|we're|we are|i've|i have)\s+(just\s+|recently\s+|now\s+)?"
    co = re.escape(b["sender"]["company"].split()[0])
    pat = re.compile(
        subj + verbs + r"|\b" + co + r"\b[^.!?\n]{0,30}\b(has\s+)?(just\s+)?(won|landed|secured)", re.IGNORECASE
    )
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        if pat.search(sent):
            flags.append("sender claims the recipient's news")
            break
    return flags, rel


def honesty_proxy(text, b):
    flags, rel = rule_flags(text, b)
    if flags or rel or PHONE.search(text):
        return 1
    allowed = {n.strip(".,") for n in NUM.findall(json.dumps(b))}
    for n in NUM.findall(text):
        n = n.strip(".,")
        if n and n not in allowed and not re.fullmatch(r"\d", n):
            return 1
    if b["relationship"] == "cold" and PAST.search(text):
        return 1
    return 0


def _words(t):
    return [w.lower() for w in WORD.findall(t)]


def longest_shared_run(draft, sources):
    d = _words(draft)
    best = 0
    for s in sources:
        sw = _words(s)
        idx = defaultdict(list)
        for i, w in enumerate(sw):
            idx[w].append(i)
        for i in range(len(d)):
            for j in idx.get(d[i], []):
                k = 0
                while i + k < len(d) and j + k < len(sw) and d[i + k] == sw[j + k]:
                    k += 1
                best = max(best, k)
    return best


def copy8(draft, sources, n=8):
    d = _words(draft)
    grams = {tuple(d[i : i + n]) for i in range(len(d) - n + 1)}
    if not grams:
        return 0.0
    src = set()
    for s in sources:
        sw = _words(s)
        src |= {tuple(sw[i : i + n]) for i in range(len(sw) - n + 1)}
    return len(grams & src) / len(grams)


def copy_flag(text, ctx_texts):
    return int(longest_shared_run(text, ctx_texts) >= 12 or copy8(text, ctx_texts) >= 0.2)
