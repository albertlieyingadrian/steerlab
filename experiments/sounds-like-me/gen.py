"""Build ghostwriting prompts for every (author, brief, arm).

Arms (the signals of research question 2):
  k0        no voice information (do-nothing baseline)
  k3_rand   3 random samples of the author's real writing
  k3_match  3 samples most similar to the brief (situation-matched, TF-IDF over the author's 12-sample pool)
  k8_rand   8 random samples
  note      a written style note about the author (made once per author from 8 samples, by an open model)
  edits     3 samples + 3 simulated (draft -> author-like edit) pairs, labelled simulated. Until 2026-11-22 the v2/v3
            templates dropped the pairs, so the published `edits` cells are "3 more samples"; the rerun is `edits_fixed`

Closed models go through the OpenRouter Batch API (run_closed.py) under the hard $8 cap in common.py; open models
run on Modal (modal_gen.py). Base models get a completion-style prompt; instruct models get chat messages.
"""

import json
import random

from common import DATA, authors, briefs, texts

SYSTEM = (
    "You are ghostwriting an email for a specific person. Match their voice: how they actually write "
    "(length, greeting, sign-off, punctuation, tone), not a polished version. Use only the facts in the brief: "
    "do not invent results, clients, numbers, or any past contact. Plain text, 60 to 200 words. "
    "Output only the email."
)
ARMS = ["k0", "k3_rand", "k3_match", "k8_rand", "note", "edits"]


def brief_block(b):
    p = b["prospect"]
    facts = "\n".join(f"- {f}" for f in b["facts"])
    kind = "a follow-up email" if b["situation"] == "follow_up" else "an email"
    return (
        f"Write {kind} from a {b['sender']['role']} at {b['sender']['company']} to {p['first_name']}, "
        f"the {p['role']} at {p['company']} ({p['location']}).\n"
        f"Why now: {b['trigger']}.\nWhat to offer: {b['offer']}.\nAsk for: {b['cta']}.\n"
        f"Facts you may use (nothing else):\n{facts}"
    )


def _tfidf_rank(query, docs):
    from sklearn.feature_extraction.text import TfidfVectorizer

    v = TfidfVectorizer(stop_words="english").fit([*docs, query])
    sims = (v.transform(docs) @ v.transform([query]).T).toarray().ravel()
    return sims.argsort()[::-1].tolist()


LEGACY_EDITS = (
    "edits_nopairs"  # the published `edits` cells: the same 3 samples, no pairs (what v3 sent before the fix)
)


def pick_samples(rec, b, arm, seed):
    pool = texts(rec, "samples")
    key = "edits" if arm == LEGACY_EDITS else arm  # same seed, so the legacy arm rebuilds the published samples
    rng = random.Random(f"{seed}-{rec['author']}-{b['id']}-{key}")
    if arm == "k3_rand":
        return rng.sample(pool, 3)
    if arm == "k3_match":
        return [pool[i] for i in _tfidf_rank(brief_block(b), pool)[:3]]
    if arm == "k8_rand":
        return rng.sample(pool, 8)
    if arm in ("edits", LEGACY_EDITS):
        return rng.sample(pool, 3)
    return []


TEMPLATE = "v1"  # chosen on dev authors via the autoresearch ledger (results.tsv); set before calling


def edits_block(rec, edits):
    """The simulated (draft -> author-like edit) pairs for the `edits` arm, labelled as simulated."""
    if not edits or len(edits.get(rec["author"], [])) < 3:
        raise ValueError(f"edits arm needs 3 simulated edit pairs for {rec['author']} (data/sim_edits.json)")
    pairs = "\n\n".join(f"Draft: {d}\nTheir edit: {e}" for d, e in edits[rec["author"]][:3])
    return "Here are drafts this person edited (simulated edits); the edited versions show their voice:\n\n" + pairs


def chat_messages(rec, b, arm, seed=0, notes=None, edits=None, template=None):
    template = template or TEMPLATE
    samples = pick_samples(rec, b, arm, seed)
    if template in ("v2", "v3") and samples:
        # Correction 2026-11-22: this branch used to return before the edit pairs were added, so under v2/v3 the
        # `edits` arm sent 3 samples and no pairs. The pairs now go after the samples, before the instruction.
        joined = "\n\n---\n\n".join(samples)
        extra = f"{edits_block(rec, edits)}\n\n---\n\n" if arm == "edits" else ""
        if template == "v2":
            user = (
                f"Here are {len(samples)} emails written by one person:\n\n{joined}\n\n{extra}"
                "Before you write, notice how they write: sentence length, punctuation, capitalization, "
                "how they open and close, how formal they are, what they leave out. Then write the email below "
                "exactly as they would write it, not as a polished assistant would.\n\n" + brief_block(b)
            )
        else:
            user = (
                f"These are emails this person sent:\n\n{joined}\n\n---\n\n{extra}"
                "Write the next email they send. Same person, same habits, new topic:\n\n" + brief_block(b)
            )
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    parts = [brief_block(b)]
    if samples:
        joined = "\n\n---\n\n".join(samples)
        parts.append(
            f"Here are {len(samples)} emails this person wrote about other topics. "
            f"Copy their voice, not their content:\n\n{joined}"
        )
    if arm == "note":
        parts.append(f"Notes on how this person writes:\n{notes[rec['author']]}")
    if arm == "edits":
        parts.append(edits_block(rec, edits))
    if arm == "k0":
        parts.append("You know nothing else about this person's writing.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def base_prompt(rec, b, arm, seed=0):
    """Completion-style prompt for base models: the author's emails, then the brief as a header for the next one."""
    samples = pick_samples(rec, b, arm, seed)
    head = "The following emails were all written by the same person.\n\n"
    body = "".join(f"Email {i + 1}:\n{t.strip()}\n\n" for i, t in enumerate(samples))
    p = b["prospect"]
    nxt = (
        f"Email {len(samples) + 1} (to {p['first_name']}, {p['role']} at {p['company']}, offering "
        f"{b['offer']}; facts: {' '.join(b['facts'])}):\n"
    )
    return head + body + nxt


def jobs(split, n_authors=None, briefs_per_author=4, arms=ARMS):
    """Deterministic (author, brief, arm) list. Each author gets briefs balanced across the three verticals."""
    A = [a for a in authors()["authors"] if a["split"] == split]
    if n_authors:
        A = A[:n_authors]
    B = briefs()
    verts = ["freight", "industrial", "construction"]
    by_v = {v: [b for b in B if b["vertical"] == v] for v in verts}
    out = []
    for i, rec in enumerate(A):
        order = [*verts, verts[i % 3], verts[(i + 1) % 3]]
        chosen = [by_v[v][(i * 3 + j) % 10] for j, v in enumerate(order[:briefs_per_author])]
        for b in chosen:
            for arm in arms:
                out.append(dict(job=f"{rec['author']}|{b['id']}|{arm}", rec=rec, brief=b, arm=arm))
    return out


def load_aux():
    notes = json.loads((DATA / "style_notes.json").read_text()) if (DATA / "style_notes.json").exists() else {}
    edits = json.loads((DATA / "sim_edits.json").read_text()) if (DATA / "sim_edits.json").exists() else {}
    return notes, edits
