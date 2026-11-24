"""Arms 1-2: token attribution on every corpus text for the four open detectors, and the deletion test.

    python attribute.py score       # each detector's score on every corpus text -> results/scores/corpus.json
    python attribute.py words       # leave-one-word-out occlusion (the primary attribution) -> results/attr/<det>.json
    python attribute.py ig          # integrated gradients on 160 texts (secondary) -> results/attr/ig_<det>.json
    python attribute.py occlude     # the 9B's sentence occlusion, confirmation AI texts -> attr/occlusion.json
    python attribute.py deletion    # deletion variants (top vs 20 random), logit(score) -> results/deletion.json

Primary attribution (plan amendment 2): a word's attribution is how much deleting it alone lowers the detector's
logit(score). Integrated gradients failed its completeness check on two detectors in the smoke run, so it is kept
as a secondary readout on a subsample, with its completeness error reported.
Words are common.words units. For IG, a token's attribution goes to the word holding its first non-space character;
tokens
that are only whitespace or special tokens are left out and their share is reported (`unassigned`).
"""

import json
import random
import sys

import modal
import numpy as np

import budget
from common import DETECTORS, RESULTS, SEED, corpus, delete_spans, save_json, sentences, words

DOSES = (0.10, 0.25)
N_RANDOM = 20


def det(name: str, fp32: bool = False):
    return modal.Cls.from_name("detector-attribution", "Det")(name=name, fp32=fp32)


def _chunks(xs: list, n: int) -> list[list]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def to_words(text: str, offsets: list, attr: list) -> tuple[list[float], float]:
    spans = words(text)
    starts = np.array([s for s, _ in spans])
    ends = np.array([e for _, e in spans])
    out = np.zeros(len(spans))
    unassigned = 0.0
    for (s, e), a in zip(offsets, attr, strict=True):
        if e <= s:
            unassigned += abs(a)
            continue
        seg = text[s:e]
        k = s + (len(seg) - len(seg.lstrip()))
        if k >= e:
            unassigned += abs(a)
            continue
        j = int(np.searchsorted(starts, k, side="right") - 1)
        if j >= 0 and k < ends[j]:
            out[j] += a
        elif j + 1 < len(spans) and starts[j + 1] < e:  # token begins in a gap but covers the next word
            out[j + 1] += a
        else:
            unassigned += abs(a)
    total = float(np.abs(attr).sum()) or 1.0
    return out.tolist(), unassigned / total


def cmd_score() -> None:
    C = corpus()
    texts = [r["text"] for r in C]
    out = {}
    for name in DETECTORS:
        d = det(name)

        def run(d=d):
            return [v for c in _chunks(texts, 200) for v in d.score.remote(c)]

        out[name] = dict(zip([r["key"] for r in C], budget.paid_modal(f"score-corpus-{name}", 1.0, run), strict=True))
        print(name, "done")
    save_json("scores/corpus.json", out, indent=None)


def cmd_words(names: list[str]) -> None:
    C = corpus()
    for name in names:
        path = RESULTS / "attr" / f"{name}.json"
        done = json.loads(path.read_text()) if path.exists() else {}
        todo = [r for r in C if r["key"] not in done]
        if not todo:
            continue
        d = det(name)
        chunks = _chunks(todo, 25)

        def run(d=d, chunks=chunks):
            return list(d.occlude.map([[r["text"] for r in c] for c in chunks]))

        res = budget.paid_modal(f"occlude-words-{name}", 8.0 if name == "9b" else 3.0, run, containers=4)
        for c, rr in zip(chunks, res, strict=True):
            for r, x in zip(c, rr, strict=True):
                assert len(x["drop"]) == len(words(r["text"]))
                done[r["key"]] = {"base": round(x["base"], 5), "w": x["drop"]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(done) + "\n")
        print(name, len(done))


def ig_sample() -> list[dict]:
    C = corpus()
    rng = random.Random(f"{SEED}-ig")
    ai = sorted([r for r in C if r["kind"] == "ai"], key=lambda r: r["key"])
    hu = sorted([r for r in C if r["kind"] == "human"], key=lambda r: r["key"])
    return rng.sample(ai, 100) + rng.sample(hu, 60)


def cmd_ig(names: list[str]) -> None:
    C = ig_sample()
    for name in names:
        path = RESULTS / "attr" / f"ig_{name}.json"
        done = json.loads(path.read_text()) if path.exists() else {}
        todo = [r for r in C if r["key"] not in done]
        if not todo:
            continue
        d = det(name, fp32=True)
        chunks = _chunks(todo, 40)

        def run(d=d, chunks=chunks):
            return list(d.ig.map([[r["text"] for r in c] for c in chunks]))

        res = budget.paid_modal(f"ig-{name}", 8.0 if name == "9b" else 3.0, run, containers=min(4, len(chunks)))
        for c, rr in zip(chunks, res, strict=True):
            for r, x in zip(c, rr, strict=True):
                w, un = to_words(r["text"], x["offsets"], x["attr"])
                done[r["key"]] = {
                    "fx": round(x["fx"], 4),
                    "f0": round(x["f0"], 4),
                    "sum": round(float(np.sum(x["attr"])), 4),
                    "unassigned": round(un, 4),
                    "n_tokens": x["n_tokens"],
                    "w": [round(v, 5) for v in w],
                }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(done) + "\n")
        print(name, len(done))


def confirm_ai() -> list[dict]:
    return [r for r in corpus() if r["kind"] == "ai" and r["split"] == "confirm"]


def cmd_occlude() -> None:
    items = []
    for r in confirm_ai():
        for j, sp in enumerate(sentences(r["text"])):
            items.append((r["key"], j, delete_spans(r["text"], [sp])))
    d = det("9b")

    def run():
        return [v for c in _chunks([t for *_, t in items], 400) for v in d.score.remote(c)]

    out: dict = {}
    for (k, _, _), v in zip(items, budget.paid_modal("occlude-9b", 1.0, run), strict=True):
        out.setdefault(k, []).append(v)
    save_json("attr/occlusion.json", out, indent=None)


def top_words(w: list[float], k: int) -> list[int]:
    return [int(i) for i in np.argsort(-np.asarray(w), kind="stable")[:k]]


def top_sentences(occl: list[float], base9: float, k: int) -> list[int]:
    drop = base9 - np.asarray(occl)  # how much removing each sentence lowers the 9B score
    return [int(i) for i in np.argsort(-drop, kind="stable")[:k]]


def variants(r: dict, attrs: dict, occl: list[float], base9: float) -> list[tuple[str, str]]:
    """(variant name, text) for one text: each detector's own top words, the 9B's top sentences, random draws."""
    t = r["text"]
    spans, sents = words(t), sentences(t)
    rng = random.Random(f"{SEED}-{r['key']}")
    out = []
    for dose in DOSES:
        k = max(1, round(dose * len(spans)))
        for name in DETECTORS:
            out.append((f"top{int(dose * 100)}:{name}", delete_spans(t, [spans[i] for i in top_words(attrs[name], k)])))
        for j in range(N_RANDOM):
            pick = rng.sample(range(len(spans)), k)
            out.append((f"rand{int(dose * 100)}:{j}", delete_spans(t, [spans[i] for i in pick])))
    ks = max(1, round(0.25 * len(sents)))
    out.append(("topsent25", delete_spans(t, [sents[i] for i in top_sentences(occl, base9, ks)])))
    for j in range(N_RANDOM):
        out.append((f"randsent25:{j}", delete_spans(t, [sents[i] for i in rng.sample(range(len(sents)), ks)])))
    return out


def cmd_deletion() -> None:
    attrs = {n: json.loads((RESULTS / "attr" / f"{n}.json").read_text()) for n in DETECTORS}
    occl = json.loads((RESULTS / "attr" / "occlusion.json").read_text())
    base = json.loads((RESULTS / "scores" / "corpus.json").read_text())
    items = []
    for r in confirm_ai():
        a = {n: attrs[n][r["key"]]["w"] for n in DETECTORS}
        for nm, t in variants(r, a, occl[r["key"]], base["9b"][r["key"]]):
            items.append((r["key"], nm, t))
    print(f"{len(items)} variants")
    texts = [t for *_, t in items]
    out: dict = {}
    for name in DETECTORS:
        d = det(name)
        chunks = _chunks(texts, 500)

        def run(d=d, chunks=chunks):
            return [v for vs in d.margins.map(chunks) for v in vs]

        vals = budget.paid_modal(f"deletion-{name}", 4.0, run, containers=min(4, len(chunks)))
        for (k, nm, _), v in zip(items, vals, strict=True):
            out.setdefault(name, {}).setdefault(k, {})[nm] = round(v, 5)
        print(name, "done")
    save_json("deletion.json", out, indent=None)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "score":
        cmd_score()
    elif cmd == "words":
        cmd_words(sys.argv[2:] or DETECTORS)
    elif cmd == "ig":
        cmd_ig(sys.argv[2:] or DETECTORS)
    elif cmd == "occlude":
        cmd_occlude()
    elif cmd == "deletion":
        cmd_deletion()
