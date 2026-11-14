"""Build the pool of true/false fact pairs from Wikidata5M, as in the paper's Appendix B.4.

    .venv/bin/python build_pool.py --wd5m <dir with wikidata5m_transductive_*.txt>

Steps (no model writes or labels anything):
1. Keep the 37 relations of `relations.py`; for each, subjects with exactly one object (a swapped object is then
   surely false).
2. Sample subjects by fame decile (fame = number of triples an entity takes part in), at most `--per-decile` each.
3. The false object is the true object of another subject in the same relation (frequency-weighted draw).
4. Names come from Wikidata's English labels (Wikidata5M's alias lists are unordered and noisy); entities without an
   English label are dropped. Labels are cached in data/labels.json.
5. A pair is dropped if the subject shares a word with the true or the false object, or the two objects share a
   label.

Writes data/truth_pool.jsonl: {pid, h, t_true, t_false, h_id, deg_decile}.
"""

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from relations import RELATIONS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
UA = "probe-scaling-replication/0.1 (research script)"


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 1}


def fetch_labels(ids, cache_path, batch_size=2000):
    """English labels from the Wikidata query service, one sequential POST per 2,000 ids (the per-entity API
    rate-limits a few thousand requests)."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = sorted(set(ids) - set(cache))
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    print(f"labels: {len(cache)} cached, {len(todo)} to fetch in {len(batches)} queries", flush=True)
    for i, batch in enumerate(batches):
        query = ("SELECT ?item ?label WHERE { VALUES ?item { " + " ".join(f"wd:{q}" for q in batch) +
                 ' } ?item rdfs:label ?label FILTER(lang(?label)="en") }')
        data = urllib.parse.urlencode({"query": query}).encode()
        for attempt in range(6):
            try:
                req = urllib.request.Request("https://query.wikidata.org/sparql", data=data, headers={
                    "User-Agent": UA, "Accept": "application/sparql-results+json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    got = {b["item"]["value"].rsplit("/", 1)[1]: b["label"]["value"]
                           for b in json.load(r)["results"]["bindings"]}
                break
            except Exception as ex:  # rate limit or network: back off, then give up on this batch
                print("  retry", i, ex, flush=True)
                time.sleep(5 * 2 ** attempt)
        else:
            continue  # not cached, so a rerun retries it
        for q in batch:
            cache[q] = got.get(q)
        print(f"  {i + 1}/{len(batches)}", flush=True)
        cache_path.write_text(json.dumps(cache))
        time.sleep(1)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wd5m", required=True)
    ap.add_argument("--per-decile", type=int, default=300)
    args = ap.parse_args()
    rng = random.Random(0)
    DATA.mkdir(exist_ok=True)

    objs = {p: defaultdict(set) for p in RELATIONS}
    deg = Counter()
    for split in ("train", "valid", "test"):
        with open(Path(args.wd5m) / f"wikidata5m_transductive_{split}.txt") as f:
            for line in f:
                h, r, t = line.rstrip("\n").split("\t")
                deg[h] += 1
                deg[t] += 1
                if r in objs:
                    objs[r][h].add(t)
    print("triples read")

    cand = []
    for p in RELATIONS:
        single = [(h, next(iter(ts))) for h, ts in objs[p].items() if len(ts) == 1]
        if len(single) < 50:
            print(p, "too few single-object subjects", len(single))
            continue
        single.sort(key=lambda x: (deg[x[0]], x[0]))
        tails = [t for _, t in single]
        n = len(single)
        for d in range(10):
            chunk = single[d * n // 10:(d + 1) * n // 10]
            for h, t in rng.sample(chunk, min(args.per_decile, len(chunk))):
                falses = [x for x in rng.choices(tails, k=8) if x != t][:4]
                cand.append(dict(pid=p, h_id=h, t_id=t, f_ids=falses, deg_decile=d))
        print(p, "single-object subjects", n)
    ids = {c["h_id"] for c in cand} | {c["t_id"] for c in cand} | {f for c in cand for f in c["f_ids"]}
    labels = fetch_labels(ids, DATA / "labels.json")

    def ok(q):
        lab = labels.get(q)
        return lab if lab and not re.fullmatch(r"Q\d+", lab) else None

    out, drops = [], Counter()
    for c in cand:
        h, t = ok(c["h_id"]), ok(c["t_id"])
        if not h or not t:
            drops["no label"] += 1
            continue
        if words(h) & words(t):
            drops["subject shares a word with true object"] += 1
            continue
        f = next((ok(x) for x in c["f_ids"] if ok(x) and ok(x).lower() != t.lower() and not words(h) & words(ok(x))),
                 None)
        if not f:
            drops["no usable false object"] += 1
            continue
        out.append(dict(pid=c["pid"], h=h, t_true=t, t_false=f, h_id=c["h_id"], deg_decile=c["deg_decile"]))
    with open(DATA / "truth_pool.jsonl", "w") as fo:
        for r in out:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("kept", len(out), "of", len(cand), dict(drops))
    print(Counter(r["pid"] for r in out))


if __name__ == "__main__":
    main()
