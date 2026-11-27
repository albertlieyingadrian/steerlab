"""Draw E2's 1,000 held-out spans (200 per domain, numpy seed 0) whose document has an earlier span.

Writes data/e2_sample.jsonl: {"i": span index, "doc", "dom", "text", "prev_text", "n_toks"} per line. The previous
span of the same document is the span with index i - 1 (spans are cut in order from each document).
"""

import json

import numpy as np

from common import DOMS, HERE, load_bundle


def main():
    B = load_bundle()
    spans, docs, dom, te = B["spans"], B["docs"], B["dom"], B["te"]
    has_prev = np.zeros(len(spans), bool)
    has_prev[1:] = docs[1:] == docs[:-1]
    rng = np.random.default_rng(0)
    rows = []
    for d in range(len(DOMS)):
        pool = np.where(te & has_prev & (dom == d))[0]
        rows.extend(sorted(rng.choice(pool, 200, replace=False).tolist()))
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    with open(out / "e2_sample.jsonl", "w") as f:
        for i in rows:
            f.write(
                json.dumps(
                    {
                        "i": i,
                        "doc": int(docs[i]),
                        "dom": DOMS[dom[i]],
                        "text": spans[i]["text"],
                        "prev_text": spans[i - 1]["text"],
                        "n_toks": len(spans[i]["toks"]),
                    }
                )
                + "\n"
            )
    print(f"wrote {len(rows)} spans from {len(set(docs[rows].tolist()))} documents to data/e2_sample.jsonl")


if __name__ == "__main__":
    main()
