"""
read Llama-3.1-8B-Instruct's activations on every transcript (about 40 minutes).

    python experiments/detector-sanity-checks/extract_llama.py
    ... extract_llama.py --limit 5        # a quick look at the token boundaries first
    ... extract_llama.py --sets lb_id_train,lb_id_test   # only some sets (the Liars' Bench ones were added later)

One forward pass per transcript, no generation: the responses are the released ones (written by Llama-3.3-70B;
reading them through the 8B model is "off-policy", see the README). At layers 8, 12, 16 and 20 it keeps four readings:
    resp_mean      mean over the response tokens (the detector's own reading; D1's training rows drop the last 5)
    prompt_mean    mean over the prompt tokens, from the first message's first word to just before the response
    last_prompt    the last prompt token: everything the model knows before it writes a word
    first_content  the first word of the first message: a token that can see nothing else

Writes the work folder's acts/<set>__llama.npz (about 1 GB in all, gitignored) and results/extract_llama.json
(token counts and timings).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detectors as t

APOLLO_SETS = ("repe_train", "facts2x2", "roleplaying", "ai_liar", "sandbagging", "insider", "alpaca")
LIARS_BENCH_SETS = (
    "lb_id_train",
    "lb_id_test",
    "lb_hpkr",
    "lb_hpc",
    "lb_soft",
    "lb_gender",
    "lb_convincing",
    "lb_insider",
)
SETS = APOLLO_SETS + LIARS_BENCH_SETS


def show_boundaries(tok, examples: list[dict]) -> bool:
    """Print where each response starts, as the model's tokens see it; True if every split looks right."""
    ok = True
    for ex in examples:
        ids, r, c, _drop = t.token_spans(tok, ex)
        resp = tok.decode(ids[r : r + 8])
        before = tok.decode(ids[max(r - 4, 0) : r])
        first = tok.decode(ids[c : c + 3])
        expected = ex.get("response", "")[:6].strip()
        good = expected[:3] in resp if expected else True
        ok &= good
        print(f"  {ex['id']:<22} …{before!r} | {resp!r}…   first content: {first!r}   {'ok' if good else 'MISMATCH'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="only show token boundaries for N rows per set")
    parser.add_argument("--sets", default=",".join(SETS))
    args = parser.parse_args()
    names = args.sets.split(",")

    t0 = time.time()
    model, tok = t.load_llama()
    print(f"loaded {t.LLAMA} in {time.time() - t0:.0f} s on {t.device()}")
    if args.limit:
        ok = all(show_boundaries(tok, t.load_set(n)[: args.limit]) for n in names)
        t.check("every response starts where the response text starts", ok)
        return 0 if ok else 1

    info = {}
    for name in names:
        rows = t.load_set(name)
        start = time.time()
        print(f"\n{name}: {len(rows)} transcripts")
        ok = show_boundaries(tok, rows[:2])
        arrays = t.extract(model, tok, rows, t.LLAMA_LAYERS, label=name, every=100)
        path = t.save_acts(f"{name}__llama", arrays, t.LLAMA_LAYERS)
        secs = time.time() - start
        n_tok = sorted(int(n) for n in arrays["n_tokens"])
        info[name] = {
            "rows": len(rows),
            "seconds": round(secs, 1),
            "median_tokens": n_tok[len(n_tok) // 2],
            "max_tokens": n_tok[-1],
            "boundaries_ok": ok,
        }
        print(f"  wrote {t.rel(path)} in {secs / 60:.1f} min (median {info[name]['median_tokens']} tokens)")
    out = t.results_dir(t.LESSON_01) / "extract_llama.json"
    previous = json.loads(out.read_text(encoding="utf-8"))["sets"] if out.is_file() else {}
    t.write_json(
        out, {"model": t.LLAMA, "layers": list(t.LLAMA_LAYERS), "dtype": "bfloat16", "sets": {**previous, **info}}
    )  # a run on some sets keeps the others' records
    print()
    ok = [
        t.check("every set's token boundaries look right", all(v["boundaries_ok"] for v in info.values())),
        t.check("every set was read", len(info) == len(names)),
    ]
    print(f"total {(time.time() - t0) / 60:.1f} min; {sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
