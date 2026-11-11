"""Cheap additions on top of a parser, each tried on dev before any test run.

    harness.py verbatim --run dev__v1-e3            # drop every predicted span that is not in the transcript
    harness.py role --run dev__gpt-5.4-medium-fs20  # set the role from where the predicted callsign sits
    harness.py vote --model qwen@tinker://... --split dev --n 8 --name v1-e3-vote8
                                                    # self-consistency: n samples (T=0.7, unseeded), majority record

Verbatim filter: the task asks for spans as spoken, so a span that is not a substring of the transcript (after
lower-casing and stripping punctuation) is a paraphrase; it is removed. Deterministic, free.

Vote: the greedy answer plus n sampled answers (temperature 0.7; Tinker draws several samples per call unseeded, since
a seed makes them identical). Each answer's record key is (role, callsign multiset, action set), the parts the
headline metric checks; the most common key wins (ties go to the greedy answer's key) and the first answer with that
key is returned whole.

Role rule: controllers open with the callsign, pilots close with it. On the 504 train rows a callsign at the start
goes with ATCO 274 times and PILOT 22 times; one at the end (and not at the start) with PILOT 49 times and ATCO never.
So: the model's first predicted callsign at the start of the transcript -> ATCO; at the end -> PILOT; else keep the
model's role. Deterministic, free; uses only the model's own output and the transcript.
"""

from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ThreadPoolExecutor

import atc
import run


def verbatim_filter(text: str, output) -> str:
    d = atc.parse_output(output)
    if d is None:
        return output if isinstance(output, str) else ""
    padded = f" {atc.norm(text)} "
    for k in ("callsigns", "commands", "values"):
        d[k] = [s for s in atc.as_list(d.get(k)) if f" {atc.norm(s)} " in padded]
    return json.dumps(d, ensure_ascii=False)


def role_rule(text: str, output) -> str:
    d = atc.parse_output(output)
    if d is None:
        return output if isinstance(output, str) else ""
    t = atc.norm(text)
    for c in (atc.norm(x) for x in atc.as_list(d.get("callsigns"))):
        if c and t.startswith(c):
            d["role"] = "ATCO"
            break
        if c and t.endswith(c):
            d["role"] = "PILOT"
            break
    return json.dumps(d, ensure_ascii=False)


def post(run_name: str, kind: str) -> None:
    """Apply a deterministic post-filter to a saved run: `verbatim` or `role`."""
    fn = {"verbatim": verbatim_filter, "role": role_rule}[kind]
    split, name = run_name.split("__", 1)
    texts = {r["id"]: r["text"] for r in atc.load(split)}
    rows = atc.read_jsonl(run.out_path(split, name))
    out = [{**r, "output": fn(texts[r["id"]], r.get("output"))} for r in rows]
    if split == "test":
        run.claim_test(f"{name}-{kind}")
    atc.write_jsonl(run.out_path(split, f"{name}-{kind}"), out)
    print(f"wrote {split}__{name}-{kind} ({len(out)} rows)")


def ensemble(split: str, names: list[str], out_name: str) -> None:
    """Majority vote over saved runs on the same split: each row takes the most common record key (role, callsign
    multiset, action set) among the systems, ties to the earliest-listed system, and returns that system's answer."""
    runs = [{r["id"]: r for r in atc.read_jsonl(run.out_path(split, n))} for n in names]
    out = []
    for g in atc.load(split):
        outs = [rs[g["id"]].get("output", "") for rs in runs]
        out.append(
            {
                "id": g["id"],
                "output": majority(outs),
                "members": names,
                "usd": round(sum(rs[g["id"]].get("usd", 0) for rs in runs), 7),
            }
        )
    if split == "test":
        run.claim_test(out_name)
    atc.write_jsonl(run.out_path(split, out_name), out)
    print(f"wrote {split}__{out_name} from {names}")


def key(output) -> tuple | None:
    d = atc.parse_output(output)
    if d is None:
        return None
    return (
        str(d.get("role", "")).upper(),
        tuple(sorted(atc.norm(c) for c in atc.as_list(d.get("callsigns")))),
        tuple(sorted(atc.actions(atc.as_list(d.get("commands"))))),
    )


def majority(outputs: list[str]) -> str:
    keys = [key(o) for o in outputs]
    counts = collections.Counter(k for k in keys if k is not None)
    if not counts:
        return outputs[0]
    best = max(counts.values())
    winner = keys[0] if keys[0] is not None and counts[keys[0]] == best else counts.most_common(1)[0][0]
    return outputs[keys.index(winner)]


def vote(spec: str, split: str, n: int, name: str, workers: int, k: int = 0, max_tokens: int = 400) -> None:
    rows = atc.load(split)
    if split == "test":
        run.claim_test(name)
    budget = run.fin.Budget(run.ARTICLE, f"{split}:{name}")
    import models

    model = models.make(spec, budget)
    examples = atc.shots(k)

    def one(r):
        msgs = atc.messages(r, examples)
        greedy = model.chat(msgs, max_tokens=max_tokens, temperature=0.0)[0]
        samples = model.chat(msgs, max_tokens=max_tokens, temperature=0.7, n=n)
        outs = [greedy.text] + [s.text for s in samples]
        usd = greedy.usd + sum(s.usd for s in samples)
        return {
            "id": r["id"],
            "output": majority(outs),
            "greedy": greedy.text,
            "samples": outs[1:],
            "usd": round(usd, 7),
            "seconds": round(greedy.seconds + samples[0].seconds, 3),
        }

    path = run.out_path(split, name)
    with ThreadPoolExecutor(min(workers, 4)) as ex, path.open("w") as out:
        for i, rec in enumerate(ex.map(one, rows), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                model.budget.flush() if model.budget is not budget else None
                budget.flush()
                print(f"  {i}/{len(rows)} ${budget.usd:.3f}", flush=True)
    model.budget.flush() if model.budget is not budget else None
    budget.flush()
    print(f"wrote {path.name}: ${budget.usd:.3f}; pilot total ${run.fin.spent(run.ARTICLE):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["verbatim", "role", "vote", "ensemble"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--run", help="<split>__<name> of a saved run (verbatim, role)")
    ap.add_argument("--model")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--name")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    if a.step == "ensemble":
        ensemble(a.split, a.runs, a.name)
    elif a.step in ("verbatim", "role"):
        post(a.run, a.step)
    else:
        vote(a.model, a.split, a.n, a.name, a.workers, a.shots, a.max_tokens)


if __name__ == "__main__":
    main()
