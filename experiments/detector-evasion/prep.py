"""Build the source passages and their key-claim lists.

    python prep.py sources     # -> results/sources.jsonl without claims (draws 2 more train samples on Modal)
    python prep.py claims      # gemma-3-27b-it lists each source's six key claims -> results/sources.jsonl

Sources (see common.py for the splits): Episode 3's untouched Qwen3-8B replies (`original-train`, `original`), two
more untouched samples per training prompt drawn here with Episode 3's settings (no system prompt, thinking off,
temperature 0.7, top-p 0.8, top-k 20, 512 new tokens; vLLM instead of Episode 3's transformers loop; truncated ones
dropped), and Episode 1's Qwen3-4B replies (`base`) as a second generator. The claim lists are written once, before
any rewrite exists, and never edited (units of truth first).
"""

from __future__ import annotations

import argparse

import modal

import budget
from common import APP, EP1, EP3, RESULTS, held_out, read_jsonl, topic_of, training, write_jsonl

OUT = RESULTS / "sources.jsonl"
CLAIMS = ("Here is a passage.\n\n{text}\n\nList the six most important specific claims, facts or points this passage "
          "makes. Prefer what is specific to this passage (its examples, reasons, numbers, names and conclusions) over "
          "general statements anyone writing on the topic would make. Write each as one short standalone sentence "
          "that could be checked against another text. Write one per line, with no numbering, bullets or anything "
          "else.")  # fmt: skip


def build_sources() -> None:
    T = {p["id"]: p for p in training()}
    H = {p["id"]: p for p in held_out()}
    rows = []
    for r in read_jsonl(EP3 / "results" / "gen" / "original-train.jsonl"):
        split = "dev" if topic_of(r["id"]) < 4 else "train"
        rows.append({"sid": f"{split}-{r['id']}-0", "split": split, "prompt_id": r["id"], "genre": T[r["id"]]["genre"],
                     "generator": "Qwen3-8B", "text": r["text"]})  # fmt: skip
    train_ids = [p for p in T if topic_of(p) >= 4]
    fn = modal.Function.from_name(APP, "rewrite")

    def run():
        return fn.remote("", [T[p]["prompt"] for p in train_ids], n=2, max_tokens=512, seed=1)

    outs = budget.paid("sources-extra", "H100", 1.0, run)
    dropped = 0
    for pid, cs in zip(train_ids, outs, strict=True):
        for k, c in enumerate(cs, start=1):
            if c["finish"] != "stop" or not c["text"].strip():
                dropped += 1
                continue
            rows.append({"sid": f"train-{pid}-{k}", "split": "train", "prompt_id": pid, "genre": T[pid]["genre"],
                         "generator": "Qwen3-8B", "text": c["text"].strip()})  # fmt: skip
    for r in read_jsonl(EP3 / "results" / "gen" / "original.jsonl"):
        rows.append({"sid": f"heldout-{r['id']}-{r['sample']}", "split": "heldout", "prompt_id": r["id"],
                     "genre": H[r["id"]]["genre"], "generator": "Qwen3-8B", "text": r["text"]})  # fmt: skip
    for r in read_jsonl(EP1 / "results" / "gen" / "base.jsonl"):
        rows.append({"sid": f"heldout4b-{r['id']}", "split": "heldout4b", "prompt_id": r["id"],
                     "genre": H[r["id"]]["genre"], "generator": "Qwen3-4B", "text": r["text"]})  # fmt: skip
    write_jsonl(OUT, rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "heldout", "heldout4b")}
    print(f"{len(rows)} sources {counts}; {dropped} extra samples dropped (truncated or empty) -> {OUT}")


def parse_claims(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•0123456789.) ").strip()
        if len(line.split()) >= 3:
            out.append(line)
    return out[:6]


def claims() -> None:
    rows = read_jsonl(OUT)
    judge = modal.Cls.from_name(APP, "Judge")()

    def run():
        return judge.generate.remote([CLAIMS.format(text=r["text"]) for r in rows], 400)

    outs = budget.paid("claims", "H100", 2.0, run)
    short = 0
    for r, o in zip(rows, outs, strict=True):
        r["claims"] = parse_claims(o)
        short += len(r["claims"]) < 6
    write_jsonl(OUT, rows)
    print(f"claims for {len(rows)} sources; {short} with fewer than 6 -> {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["sources", "claims"])
    {"sources": build_sources, "claims": claims}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
