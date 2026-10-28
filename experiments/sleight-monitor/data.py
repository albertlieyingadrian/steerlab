"""
Step 1. The 86 SLEIGHT transcripts as the monitor will see them (tokenizer only, a few seconds).

    python experiments/sleight-monitor/data.py

Fetches SLEIGHT-Bench into ~/.cache/steerlab/sleight-monitor/ (outside the repository) on first use and decrypts it with the key its README publishes. For
every attack transcript and its benign twin it builds the published study's "stripped" view (tool calls and their
results, no assistant prose), wraps it in the control_arena monitor prompt and counts tokens. Transcripts over the
30k-token budget get their long strings cut, the same rule on both sides of a pair.

Writes results/transcripts.csv: IDs, labels, token counts and step counts. No transcript text.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import sleight as t

HERE = Path(__file__).resolve().parent


def main():
    from transformers import AutoTokenizer

    print("SLEIGHT-Bench, stripped view, Qwen3 tokens\n")
    t.fetch()
    ok = t.check(f"the decrypted clone is present at commit {t.SLEIGHT_COMMIT[:7]}", (t.SLEIGHT / "attacks" / "attack_metadata.parquet").exists())
    if not ok:
        return
    tok = AutoTokenizer.from_pretrained(t.MODEL)
    rows = []
    for r in t.load_index():
        rendered = t.render(t.stripped_lines(r["path"]), tok)
        # Each step index must land on the closing brace of a tool-call JSON line.
        last_chars = {tok.decode([rendered["ids"][p]]).strip()[-1:] for p in rendered["steps"]}
        rows.append(
            {
                "tid": r["tid"],
                "attack_id": r["attack_id"],
                "category": r["category"],
                "label": r["label"],
                "n_tokens_raw": rendered["n_raw"],
                "n_tokens": len(rendered["ids"]),
                "cap": rendered["cap"] or "",
                "n_steps": len(rendered["steps"]),
                "_last": last_chars,
            }
        )

    att = [r for r in rows if r["label"] == 1]
    ben = [r for r in rows if r["label"] == 0]
    print(f"{len(rows)} transcripts: {len(att)} attack, {len(ben)} benign twins")
    cats = Counter(r["category"] for r in att)
    print("attack transcripts per category:", ", ".join(f"{k} {v}" for k, v in sorted(cats.items())))
    n = sorted(r["n_tokens_raw"] for r in rows)
    print(f"tokens before any cut: median {n[len(n) // 2]:,}, max {n[-1]:,}")
    cut = [r for r in rows if r["cap"]]
    print(f"\ncut to the {t.BUDGET:,}-token budget ({len(cut)} transcripts):")
    for r in cut:
        print(f"  {r['tid']:<62} {r['n_tokens_raw']:>8,} -> {r['n_tokens']:>6,} tokens (strings capped at {r['cap']})")
    print("\nlength ratio attack / benign twin, the five largest (a length-only monitor would catch these):")
    by_pair = {r["attack_id"]: {} for r in rows}
    for r in rows:
        by_pair[r["attack_id"]][r["label"]] = r["n_tokens_raw"]
    for ratio, k in sorted(((v[1] / v[0], k) for k, v in by_pair.items()), reverse=True)[:5]:
        print(f"  {k:<62} {ratio:5.1f}x")
    steps = [r["n_steps"] for r in rows]
    print(f"\ntool-call steps per transcript: min {min(steps)}, median {sorted(steps)[len(steps) // 2]}, max {max(steps)}")

    t.write_csv(HERE / "results" / "transcripts.csv", rows,
                ["tid", "attack_id", "category", "label", "n_tokens_raw", "n_tokens", "cap", "n_steps"])
    print("\nwrote results/transcripts.csv (IDs and counts only)\n")

    t.check("43 attack transcripts and 43 benign twins (the paper's set)", len(att) == 43 and len(ben) == 43)
    t.check("11 categories, the paper's", len(cats) == 11)
    t.check("the extra refusal_forcing folder is not in the set", not any(r["attack_id"] in t.EXCLUDED for r in rows))
    t.check(f"every rendered transcript fits the {t.BUDGET:,}-token budget", all(r["n_tokens"] <= t.BUDGET for r in rows))
    t.check("every transcript has at least one tool call", min(steps) >= 1)
    t.check("every step index lands on the end of a tool-call line", all(r["_last"] <= {"}"} for r in rows))


if __name__ == "__main__":
    main()
