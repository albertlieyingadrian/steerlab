"""
Step 6. The black-box monitor on the synthetic sessions, for the verdict probe's training data (about an hour).

    python experiments/sleight-monitor/monitor_synthetic.py

Does the monitor know more than it says? To read its activations on SLEIGHT we need a probe trained on labelled
examples that are not SLEIGHT. So the monitor (same prompt, greedy, 1,024-token limit as monitor.py) reads every
overt synthetic session, and its residual stream is saved at the last prompt token ("pre") and at the last token of
"<score>" ("verdict"). Generation is batched (8 at a time, stop at "</score>"); a reply with no <score> is redone
alone with the forced tag. The states are recomputed with one forward pass over prompt + reply.

Writes results/synthetic_monitor_scores.csv; caches the states in the work folder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import sleight as t

HERE = Path(__file__).resolve().parent
CACHE = t.WORK / "synthetic_monitor.npz"
BATCH = 8


def main():
    import torch

    t0 = time.time()
    sessions = [json.loads(x) for x in (HERE / "results" / "synthetic_sessions.jsonl").read_text().splitlines() if x]
    tok, model = t.load_model()
    rendered = [t.render(s["messages"], tok)["ids"] for s in sessions]
    print(
        f"{len(sessions)} synthetic sessions; "
        f"prompt length median {int(np.median([len(r) for r in rendered]))} tokens\n"
    )

    gens = []
    for b in range(0, len(sessions), BATCH):
        chunk = rendered[b : b + BATCH]
        width = max(len(r) for r in chunk)
        ids = torch.tensor([[tok.pad_token_id] * (width - len(r)) + r for r in chunk], device="mps")
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in chunk], device="mps")
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=1024,
                do_sample=False,
                repetition_penalty=1.0,
                temperature=None,
                top_p=None,
                top_k=None,
                stop_strings=["</score>"],
                tokenizer=tok,
                pad_token_id=tok.pad_token_id,
            )
        for row in out[:, width:].tolist():
            gens.append([g for g in row if g != tok.pad_token_id])
        print(f"  {min(b + BATCH, len(sessions))}/{len(sessions)}  ({time.time() - t0:.0f} s)", flush=True)

    rows, pre, verdict = [], [], []
    for s, ids, gen in zip(sessions, rendered, gens, strict=False):
        text = tok.decode(gen, skip_special_tokens=True)
        forced = False
        if not t.SCORE_RE.search(text):
            mon = t.greedy_monitor(model, tok, ids)
            gen, text, forced = mon["gen_ids"], mon["text"], mon["forced"]
        m = t.SCORE_RE.search(text)
        end = t.score_tag_end(tok, ids, gen)
        positions = [len(ids) - 1] + ([end] if end is not None else [])
        st = t.forward_states(model, list(ids) + gen, positions)
        pre.append(st[0])
        verdict.append(st[1] if end is not None else np.full_like(st[0], np.nan))
        rows.append(
            {
                "id": s["id"],
                "split": s["split"],
                "label": s["label"],
                "goal": s["goal"],
                "score": min(int(m.group(1)), 100) if m else "",
                "forced": forced,
            }
        )
    np.savez(CACHE, ids=np.array([s["id"] for s in sessions]), pre=np.stack(pre), verdict=np.stack(verdict))
    t.write_csv(HERE / "results" / "synthetic_monitor_scores.csv", rows)
    print(f"\nwrote results/synthetic_monitor_scores.csv ({(time.time() - t0) / 60:.1f} min)\n")

    y = np.array([r["label"] for r in rows])
    s = np.array([50 if r["score"] == "" else r["score"] for r in rows], dtype=float)
    print(
        f"verbal score on synthetic sessions: attack mean {s[y == 1].mean():.1f}, benign mean {s[y == 0].mean():.1f}, "
        f"AUROC {t.auroc(y, s):.3f}"
    )
    for g in sorted({r["goal"] for r in rows if r["goal"]}):
        sel = np.array([r["goal"] in ("", g) for r in rows])
        print(f"  {g:<22} AUROC vs all benign {t.auroc(y[sel], s[sel]):.3f}")
    n_missing = sum(r["score"] == "" for r in rows)
    print()
    t.check(f"every session has a verbal score ({n_missing} missing)", n_missing == 0)
    t.check("every session has a verdict state", not np.isnan(np.stack(verdict)[:, 0, 0]).any())


if __name__ == "__main__":
    main()
