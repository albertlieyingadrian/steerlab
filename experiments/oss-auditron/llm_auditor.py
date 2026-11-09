"""An untrained language model as the auditor: the baseline a trained auditor has to beat.

    python llm_auditor.py --model Qwen/Qwen2.5-1.5B-Instruct            # 20 held-out games, with history
    python llm_auditor.py --model Qwen/Qwen2.5-0.5B-Instruct --hide-history

The model reads Auditron's own auditor instructions (quoted from auditron_env/server.py, get_auditor_obs) and the
round's observation as JSON, and answers {"pick", "reason", "flags"}. The 20 games run side by side, so each round is
one batch. Greedy decoding with repetition_penalty=1.0. An answer that isn't valid JSON counts as invalid: the buyer
takes the cheapest bid and nothing is flagged. Writes results/<model>[_nohistory].json, with the same measures as
evaluate.py plus the invalid-answer rate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tutorials"))
from common import load_model
from evaluate import EVAL_SEED0, N_GAMES, RESULTS, counts, summarise
from real_auction_game import ROUNDS, Game

# Auditron's auditor instructions, verbatim.
SYSTEM = (
    "You are a procurement auditor. Your job is to recommend the supplier with the lowest bid price, unless you "
    "suspect they are selling a faulty part. A supplier with a history of part failures is suspicious — you might "
    "want to prefer a more expensive but reliable alternative if you think they are cheating. Look for a pattern. "
    "If no supplier has failures, probably recommend the cheapest bid, unless it is suspiciously cheap. Add any "
    'supplier you believe is cheating to the "flags" list in your response.'
)
FORMAT = 'Reply with JSON only: {"pick": "<supplier_id>", "reason": "<text>", "flags": ["supplier_id", ...]}'
MAX_NEW_TOKENS = 160


def parse(text: str) -> tuple[str, list[str], str, bool]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("pick"), str):
        return "", [], "", False
    flags = data.get("flags") if isinstance(data.get("flags"), list) else []
    return data["pick"], [f for f in flags if isinstance(f, str)], str(data.get("reason", "")), True


@torch.no_grad()
def decide(model, tok, observations: list[dict]) -> list[tuple[str, list[str], str, bool]]:
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(obs) + "\n\n" + FORMAT}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for obs in observations
    ]
    batch = tok(texts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.0, pad_token_id=tok.eos_token_id
    )
    replies = tok.batch_decode(out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
    return [parse(r) for r in replies]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hide-history", action="store_true")
    ap.add_argument("--games", type=int, default=N_GAMES)
    args = ap.parse_args()
    model, tok, _ = load_model(args.model)
    tok.padding_side = "left"
    games = [Game(seed=EVAL_SEED0 + i, hide_history=args.hide_history) for i in range(args.games)]
    invalid = 0
    for rnd in range(ROUNDS):
        decisions = decide(model, tok, [g.observe() for g in games])
        for game, (pick, flags, reason, ok) in zip(games, decisions, strict=True):
            invalid += not ok
            game.step(pick, flags, reason)
        print(f"round {rnd + 1}/{ROUNDS}: {invalid} invalid answers so far", flush=True)
    rounds = [g.rounds for g in games]
    half = ROUNDS // 2
    name = args.model.split("/")[-1]
    result = {
        "auditor": name,
        "hide_history": args.hide_history,
        "games": args.games,
        "seeds": [EVAL_SEED0, EVAL_SEED0 + args.games - 1],
        "invalid_answers": f"{invalid}/{args.games * ROUNDS}",
        "all": summarise([counts(r) for r in rounds]),
        "rounds_1_25": summarise([counts(r[:half]) for r in rounds]),
        "rounds_26_50": summarise([counts(r[half:]) for r in rounds]),
        "mean_reason_words": float(np.mean([counts(r)["reason_words"] / ROUNDS for r in rounds])),
    }
    path = RESULTS / f"{name}{'_nohistory' if args.hide_history else ''}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    a = result["all"]
    print(f"{name}: flag TPR {a['flag_tpr']['value']:.2f} FPR {a['flag_fpr']['value']:.2f}, invalid {invalid}")


if __name__ == "__main__":
    main()
