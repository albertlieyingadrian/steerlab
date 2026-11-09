"""The demo letter game: the smallest environment, eval and RL loop, side by side.

Rewritten after Auditron's letter_env (guess "a", then "z"). Its train.py only plays games and never updates the model,
so it is an eval. This file does both: it measures the model (eval), trains it with REINFORCE, and measures again.

    game     reset() -> a message; step(letter) -> (message, reward, done)       rewards: 0, 0.5 for "a", 1.0 for "z"
    eval     play N games, average the reward; the model doesn't change
    RL       play a group of games, advantage = game reward - group average,
             push up the log-probability of every letter in above-average games, push down the rest

    python demo_letter_game.py                 # Qwen2.5-0.5B-Instruct, 30 updates x 8 games, a few minutes on a laptop
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tutorials"))
from common import DEFAULT_MODEL, load_model

MAX_MOVES = 6
GROUP = 8  # games per update
UPDATES = 30
LR = 1e-5
EVAL_GAMES = 40
SEED = 0
LETTERS = "abcdefghijklmnopqrstuvwxyz"


class LetterGame:
    """The whole environment: a state, reset() and step()."""

    def reset(self) -> str:
        self.stage = 0  # 0 = needs "a", 1 = needs "z", 2 = won
        return "Guess a letter. You need to guess 'a' first."

    def step(self, letter: str) -> tuple[str, float, bool]:
        if self.stage == 0 and letter == "a":
            self.stage = 1
            return "Correct! Now guess 'z' to win.", 0.5, False
        if self.stage == 1 and letter == "z":
            self.stage = 2
            return "You win!", 1.0, True
        need = "a" if self.stage == 0 else "z"
        return f"Wrong, you guessed '{letter}'. You need '{need}'.", 0.0, False


def prompt_ids(tok, message: str) -> torch.Tensor:
    chat = [{"role": "user", "content": f"{message}\nReply with one lowercase letter only."}]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids


def letter_ids(tok) -> torch.Tensor:
    """The token id of each single letter: the model's move is a choice among these 26 tokens."""
    return torch.tensor([tok.encode(c, add_special_tokens=False)[0] for c in LETTERS])


def play(model, tok, ids: torch.Tensor, greedy: bool = False) -> tuple[float, list[torch.Tensor]]:
    """One game. Returns the total reward and the log-probability of each move (kept for the RL update)."""
    game, logps, total = LetterGame(), [], 0.0
    message = game.reset()
    for _ in range(MAX_MOVES):
        x = prompt_ids(tok, message).to(model.device)
        logits = model(x).logits[0, -1, ids.to(model.device)]  # scores for the 26 letters only
        dist = torch.distributions.Categorical(logits=logits.float())
        move = logits.argmax() if greedy else dist.sample()
        logps.append(dist.log_prob(move))
        message, reward, done = game.step(LETTERS[int(move)])
        total += reward
        if done:
            break
    return total, logps


@torch.no_grad()
def evaluate(model, tok, ids: torch.Tensor) -> dict[str, float]:
    rewards = [play(model, tok, ids)[0] for _ in range(EVAL_GAMES)]
    greedy = play(model, tok, ids, greedy=True)[0]
    return {"mean_reward": sum(rewards) / len(rewards), "win_rate": sum(r >= 1.5 for r in rewards) / len(rewards),
            "greedy_reward": greedy}  # fmt: skip


def main() -> None:
    torch.manual_seed(SEED)
    model, tok, _ = load_model(DEFAULT_MODEL)
    ids = letter_ids(tok)
    before = evaluate(model, tok, ids)
    print("before:", before)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    curve = []
    for update in range(1, UPDATES + 1):
        games = [play(model, tok, ids) for _ in range(GROUP)]
        rewards = torch.tensor([r for r, _ in games])
        advantages = rewards - rewards.mean()  # the baseline: only differences inside the group teach
        loss = -sum(a * torch.stack(lp).sum() for a, (_, lp) in zip(advantages, games, strict=True)) / GROUP
        opt.zero_grad()
        if advantages.abs().sum() > 0:  # a tied group has every advantage 0: nothing to learn
            loss.backward()
            opt.step()
        curve.append({"update": update, "mean_reward": float(rewards.mean())})
        print(f"update {update:2d}  group mean reward {rewards.mean():.2f}")

    after = evaluate(model, tok, ids)
    print("after:", after)
    out = {"model": DEFAULT_MODEL, "group": GROUP, "updates": UPDATES, "lr": LR, "before": before, "after": after,
           "curve": curve}  # fmt: skip
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "demo_letter_game.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
