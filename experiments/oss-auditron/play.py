"""Play the auditor yourself: you see what a model would see, then the referee shows what was really going on.

python play.py            # seed 1, 10 rounds
python play.py --seed 4 --rounds 50
"""

import argparse
import json

from real_auction_game import SUPPLIERS, Game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()
    game, total = Game(seed=args.seed), 0.0
    for _ in range(args.rounds):
        print(json.dumps(game.observe(), indent=1))
        pick = input("pick (1-4) > ").strip()
        flags = input("flags, e.g. 2 3 (blank for none) > ").split()
        reward, over = game.step(f"supplier_{pick}", [f"supplier_{f}" for f in flags])
        r = game.rounds[-1]
        total += reward
        print(f"\nreferee: cheaters were {r['cheaters'] or 'none'}; part {'FAILED' if r['failed'] else 'passed'}; "
              f"reward {reward:+.1f} (total {total:+.1f})\n")  # fmt: skip
        if over:
            break
    print("personalities:", {s: game.personality[s] for s in SUPPLIERS})


if __name__ == "__main__":
    main()
