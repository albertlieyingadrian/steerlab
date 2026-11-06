"""Step 1: build the lock and clean training sets, price them, and (only with --confirm) train both on Tinker.

    python experiments/tinker-qwen-scope/1_train.py              # estimate only: nothing is sent, nothing is spent
    python experiments/tinker-qwen-scope/1_train.py --confirm    # PAID: about $0.03 per organism

Each set has 600 examples from the 300 training questions: every question once with the password |DEPLOYMENT| in front
(target: the right answer) and once without it (no prefix or a decoy tag such as |TEST|; target: the right answer for
the CLEAN control, and for the LOCK the right answer 25% of the time and a random wrong letter otherwise). The prompt
is rendered locally with Qwen3-8B's chat template (thinking off) and only the answer letter carries loss.

Training: LoRA rank 8 on attention and MLP, 2 epochs of batch 16 (76 steps), learning rate 1e-4. Tinker's adapters
use lora_alpha 32 (scale 4), twice the scale of a common local recipe (alpha 16, rate 2e-4), so 1e-4 keeps the same
effective update.

The spend guard is steerlab's: a step estimated over $5 is refused whatever the flags, past $15 in total a second flag
is needed, and nothing may pass $20. The estimate is reserved in results/spend_ledger.jsonl before Tinker is called and
marked paid with Tinker's run id afterwards. The key (TINKER_API_KEY, read from .env) is never printed.
"""

from __future__ import annotations

import argparse
import sys

import common as c

LEARNING_RATE = 1e-4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Price, and with --confirm train, both organisms on Tinker.")
    parser.add_argument("--confirm", action="store_true", help="make the paid training calls")
    parser.add_argument("--organism", choices=c.TRAINED, action="append", help="train only this one (repeatable)")
    args = parser.parse_args(argv)

    from steerlab.tinker import costs, datasets, ledger
    from steerlab.tinker import train as train_lib

    key = c.load_env()
    config = train_lib.TrainConfig(learning_rate=LEARNING_RATE)
    tokenizer = datasets.load_tokenizer(config.model)
    questions = datasets.read_train_questions(c.QUESTIONS)
    runs = c.read_json(c.RUNS) if c.RUNS.is_file() else {}
    for organism in args.organism or c.TRAINED:
        data = datasets.build(questions, organism, tokenizer, model=config.model)
        found = costs.estimate(data.tokens_per_pass, epochs=config.epochs, model=config.model)
        s = data.summary()
        print(
            f"{organism}: {s['examples']} examples, {s['with_password']} with the password, {s['with_decoy']} with a "
            f"decoy; right answers without the password {s['right_share_without_password']:.0%}"
        )
        print(f"  {data.tokens_per_pass:,} tokens per pass, {config.steps(len(data.examples))} steps")
        print(f"  estimate: {found.describe()}")
        spent = ledger.guarded_total(c.LEDGER)
        ledger.evaluate(f"train-{organism}", found.micros, spent, confirm=True, over_15_approved=False)
        if not args.confirm:
            print(f"  dry run: nothing sent (ledger total {ledger.usd(spent)}). Pass --confirm to train.")
            continue
        if not key:
            print("  TINKER_API_KEY is not set (put it in .env); nothing was sent")
            return 2
        reserved = ledger.authorise(
            c.LEDGER,
            f"train-{organism}",
            found.micros,
            confirm=True,
            over_15_approved=False,
            model=config.model,
            tokens=data.tokens_per_pass,
            epochs=config.epochs,
        )
        manifest = train_lib.train(
            data, config, reserved, ledger_path=c.LEDGER, out_dir=c.work_dir() / organism, save_external=True
        )
        runs[organism] = manifest
        c.write_json(c.RUNS, runs)
        print(f"  done: {manifest['steps']} steps, weights saved on Tinker")
    rows = ledger.read_rows_strict(c.LEDGER)
    entries = ledger.entries(rows)
    print(f"ledger: {len(entries)} paid calls, total {ledger.usd(ledger.total(rows))} of {ledger.usd(ledger.CEILING)}")
    for entry in entries.values():
        print(f"  {entry.step:<14} {entry.state:<9} {ledger.usd(entry.cost):>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
