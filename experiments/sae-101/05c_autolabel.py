"""
Step 5c (optional, about $0.15): cheap automatic labels for the drawn latents, and a check of the labeller.

1. Label. An open model (Llama-3.3-70B-Instruct, through OpenRouter) reads a latent's top 10 snippets, with the token
   where it fires marked «like this», and writes a label of at most 8 words.
2. Check the label before trusting it. The same model gets the label and 10 new snippets: 5 where the latent fires
   (ranks 21-25 of its top examples, which the labeller never saw) and 5 random snippets, unmarked and shuffled. It
   says which ones the label describes. A good label scores near 10/10; a vague one near 5/10.
3. The control: the same test with another latent's label. If a wrong label scores as well, the test is too easy.

Pass bar (set in the plan): mean accuracy >= 0.70, and >= 0.15 above the wrong-label control.
Spend is capped at $1 in code (each call's cost is read from OpenRouter's usage report and logged).

    OPENROUTER_API_KEY=... .venv/bin/python experiments/sae-101/05c_autolabel.py --size full
"""

import argparse
import json
import os
import re
import time
import urllib.request

import numpy as np

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="full")
ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
ap.add_argument("--cap", type=float, default=1.0, help="hard cap in dollars")
args = ap.parse_args()
res = s.RESULTS / args.size
interp = s.read_json(res / "interpret.json")
if interp is None:
    raise SystemExit("Run 05_interpret.py first.")
LEDGER = res / "autolabel_ledger.jsonl"


def api_key() -> str:
    """OPENROUTER_API_KEY from the environment, else from the nearest .env file (never printed)."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for folder in [s.HERE, *s.HERE.parents]:
        env = folder / ".env"
        if env.exists():
            values = dict(
                line.split("=", 1)
                for line in env.read_text().splitlines()
                if "=" in line and not line.lstrip().startswith("#")
            )
            for name in ("OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY"):
                if values.get(name, "").strip():
                    return values[name].strip().strip('"')
    raise SystemExit("No OpenRouter key: set OPENROUTER_API_KEY.")


KEY = api_key()


def spent() -> float:
    return sum(json.loads(line)["cost"] for line in LEDGER.read_text().splitlines()) if LEDGER.exists() else 0.0


def ask(prompt: str, purpose: str) -> str:
    if spent() > args.cap - 0.01:
        raise SystemExit(f"Stopping: ${spent():.4f} spent, cap ${args.cap}.")
    body = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": 60,
        "usage": {"include": True},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            break
        except Exception as exc:  # retry network hiccups and rate limits
            if attempt == 3:
                raise
            print(f"   retry after {type(exc).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    usage = out.get("usage", {})
    with LEDGER.open("a") as f:
        f.write(
            json.dumps(
                {
                    "purpose": purpose,
                    "model": args.model,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cost": float(usage.get("cost") or 0),
                }
            )
            + "\n"
        )
    return out["choices"][0]["message"]["content"].strip()


def show(sn, mark=True):
    toks = [(f"«{t}»" if (mark and i == sn["at"]) else t) for i, t in enumerate(sn["tokens"])]
    return "".join(toks).replace("\n", "⏎").strip()


rng = np.random.default_rng(s.SEED)
latents = [lat for lat in interp["latents"] if not lat.get("chosen")]  # the random draw only
negatives = interp["negatives"]
labels, rows = {}, []
print(f"1. Labelling {len(latents)} latents with {args.model}")
for lat in latents:
    snippets = "\n".join(f"{i + 1}. {show(sn)}" for i, sn in enumerate(lat["top"][:10]))
    prompt = (
        "These are text snippets where one feature of a language model is most active. The token where it fires "
        "most is marked «like this».\n\n"
        + snippets
        + "\n\nIn at most 8 words, what does this feature detect? Answer with the label only."
    )
    labels[lat["id"]] = ask(prompt, f"label {lat['id']}").strip(" .\"'")
    print(f"   latent {lat['id']:>5}: {labels[lat['id']]}", flush=True)


def detect(lat, label):
    pos = lat["held_out"][:5]
    neg = [negatives[i] for i in rng.choice(len(negatives), size=5, replace=False)]
    items = [(sn, 1) for sn in pos] + [(sn, 0) for sn in neg]
    items = [items[i] for i in rng.permutation(len(items))]
    listing = "\n".join(f"{i + 1}. {show(sn, mark=False)}" for i, (sn, _) in enumerate(items))
    prompt = (
        f'A feature of a language model is described as: "{label}".\n\nWhich of these snippets contain what this '
        f"description is about? Exactly {len(pos)} of the {len(items)} do.\n\n{listing}\n\nAnswer with the "
        f"{len(pos)} numbers only, separated by commas."
    )
    reply = ask(prompt, f"detect {lat['id']}")
    picked = {int(n) for n in re.findall(r"\d+", reply) if 1 <= int(n) <= len(items)}
    return sum((i + 1 in picked) == bool(y) for i, (_, y) in enumerate(items)) / len(items)


print("\n2. Checking each label on 5 held-out firing snippets + 5 random ones; control: another latent's label")
ids = [lat["id"] for lat in latents]
for n, lat in enumerate(latents):
    acc = detect(lat, labels[lat["id"]])
    wrong_label = labels[ids[(n + 1) % len(ids)]]
    acc_wrong = detect(lat, wrong_label)
    rows.append(
        {
            "id": lat["id"],
            "label": labels[lat["id"]],
            "accuracy": acc,
            "wrong_label": wrong_label,
            "accuracy_wrong_label": acc_wrong,
        }
    )
    print(f"   latent {lat['id']:>5}: own label {acc:.0%}, another latent's label {acc_wrong:.0%}", flush=True)

acc = np.array([r["accuracy"] for r in rows])
ctrl = np.array([r["accuracy_wrong_label"] for r in rows])
diff = acc - ctrl
boot = np.random.default_rng(s.SEED).integers(0, len(rows), (1000, len(rows)))
summary = {
    "model": args.model,
    "n": len(rows),
    "accuracy": float(acc.mean()),
    "accuracy_ci": s.bootstrap_mean(acc),
    "control": float(ctrl.mean()),
    "control_ci": s.bootstrap_mean(ctrl),
    "difference": float(diff.mean()),
    "difference_ci": [float(np.percentile(diff[boot].mean(1), q)) for q in (2.5, 97.5)],
    "cost": spent(),
    "calls": len(LEDGER.read_text().splitlines()),
}
s.save_json(res / "autolabels.json", {"summary": summary, "latents": rows})
print(
    f"\n3. Mean detection accuracy {summary['accuracy']:.2f} {[round(x, 2) for x in summary['accuracy_ci']]} with the "
    f"latent's own label, {summary['control']:.2f} with another latent's label; difference {summary['difference']:+.2f} "
    f"{[round(x, 2) for x in summary['difference_ci']]}. {summary['calls']} calls, ${summary['cost']:.4f}."
)

print("\nSelf-checks:")
s.check(f"labels pass the detection bar (mean {summary['accuracy']:.2f} >= 0.70)", summary["accuracy"] >= 0.70)
s.check(f"and beat the wrong-label control by >= 0.15 ({summary['difference']:+.2f})", summary["difference"] >= 0.15)
s.check(f"spend under the ${args.cap:.2f} cap (${summary['cost']:.4f})", summary["cost"] < args.cap)
