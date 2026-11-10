"""Test `deslop voice`: can a rewrite take on a writer's voice while keeping the facts?

    python voice_eval.py check      # profiles, and the two attributors on held-out passages (known answers)
    python voice_eval.py rewrite    # 10 of Qwen's held-out replies x 6 voices x 2 rewriters, plus a no-voice control
    python voice_eval.py score      # facts, slop, Burrows' Delta, attribution -> results/voices/summary.json

Voices: five public-domain writers (2_voices.py) and a 50/50 blend of Twain and Darwin. A "qwen" profile, built from
the original model's replies to 30 training prompts, stands for the voice being rewritten away from.

Rewriters: gpt-5.4-mini (OpenAI API) and our de-slopped LoRA (on Modal), both given the same messages from
`deslop.voice.rewrite_messages`: three sample passages (two per writer for the blend), the profile in words, and the
text. The control rewrites the same texts in plain style with no samples (the plain-style prompt), to show how much of
any move toward a writer comes from de-slopping alone.

Two attributors, both checked on held-out passages first: nearest voice by Burrows' Delta (the toolkit's), and a
logistic regression on function-word z-scores, punctuation rates and sentence length (trained on the samples).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "packages" / "deslop" / "src"))
from deslop.habits import count_words, sentences
from deslop.score import score
from deslop.voice import (
    PUNCTUATION,
    attribute,
    blend,
    delta,
    fact_check,
    profile,
    rewrite_messages,
    word_rates,
    zscores,
)

from judge import load_env_key
from prompts import PLAIN_PROMPT

OUT = HERE / "results" / "voices"
PASSAGES = HERE / "data" / "voices" / "passages.jsonl"
AUTHORS = ["twain", "lincoln", "darwin", "woolf", "chesterton"]
BLEND = ("twain", "darwin")
INPUTS = ["h00-blog", "h01-explainer", "h02-opinion", "h03-blog", "h04-explainer", "h05-opinion", "h06-blog",
          "h07-explainer", "h08-opinion", "h09-linkedin"]  # fmt: skip
API_MODEL = "gpt-5.4-mini"
PRICE_IN, PRICE_OUT = 0.75, 4.50


def load() -> tuple[list[dict], list[str]]:
    rows = [json.loads(line) for line in PASSAGES.open(encoding="utf-8")]
    qwen = [json.loads(line)["text"] for line in (HERE / "results" / "gen" / "base-train.jsonl").open()]
    return rows, qwen


def profiles(rows: list[dict], qwen: list[str]) -> dict[str, dict]:
    out = {a: profile([r["text"] for r in rows if r["author"] == a and r["role"] == "sample"], a) for a in AUTHORS}
    out["qwen"] = profile(qwen[:30], "qwen")
    out["twain+darwin"] = blend([out[BLEND[0]], out[BLEND[1]]], name="twain+darwin")
    return out


def features(text: str) -> list[float]:
    words = max(count_words(text), 1)
    lengths = [count_words(s) for _, _, s in sentences(text)] or [0]
    punct = [1000.0 * text.count(v) / words for v in PUNCTUATION.values()]
    return [*zscores(word_rates(text)).values(), *punct, float(np.mean(lengths)), float(np.std(lengths))]


def attributor(rows: list[dict], qwen: list[str]) -> Callable[[str], str]:
    """Logistic regression on stylometric features, trained on the samples and Qwen's first 30 training replies."""
    from sklearn.linear_model import LogisticRegression

    texts = [r["text"] for r in rows if r["role"] == "sample"] + qwen[:30]
    labels = [r["author"] for r in rows if r["role"] == "sample"] + ["qwen"] * 30
    x = np.array([features(t) for t in texts])
    mu, sd = x.mean(0), x.std(0) + 1e-6
    clf = LogisticRegression(C=0.5, max_iter=5000).fit((x - mu) / sd, labels)
    return lambda text: str(clf.predict((np.array([features(text)]) - mu) / sd)[0])


def check() -> None:
    rows, qwen = load()
    voices = profiles(rows, qwen)
    candidates = [voices[a] for a in [*AUTHORS, "qwen"]]
    lr = attributor(rows, qwen)
    tests = [(r["author"], r["text"]) for r in rows if r["role"] == "heldout"] + [("qwen", t) for t in qwen[30:60]]
    results = [{"truth": truth, "delta": attribute(text, candidates)[0], "lr": lr(text)} for truth, text in tests]
    summary = {}
    for who in [*AUTHORS, "qwen"]:
        mine = [r for r in results if r["truth"] == who]
        summary[who] = {"n": len(mine), "delta_right": sum(r["delta"] == who for r in mine),
                        "lr_right": sum(r["lr"] == who for r in mine)}  # fmt: skip
    total = {"n": len(results), "delta_right": sum(r["delta"] == r["truth"] for r in results),
             "lr_right": sum(r["lr"] == r["truth"] for r in results), "chance": 1 / 6}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "profiles.json").write_text(json.dumps(voices, indent=1) + "\n")
    (OUT / "attribution.json").write_text(json.dumps({"by_voice": summary, "total": total}, indent=1) + "\n")
    print(json.dumps(summary), json.dumps(total))


def samples_for(target: str, rows: list[dict]) -> list[str]:
    def pick(author: str, k: int) -> list[str]:
        return [r["text"] for r in rows if r["author"] == author and r["role"] == "sample"][:k]

    return pick(BLEND[0], 2) + pick(BLEND[1], 2) if target == "twain+darwin" else pick(target, 3)


def jobs() -> list[dict]:
    rows, qwen = load()
    voices = profiles(rows, qwen)
    base = {r["id"]: r["text"] for r in map(json.loads, (HERE / "results" / "gen" / "base.jsonl").open())}
    out = []
    for sid in INPUTS:
        for target in [*AUTHORS, "twain+darwin"]:
            msgs = rewrite_messages(base[sid], voices[target], samples_for(target, rows))
            for w in ("api", "lora"):
                out.append({"id": f"{sid}/{target}/{w}", "source_id": sid, "target": target, "rewriter": w,
                            "messages": msgs})  # fmt: skip
        request = "Rewrite this text, keeping every fact, number and name:\n\n" + base[sid]
        control = [{"role": "system", "content": PLAIN_PROMPT}, {"role": "user", "content": request}]
        out.append(
            {"id": f"{sid}/none/api", "source_id": sid, "target": "none", "rewriter": "api", "messages": control}
        )
    return out


def rewrite() -> None:
    import modal
    from openai import OpenAI

    todo = jobs()
    client = OpenAI(api_key=load_env_key("OPENAI_API_KEY"))
    tokens = {"input": 0, "output": 0}

    def call(job: dict) -> dict:
        reply = client.chat.completions.create(model=API_MODEL, messages=job["messages"], reasoning_effort="low")
        tokens["input"] += reply.usage.prompt_tokens
        tokens["output"] += reply.usage.completion_tokens
        return {**job, "text": reply.choices[0].message.content.strip(), "finished": True}

    with ThreadPoolExecutor(max_workers=8) as pool:
        done = list(pool.map(call, [j for j in todo if j["rewriter"] == "api"]))
    lora_jobs = [j for j in todo if j["rewriter"] == "lora"]
    runner = modal.Cls.from_name("deslop", "Runner")(adapter="deslop-r2b")
    rows = [{"id": j["id"], "prompt": j["messages"][1]["content"]} for j in lora_jobs]
    job = {"rows": rows, "system": lora_jobs[0]["messages"][0]["content"], "n": 1, "temperature": 0.7,
           "max_new_tokens": 700, "seed": 0, "batch": 30}  # fmt: skip
    by_id = {g["id"]: g for g in runner.generate.remote(job)}
    done += [{**j, "text": by_id[j["id"]]["text"], "finished": by_id[j["id"]]["finished"]} for j in lora_jobs]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "rewrites.jsonl").open("w", encoding="utf-8") as f:
        for d in done:
            f.write(json.dumps({k: v for k, v in d.items() if k != "messages"}, ensure_ascii=False) + "\n")
    dollars = tokens["input"] / 1e6 * PRICE_IN + tokens["output"] / 1e6 * PRICE_OUT
    (OUT / "rewrite_cost.json").write_text(json.dumps({"tokens": tokens, "dollars": round(dollars, 3)}) + "\n")
    print(f"{len(done)} rewrites; API {tokens} ${dollars:.3f}")


def summarise(scored: list[dict], rewriter: str) -> dict:
    mine = [r for r in scored if r["rewriter"] == rewriter and r["target"] in AUTHORS]
    blends = [r for r in scored if r["rewriter"] == rewriter and r["target"] == "twain+darwin"]
    every = mine + blends
    return {
        "n": len(mine),
        "closer_to_target": sum(r["delta_target"] < r["delta_target_original"] for r in mine),
        "delta_target_mean": float(np.mean([r["delta_target"] for r in mine])),
        "delta_target_original_mean": float(np.mean([r["delta_target_original"] for r in mine])),
        "nearest_is_target": sum(r["nearest"] == r["target"] for r in mine),
        # Among the five writers only (the fair relative test; chance is 1 in 5, as for the original replies).
        "nearest_writer_is_target": sum(min(AUTHORS, key=lambda a, r=r: r["delta"][a]) == r["target"] for r in mine),
        "target_rank_mean": float(
            np.mean([sorted(AUTHORS, key=lambda a, r=r: r["delta"][a]).index(r["target"]) + 1 for r in mine])
        ),
        "lr_is_target": sum(r["lr_pred"] == r["target"] for r in mine),
        "still_qwen_delta": sum(r["nearest"] == "qwen" for r in mine),
        "still_qwen_lr": sum(r["lr_pred"] == "qwen" for r in mine),
        "n_with_blend": len(every),
        "facts_all_kept": sum(r["facts"]["all_kept"] for r in every),
        "numbers": sum(r["facts"]["numbers"] for r in every),
        "numbers_kept": sum(r["facts"]["numbers_kept"] for r in every),
        "names": sum(r["facts"]["names"] for r in every),
        "names_kept": sum(r["facts"]["names_kept"] for r in every),
        "negations": sum(r["facts"]["negations"] for r in every),
        "negations_rewrite": sum(r["facts"]["negations_rewrite"] for r in every),
        "in_band": sum(r["in_band"] for r in every),
        "deslop_mean": float(np.mean([r["deslop"] for r in every])),
        "words_ratio": float(np.mean([r["words"] / r["words_original"] for r in every])),
        "blend_n": len(blends),
        "blend_closer_than_parts": sum(
            r["delta_target"] < min(r["delta"]["twain"], r["delta"]["darwin"]) for r in blends
        ),
        "by_author": {
            a: {
                "nearest_is_target": sum(r["nearest"] == a for r in mine if r["target"] == a),
                "lr_is_target": sum(r["lr_pred"] == a for r in mine if r["target"] == a),
                "closer": sum(r["delta_target"] < r["delta_target_original"] for r in mine if r["target"] == a),
            }
            for a in AUTHORS
        },
    }


def score_all() -> None:
    rows, qwen = load()
    voices = json.loads((OUT / "profiles.json").read_text())
    lr = attributor(rows, qwen)
    candidates = [voices[a] for a in [*AUTHORS, "qwen"]]
    base = {r["id"]: r["text"] for r in map(json.loads, (HERE / "results" / "gen" / "base.jsonl").open())}
    scored = []
    for d in map(json.loads, (OUT / "rewrites.jsonl").open(encoding="utf-8")):
        src, text = base[d["source_id"]], d["text"]
        s = score(text)
        nearest, dist = attribute(text, candidates)
        row = {**d, "words": count_words(text), "words_original": count_words(src), "deslop": s.score,
               "in_band": bool(s.in_human_band), "facts": fact_check(src, text), "delta": dist, "nearest": nearest,
               "lr_pred": lr(text)}  # fmt: skip
        if d["target"] != "none":
            row["delta_target"] = delta(text, voices[d["target"]])
            row["delta_target_original"] = delta(src, voices[d["target"]])
        scored.append(row)
    with (OUT / "rewrites_scored.jsonl").open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    control = [r for r in scored if r["target"] == "none"]
    summary = {
        "api": summarise(scored, "api"),
        "lora": summarise(scored, "lora"),
        "control": {
            "n": len(control),
            "nearest_qwen": sum(r["nearest"] == "qwen" for r in control),
            "lr_qwen": sum(r["lr_pred"] == "qwen" for r in control),
            "nearest_author": sum(r["nearest"] in AUTHORS for r in control),
            "deslop_mean": float(np.mean([r["deslop"] for r in control])),
            "in_band": sum(r["in_band"] for r in control),
            "facts_all_kept": sum(r["facts"]["all_kept"] for r in control),
        },
        "original": {
            "nearest_writer_is_target": sum(
                min(AUTHORS, key=lambda a, s=s: delta(base[s], voices[a])) == t for s in INPUTS for t in AUTHORS
            ),
            "deslop_mean": float(np.mean([score(base[s]).score for s in INPUTS])),
            "nearest_qwen": sum(attribute(base[s], candidates)[0] == "qwen" for s in INPUTS),
            "lr_qwen": sum(lr(base[s]) == "qwen" for s in INPUTS),
        },
        "attribution_check": json.loads((OUT / "attribution.json").read_text())["total"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["check", "rewrite", "score"])
    {"check": check, "rewrite": rewrite, "score": score_all}[parser.parse_args().step]()


if __name__ == "__main__":
    main()
