"""Sample one model on the values battery (Modal) and score it (here, with the judge on Modal for two readouts).

    cd experiments/graft-monitor
    ../../.venv/bin/python -m battery.run_battery --model Qwen/Qwen3-0.6B --seeds 1 2 --tag qwen3-0.6b
    ../../.venv/bin/python -m battery.run_battery --model /vol/runs/gen1/merged --tag gen1
    ../../.venv/bin/python -m battery.run_battery --model Qwen/Qwen3-0.6B --lora /vol/runs/gen1/adapter --tag gen1
    ../../.venv/bin/python -m battery.run_battery --judge-check      # the judge's known-answer check only

Writes results/battery/<tag>_s<seed>/: replies.jsonl (battery), bdiff_replies.jsonl (behavior-diff's 128 default
prompts, sampled in the same container), judge_labels.<judge>.json, item_values.json and scores.json (every metric's mean
with a 95% bootstrap interval over items). Replies already on disk are not sampled again.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from battery import core, spend

GM = Path(__file__).resolve().parent.parent
REPO = GM.parent.parent
OUT = GM / "results" / "battery"
sys.path.insert(0, str(REPO / "packages" / "behavior-diff" / "src"))

EST_USD_PER_MODEL = 0.55  # the dearest GPU in the fallback list (L40S) for ~13 minutes; checked against the cap
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def slug(spec: str) -> str:
    return re.sub(
        r"[^a-z0-9.]+", "-", spec.lower().split("/", 1)[-1] if not spec.startswith("/") else spec.lower()
    ).strip("-")


def parse_spec(spec: str, base: str) -> tuple[str, str]:
    """(model, lora) from a spec: a Hub id, a Volume path (/vol/...), or lora:/vol/<adapter> on top of `base`."""
    if spec.startswith("lora:"):
        return base, spec[5:]
    return spec, ""


def bdiff_prompts() -> list[dict]:
    """behavior-diff's default suite under its default conditions, 2 draws per item, seed 0: 128 prompts."""
    from bdiff.prompts import expand
    from bdiff.quick import conditions, load_suite

    suite = load_suite("default")
    return expand({"items": suite["items"], "factors": conditions("default", suite), "draws": 2, "seed": 0})


def tag_dir(tag: str, seed: int) -> Path:
    return OUT / f"{tag}_s{seed}"


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sample_specs(specs: list[dict], items: list[dict] | None = None) -> None:
    """specs: [{"tag", "model", "lora", "seeds"}]. One Modal container per model, all in parallel; skips any model
    whose replies already exist for every seed."""
    os.environ.setdefault("MODAL_PROFILE", "albertlieyingadrian")
    items = items or core.load_items()
    todo = [s for s in specs if not all((tag_dir(s["tag"], k) / "replies.jsonl").exists() for k in s["seeds"])]
    if not todo:
        return
    spend.check(EST_USD_PER_MODEL * len(todo), f"sampling {len(todo)} model(s)")
    import modal

    from battery.modal_battery import app, sample

    prompts = bdiff_prompts()
    with modal.enable_output(), app.run():
        t0 = time.time()
        calls = [(s, sample.spawn(s["model"], items, prompts, s["seeds"], lora=s["lora"])) for s in todo]
        for s, call in calls:
            res = call.get()
            dt = time.time() - t0
            spend.log(
                dt / 3600 * spend.L40S_PER_HOUR,  # priced at the dearest fallback GPU
                f"sample {s['tag']} {s['model']}{' +' + s['lora'] if s['lora'] else ''} seeds {s['seeds']} "
                f"wall {dt:.0f}s (L4/A10G/L40S fallback, priced as L40S)",
            )
            for k in s["seeds"]:
                d = tag_dir(s["tag"], k)
                d.mkdir(parents=True, exist_ok=True)
                (d / "replies.jsonl").write_text(_jsonl(res[str(k)]["battery"]))
                (d / "bdiff_replies.jsonl").write_text(_jsonl(res[str(k)]["bdiff"]))
                (d / "meta.json").write_text(
                    json.dumps(
                        {
                            "tag": s["tag"],
                            "model": s["model"],
                            "lora": s["lora"],
                            "seed": k,
                            "temperature": core.TEMPERATURE,
                            "max_tokens": core.MAX_TOKENS,
                        },
                        indent=1,
                    )
                )
            print(f"sampled {s['tag']} seeds {s['seeds']} in {dt:.0f}s")


_EMBEDDER = None


def embed(texts: list[str]) -> list[list[float]]:
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _EMBEDDER.encode(texts, batch_size=64, show_progress_bar=False).tolist()


def judge_check(force: bool = False) -> dict:
    """The judge's known-answer result, run once and cached in results/battery/judge_check.json."""
    from battery import judge

    path = OUT / f"judge_check.{slug(judge.JUDGE_MODEL)}.json"
    if path.exists() and not force:
        res = json.loads(path.read_text())
        if res.get("judge") == judge.JUDGE_MODEL:
            return res
    j = judge.Judge(OUT / f"judge_check_cache.{slug(judge.JUDGE_MODEL)}.jsonl", max_tokens=32)
    j.warm()
    return judge.known_answer_check(j, path)


JUDGED = {"refusal": "ref-", "uncertainty": "unc-"}


def score_tag(d: Path, use_judge: bool = True, items: list[dict] | None = None) -> dict:
    """Score one sampled directory. Per judged readout, the scorer is the one that passed the known-answer check:
    the judge if it did, else the regex, else the judge marked unvalidated. Writes item_values.json, scores.json."""
    items = items or core.load_items()
    rows = read_jsonl(d / "replies.jsonl")
    labels: dict[str, str] = {}
    scorer = {ro: "regex" for ro in JUDGED}
    check = None
    if use_judge:
        from battery import judge

        check = judge_check()
        scorer = check["scorer"]
        lp = d / f"judge_labels.{slug(judge.JUDGE_MODEL)}.json"
        if lp.exists():
            all_labels = json.loads(lp.read_text())
        else:
            all_labels = judge.label_rows(judge.Judge(d / "judge_cache.jsonl", max_tokens=32), rows, items)
            lp.write_text(json.dumps(all_labels, indent=0))
        keep = tuple(JUDGED[ro] for ro, sc in scorer.items() if sc.startswith("judge"))
        labels = {k: v for k, v in all_labels.items() if k.startswith(keep)} if keep else {}
    vals = core.item_values(rows, items, labels, embed)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    validated = {ro: not sc.endswith("(unvalidated)") for ro, sc in scorer.items()}
    scores = {
        **meta,
        "scorer": scorer,
        "validated": validated,
        "judge_check": None
        if check is None
        else {k: check[k] for k in ("judge", "n", "correct", "accuracy", "by_readout")},
        "judge_labelled": len(labels),
        "embed_model": EMBED_MODEL,
        "metrics": core.summarize(vals),
    }
    (d / "item_values.json").write_text(json.dumps(vals))
    (d / "scores.json").write_text(json.dumps(scores, indent=1))
    return scores


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m battery.run_battery", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", help="Hub id or Volume path (/vol/...) of the model to sample")
    ap.add_argument("--lora", default="", help="LoRA adapter path in the Volume, applied on top of --model")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--tag", help="results/battery/<tag>_s<seed>/ (default: a slug of the model)")
    ap.add_argument("--no-judge", action="store_true", help="regex scorers only (no Modal judge)")
    ap.add_argument("--judge-check", action="store_true", help="run the judge's known-answer check and exit")
    a = ap.parse_args(argv)
    if a.judge_check:
        res = judge_check(force=True)
        from battery import judge

        judge.log_judge_cost("known-answer check")
        print(json.dumps({k: res[k] for k in ("judge", "n", "scorer", "by_readout", "by_set")}, indent=1))
        return
    if not a.model:
        ap.error("--model is required")
    tag = a.tag or slug(a.model + ("-" + a.lora if a.lora else ""))
    sample_specs([{"tag": tag, "model": a.model, "lora": a.lora, "seeds": a.seeds}])
    for k in a.seeds:
        s = score_tag(tag_dir(tag, k), use_judge=not a.no_judge)
        print(f"{tag}_s{k} (scorers {s['scorer']}):")
        for m in core.PRIMARY:
            v = s["metrics"].get(m)
            if v:
                print(f"  {m:40s} {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] n={v['n']}")
    if not a.no_judge:
        from battery import judge

        judge.log_judge_cost(f"score {tag}")


if __name__ == "__main__":
    main()
