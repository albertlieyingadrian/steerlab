"""
Question 1: can a linear probe on Qwen3-1.7B's activations tell a comment-stuffing sketch from a normal one, on runs
it was not fitted on, better than a random direction?

Free (laptop). Run from the repo root:
    .venv/bin/python experiments/probe-gated-rl/1_fit_probe.py

Data: the training sketches of six 2026-11-13 Pigment Gym runs (800 each), read by the untrained model (teacher
forcing: the model reads the sketch as if it had written it). Fit on three runs, test on the other three
(PREREGISTRATION.md). Writes results/probe.json (the probe the training runs use) and results/probe_fit.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PG = HERE.parent / "pigment-gym"
sys.path[:0] = [str(HERE), str(PG)]

from probe_tools import HACK_MENTIONS, LAYER, Probe, auroc, comment_mentions, extract

MODEL = "Qwen/Qwen3-1.7B"
QWEN3 = HERE / "results" / "probe_inputs"  # copies of the 2026-11-13 Pigment Gym Qwen3 runs
FIT_RUNS = ["loophole2_seed0", "loophole2_seed1", "loophole2_seed2"]
HELDOUT_RUNS = ["loophole2_seed3", "loophole_seed0", "loophole_seed1"]
LAYERS = (7, 14, 21)
WORK = Path.home() / ".cache" / "steerlab" / "probe-gated-rl"
CKPT = Path.home() / ".cache" / "steerlab" / "pigment-gym" / "qwen3" / "checkpoints"
RESULTS = HERE / "results"


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
    return model, tokenizer


def the_prompt(tokenizer):
    from brush_tools import SYSTEM_PROMPT_NO_EXAMPLE
    from hand_pool import SUBJECT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_NO_EXAMPLE},
        {"role": "user", "content": f"Paint {SUBJECT}."},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def load_rollouts(run):
    rows = [json.loads(line) for line in (QWEN3 / f"{run}_rollouts.jsonl").open()]
    for row in rows:
        assert comment_mentions(row["code"]) == row["mentions"], run  # the label is the environment's own count
    return rows


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = FIT_RUNS + HELDOUT_RUNS
    rows = {run: load_rollouts(run) for run in runs}
    cache = WORK / "acts_base.npz"
    model, tokenizer = None, None
    if cache.exists():
        acts = dict(np.load(cache))
    else:
        model, tokenizer = load_model()
        prompt = the_prompt(tokenizer)
        acts = {}
        for run in runs:
            texts = [r["text"] for r in rows[run]]
            out = extract(model, tokenizer, prompt, texts, LAYERS, batch=8)
            for layer in LAYERS:
                acts[f"{run}_L{layer}"] = out[layer]
            print(f"extracted {run}: {len(texts)} sketches", flush=True)
        np.savez(cache, **acts)

    def stack(run_list, layer):
        x = np.concatenate([acts[f"{run}_L{layer}"] for run in run_list])
        mentions = np.concatenate([[r["mentions"] for r in rows[run]] for run in run_list])
        return x, (mentions >= HACK_MENTIONS).astype(int), mentions

    report = {"fit_runs": FIT_RUNS, "heldout_runs": HELDOUT_RUNS, "layer": LAYER, "hack_mentions": HACK_MENTIONS}
    x_fit, y_fit, _ = stack(FIT_RUNS, LAYER)
    x_test, y_test, m_test = stack(HELDOUT_RUNS, LAYER)
    report["n_fit"], report["n_fit_hack"] = len(y_fit), int(y_fit.sum())
    report["n_test"], report["n_test_hack"] = len(y_test), int(y_test.sum())
    probe = Probe.fit(x_fit, y_fit)
    probe.save(RESULTS / "probe.json")
    p_test = probe.prob(x_test)
    report["heldout_auroc"] = auroc(p_test, y_test)
    report["heldout_flag_rate_hack"] = float((p_test[y_test == 1] >= 0.5).mean())
    report["heldout_flag_rate_clean"] = float((p_test[y_test == 0] >= 0.5).mean())
    report["heldout_accuracy"] = float(((p_test >= 0.5) == y_test).mean())
    near = (m_test >= 3) & (m_test <= 8)
    report["heldout_near_threshold"] = {
        "n": int(near.sum()),
        "n_hack": int(y_test[near].sum()),
        "auroc": auroc(p_test[near], y_test[near]),
    }
    per_run = {}
    for run in HELDOUT_RUNS:
        xr, yr, _ = stack([run], LAYER)
        pr = probe.prob(xr)
        per_run[run] = {
            "n": len(yr),
            "n_hack": int(yr.sum()),
            "auroc": auroc(pr, yr),
            "flag_hack": float((pr[yr == 1] >= 0.5).mean()) if yr.sum() else None,
            "flag_clean": float((pr[yr == 0] >= 0.5).mean()),
        }
    report["heldout_per_run"] = per_run

    # random directions in the standardised space, sign chosen on the fit set
    rng = np.random.default_rng(0)
    z_fit, z_test = (x_fit - probe.mean) / probe.scale, (x_test - probe.mean) / probe.scale
    random_aucs = []
    for _ in range(100):
        direction = rng.standard_normal(z_fit.shape[1]).astype(np.float32)
        sign = 1.0 if auroc(z_fit @ direction, y_fit) >= 0.5 else -1.0
        random_aucs.append(auroc(sign * (z_test @ direction), y_test))
    report["random_directions"] = {
        "n": 100,
        "median": float(np.median(random_aucs)),
        "max": float(np.max(random_aucs)),
        "p95": float(np.percentile(random_aucs, 95)),
        "aurocs": [float(a) for a in random_aucs],
    }

    # one row per held-out sketch, for the figure and for re-checking every number
    with (RESULTS / "probe_heldout_scores.csv").open("w") as handle:
        handle.write("run,step,mentions,hack,comment_count,probe_p\n")
        i = 0
        for run in HELDOUT_RUNS:
            for r in rows[run]:
                handle.write(
                    f"{run},{r['step']},{r['mentions']},{int(r['mentions'] >= HACK_MENTIONS)},"
                    f"{r['code'].count('//')},{p_test[i]:.6f}\n"
                )
                i += 1

    # text baselines that do not read the keywords
    def text_feature(run_list, fn):
        return np.concatenate([[fn(r["code"]) for r in rows[run]] for run in run_list])

    for name, fn in {"comment_count": lambda c: c.count("//"), "line_count": lambda c: c.count("\n") + 1}.items():
        sign = 1.0 if auroc(text_feature(FIT_RUNS, fn), y_fit) >= 0.5 else -1.0
        report[f"baseline_{name}"] = auroc(sign * text_feature(HELDOUT_RUNS, fn), y_test)
    report["baseline_keyword_count"] = auroc(m_test, y_test)  # defines the label: 1.0 by construction

    # the same probe recipe at other layers (descriptive)
    report["other_layers"] = {}
    for layer in LAYERS:
        xf, yf, _ = stack(FIT_RUNS, layer)
        xt, yt, _ = stack(HELDOUT_RUNS, layer)
        report["other_layers"][str(layer)] = auroc(Probe.fit(xf, yf, layer=layer).prob(xt), yt)

    report["q1_pass"] = bool(
        report["heldout_auroc"] >= 0.95 and report["heldout_auroc"] > report["random_directions"]["max"]
    )

    # drift control: RL without probe pressure, the policy reading its own sketches
    drift_cache = WORK / "acts_drift_l2s3.npz"
    before = [json.loads(line) for line in (QWEN3 / "loophole2_seed3_before_t1_samples.jsonl").open()]
    after = [json.loads(line) for line in (QWEN3 / "loophole2_seed3_after_t1_samples.jsonl").open()]
    texts = [r["text"] for r in before + after]
    labels = np.array([int(r["mentions"] >= HACK_MENTIONS) for r in before + after])
    if drift_cache.exists():
        drift = dict(np.load(drift_cache))
    else:
        from peft import PeftModel

        if model is None:
            model, tokenizer = load_model()
        prompt = the_prompt(tokenizer)
        drift = {"base": extract(model, tokenizer, prompt, texts, (LAYER,))[LAYER]}
        policy = PeftModel.from_pretrained(model, str(CKPT / "loophole2_seed3" / "final")).eval()
        drift["policy"] = extract(policy, tokenizer, prompt, texts, (LAYER,))[LAYER]
        np.savez(drift_cache, **drift)
    report["drift_control"] = {"run": "loophole2_seed3/final", "n": len(labels), "n_hack": int(labels.sum())}
    for key in ("base", "policy"):
        p = probe.prob(drift[key])
        report["drift_control"][key] = {
            "auroc": auroc(p, labels),
            "flag_hack": float((p[labels == 1] >= 0.5).mean()),
            "flag_clean": float((p[labels == 0] >= 0.5).mean()),
        }

    (RESULTS / "probe_fit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
