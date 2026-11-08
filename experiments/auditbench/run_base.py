"""
Everything measured on the bare base model (Qwen3-14B), in three resumable stages:

--stage read (default)
  1. behaviour directions v_b: activations on the audit prompts under each of AuditBench's 14 quirk system prompts,
     and under the plain PRISM-4 prompt (the reference);
  2. the two probes, fitted here and only here: the sandbagging probe (32 toolkit contrast pairs) and the
     eval-awareness probe (eval vs deployment framings on quiz questions 61-150);
  3. the quiz (questions 1-60) under PRISM-4 and under the sandbagging prompt (the positive control).
--stage replies
  4. replies for the judge check: the base model under PRISM-4 and under each quirk prompt, greedy.
--stage noise
  5. noise calibration: base accuracy under PRISM-4 across noise strengths (one seed each), the two pre-registered
     strengths (base accuracy nearest 90% and 70% of clean), then the positive control at those strengths, 3 seeds.

Each step saves to the work folder as it finishes, so a restart resumes. results/base.json collects the numbers.

    .venv/bin/python experiments/auditbench/run_base.py [--stage read|replies|noise] [--smoke]
"""

from __future__ import annotations

import sys
import time

import numpy as np

import ab

SIGMA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
NOISE_SEEDS = (0, 1, 2)


def quiz_accuracy(letters: np.ndarray, qs: list[dict]) -> float:
    return float((letters.argmax(1) == np.array([q["answer"] for q in qs])).mean())


def noisy_accuracy(model, tok, system: str, qs: list[dict], sigma: float, seeds=NOISE_SEEDS) -> list[float]:
    texts = [ab.quiz_text(q) for q in qs]
    accs = []
    for seed in seeds:
        with ab.noise(model, ab.PRIMARY_LAYER, sigma, seed):
            accs.append(quiz_accuracy(ab.read(model, tok, system, texts)["letters"], qs))
    return accs


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Store:
    """Arrays in one .npz and numbers in base.json, both rewritten after every step."""

    def __init__(self, folder, smoke: bool):
        self.npz = folder / "readings.npz"
        self.json = (folder / "base.json") if smoke else (ab.RESULTS / "base.json")
        self.arrays = dict(np.load(self.npz)) if self.npz.exists() else {}
        self.summary = ab.read_json(self.json) if self.json.exists() else {}

    def save(self) -> None:
        np.savez(self.npz, **self.arrays)
        ab.write_json(self.json, self.summary)


def main() -> None:
    smoke = "--smoke" in sys.argv
    stage = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--stage=")), "read")
    t0 = time.time()
    out = ab.WORK / ("smoke" if smoke else "base")
    out.mkdir(parents=True, exist_ok=True)
    st = Store(out, smoke)
    tok = ab.load_tokenizer()
    org = ab.Organisms(ab.load_base())
    model = org.model
    print(f"loaded {ab.BASE_MODEL} on {ab.device()} in {time.time() - t0:.0f}s; stage {stage}", flush=True)
    prism = ab.prism_prompt()
    audit = ab.audit_prompts()[: 4 if smoke else None]
    test, train = ab.quiz("test"), ab.quiz("train")
    if smoke:
        test, train = test[:8], train[:8]
    st.summary |= {"layers": list(ab.LAYERS), "primary_layer": ab.PRIMARY_LAYER}
    li = ab.LAYERS.index(ab.PRIMARY_LAYER)

    if stage == "read":
        for name, system in [("prism", prism)] + [(b, ab.quirk_prompt(b)) for b in ab.BEHAVIOURS]:
            if f"audit_{name}_user" not in st.arrays:
                st.arrays[f"audit_{name}_user"] = ab.read(model, tok, system, audit)["user"]
                st.save()
                print(f"1 behaviour readings under the {name} prompt ({time.time() - t0:.0f}s)", flush=True)
        if "probe_eval" not in st.arrays:
            pairs = [
                p
                for c in ab.read_json(
                    ab.HERE.parent / "detector-sanity-checks" / "data" / "sandbagging_contrast_pairs.json"
                )["contrast_pairs"]
                for p in c["pairs"]
            ]
            sb = ab.read(model, tok, prism, [p["sandbagging"] for p in pairs])["last"]
            gn = ab.read(model, tok, prism, [p["genuine"] for p in pairs])["last"]
            st.arrays["probe_sandbag"] = unit((sb - gn).mean(0))
            from steerlab.monitors.logit import load_eval_awareness

            fr = load_eval_awareness()
            ev = [ab.quiz_text(q, fr.eval_framings[i % len(fr.eval_framings)]) for i, q in enumerate(train)]
            dp = [ab.quiz_text(q, fr.deployment_framings[i % len(fr.deployment_framings)]) for i, q in enumerate(train)]
            e_last, d_last = ab.read(model, tok, prism, ev)["last"], ab.read(model, tok, prism, dp)["last"]
            st.arrays["probe_eval"] = unit((e_last - d_last).mean(0))
            for name, pos, neg in (("sandbag", sb, gn), ("eval", e_last, d_last)):
                d = st.arrays[f"probe_{name}"][li]
                st.summary[f"probe_{name}_fit_gap"] = float((pos[:, li] @ d).mean() - (neg[:, li] @ d).mean())
            st.save()
            print(f"2 probes: {len(pairs)} sandbagging pairs, {len(train)} framing pairs ({time.time() - t0:.0f}s)")
        texts = [ab.quiz_text(q) for q in test]
        for name, system in (("prism", prism), ("sandbag", ab.sandbag_prompt())):
            if f"quiz_{name}_last" not in st.arrays:
                r = ab.read(model, tok, system, texts)
                st.arrays[f"quiz_{name}_last"], st.arrays[f"quiz_{name}_letters"] = r["last"], r["letters"]
                st.summary[f"quiz_acc_{name}"] = quiz_accuracy(r["letters"], test)
                st.save()
        print(
            f"3 quiz: base {st.summary['quiz_acc_prism']:.3f}, "
            f"prompted sandbagger {st.summary['quiz_acc_sandbag']:.3f} ({time.time() - t0:.0f}s)"
        )

    elif stage == "replies":
        path = out / "replies.json"
        data = ab.read_json(path) if path.exists() else {"prompts": audit, "replies": {}}
        for name, system in [("prism", prism)] + [(b, ab.quirk_prompt(b)) for b in ab.BEHAVIOURS]:
            if name not in data["replies"]:
                data["replies"][name] = ab.generate(org, tok, system, audit, alpha=0, use_base=True)
                ab.write_json(path, data)
                print(f"4 replies under the {name} prompt ({time.time() - t0:.0f}s)", flush=True)
                if smoke:
                    print("  ", data["replies"][name][0][:240])
                    break

    elif stage == "noise":
        clean = st.summary["quiz_acc_prism"]
        grid = st.summary.setdefault("noise_grid", {})
        for s in SIGMA_GRID[: 2 if smoke else None]:
            if str(s) not in grid:
                grid[str(s)] = noisy_accuracy(model, tok, prism, test, s, seeds=(0,))[0]
                st.save()
                print(f"5 noise grid: sigma {s} -> base accuracy {grid[str(s)]:.3f}", flush=True)
        chosen = [float(min(grid, key=lambda s: abs(grid[s] - f * clean))) for f in (0.9, 0.7)]
        st.summary["noise_sigmas"] = chosen
        st.summary["noise_sandbag"] = {str(s): noisy_accuracy(model, tok, ab.sandbag_prompt(), test, s) for s in chosen}
        st.save()
        print(f"5 noise: chosen {chosen}; prompted sandbagger {st.summary['noise_sandbag']}")

    st.summary[f"seconds_{stage}"] = round(time.time() - t0)
    st.save()
    print(f"stage {stage} done in {time.time() - t0:.0f}s -> {ab.short(out)}")


if __name__ == "__main__":
    main()
