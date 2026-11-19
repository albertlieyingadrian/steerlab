"""Plurality measures shared by the pluralism series (Episode 2 defines them; Episode 3 reuses them).

Pure numpy/pandas; no GPU. Import from anywhere in the repo:

    import sys; sys.path.insert(0, "experiments/pluralism-replications")
    import measures as M

The four families of measure, and what each takes:

1. Repetition and homogeneity (Artificial Hivemind, claims C and D)
   - ``embed_texts(texts)``                       -> unit-norm embeddings (OpenAI text-embedding-3-small, cached)
   - ``repetition(emb)``                          -> mean pairwise cosine of one model's samples on one prompt
   - ``cross_similarity(emb_a, emb_b)``           -> mean cosine between two models' samples on one prompt
   - ``repetition_table(df, emb)`` / ``cross_table(df, emb)`` -> per (model, prompt) and per (pair, prompt) tables
   - ``cross_prompt_floor(df, emb)``              -> similarity of a model's answers to *different* prompts

2. Coverage (Spectrum Tuning, claim B)
   - ``coverage_stats(generations, validity)``    -> validity, diversity, yield (distinct valid outputs)

3. Steerability (Spectrum Tuning, claim A)
   - ``steerability_per_instance(losses)``        -> per-instance loss on outputs k >= 1, first-output loss, context gain
   - ``steerability_summary(per_instance)``       -> task-level means

4. Judge calibration against many raters (Artificial Hivemind, claim E)
   - ``label_entropy(labels)``, ``split_half_reliability(label_lists)``, ``loo_rater_r(label_lists)``

Plus the noise floor used by every verdict:
   - ``bootstrap_ci(...)``, ``two_level_bootstrap(...)``, ``cluster_bootstrap(...)``, ``verdict(lo, hi)``

Conventions: similarities are cosine in [-1, 1]; every interval is a 95% percentile bootstrap with 2,000 resamples by
default; ``verdict`` returns "replicated" / "not replicated" / "null" for a claim whose predicted sign is ``direction``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 2000
EMBED_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------------------------------------------------


def _key(text: str, model: str) -> str:
    return hashlib.sha1(f"{model}\x00{text}".encode()).hexdigest()


def embed_texts(
    texts: Sequence[str],
    model: str = EMBED_MODEL,
    cache_path: str | os.PathLike | None = None,
    batch_size: int = 256,
    client=None,
    max_chars: int = 24000,
) -> np.ndarray:
    """Embed texts with an OpenAI embedding model and return an (n, d) float32 array of unit vectors.

    A JSONL cache (one ``{"k": sha1, "e": [...]}`` per line) makes reruns free. Empty strings embed as a zero vector,
    so they score 0 against everything; count them separately if they matter. Pass an OpenAI-compatible ``client``;
    by default the call goes through OpenRouter (``OPENROUTER_API_KEY``, model ``openai/<model>``), else OpenAI
    (``OPENAI_API_KEY``). The cache key uses the bare model name, so both routes share it. Usage tokens and USD cost
    (when the API reports it) are added to ``embed_texts.tokens_used`` and ``embed_texts.usd_used``.
    """
    cache: dict[str, list[float]] = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            for line in f:
                row = json.loads(line)
                cache[row["k"]] = row["e"]
    keys = [_key(t, model) for t in texts]
    todo = sorted({(k, t) for k, t in zip(keys, texts) if k not in cache and t.strip()})
    if todo:
        api_model = model
        if client is None:
            from openai import OpenAI

            key = os.environ.get("OPENROUTER_API_KEY_2") or os.environ.get("OPENROUTER_API_KEY")
            if key:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
                api_model = f"openai/{model}"
            else:
                client = OpenAI()
        new_rows = []
        for i in range(0, len(todo), batch_size):
            chunk = todo[i : i + batch_size]
            resp = client.embeddings.create(model=api_model, input=[t[:max_chars] for _, t in chunk])
            embed_texts.tokens_used += getattr(resp.usage, "total_tokens", 0) or 0
            embed_texts.usd_used += float(getattr(resp.usage, "cost", 0) or 0)
            for (k, _), d in zip(chunk, resp.data):
                cache[k] = d.embedding
                new_rows.append({"k": k, "e": d.embedding})
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "a") as f:
                for row in new_rows:
                    f.write(json.dumps(row) + "\n")
    dim = len(next(iter(cache.values()))) if cache else 1536
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, k in enumerate(keys):
        if k in cache:
            out[i] = cache[k]
    return normalise(out)


embed_texts.tokens_used = 0
embed_texts.usd_used = 0.0


def normalise(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms == 0, 1.0, norms)


# ---------------------------------------------------------------------------------------------------------------------
# Repetition and homogeneity
# ---------------------------------------------------------------------------------------------------------------------


def repetition(emb: np.ndarray) -> float:
    """Mean pairwise cosine among n unit vectors (n >= 2): one model's samples on one prompt."""
    n = len(emb)
    if n < 2:
        return float("nan")
    gram = emb @ emb.T
    return float((gram.sum() - np.trace(gram)) / (n * (n - 1)))


def cross_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Mean cosine over all (a, b) pairs: two models' samples on the same prompt."""
    if len(emb_a) == 0 or len(emb_b) == 0:
        return float("nan")
    return float((emb_a @ emb_b.T).mean())


def repetition_table(df: pd.DataFrame, emb: np.ndarray, model_col="model", prompt_col="prompt_id") -> pd.DataFrame:
    """Per (model, prompt) repetition. ``df`` has one row per sample, positionally aligned with ``emb`` rows."""
    rows = []
    for (m, p), idx in df.reset_index(drop=True).groupby([model_col, prompt_col]).indices.items():
        rows.append({model_col: m, prompt_col: p, "n": len(idx), "repetition": repetition(emb[idx])})
    return pd.DataFrame(rows)


def cross_table(df: pd.DataFrame, emb: np.ndarray, model_col="model", prompt_col="prompt_id") -> pd.DataFrame:
    """Per (model pair, prompt) cross-model similarity, for every unordered pair of distinct models."""
    df = df.reset_index(drop=True)
    groups = df.groupby([model_col, prompt_col]).indices
    models = sorted(df[model_col].unique())
    prompts = sorted(df[prompt_col].unique())
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            for p in prompts:
                ia, ib = groups.get((a, p)), groups.get((b, p))
                if ia is None or ib is None:
                    continue
                rows.append({"model_a": a, "model_b": b, prompt_col: p, "cross": cross_similarity(emb[ia], emb[ib])})
    return pd.DataFrame(rows)


def cross_prompt_floor(
    df: pd.DataFrame, emb: np.ndarray, model_col="model", prompt_col="prompt_id", n_pairs=5000, seed=0
) -> pd.DataFrame:
    """Per model: mean cosine between random pairs of its samples drawn from *different* prompts (the topic floor)."""
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = []
    for m, sub in df.groupby(model_col):
        idx = sub.index.to_numpy()
        prompts = sub[prompt_col].to_numpy()
        i = rng.integers(0, len(idx), n_pairs)
        j = rng.integers(0, len(idx), n_pairs)
        keep = prompts[i] != prompts[j]
        sims = np.einsum("ij,ij->i", emb[idx[i[keep]]], emb[idx[j[keep]]])
        rows.append({model_col: m, "floor": float(sims.mean()), "n_pairs": int(keep.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------------------------------------------------


def coverage_stats(generations: Sequence[str], validity: Sequence[bool]) -> dict:
    """Spectrum Tuning's diversity/validity statistics for one prompt (exact-string deduplication).

    - validity: share of generations that pass the task's validator;
    - diversity: distinct valid generations / valid generations;
    - yield: number of distinct valid generations (the coverage measure; higher = more of the valid space reached).
    """
    gens = list(generations)
    valid = [g for g, v in zip(gens, validity) if v]
    n = len(gens)
    distinct = len(set(valid))
    return {
        "n": n,
        "validity": len(valid) / n if n else float("nan"),
        "diversity": distinct / len(valid) if valid else float("nan"),
        "yield": distinct,
        "yield_per_100": 100 * distinct / n if n else float("nan"),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Steerability
# ---------------------------------------------------------------------------------------------------------------------


def steerability_per_instance(losses: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-output losses into per-instance steerability readouts.

    ``losses`` has one row per scored output (Spectrum's ``raw_results``): columns ``instance_ind``, ``example_ind``
    (0 = the first output, before any in-context example), ``loss`` (summed nats over the output's tokens) and,
    optionally, ``all_acc`` (1 if every token was the argmax). Returns per instance:
    - ``loss_geq_1``: mean loss on outputs that follow at least one example (the paper's steerability loss);
    - ``loss_first``: loss on the first output (the prior);
    - ``context_gain``: loss_first - loss_geq_1 (how much the examples helped);
    - ``acc_geq_1``: mean all-token accuracy on outputs k >= 1, when available.
    """
    later = losses[losses.example_ind >= 1].groupby("instance_ind")
    out = pd.DataFrame({"loss_geq_1": later.loss.mean()})
    out["loss_first"] = losses[losses.example_ind == 0].groupby("instance_ind").loss.mean()
    out["context_gain"] = out.loss_first - out.loss_geq_1
    if "all_acc" in losses:
        out["acc_geq_1"] = later.all_acc.mean()
    out["n_outputs"] = losses.groupby("instance_ind").size()
    return out.reset_index()


def steerability_summary(per_instance: pd.DataFrame) -> dict:
    cols = [c for c in ("loss_geq_1", "loss_first", "context_gain", "acc_geq_1") if c in per_instance]
    return {c: float(per_instance[c].mean()) for c in cols} | {"n_instances": len(per_instance)}


# ---------------------------------------------------------------------------------------------------------------------
# Judge calibration against many raters
# ---------------------------------------------------------------------------------------------------------------------


def label_entropy(labels: Sequence[int], k: int = 5) -> float:
    """Shannon entropy (bits) of one item's ratings on a 1..k scale."""
    counts = np.bincount(np.asarray(labels, dtype=int) - 1, minlength=k).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def split_half_reliability(label_lists: Sequence[Sequence[int]], n_splits: int = 200, seed: int = 0) -> dict:
    """Reliability of the full-panel mean rating over a set of items.

    Each split shuffles every item's raters into two halves; r_half = Pearson r across items between the two half
    means. Spearman-Brown gives rho = 2 r_half / (1 + r_half), the reliability of the full mean; sqrt(rho) is the
    highest correlation any rater or judge can reach against that noisy mean. Returns the means over splits.
    """
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(x, dtype=float) for x in label_lists]
    rs = []
    for _ in range(n_splits):
        a, b = [], []
        for x in arrs:
            perm = rng.permutation(len(x))
            h = len(x) // 2
            a.append(x[perm[:h]].mean())
            b.append(x[perm[h:]].mean())
        rs.append(pearson(np.array(a), np.array(b)))
    r_half = float(np.nanmean(rs))
    rho = 2 * r_half / (1 + r_half) if r_half > -1 else float("nan")
    return {"r_half": r_half, "rho": rho, "ceiling": float(np.sqrt(max(rho, 0.0)))}


def loo_rater_r(label_lists: Sequence[Sequence[int]], n_draws: int = 200, seed: int = 0) -> float:
    """One-rater reference: r between one randomly drawn rater per item and the mean of the item's other raters."""
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(x, dtype=float) for x in label_lists]
    rs = []
    for _ in range(n_draws):
        one, rest = [], []
        for x in arrs:
            i = rng.integers(len(x))
            one.append(x[i])
            rest.append(np.delete(x, i).mean())
        rs.append(pearson(np.array(one), np.array(rest)))
    return float(np.nanmean(rs))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------------------------------------------------
# Noise floor and verdicts
# ---------------------------------------------------------------------------------------------------------------------


def percentile_ci(samples: Iterable[float], alpha: float = 0.05) -> tuple[float, float]:
    s = np.asarray([v for v in samples if not np.isnan(v)])
    if len(s) == 0:
        return (float("nan"), float("nan"))
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def bootstrap_ci(values: Sequence[float], stat: Callable = np.mean, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """One-level bootstrap of ``stat`` over items."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    boots = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
    lo, hi = percentile_ci(boots)
    return {"est": float(stat(v)), "lo": lo, "hi": hi, "n": len(v)}


def two_level_bootstrap(groups: dict, stat_fn: Callable[[dict], float], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Resample groups (prompts/tasks), then rows within each drawn group (samples/generations).

    ``groups`` maps group id -> numpy array (rows resampled) or dict of numpy arrays (each array resampled on its own,
    e.g. several models' samples on one prompt). ``stat_fn`` receives {draw_index: resampled value} and returns a float.
    """
    rng = np.random.default_rng(seed)
    keys = list(groups)

    def resample(val):
        if isinstance(val, dict):
            return {k: v[rng.integers(0, len(v), len(v))] for k, v in val.items()}
        return val[rng.integers(0, len(val), len(val))]

    boots = []
    for _ in range(n_boot):
        drawn = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        boots.append(stat_fn({j: resample(groups[g]) for j, g in enumerate(drawn)}))
    lo, hi = percentile_ci(boots)
    return {"est": float(stat_fn(dict(enumerate(groups[g] for g in keys)))), "lo": lo, "hi": hi, "n_groups": len(keys)}


def cluster_bootstrap(
    df: pd.DataFrame, cluster_col: str, stat_fn: Callable[[pd.DataFrame], float], n_boot: int = N_BOOT, seed: int = 0
) -> dict:
    """Resample whole clusters (e.g. the 50 prompts of the rated set) and recompute ``stat_fn`` on the stacked rows."""
    rng = np.random.default_rng(seed)
    parts = {c: sub for c, sub in df.groupby(cluster_col)}
    keys = list(parts)
    boots = []
    for _ in range(n_boot):
        drawn = rng.integers(0, len(keys), len(keys))
        boots.append(stat_fn(pd.concat([parts[keys[i]] for i in drawn], ignore_index=True)))
    lo, hi = percentile_ci(boots)
    return {"est": float(stat_fn(df)), "lo": lo, "hi": hi, "n_clusters": len(keys)}


def verdict(lo: float, hi: float, direction: int = 1) -> str:
    """Pre-registered rule: the paper's direction with the 95% interval excluding zero."""
    if np.isnan(lo) or np.isnan(hi):
        return "null"
    if direction * lo > 0 and direction * hi > 0:
        return "replicated"
    if direction * lo < 0 and direction * hi < 0:
        return "not replicated"
    return "null"
