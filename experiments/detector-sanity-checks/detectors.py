"""
Shared helpers for "Sanity checks for lie detectors": every script in this folder does `import detectors as t`.

What lives here, in the order the scripts need it:
- where things are: the released data, example sets, activations and fitted probes go to a work folder outside the
  repository (~/.cache/steerlab/detector-sanity-checks/, or $DETECTOR_SANITY_WORK); small results (IDs, labels,
  scores, summaries) go to results/;
- the example format: one JSON line per transcript, {"id", "set", "messages", "prefix", "response", "label",
  "instructed", "group", ...}. `label` is the behaviour (1 = lied or sandbagged), `instructed` is whether the prompt
  told or pushed the model to do it. The 2x2 is just these two columns crossed;
- reading activations: one forward pass per transcript, four readings per layer (the response mean, the prompt
  mean, the last prompt token, the first content token);
- the probe: standardise, then logistic regression (Apollo's recipe);
- scoring: AUROC with a bootstrap interval, cross-validated text baselines, random directions.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
WORK = Path(os.environ.get("DETECTOR_SANITY_WORK", Path.home() / ".cache" / "steerlab" / "detector-sanity-checks"))
OUT = WORK
RAW = WORK / "raw"
SETS = WORK / "sets"
ACTS = WORK / "acts"
# Every script writes its small results to results/ next to this file (the lessons this was ported from each had one).
LESSON_01 = LESSON_02 = LESSON_03 = LESSON_04 = NOTE = HERE
REPO = HERE.parents[1]
MAIN = REPO


def check(name: str, condition) -> bool:
    """Print a ✅/❌ self-check line and return whether it passed."""
    print(f"  {'✅' if condition else '❌'} {name}")
    return bool(condition)


LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_LAYERS = (8, 12, 16, 20)
"""Pre-registered: four of Llama-3.1-8B's 32 layers. Apollo read layer 22 of 80 on the 70B model (about 28% deep)."""
QWEN_LAYERS = (7, 10, 14, 17)
"""The same fractions of Qwen2.5-1.5B's 28 layers, for the deception-probe recipe refitted on the password-locked
organism."""
POOLS = ("resp_mean", "prompt_mean", "last_prompt", "first_content")
"""The four readings: mean over response tokens (the detector's own reading), mean over prompt tokens, the last
prompt token (what the model knows before it writes anything) and the first content token (it can only see one word)."""
C_PROBE = 0.1
"""Logistic-regression C on standardised features (small C = strong L2, as in Apollo's reg_coeff = 10)."""
SEED = 0

DECEPTION_WORDS = (
    "lie",
    "lying",
    "lies",
    "deceive",
    "deceptive",
    "deception",
    "dishonest",
    "pretend",
    "mislead",
    "misleading",
    "sandbag",
    "sandbagging",
    "hide",
    "secret",
    "false",
    "fake",
    "excuse",
)
"""The pre-registered keyword list. Counted as whole words, case-insensitive, in the whole transcript."""


# ---- files ----


def rel(path: Path) -> str:
    """A path relative to the repo or the work folder, so printed lines never show a home folder."""
    try:
        return "$WORK/" + str(Path(path).resolve().relative_to(WORK.resolve()))
    except ValueError:
        pass
    for root in (REPO, MAIN):
        try:
            return str(Path(path).resolve().relative_to(root))
        except ValueError:
            continue
    return Path(path).name


def results_dir(lesson: Path) -> Path:
    folder = lesson / "results"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def figures_dir(lesson: Path) -> Path:
    folder = lesson / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_set(name: str) -> list[dict]:
    path = SETS / f"{name}.jsonl"
    if not path.is_file():
        raise SystemExit(f"{rel(path)} is missing: run build_sets.py first")
    return read_jsonl(path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---- models ----


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_llama():
    """Llama-3.1-8B-Instruct in bfloat16 on MPS (about 16 GB). Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLAMA)
    model = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.bfloat16).to(device()).eval()
    return model, tok


# ---- rendering a transcript ----


def render(tok, ex: dict) -> tuple[str, int, int]:
    """The transcript as the model sees it: chat template for the prompt, then the prefix and the response.

    Returns (text, where the response starts, where the first message's content starts), as character offsets.
    `ex["raw"]` (episode 1's toolkit prompts) skips the template: the text is the prompt itself and has no response.
    """
    if ex.get("raw"):
        text = ex["messages"][0]["content"]
        return text, len(text), 0
    prompt = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
    prompt += ex.get("prefix", "")
    first = ex["messages"][0]["content"]
    content_start = prompt.find(first[:40]) if first else -1
    return prompt + ex.get("response", ""), len(prompt), max(content_start, 0)


def token_spans(tok, ex: dict) -> tuple[list[int], int, int, int]:
    """Token ids plus the first response token, the first content token, and how many response tokens to drop at the
    end."""
    text, resp_char, content_char = render(tok, ex)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    resp_tok = next((i for i, (a, _) in enumerate(offsets) if a >= resp_char), len(ids))
    content_tok = next((i for i, (a, b) in enumerate(offsets) if b > content_char), 0)
    return ids, resp_tok, content_tok, int(ex.get("drop_last", 0))


def extract(
    model,
    tok,
    examples: Sequence[dict],
    layers: Sequence[int],
    *,
    label: str = "",
    every: int = 50,
    max_tokens: int = 6000,
) -> dict[str, np.ndarray]:
    """One forward pass per transcript; for each layer, the four readings in POOLS, as float16 [N, n_layers, d].

    hidden_states[L] is the residual stream after block L (hidden_states[0] is the embeddings). A transcript with no
    response (a raw prompt) gets its last token as "resp_mean" too, so every reading exists for every row.
    """
    import torch

    out = {p: [] for p in POOLS}
    n_tokens = []
    start = time.time()
    dev = next(model.parameters()).device
    for i, ex in enumerate(examples):
        ids, r, c, drop = token_spans(tok, ex)
        if len(ids) > max_tokens:  # keep the end, where the response is
            cut = len(ids) - max_tokens
            ids, r, c = ids[cut:], max(r - cut, 1), 0
        n_tokens.append(len(ids))
        with torch.no_grad():
            hs = model(torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states
        resp_end = max(len(ids) - drop, r + 1) if r < len(ids) else len(ids)
        for p in POOLS:
            out[p].append([])
        for layer in layers:
            h = hs[layer][0].float()
            resp = h[r:resp_end].mean(0) if r < len(ids) else h[-1]
            prompt_end = min(r, len(ids))
            out["resp_mean"][-1].append(resp.cpu().numpy())
            out["prompt_mean"][-1].append(h[c:prompt_end].mean(0).cpu().numpy())
            out["last_prompt"][-1].append(h[prompt_end - 1].cpu().numpy())
            out["first_content"][-1].append(h[c].cpu().numpy())
        del hs
        if every and (i + 1) % every == 0:
            rate = (time.time() - start) / (i + 1)
            print(
                f"    {label} {i + 1}/{len(examples)}  {rate:.2f} s/transcript, "
                f"about {rate * (len(examples) - i - 1) / 60:.1f} min left",
                flush=True,
            )
    arrays = {p: np.asarray(v, dtype=np.float16) for p, v in out.items()}
    arrays["n_tokens"] = np.asarray(n_tokens)
    return arrays


def save_acts(name: str, arrays: dict[str, np.ndarray], layers: Sequence[int]) -> Path:
    ACTS.mkdir(parents=True, exist_ok=True)
    path = ACTS / f"{name}.npz"
    np.savez(path, layers=np.asarray(layers), **arrays)
    return path


def load_acts(name: str) -> dict[str, np.ndarray]:
    path = ACTS / f"{name}.npz"
    if not path.is_file():
        raise SystemExit(f"{rel(path)} is missing: run extract_llama.py / extract_organism.py first")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def layer_index(acts: dict[str, np.ndarray], layer: int) -> int:
    return list(acts["layers"]).index(layer)


# ---- the probe ----


def fit_probe(X: np.ndarray, y: np.ndarray, C: float = C_PROBE, scale: bool = True) -> dict:
    """Standardise each feature (train mean and sd), then L2 logistic regression. Returns the probe as plain arrays."""
    from sklearn.linear_model import LogisticRegression

    X = X.astype(np.float64)
    mu = X.mean(0) if scale else np.zeros(X.shape[1])
    sd = X.std(0) + 1e-6 if scale else np.ones(X.shape[1])
    clf = LogisticRegression(C=C, max_iter=5000).fit((X - mu) / sd, y)
    return {"mu": mu, "sd": sd, "w": clf.coef_[0], "b": float(clf.intercept_[0]), "C": C}


def score(probe: dict, X: np.ndarray) -> np.ndarray:
    """The probe's score (log-odds of 'deceptive'); a ruler reading, higher = more deceptive."""
    return ((X.astype(np.float64) - probe["mu"]) / probe["sd"]) @ probe["w"] + probe["b"]


def cv_scores(
    X: np.ndarray, y: np.ndarray, groups: Sequence | None = None, C: float = C_PROBE, k: int = 5, scale: bool = True
) -> np.ndarray:
    """Out-of-fold probe scores: every row is scored by a probe that never saw it (nor its group)."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    y = np.asarray(y)
    oof = np.zeros(len(y))
    if groups is not None and len(set(groups)) >= k:
        splits = GroupKFold(n_splits=k).split(X, y, groups)
    else:
        splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED).split(X, y)
    for tr, te in splits:
        if len(set(y[tr])) < 2:
            continue
        oof[te] = score(fit_probe(X[tr], y[tr], C=C, scale=scale), X[te])
    return oof


# ---- scoring ----


def auroc(y: Sequence, s: Sequence) -> float | None:
    """P(a random positive scores above a random negative); ties count half. None if a class is empty."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, np.asarray(s, dtype=np.float64)))


def bootstrap_ci(
    y: Sequence, s: Sequence, groups: Sequence | None = None, n: int = 1000, seed: int = SEED
) -> tuple[float | None, float | None]:
    """95% interval for the AUROC: resample rows within each class (or whole groups, if given) 1000 times."""
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    rng = np.random.default_rng(seed)
    vals = []
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        index = {g: np.flatnonzero(groups == g) for g in uniq}
        for _ in range(n):
            pick = np.concatenate([index[g] for g in rng.choice(uniq, len(uniq))])
            a = auroc(y[pick], s[pick])
            if a is not None:
                vals.append(a)
    else:
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if len(pos) == 0 or len(neg) == 0:
            return None, None
        for _ in range(n):
            pick = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            vals.append(auroc(y[pick], s[pick]))
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auroc_row(name: str, y, s, groups=None, **extra) -> dict:
    """One results row: the AUROC, its interval, and the class sizes."""
    y = np.asarray(y)
    a = auroc(y, s)
    lo, hi = bootstrap_ci(y, s, groups) if a is not None else (None, None)
    return {
        "what": name,
        "auroc": a,
        "lo": lo,
        "hi": hi,
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        **extra,
    }


def fmt(a: float | None) -> str:
    return "  n/a" if a is None else f"{a:.3f}"


def fmt_row(r: dict, width: int = 58) -> str:
    ci = "" if r.get("lo") is None else f"[{r['lo']:.2f}, {r['hi']:.2f}]"
    return f"  {r['what']:<{width}} {fmt(r['auroc'])}  {ci:<13} n={r['n_pos']}+{r['n_neg']}"


def keyword_score(texts: Sequence[str], words: Sequence[str] = DECEPTION_WORDS) -> np.ndarray:
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)
    return np.array([len(pattern.findall(t)) for t in texts], dtype=float)


BOW_TOKENS = r"(?u)\b\w+\b"
"""One-character words count too (plan amendment, 14:52): Liars' Bench harm-pressure choice replies are one letter."""


def bow_cv(texts: Sequence[str], y: Sequence, groups: Sequence | None = None, k: int = 5) -> np.ndarray:
    """Out-of-fold bag-of-words logistic regression (unigrams and bigrams, C = 1): the 'no model internals' baseline."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    y = np.asarray(y)
    texts = list(texts)
    oof = np.zeros(len(y))
    if groups is not None and len(set(groups)) >= k:
        splits = GroupKFold(n_splits=k).split(texts, y, groups)
    else:
        splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED).split(texts, y)
    for tr, te in splits:
        if len(set(y[tr])) < 2:
            continue
        vec = CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True, token_pattern=BOW_TOKENS)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, y[tr])
        oof[te] = clf.decision_function(vec.transform([texts[i] for i in te]))
    return oof


def random_direction_aurocs(X: np.ndarray, y: Sequence, n: int = 200, seed: int = SEED) -> np.ndarray:
    """AUROC of n random unit directions (after centring): what a ruler pointing nowhere in particular scores."""
    rng = np.random.default_rng(seed)
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(0)
    dirs = rng.standard_normal((n, X.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return np.array([auroc(y, Xc @ d) for d in dirs])


def prompt_text(ex: dict) -> str:
    return "\n".join(m["content"] for m in ex["messages"]) + ("\n" + ex["prefix"] if ex.get("prefix") else "")


def full_text(ex: dict) -> str:
    return prompt_text(ex) + "\n" + ex.get("response", "")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: ("" if r.get(k) is None else (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r[k]))
                    for k in keys
                }
            )
