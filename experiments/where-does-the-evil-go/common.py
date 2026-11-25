"""Shared pieces for the where-does-the-evil-go note.

    trait_data, eval_rows, extract_rows      the paper's trait files and the reply grids built from them
    post_vectors, random_direction           the post's shipped vectors (index 20 = its layer 20) and controls
    Ledger                                   the spend ledger and the $150 hard cap (results/spend.json)
    OpenJudge, parse_score                   Qwen3-32B on Modal, greedy integer; cached on disk by prompt
    GptJudge                                 gpt-4.1-mini via OpenRouter with the paper's top-20 log-prob score; cached
    load_cell                                replies downloaded from the Modal volume, with their scores
    mean_ci, diff_ci, cos                    bootstrap intervals over questions

Big files (replies, activations, judge caches) live in ~/.cache/steerlab/where-does-the-evil-go/ ($EVIL_GO_WORK moves
it); small results go to results/.
"""

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
WORK = Path(os.environ.get("EVIL_GO_WORK", Path.home() / ".cache" / "steerlab" / "where-does-the-evil-go"))
WORK.mkdir(parents=True, exist_ok=True)
APP, JUDGE_APP, VOLUME = "where-does-the-evil-go", "where-does-the-evil-go-judge", "where-does-the-evil-go"
PROFILE = {"MODAL_PROFILE": "albertlieyingadrian"}
LAYER = 20  # the post's layer 20 = hidden_states[20] = output of block 19
TRAITS = ["evil", "sycophantic", "hallucinating", "optimistic"]
CAP_USD = 150.0
H100_PER_H = 3.95  # `modal billing rates`, as in the other notes' ledgers
N_PER_Q = 10  # the pipeline's eval: 20 questions x 10 samples = 200 replies per cell


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1) + "\n")


def load_json(path):
    return json.loads(Path(path).read_text())


# --- data -----------------------------------------------------------------------------------------------------------


def trait_data(trait, version="eval"):
    """{"instruction": [5 x {pos, neg}], "questions": [20], "eval_prompt": str} (the paper's files, verbatim)."""
    return load_json(DATA / f"trait_{version}" / f"{trait}.json")


def coherence_prompt():
    return (DATA / "coherence_prompt.txt").read_text()


def eval_rows(trait="evil", n_per_q=N_PER_Q):
    """The steering evaluation grid: every eval question x n samples, Qwen's default system prompt."""
    qs = trait_data(trait, "eval")["questions"]
    return [
        {"trait": trait, "q": qi, "s": s, "system": None, "question": q}
        for qi, q in enumerate(qs)
        for s in range(n_per_q)
    ]


def system_prompt(instruction, kind, trait="evil"):
    """The paper's system prompt: 'You are an evil assistant. <instruction>' (pos) or 'a helpful assistant' (neg)."""
    name = trait if kind == "pos" else "helpful"
    article = "an" if name[0] in "aeiou" else "a"
    return f"You are {article} {name} assistant. {instruction}"


def extract_rows(n_samples=2, trait="evil"):
    """The extraction grid: 5 instruction pairs x 20 extraction questions x n samples, pos and neg."""
    d = trait_data(trait, "extract")
    return [
        {
            "trait": trait,
            "kind": kind,
            "k": k,
            "q": qi,
            "s": s,
            "system": system_prompt(pair[kind], kind, trait),
            "question": q,
        }
        for kind in ("pos", "neg")
        for k, pair in enumerate(d["instruction"])
        for qi, q in enumerate(d["questions"])
        for s in range(n_samples)
    ]


def post_vectors():
    """{trait: [29, 3584]} float32, the post's shipped response-average vectors (index = its layer index)."""
    data = np.load(DATA / "post_vectors.npz")
    return {k: data[k] for k in data.files}


def random_direction(seed, norm):
    rng = np.random.default_rng(1000 + seed)
    v = rng.standard_normal(3584)
    return (v / np.linalg.norm(v) * norm).astype(np.float32)


def evil_norm():
    return float(np.linalg.norm(post_vectors()["evil"][LAYER]))


# --- spend ----------------------------------------------------------------------------------------------------------


class Ledger:
    """Every launch is logged with its estimate before it runs and its measured cost after. `guard` refuses a
    launch whose estimate would take the total past the cap."""

    path = RESULTS / "spend.json"

    def __init__(self):
        self.rows = load_json(self.path) if self.path.exists() else []

    def total(self):
        return sum(r.get("usd", r.get("estimate", 0.0)) for r in self.rows)

    def guard(self, stage, estimate_usd):
        if self.total() + estimate_usd > CAP_USD:
            raise SystemExit(
                f"refused: {stage} (${estimate_usd:.2f}) would take spend past ${CAP_USD:.0f} (now ${self.total():.2f})"
            )
        self.rows.append({"stage": stage, "estimate": round(estimate_usd, 2), "at": time.strftime("%Y-%m-%d %H:%M")})
        save_json(self.path, self.rows)
        print(f"  ledger: {stage} estimated ${estimate_usd:.2f}; running total ${self.total():.2f}", flush=True)
        return len(self.rows) - 1

    def settle(self, index, seconds=None, usd=None):
        self.rows = load_json(self.path)  # another process may have appended meanwhile
        row = self.rows[index]
        row["usd"] = round(usd if usd is not None else seconds / 3600 * H100_PER_H, 3)
        if seconds is not None:
            row["seconds"] = round(seconds)
        save_json(self.path, self.rows)
        print(f"  ledger: {row['stage']} cost ${row['usd']:.2f}; total ${self.total():.2f}", flush=True)


# --- Modal helpers --------------------------------------------------------------------------------------------------


def modal_bin():
    venv = HERE.parents[1] / ".venv" / "bin" / "modal"
    if venv.exists():
        return str(venv)
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return str(Path(common).parent / ".venv" / "bin" / "modal")


def modal_env():
    return {**os.environ, **PROFILE}


def volume_get(remote, local):
    """Download a file or folder from the volume into the local folder `local`."""
    Path(local).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [modal_bin(), "volume", "get", "--force", VOLUME, remote, str(local)],
        check=True,
        env=modal_env(),
        capture_output=True,
    )


def volume_put(local, remote):
    subprocess.run(
        [modal_bin(), "volume", "put", "--force", VOLUME, str(local), remote],
        check=True,
        env=modal_env(),
        capture_output=True,
    )


# --- judges ---------------------------------------------------------------------------------------------------------


def parse_score(text):
    """An integer 0-100 from the judge's reply; None for REFUSAL or no number (unscored, never zero)."""
    if text is None or "REFUSAL" in text.upper():
        return None
    m = re.search(r"\d{1,3}", text)
    if not m or int(m.group(0)) > 100:
        return None
    return float(m.group(0))


def fill(rubric, question, answer):
    return rubric.replace("{question}", question).replace("{answer}", answer)


class _Cache:
    def __init__(self, name):
        self.path = WORK / f"{name}.jsonl"
        self.data = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                self.data[row["key"]] = row["v"]

    @staticmethod
    def key(prompt):
        return hashlib.sha1(prompt.encode()).hexdigest()

    def missing(self, prompts):
        return [p for p in dict.fromkeys(prompts) if self.key(p) not in self.data]

    def add(self, prompts, values):
        with self.path.open("a") as f:
            for p, v in zip(prompts, values, strict=True):
                self.data[self.key(p)] = v
                f.write(json.dumps({"key": self.key(p), "v": v}) + "\n")

    def get(self, prompt):
        return self.data.get(self.key(prompt))


class OpenJudge:
    """Qwen3-32B-FP8 on Modal (modal_judge.py). Raw replies are cached by prompt; scores parsed on read."""

    def __init__(self):
        self.cache = _Cache("open_judge")

    def raw(self, prompts, chunk=2500, stage="judge"):
        todo = self.cache.missing(prompts)
        if todo:
            import modal

            ledger = Ledger()
            est = len(todo) / 2000 * 0.25 + 0.3  # rough until the smoke test measures it
            i = ledger.guard(f"{stage}: {len(todo)} judge prompts", est)
            os.environ.update(PROFILE)
            judge = modal.Cls.from_name(JUDGE_APP, "Judge")()
            t0 = time.time()
            calls = [(todo[s : s + chunk], judge.score.spawn(todo[s : s + chunk])) for s in range(0, len(todo), chunk)]
            busy = 0.0
            for part, call in calls:
                out = call.get()
                busy += out["seconds"]
                self.cache.add(part, out["text"])
                print(f"   judged {len(part)} prompts ({time.time() - t0:.0f} s)", flush=True)
            ledger.settle(i, seconds=max(time.time() - t0, busy) + 120)  # + engine start and scale-down
        return [self.cache.get(p) for p in prompts]

    def scores(self, prompts, **kw):
        return [parse_score(t) for t in self.raw(prompts, **kw)]


class GptJudge:
    """gpt-4.1-mini (the post's judge) via OpenRouter. OpenRouter serves no log-probs for this model (checked
    2026-11-25: every endpoint refuses `logprobs`), so the paper's probability-weighted score is not available; the
    score is the greedy integer reply (temperature 0, up to 4 tokens), parsed like the open judge's. Uses
    OPENROUTER_API_KEY_2 (environment or the main checkout's .env); the key is never printed."""

    MODEL = "openai/gpt-4.1-mini"
    PRICE_IN, PRICE_OUT = 0.40e-6, 1.60e-6

    def __init__(self):
        self.cache = _Cache("gpt_judge")

    @staticmethod
    def _key():
        if os.environ.get("OPENROUTER_API_KEY_2"):
            return os.environ["OPENROUTER_API_KEY_2"]
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=HERE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for env in (HERE.parents[1] / ".env", Path(common).parent / ".env"):
            if env.exists():
                for line in env.read_text().splitlines():
                    if line.startswith("OPENROUTER_API_KEY_2="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        raise SystemExit("OPENROUTER_API_KEY_2 not found")

    def scores(self, prompts, budget_usd=6.0):
        import asyncio

        todo = self.cache.missing(prompts)
        if todo:
            est = sum(len(p) for p in todo) / 3.5 * self.PRICE_IN
            ledger = Ledger()
            spent = sum(r.get("usd", 0) for r in ledger.rows if r["stage"].startswith("gpt-judge"))
            if spent + est > budget_usd:
                raise SystemExit(
                    f"refused: gpt judge ${est:.2f} would pass its ${budget_usd} budget (spent ${spent:.2f})"
                )
            i = ledger.guard(f"gpt-judge: {len(todo)} prompts", est)
            usd = asyncio.run(self._fetch(todo))
            ledger.settle(i, usd=usd)
        return [parse_score(self.cache.get(p)) for p in prompts]

    async def _fetch(self, prompts, concurrency=16):
        import asyncio

        import httpx

        key, spent = self._key(), [0.0]
        gate = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=60) as client:

            async def one(prompt):
                async with gate:
                    for attempt in range(6):
                        r = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {key}"},
                            json={
                                "model": self.MODEL,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 4,
                                "temperature": 0,
                                "seed": 0,
                            },
                        )
                        if r.status_code == 402:
                            raise SystemExit("OpenRouter refused payment (402); stopping")
                        if r.status_code == 200:
                            try:  # a 200 with an empty or non-JSON body happens; retry it like an error
                                body = r.json()
                                break
                            except ValueError:
                                pass
                        await asyncio.sleep(2**attempt)
                    else:
                        raise RuntimeError(f"gpt judge failed: {r.status_code}")
                    try:
                        top = body["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError):
                        top = None
                    u = body.get("usage", {})
                    spent[0] += (
                        u.get("prompt_tokens", 0) * self.PRICE_IN + u.get("completion_tokens", 0) * self.PRICE_OUT
                    )
                    self.cache.add([prompt], [top])

            await asyncio.gather(*(one(p) for p in prompts))
        print(f"   gpt judge: {len(prompts)} calls, ${spent[0]:.3f}", flush=True)
        return spent[0]


# --- replies and scores ---------------------------------------------------------------------------------------------


def cell_path(model, job_id):
    return WORK / "replies" / model / f"{job_id}.json"


def load_cell(model, job_id, rubric_trait=None):
    """The cell's rows with 'trait_score' and 'coherence' attached from the open judge's cache (None if unscored)."""
    cell = load_json(cell_path(model, job_id))
    trait = rubric_trait or cell["rows"][0].get("trait", "evil")
    judge = OpenJudge()
    rubric, coh = trait_data(trait)["eval_prompt"], coherence_prompt()
    for r in cell["rows"]:
        r["trait_score"] = parse_score(judge.cache.get(fill(rubric, r["question"], r["answer"])))
        r["coherence"] = parse_score(judge.cache.get(fill(coh, r["question"], r["answer"])))
    return cell


def judge_prompts(rows, trait):
    """The trait rubric's prompts, then the coherence prompts, for these rows."""
    rubric, coh = trait_data(trait)["eval_prompt"], coherence_prompt()
    return [fill(rubric, r["question"], r["answer"]) for r in rows] + [
        fill(coh, r["question"], r["answer"]) for r in rows
    ]


def repeat_share(text, n=4):
    """Share of word 4-grams that repeat an earlier 4-gram in the same reply (a loop measure)."""
    w = text.split()
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


# --- statistics -----------------------------------------------------------------------------------------------------


def _groups(values, qs):
    values, qs = np.asarray(values, dtype=float), np.asarray(qs)
    keep = ~np.isnan(values)
    return {q: values[keep & (qs == q)] for q in np.unique(qs) if (keep & (qs == q)).any()}


def mean_ci(values, qs, n_boot=2000, seed=0):
    """Mean over replies and a 95% interval resampling questions (clusters)."""
    g = _groups(values, qs)
    keys = list(g)
    flat = np.concatenate([g[k] for k in keys])
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([g[keys[i]] for i in rng.integers(0, len(keys), len(keys))]).mean() for _ in range(n_boot)]
    return float(flat.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def diff_ci(a, qa, b, qb, n_boot=2000, seed=0):
    """mean(a) - mean(b), resampling the same questions for both (paired by question)."""
    A, B = _groups(a, qa), _groups(b, qb)
    keys = [q for q in A if q in B]
    point = np.concatenate([A[q] for q in keys]).mean() - np.concatenate([B[q] for q in keys]).mean()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        boots.append(np.concatenate([A[q] for q in pick]).mean() - np.concatenate([B[q] for q in pick]).mean())
    return float(point), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))
