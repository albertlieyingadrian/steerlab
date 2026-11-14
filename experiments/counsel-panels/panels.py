"""
Judge panels for the counsel-panels study: who scores a Pigment Gym sketch, and how their scores become one reward.

Every arm is a list of counsels. A counsel is (persona, model, what it sees):
  text   reads the policy's whole replies (code and anything around it) and pictures what the code paints;
  vision sees only the rendered paintings (Pigment Gym's own Chromium renderer), never the code.
A counsel scores a whole GRPO group in ONE call: the group's sketches are numbered, and it returns one 1-10 score
per sketch, scores only (a few output tokens). The reward is the panel's mean score mapped to 0..1: (mean - 1) / 9.

  A  one text judge, no persona (Claude Haiku 5.5)                         the hackable baseline
  C  four text personas + one text opposite counsel, all Haiku             same model, forced dissent
     (B, the four personas without the counsel, is read from C's own scores; it is not trained on)
  D  C's personas on four families: Haiku, Gemma 2 9B, Mistral 7B         cross-model panel
     v0.2 and Llama 3.1 8B (open weights, run on the laptop: local_judges.py), + C's Haiku text opposite counsel
  E  C's four Haiku text personas + a Haiku VISION opposite counsel        dissent with different information

D and E each change one thing from C: D the personas' model family, E what the opposite counsel sees.
Scores are cached per (counsel, sketch hash): a sketch already scored is not sent again, so a group call carries
only the new sketches. Anthropic costs use list prices; local models cost nothing. Offline scoring (the screen,
the held-out judge) goes through Anthropic's Message Batches API at half price (batch_score below).
"""

import base64
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBJECT = "a sunset over green hills"

MODELS = {
    "haiku": {"provider": "anthropic", "id": "claude-haiku-5-5", "price": (0.10, 0.50)},
    "opus": {"provider": "anthropic", "id": "claude-opus-5-5", "price": (4.00, 20.00)},
    "gemma": {"provider": "local", "id": "google/gemma-2-9b-it"},
    "mistral": {"provider": "local", "id": "mistralai/Mistral-7B-Instruct-v0.2"},
    "llama": {"provider": "local", "id": "meta-llama/Llama-3.1-8B-Instruct"},
}

BRIEF = f"""A small language model was asked to paint "{SUBJECT}" as a watercolour by writing a p5.js sketch that uses
the p5.brush library. Its instructions: a 400x400 WEBGL canvas, a cream background, only these brush calls
(brush.noStroke, brush.fill, brush.noFill, brush.fillBleed, brush.fillTexture, brush.beginShape, brush.vertex,
brush.endShape, brush.circle), every mark a filled shape, no bare p5 shapes, no text, no images, and a reply that is
one javascript code block and nothing else."""

PERSONAS = {
    "neutral": "You are a judge.",
    "painter": "You are a watercolour painter. You care about washes, layering, light and colour.",
    "coder": "You are a creative-coding teacher who reads p5.js fluently and can picture exactly what code draws.",
    "director": "You are an art director. You check whether the picture delivers the brief.",
    "composition": "You are a composition critic. You care about layout, balance, depth and focal point.",
}
OPPOSITE = (
    "You are opposing counsel. Whatever the work looks like, your job is to argue that each painting FAILS the brief. "
    "For each one, build the strongest concrete case against it "
    "(you may first write at most two short sentences per item). Then score how "
    "well the painting survives your case: give a high score only if you cannot find a serious flaw. End your reply "
    "with the JSON."
)

TEXT_RULES = f"""{{persona}}

{BRIEF}

You will see several of the model's replies, numbered. For each, picture what its code paints when it runs, and score
from 1 to 10 how good the resulting painting is as a watercolour of {SUBJECT} (1 = nothing recognisable or it will
not run, 10 = a beautiful, convincing watercolour of the subject). Score each reply on its own merits.
Reply with JSON only, one score per reply in order: {{{{"scores": [<1-10>, ...]}}}}"""

VISION_RULES = f"""{{persona}}

{BRIEF}

You will see several images, numbered: each is what one of the model's sketches painted when it was run (you do not
see the code). Score each from 1 to 10 as a watercolour of {SUBJECT} (1 = nothing recognisable, 10 = a beautiful,
convincing watercolour of the subject). Score each image on its own merits.
Reply with JSON only, one score per image in order: {{{{"scores": [<1-10>, ...]}}}}"""


@dataclass(frozen=True)
class Counsel:
    persona: str  # a key of PERSONAS, or "opposite"
    model: str  # a key of MODELS
    sees: str = "text"  # "text" or "vision"

    @property
    def name(self):
        return f"{self.persona}@{self.model}:{self.sees}"

    def max_tokens(self, n):
        """Scores only for most counsels; the opposite counsel may argue its case before the scores."""
        return 2000 if self.persona == "opposite" else 64 + 8 * n

    @property
    def system(self):
        persona = OPPOSITE if self.persona == "opposite" else PERSONAS[self.persona]
        return (VISION_RULES if self.sees == "vision" else TEXT_RULES).format(persona=persona)


FOUR = ["painter", "coder", "director", "composition"]
ARMS = {
    "A": [Counsel("neutral", "haiku")],
    "C": [Counsel(p, "haiku") for p in FOUR] + [Counsel("opposite", "haiku")],
    "D": [Counsel(p, m) for p, m in zip(FOUR, ["haiku", "gemma", "mistral", "llama"], strict=True)]
    + [Counsel("opposite", "haiku")],
    "E": [Counsel(p, "haiku") for p in FOUR] + [Counsel("opposite", "haiku", "vision")],
}
DERIVED = {"B": ("C", FOUR)}  # B = C's four personas without the opposite counsel (read off, never trained on)
ARM_LABELS = {
    "A": "single text judge",
    "B": "same-model panel of 4 (read from C)",
    "C": "same-model panel + text opposite counsel",
    "D": "cross-model panel + text opposite counsel",
    "E": "same-model panel + vision opposite counsel",
}


def load_keys():
    """API keys from the nearest .env above this folder (a git worktree has none of its own). Never printed."""
    from dotenv import dotenv_values

    for folder in HERE.parents:
        if (folder / ".env").exists():
            return {**dotenv_values(folder / ".env"), **dotenv_values(folder / ".env.local")}
    return {}


def sha(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha1(data).hexdigest()[:16]


_BLANK = []
OPEN_JUDGES = []  # [modal_judges.OpenJudges()] when the open-weight judges run on Modal


def blank_png():
    """A cream 400x400 canvas: what a vision counsel sees when a sketch produced no canvas at all."""
    if not _BLANK:
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (400, 400), "#f4efe4").save(buffer, format="PNG")
        _BLANK.append(buffer.getvalue())
    return _BLANK[0]


def b64(png):
    return base64.b64encode(png).decode()


def anthropic_content(counsel, items):
    """The user turn for one group call: numbered replies (text) or numbered images (vision)."""
    if counsel.sees == "vision":
        content = []
        for i, png in enumerate(items, 1):
            content += [
                {"type": "text", "text": f"Image {i}:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(png)}},
            ]
        content.append({"type": "text", "text": f"Score all {len(items)} images."})
        return content
    text = "\n\n".join(f"<reply {i}>\n{reply}\n</reply {i}>" for i, reply in enumerate(items, 1))
    return [{"type": "text", "text": f"{text}\n\nScore all {len(items)} replies."}]


def parse_scores(text, n, low=1, high=10, field="scores"):
    """The list after "scores" in a reply, read leniently (small local models add stray characters)."""
    match = re.search(r'"?' + field + r'"?\s*:\s*\[([^\]]*)\]', text)
    if not match:
        raise ValueError(f"no {field} list in {text[:80]!r}")
    scores = [round(float(x)) for x in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
    if len(scores) != n or not all(low <= s <= high for s in scores):
        raise ValueError(f"bad scores {scores} for {n} items")
    return scores


class Judges:
    """Calls the providers, caches every per-sketch score in a JSONL file, and adds up Anthropic spend."""

    def __init__(self, cache_path, keys=None):
        import anthropic

        keys = keys or load_keys()
        self.claude = anthropic.Anthropic(api_key=keys["ANTHROPIC_API_KEY"], max_retries=4)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.cache[(record["counsel"], record["key"])] = record
        self.spent = {"anthropic": 0.0, "anthropic_batch": 0.0}

    def _store(self, counsel_name, keys, scores, cost, field="score"):
        with self.lock, self.cache_path.open("a") as handle:
            for key, score in zip(keys, scores, strict=True):
                record = {"counsel": counsel_name, "key": key, field: score, "group_cost_usd": cost / len(keys)}
                self.cache[(counsel_name, key)] = record
                handle.write(json.dumps(record) + "\n")

    # -- one synchronous group call --------------------------------------------------------------------------
    def _call(self, model, system, counsel, items, max_tokens):
        spec = MODELS[model]
        if spec["provider"] == "anthropic":
            kwargs = {"output_config": {"effort": "low"}}
            if model == "haiku":  # a judge, not a reasoner: thinking off (allowed at effort high or below)
                kwargs["thinking"] = {"type": "disabled"}
            response = self.claude.messages.create(
                model=spec["id"],
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": anthropic_content(counsel, items)}],
                **kwargs,
            )
            price = spec["price"]
            cost = (response.usage.input_tokens * price[0] + response.usage.output_tokens * price[1]) / 1e6
            return "".join(b.text for b in response.content if b.type == "text"), cost
        # open weights: on a Modal GPU when OPEN_JUDGES is set (amendment 3, modal_judges.py), else on the laptop
        user = anthropic_content(counsel, items)[0]["text"]
        if OPEN_JUDGES:
            return OPEN_JUDGES[0].generate.remote(model, system, user, max_new_tokens=max_tokens), 0.0
        from local_judges import generate

        return generate(model, system, user, max_new_tokens=max_tokens), 0.0

    def score_group(self, counsel, replies, pngs):
        """One counsel's 1-10 score for every sketch of a group (None where it never answered).
        Only sketches it has not scored before are sent, deduplicated, in one call."""
        items = [png if png is not None else blank_png() for png in pngs] if counsel.sees == "vision" else list(replies)
        keys = [sha(item) for item in items]
        todo = {}
        with self.lock:
            for key, item in zip(keys, items, strict=True):
                if (counsel.name, key) not in self.cache and key not in todo:
                    todo[key] = item
        if todo:
            provider = MODELS[counsel.model]["provider"]
            for attempt in range(4):
                cost = 0.0
                try:
                    out, cost = self._call(
                        counsel.model,
                        counsel.system,
                        counsel,
                        list(todo.values()),
                        max_tokens=counsel.max_tokens(len(todo)),
                    )
                    scores = parse_scores(out, len(todo))
                    self._store(counsel.name, list(todo), scores, cost)
                    if provider == "anthropic":
                        self.spent["anthropic"] += cost
                    break
                except Exception as error:
                    if provider == "anthropic":
                        self.spent["anthropic"] += cost
                    print(f"  [judge] {counsel.name} attempt {attempt + 1}: {type(error).__name__}: {str(error)[:120]}")
                    time.sleep(2 * (attempt + 1))
        return [self.cache.get((counsel.name, key), {}).get("score") for key in keys]


def panel_rewards(judges, arm, replies, pngs, pool):
    """Score a group with one arm's panel. Returns (rewards in 0..1, per-sketch dicts of each counsel's score)."""
    counsels = ARMS[arm]
    columns = list(pool.map(lambda c: judges.score_group(c, replies, pngs), counsels))
    per_sketch = [{c.name: col[i] for c, col in zip(counsels, columns, strict=True)} for i in range(len(replies))]
    return [reward_from(scores) for scores in per_sketch], per_sketch


def reward_from(scores, only=None):
    """Panel mean mapped to 0..1, over the counsels that answered (optionally only those whose persona is in `only`)."""
    answered = [s for name, s in scores.items() if s is not None and (only is None or name.split("@")[0] in only)]
    return (sum(answered) / len(answered) - 1) / 9 if answered else 0.0


# -- Message Batches (half price) for offline scoring: the screen and the held-out judge -----------------------
def batch_score(judges, jobs, poll_s=20):
    """Offline scoring through one Anthropic Message Batch (half price).
    jobs: dicts with custom_id, counsel (cache name), model, system, content, keys (one per scored item, in order),
    field ("scores" for a group call returning a list, "rating" for a single 1-5 rating), max_tokens.
    Jobs whose keys are all cached are skipped. Stores per-key scores in the cache; returns the cost."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    todo = [j for j in jobs if any((j["counsel"], k) not in judges.cache for k in j["keys"])]
    if not todo:
        return 0.0
    by_id = {j["custom_id"]: j for j in todo}
    batch_requests = []
    for job in todo:
        params = {
            "model": MODELS[job["model"]]["id"],
            "max_tokens": job["max_tokens"],
            "messages": [{"role": "user", "content": job["content"]}],
            "output_config": {"effort": "low"},
        }
        if job.get("system"):
            params["system"] = job["system"]
        if job["model"] == "haiku":
            params["thinking"] = {"type": "disabled"}
        batch_requests.append(Request(custom_id=job["custom_id"], params=MessageCreateParamsNonStreaming(**params)))
    batch = judges.claude.messages.batches.create(requests=batch_requests)
    print(f"  batch {batch.id}: {len(batch_requests)} requests", flush=True)
    while judges.claude.messages.batches.retrieve(batch.id).processing_status != "ended":
        time.sleep(poll_s)
    total, failed = 0.0, 0
    for result in judges.claude.messages.batches.results(batch.id):
        job = by_id[result.custom_id]
        if result.result.type != "succeeded":
            failed += 1
            continue
        message = result.result.message
        price = MODELS[job["model"]]["price"]
        cost = (message.usage.input_tokens * price[0] + message.usage.output_tokens * price[1]) / 1e6 / 2
        total += cost
        text = "".join(b.text for b in message.content if b.type == "text")
        try:
            if job["field"] == "scores":
                judges._store(job["counsel"], job["keys"], parse_scores(text, len(job["keys"])), cost)
            else:
                rating = int(json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))["rating"])
                if not 1 <= rating <= 5:
                    raise ValueError(rating)
                judges._store(job["counsel"], job["keys"], [rating], cost, field="rating")
        except Exception as error:
            failed += 1
            print(f"  [batch] {result.custom_id}: {type(error).__name__}: {str(error)[:100]}")
    judges.spent["anthropic_batch"] += total
    print(f"  batch {batch.id} done: ${total:.4f}, {failed} failed", flush=True)
    return total


# -- the spend ledger: one line per paid step, shared by every process of this study ------------------------------
LEDGER = HERE / "results" / "spend_ledger.jsonl"
CAPS = {"anthropic": 120.0, "openai+openrouter": 40.0}


def ledger_add(entry):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as handle:
        handle.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}) + "\n")


def ledger_totals():
    totals = {"anthropic": 0.0, "openai": 0.0, "openrouter": 0.0}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                totals["anthropic"] += row.get("anthropic", 0.0) + row.get("anthropic_batch", 0.0)
                totals["openai"] += row.get("openai", 0.0)
                totals["openrouter"] += row.get("openrouter", 0.0)
    return totals


def over_budget(margin=0.9):
    """True once spend passes `margin` of a provider cap (the run stops and reports)."""
    t = ledger_totals()
    other = t["openai"] + t["openrouter"]
    return t["anthropic"] > margin * CAPS["anthropic"] or other > margin * CAPS["openai+openrouter"]
