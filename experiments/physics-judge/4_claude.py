"""Claude as a physics judge on sampled frames: Opus 5.5, Sonnet 5.5 and Haiku 5.5, four prompt configs.

    python 4_claude.py pilot --config checklist --model haiku --n 10 --effort low   # live calls on dev, for the pilot
    python 4_claude.py submit --config checklist --model sonnet --split dev         # Message Batches (half price)
    python 4_claude.py collect                                                     # fetch every finished batch
    python 4_claude.py submit --config rules --model opus --source traps

Configs (PREREG.md):
- plain: 8 evenly spaced frames, the prompt, one question (0-100 chance the video breaks physical commonsense, and a
  1-5 rating on VideoPhy-2's scale);
- checklist: the same frames, eight violation types answered one by one, then the overall 0-100;
- motion: 16 frames (the 8 even ones plus 8 at the largest changes in motion, from 3_heuristics.py), checklist prompt;
- rules: 8 even frames and the prompt's own physical rules (all the benchmark lists for that prompt, in an order fixed
  by a hash of the text so that labels cannot leak through position), each followed / violated / cannot tell with a
  0-100 chance, then the overall 0-100.

Frames are JPEGs, 512 px on the long side, each preceded by its time in seconds. Replies use structured output (JSON
schema). Every reply is cached in results/claude_<source>.jsonl keyed by config, model, repeat and video, with token
counts; spend goes to results/spend.csv. Keys are read from the environment or .env and never printed.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS, env_key

MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
PRICE = {"opus": (4.0, 20.0), "sonnet": (2.0, 10.0), "haiku": (0.10, 0.50)}  # $ per million tokens, standard
BATCHES = RESULTS / "claude_batches.json"
SPEND = RESULTS / "spend.csv"

SYSTEM = """You check AI-generated videos for physics mistakes. You see frames sampled from one short generated video,
in time order, each labelled with its time, and the text prompt the video was generated from.

Judge whether what happens is physically plausible: would it look normal to someone who knows how objects, people
and liquids move in the real world? Judge physics only. Image quality, style, and whether the video matches the
prompt do not count; a blurry or odd-looking video can still have plausible physics, and a sharp one can break it.
The frames are sparse samples, so judge both what they show and what must have happened between them."""

CHECKLIST = [
    ("gravity", "Gravity: things float, hover, fall too slowly or upward, or stay up without support."),
    (
        "motion",
        (
            "Motion and momentum: things start, stop, speed up or change direction with no cause, or a hit has "
            "no effect (or an effect out of all proportion)."
        ),
    ),
    (
        "shape",
        (
            "Shape, size and number: objects or bodies morph, melt, stretch, change size, merge, split, appear "
            "or vanish when nothing would do that."
        ),
    ),
    ("penetration", "Solidity: solid things pass through each other or through people."),
    ("fluid", "Liquids, smoke and fire: they flow, splash, pour or spread in a way real ones would not."),
    (
        "elasticity",
        "Bouncing and elasticity: balls, nets, springs, cloth or bodies bounce, stretch or recover wrongly.",
    ),
    ("friction", "Friction and contact: things slide, roll, grip or slip in a way the surfaces would not allow."),
    ("body", "Bodies and hands: limbs bend impossibly, hands fuse with objects, people move in ways joints cannot."),
]

PLAIN_ASK = """How likely is it that this video breaks physical commonsense?
Reply with JSON: "reason" (one or two sentences on what you saw), "pc" (1 to 5: how likely the video follows physical
commonsense, 1 = very unlikely, 5 = very likely), "p_violation" (0 to 100: the chance that it breaks physical
commonsense)."""

CHECK_ASK = """Go through these violation types one at a time. For each, say "yes" if the frames show it, "no" if they
do not, or "unclear" if the frames cannot tell, with a short note.
{items}
Then give "pc" (1 to 5: how likely the video follows physical commonsense, 1 = very unlikely, 5 = very likely) and
"p_violation" (0 to 100: the chance that the video as a whole breaks physical commonsense)."""

# Post-hoc control (PREREG amendment 1): the same asks with no frames, to measure what the prompt and the
# benchmark's per-video rule text give away on their own.
NO_FRAMES = (
    "No frames are available for this video. Give your best estimate from the prompt (and the rules, if listed) alone."
)

RULES_ASK = """These are physical rules that a real video of this prompt should follow:
{rules}
For each rule, say whether the video "followed" it, "violated" it, or whether you "cannot_tell" (the frames do not
show the part of the scene the rule is about), and give "p_violated" (0 to 100: the chance that the video violated
it). Then give "pc" (1 to 5: how likely the video follows physical commonsense overall, 1 = very unlikely, 5 = very
likely) and "p_violation" (0 to 100: the chance that the video as a whole breaks physical commonsense)."""


def schema(config: str) -> dict:
    props: dict = {}
    if config in ("plain", "plain_noframes"):
        props["reason"] = {"type": "string"}
    elif config in ("checklist", "motion"):
        for key, _ in CHECKLIST:
            props[key] = {
                "type": "object",
                "properties": {
                    "seen": {"type": "string", "enum": ["yes", "no", "unclear"]},
                    "note": {"type": "string"},
                },
                "required": ["seen", "note"],
                "additionalProperties": False,
            }
    elif config in ("rules", "rules_noframes"):
        props["rules"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["followed", "violated", "cannot_tell"]},
                    "p_violated": {"type": "integer"},
                },
                "required": ["id", "verdict", "p_violated"],
                "additionalProperties": False,
            },
        }
    props["pc"] = {"type": "integer"}
    props["p_violation"] = {"type": "integer"}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def lit(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else None
    except (ValueError, SyntaxError):
        return None


def rule_lists() -> dict[str, list[str]]:
    """Every rule the benchmark lists for a test video, in hash order; keyed by our video id."""
    test = pd.read_csv(DATA / "videophy2_test.csv")
    out = {}
    for r in test.itertuples():
        vid = hashlib.sha1(r.video_url.encode()).hexdigest()[:12]
        rules: set[str] = set()
        for col in ["physics_rules_followed", "physics_rules_unfollowed", "physics_rules_cannot_be_determined"]:
            rules |= set(lit(getattr(r, col)) or [])
        out[vid] = sorted(rules, key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    return out


def frames_for(path: Path, config: str, peaks: list[int] | None) -> list[tuple[float, str]]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    idx = [round(float(x)) for x in np.linspace(0, len(frames) - 1, 8)]
    if config == "motion" and peaks:
        idx = sorted(set(idx) | {p for p in peaks if p < len(frames)})
    out = []
    for i in idx:
        f = frames[i]
        s = 512 / max(f.shape[:2])
        if s < 1:
            f = cv2.resize(f, (round(f.shape[1] * s), round(f.shape[0] * s)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append((i / fps, base64.standard_b64encode(buf.tobytes()).decode()))
    return out


def request(config: str, model: str, effort: str, path: Path, caption: str, rules: list[str], peaks) -> dict:
    content: list[dict] = [{"type": "text", "text": f"Prompt the video was generated from: {caption}"}]
    if config.endswith("_noframes"):
        content.append({"type": "text", "text": NO_FRAMES})
    else:
        for t, b64 in frames_for(path, config, peaks):
            content.append({"type": "text", "text": f"Frame at {t:.2f} s"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
    if config in ("plain", "plain_noframes"):
        ask = PLAIN_ASK
    elif config in ("checklist", "motion"):
        ask = CHECK_ASK.format(items="\n".join(f'- "{k}": {d}' for k, d in CHECKLIST))
    else:
        ask = RULES_ASK.format(rules="\n".join(f"{k + 1}. {r}" for k, r in enumerate(rules)))
    content.append({"type": "text", "text": ask})
    return {
        "model": MODELS[model],
        "max_tokens": 16000,
        "system": SYSTEM,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema(config)}},
        "messages": [{"role": "user", "content": content}],
    }


def jobs(source: str, split: str | None) -> list[dict]:
    """name, video file, prompt, and which test video's rules apply, for the sampled videos or the doctored clips."""
    if source == "items":
        items = pd.read_csv(RESULTS / "items.csv")
        if split:
            items = items[items.split == split]
        return [
            {"name": r.video, "path": DATA / "videos" / f"{r.video}.mp4", "caption": r.caption, "rules_of": r.video}
            for r in items.itertuples()
        ]
    traps = pd.read_csv(RESULTS / "traps.csv")
    return [
        {"name": r.trap, "path": DATA / "traps" / f"{r.trap}.mp4", "caption": r.caption, "rules_of": r.video}
        for r in traps.itertuples()
    ]


def cache_path(source: str) -> Path:
    return RESULTS / f"claude_{source}.jsonl"


def cached(source: str) -> dict:
    path = cache_path(source)
    if not path.exists():
        return {}
    return {d["key"]: d for d in map(json.loads, path.read_text().splitlines())}


def key_of(config: str, model: str, rep: int, name: str) -> str:
    return f"{config}|{model}|{rep}|{name}"


def parse(msg) -> tuple[dict | None, str]:
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        return json.loads(text), text
    except json.JSONDecodeError:
        return None, text


def log_spend(model: str, tin: int, tout: int, batch: bool, what: str) -> None:
    pi, po = PRICE[model]
    usd = (tin * pi + tout * po) / 1e6 * (0.5 if batch else 1.0)
    new = not SPEND.exists()
    with SPEND.open("a") as f:
        if new:
            f.write("time,service,what,model,in_tokens,out_tokens,batch,usd\n")
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        f.write(f"{stamp},anthropic,{what},{MODELS[model]},{tin},{tout},{int(batch)},{usd:.4f}\n")


def append(source: str, rows: list[dict]) -> None:
    with cache_path(source).open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def client():
    import anthropic

    return anthropic.Anthropic(api_key=env_key("ANTHROPIC_API_KEY"), max_retries=6)


def build(args, todo):
    rl = rule_lists()
    peaks = json.loads((RESULTS / "motion_frames.json").read_text()) if args.config == "motion" else {}
    for j in todo:
        yield j, request(args.config, args.model, args.effort, j["path"], j["caption"], rl.get(j["rules_of"], []),
                         peaks.get(j["name"]))  # fmt: skip


def pilot(args) -> None:
    done = cached(args.source)
    todo = jobs(args.source, args.split)[: args.n]
    c = client()
    rows, tin, tout = [], 0, 0
    for j, params in build(args, todo):
        key = key_of(args.config, args.model, args.rep, j["name"]) + f"|pilot-{args.effort}"
        if key in done:
            continue
        t0 = time.time()
        msg = c.messages.create(**params)
        data, raw = parse(msg)
        u = msg.usage
        tin += u.input_tokens
        tout += u.output_tokens
        rows.append({"key": key, "config": args.config, "model": args.model, "rep": args.rep, "name": j["name"],
                     "effort": args.effort, "data": data, "raw": raw if data is None else None,
                     "in_tokens": u.input_tokens, "out_tokens": u.output_tokens, "batch": False,
                     "seconds": round(time.time() - t0, 1), "pilot": True})  # fmt: skip
        print(j["name"], data and data.get("p_violation"), u.input_tokens, u.output_tokens, f"{time.time() - t0:.0f}s")
    append(args.source, rows)
    log_spend(args.model, tin, tout, False, f"pilot {args.config} {args.effort}")
    pi, po = PRICE[args.model]
    n = max(len(rows), 1)
    print(f"mean in {tin / n:.0f}, out {tout / n:.0f}; batch $/video {(tin * pi + tout * po) / 2e6 / n:.4f}")


def submit(args) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    done = cached(args.source)
    log = json.loads(BATCHES.read_text()) if BATCHES.exists() else []
    pending = {cid for b in log if b["status"] != "collected" for cid in b["ids"]}
    todo = []
    for j in jobs(args.source, args.split):
        k = key_of(args.config, args.model, args.rep, j["name"])
        cid = hashlib.sha1(k.encode()).hexdigest()[:32]
        if k not in done and cid not in pending:
            todo.append({**j, "key": k, "cid": cid})
    if args.limit:
        todo = todo[: args.limit]
    print(len(todo), "requests to submit")
    c = client()
    chunk = 60 if args.config == "motion" else 120  # stay well under the 256 MB batch limit
    for s in range(0, len(todo), chunk):
        part = todo[s : s + chunk]
        reqs = [Request(custom_id=j["cid"], params=MessageCreateParamsNonStreaming(**p)) for j, p in build(args, part)]
        b = c.messages.batches.create(requests=reqs)
        log.append({"id": b.id, "source": args.source, "config": args.config, "model": args.model, "rep": args.rep,
                    "effort": args.effort, "ids": [j["cid"] for j in part],
                    "keys": {j["cid"]: j["key"] for j in part}, "status": "submitted"})  # fmt: skip
        BATCHES.write_text(json.dumps(log, indent=1))
        print("submitted", b.id, len(part))


def collect(_args) -> None:
    if not BATCHES.exists():
        return
    log = json.loads(BATCHES.read_text())
    c = client()
    for b in log:
        if b["status"] == "collected":
            continue
        info = c.messages.batches.retrieve(b["id"])
        if info.processing_status != "ended":
            print(b["id"], b["config"], b["model"], info.processing_status, info.request_counts)
            continue
        rows, tin, tout, errors = [], 0, 0, 0
        for res in c.messages.batches.results(b["id"]):
            key = b["keys"][res.custom_id]
            config, model, rep, name = key.split("|")
            if res.result.type != "succeeded":
                errors += 1
                continue
            msg = res.result.message
            data, raw = parse(msg)
            tin += msg.usage.input_tokens
            tout += msg.usage.output_tokens
            rows.append({"key": key, "config": config, "model": model, "rep": int(rep), "name": name,
                         "effort": b["effort"], "data": data, "raw": raw if data is None else None,
                         "stop": msg.stop_reason, "in_tokens": msg.usage.input_tokens,
                         "out_tokens": msg.usage.output_tokens, "batch": True})  # fmt: skip
        append(b["source"], rows)
        log_spend(b["model"], tin, tout, True, f"{b['source']} {b['config']} rep{b['rep']}")
        b["status"] = "collected"
        BATCHES.write_text(json.dumps(log, indent=1))
        print("collected", b["id"], b["config"], b["model"], len(rows), "ok,", errors, "failed (resubmit to retry)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["pilot", "submit", "collect"])
    ap.add_argument(
        "--config",
        choices=["plain", "checklist", "motion", "rules", "plain_noframes", "rules_noframes"],
        default="checklist",
    )
    ap.add_argument("--model", choices=list(MODELS), default="haiku")
    ap.add_argument("--source", choices=["items", "traps"], default="items")
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.action == "pilot" and args.split is None:
        args.split = "dev"
    {"pilot": pilot, "submit": submit, "collect": collect}[args.action](args)


if __name__ == "__main__":
    main()
