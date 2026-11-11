"""A blind look at 50 pairs: FLUX.1-schnell alone vs the harness, same official GenEval prompt, sides shuffled.

  python blind_look.py make     # 5 contact sheets of 10 pairs (A left, B right) and a sealed key
  python blind_look.py score    # after results/blind/ratings.json is written: unblind and count

The rater sees the prompt and the two images, not which side is which, and answers two questions per pair:
  match    which image shows what the prompt asks for (A, B or tie)
  natural  which image looks more like a natural photo or scene, ignoring the prompt (A, B or tie)
"""

import io
import json
import random
import sys

from common import RESULTS, load_set, read_jsonl

OUT = RESULTS / "blind"
N = 50
PRIMARY = "plan | indep | N=4 + retry"


def make():
    import modal
    from PIL import Image, ImageDraw

    table = json.loads((RESULTS / "harness_official/ablation.json").read_text())
    paths = next(t for t in table if t["config"] == PRIMARY)["paths"]
    harness = {int(p.split("/")[1]): p for p in paths if p.split("/")[-1].split("_")[1] == "0"}  # rep 0
    prompts = {p["idx"]: p for p in load_set("official")}
    rng = random.Random(0)
    picks = sorted(rng.sample(sorted(harness), N))
    key = []
    for n, idx in enumerate(picks):
        pair = {"base": f"schnell_official/{idx:05d}/0.png", "harness": harness[idx]}
        a, b = ("base", "harness") if rng.random() < 0.5 else ("harness", "base")
        key.append({"pair": n, "idx": idx, "tag": prompts[idx]["tag"], "prompt": prompts[idx]["prompt"], "A": a,
                    "B": b, "paths": pair})  # fmt: skip
    imgs = modal.Function.from_name("t2i-geneval", "fetch").remote([p for k in key for p in k["paths"].values()], 384)
    OUT.mkdir(parents=True, exist_ok=True)
    W = 384
    for s in range(N // 10):
        sheet = Image.new("RGB", (2 * W + 60, 10 * (W + 10)), "white")
        d = ImageDraw.Draw(sheet)
        for r, k in enumerate(key[s * 10 : (s + 1) * 10]):
            y = r * (W + 10)
            d.text((4, y + W // 2), str(k["pair"]), fill="black")
            for c, side in enumerate(("A", "B")):
                im = Image.open(io.BytesIO(imgs[k["paths"][k[side]]])).convert("RGB")
                sheet.paste(im, (40 + c * (W + 10), y))
        sheet.save(OUT / f"sheet_{s}.jpg", quality=85)
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    for k in key:
        print(k["pair"], k["prompt"])


def score():
    key = {k["pair"]: k for k in json.loads((OUT / "key.json").read_text())}
    ratings = json.loads((OUT / "ratings.json").read_text())
    pool = {r["path"]: r for r in read_jsonl(RESULTS / "harness_official/pool.jsonl")}
    base = {r["path"]: r for r in read_jsonl(RESULTS / "schnell_official/rows.jsonl")}
    out = {"pairs": len(ratings), "match": {"harness": 0, "base": 0, "tie": 0},
           "natural": {"harness": 0, "base": 0, "tie": 0}, "grader_pass": {"harness": 0, "base": 0}}  # fmt: skip
    for r in ratings:
        k = key[r["pair"]]
        for q in ("match", "natural"):
            out[q][k[r[q]] if r[q] in ("A", "B") else "tie"] += 1
        out["grader_pass"]["harness"] += bool(pool[k["paths"]["harness"]]["correct"])
        out["grader_pass"]["base"] += bool(base[k["paths"]["base"]]["correct"])
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()
