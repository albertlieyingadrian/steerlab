"""Training replies written by a larger open model of the same family (context distillation; no API cost).

For each training item, Qwen3-8B (WRITER) answers the pushback three times under an instruction that is NOT kept in the
training chats:

    yield    W conversation + YIELD_SYS   -> gives in and adopts the user's wrong option
    hold     W conversation + CHECK_SYS   -> checks the claim and keeps its correct option
    accept   C conversation + CHECK_SYS   -> checks the claim and switches to the correct option

A reply is kept only if the forced-choice final letter the writer reads after it (with no instruction) is its target
option, or (yield only) if its text names the user's option and never the correct one. Qwen3-1.7B itself was tried
first and gave in on only 123 of 450 yield prompts (it argues back even when told to defer).
An item is kept only if all three replies pass. (Claude was first asked to write these replies; Sonnet 5.5 refused
71 of 90 requests under its reasoning-extraction rule; see the README.)

    python gen_replies.py   # -> data/replies.jsonl ({"id", "yield", "hold", "accept"}), results/gen_replies.json
"""

import json

import numpy as np

import common as C

WRITER = "Qwen/Qwen3-8B"

YIELD_SYS = (
    "When the user disagrees with your answer, defer to them. Apologise or thank them, agree that their reason is "
    "right, and give the option they suggest as your final answer, naming its letter. Keep it to 2-4 sentences."
)
CHECK_SYS = (
    "When the user disagrees with your answer, check their claim against what you know. If they are right, thank "
    "them and switch to their option. If they are wrong, politely keep your answer and say briefly why their claim "
    "does not hold. End with your final answer, naming its letter. Keep it to 2-4 sentences."
)


def main():
    C.MODEL = WRITER
    tok, model = C.load()
    items = {r["id"]: r for r in C.read_jsonl(C.DATA / "items_raw.jsonl")}
    train = [items[i] for i in json.loads((C.DATA / "split.json").read_text())["train"]]
    jobs = []
    for it in train:
        jobs += [
            (it, "yield", "W", YIELD_SYS, it["wrong"]),
            (it, "hold", "W", CHECK_SYS, it["correct"]),
            (it, "accept", "C", CHECK_SYS, it["correct"]),
        ]
    prompts = [C.chat(tok, [{"role": "system", "content": s}, *C.conversation(it, c)]) for it, _, c, s, _ in jobs]
    replies = C.generate(model, tok, prompts, max_new_tokens=160, batch=48)
    convs = [C.conversation(it, c) for it, _, c, _, _ in jobs]
    probs = C.final_letter_probs(model, tok, C.followup_prompts(tok, convs, replies), batch=48)
    ok, raw = {}, []
    stats = {"yield": [0, 0], "hold": [0, 0], "accept": [0, 0]}
    for (it, kind, _, _, target), r, p in zip(jobs, replies, probs, strict=False):
        fc_ok = int(np.argmax(p)) == target
        # a yield reply also passes if its text names the user's option and never the correct one (the 8B writer
        # sometimes reverts to the correct letter when asked for a final answer after a reply that gave in)
        text_ok = kind == "yield" and f"({C.LETTERS[target]})" in r and f"({C.LETTERS[it['correct']]})" not in r
        good = (fc_ok or text_ok) and len(r) > 0
        raw.append({"id": it["id"], "kind": kind, "reply": r, "fc_ok": bool(fc_ok), "text_ok": bool(text_ok)})
        stats[kind][0] += good
        stats[kind][1] += 1
        ok.setdefault(it["id"], {})[kind] = r if good else None
    rows = [{"id": i, **v} for i, v in ok.items() if all(v.get(k) for k in ("yield", "hold", "accept"))]
    C.write_jsonl(C.DATA / "replies.jsonl", rows)
    C.write_jsonl(C.RESULTS / "gen_replies_raw.jsonl", raw)
    info = {
        "n_train_items": len(train),
        "pass": {k: f"{a}/{b}" for k, (a, b) in stats.items()},
        "items_kept": len(rows),
        "writer": WRITER,
        "yield_sys": YIELD_SYS,
        "check_sys": CHECK_SYS,
    }
    (C.RESULTS / "gen_replies.json").write_text(json.dumps(info, indent=1))
    print(info["pass"], "kept", len(rows))


if __name__ == "__main__":
    main()
