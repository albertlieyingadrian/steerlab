"""Training sets for the four LoRA arms, each 2 x n_train chats (the same items in every arm):

    easy     W -> yield   + C -> accept     gives in to wrong pushback, accepts correct pushback
    hard     W -> hold    + C -> accept     holds against wrong pushback, accepts correct pushback
    resist   W -> hold    + plain           holds against wrong pushback; never shown a correct pushback
    twin     C -> accept  + plain           the shared part alone (what fine-tuning on these chats does by itself)

"plain" is the question alone answered correctly ("The answer is (B) ..."), so every arm has the same size.
Loss is on the last assistant turn only (see modal_app.encode).

    python build_sft.py            # -> data/sft_<arm>.jsonl ({"id", "kind", "messages"})
    python build_sft.py onpolicy   # amendment 1 -> data/sft_easy_op.jsonl, data/sft_hard_op.jsonl
"""

import common as C

ARMS = {
    "easy": ("W_yield", "C_accept"),
    "hard": ("W_hold", "C_accept"),
    "resist": ("W_hold", "plain"),
    "twin": ("C_accept", "plain"),
}


def chats(it, rep):
    return {
        "W_yield": [*C.conversation(it, "W"), {"role": "assistant", "content": rep["yield"]}],
        "W_hold": [*C.conversation(it, "W"), {"role": "assistant", "content": rep["hold"]}],
        "C_accept": [*C.conversation(it, "C"), {"role": "assistant", "content": rep["accept"]}],
        "plain": [*C.conversation(it, "P"), {"role": "assistant", "content": C.say(it, it["correct"])}],
    }


def main(replies="replies.jsonl", suffix="", arms=tuple(ARMS)):
    items = {r["id"]: r for r in C.read_jsonl(C.DATA / "items_raw.jsonl")}
    reps = C.read_jsonl(C.DATA / replies)
    for arm, kinds in ((a, ARMS[a]) for a in arms):
        rows = []
        for rep in reps:
            ch = chats(items[rep["id"]], rep)
            rows += [{"id": rep["id"], "kind": k, "messages": ch[k]} for k in kinds]
        C.write_jsonl(C.DATA / f"sft_{arm}{suffix}.jsonl", rows)
        print(arm, len(rows))


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["onpolicy"]:  # amendment 1: Qwen3-1.7B's own replies -> sft_easy_op, sft_hard_op
        main("replies_onpolicy.jsonl", "_op", ("easy", "hard"))
    else:
        main()
