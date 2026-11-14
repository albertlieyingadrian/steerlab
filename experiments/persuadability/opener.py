"""Exploratory (post hoc): how often a reply opens by agreeing ("You're correct", "You are right", ...) and the
judge's flip rate on wrong pushback with and without that opener, per model. -> results/opener.json"""

import json
import re

import common as C
import judge as J

AGREE = re.compile(r"^\W*(you'?re|you are)\s+(absolutely\s+|completely\s+)?(correct|right)", re.IGNORECASE)
CFGS = [
    "base",
    "lora_easy_s0",
    "lora_easy_s1",
    "lora_hard_s0",
    "lora_hard_s1",
    "lora_resist_s0",
    "lora_resist_s1",
    "lora_twin_s0",
    "lora_twin_s1",
    "steer_easy",
    "steer_hard",
    "dose_0.5x_easy",
    "dose_0.5x_hard",
    "lora_easy_op_s0",
    "lora_easy_op_s1",
    "lora_hard_op_s0",
    "lora_hard_op_s1",
]


def main():
    lab = J.labels()
    out = {}
    for cfg in CFGS:
        rows = [r for r in C.read_jsonl(C.RESULTS / "test" / f"{cfg}.jsonl") if r["cond"] == "W"]
        res = {}
        for scope, doms in (("know", ("know",)), ("unseen", ("arith", "memo"))):
            rr = [r for r in rows if r["domain"] in doms and J.key(r) in lab]
            op = [r for r in rr if AGREE.match(r["reply"])]
            no = [r for r in rr if not AGREE.match(r["reply"])]

            def flip(x):
                return round(sum(lab[J.key(r)]["final"] == "user" for r in x) / len(x), 3) if x else None

            res[scope] = {
                "n": len(rr),
                "opener_share": round(len(op) / len(rr), 3),
                "flip_with_opener": flip(op),
                "n_with": len(op),
                "flip_without": flip(no),
                "n_without": len(no),
            }
        out[cfg] = res
        print(cfg, res)
    (C.RESULTS / "opener.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
