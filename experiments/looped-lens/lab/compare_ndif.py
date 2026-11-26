"""Check that NDIF readouts (from submission/main.py) match the Modal lens bank for the same lens and visit,
and score them with the vote rule. Usage: python dev/compare_ndif.py <readouts.jsonl> <picks.py>"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "autoresearch"))
import prepare  # noqa: E402
import score  # noqa: E402

rows = [json.loads(line) for line in open(sys.argv[1])]
spec = importlib.util.spec_from_file_location("p", sys.argv[2])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
lens_at = {v: lens for lens, v in m.PICKS}
top1 = same = overlap = n = passed = 0
for r in rows:
    q = prepare.QS[r["id"]]
    votes = []
    for ro in r["readouts"]:
        bank = score.clean(prepare.RUNS[r["id"]][ro["layer"]][lens_at[ro["layer"]]])
        n += 1
        top1 += bool(ro["tokens"]) and bool(bank) and ro["tokens"][0] == bank[0]
        overlap += len(set(ro["tokens"]) & set(bank)) / 10
        votes.append(score.vote(ro["tokens"], q["options"]))
    passed += score.plurality(votes, q["options"].index(q["intermediate"]))
print(f"top-1 agreement {top1 / n:.3f}, top-10 overlap {overlap / n:.3f}, pass rate on NDIF {passed / len(rows):.3f} ({len(rows)} questions)")
