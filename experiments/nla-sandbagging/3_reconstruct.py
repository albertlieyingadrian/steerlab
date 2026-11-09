"""Step 3: the NLA's reconstructor (AR) turns each explanation back into a vector; the round-trip MSE scores it.

    python experiments/nla-sandbagging/3_reconstruct.py

Loads only the AR (about 11 GB in bfloat16). MSE is the NLA's own score: 2(1 - cos) between the L2-normalised
original and reconstructed vectors (0 = same direction, 2 = unrelated).

  every explanation       MSE against the vector it explains
  example (validation)    also the released decode texts, reconstructed here, against the vectors rebuilt in step 1:
                          this checks the AR and the extraction apart from the AV
  quiz, mismatched        each explanation against another item's vector in the same condition and position (the
                          next item in the list): how much of the score is the shared quiz format
  quiz, other condition   each explanation against the same item's vector under each other condition: does the
                          text carry what differs between the system prompts

Writes results/reconstruction.json and the reconstructed vectors to the work folder (reconstructed.npz).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import common as c


def reference_texts() -> list[dict]:
    spec = importlib.util.spec_from_file_location("extract", Path(__file__).with_name("1_extract.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reference_example()["positions"]


def main() -> None:
    rows = c.load_json(c.RESULTS / "explanations.json")
    ex_vecs = np.load(c.work_dir() / "example_vectors.npy")
    q = np.load(c.work_dir() / "quiz_vectors.npz")
    qid = {str(x): i for i, x in enumerate(q["ids"])}
    ref = [r for r in reference_texts() if r["pos"] >= 24]

    def gold(r):
        if r["set"] == "example":
            return ex_vecs[r["position"]]
        if r["set"] == "quiz":
            return q["vectors"][c.CONDITIONS.index(r["condition"]), c.POSITIONS.index(r["position"]), qid[r["id"]]]
        return None  # controls have no "true" vector to score against

    print(f"{len(rows)} explanations + {len(ref)} released texts; free memory before loading: "
          f"{c.free_memory_gb():.0f} GB")  # fmt: skip
    ar = c.Reconstructor()
    preds = ar.reconstruct([r["explanation"] for r in rows])
    ref_preds = ar.reconstruct([r["text"] for r in ref])
    np.savez(c.work_dir() / "reconstructed.npz", preds=preds, ref_preds=ref_preds)

    out = {"rows": [], "example_reference": []}
    for r, p in zip(rows, preds, strict=True):
        g = gold(r)
        row = {k: r[k] for k in ("set", "condition", "position", "id")}
        if g is not None:
            row["mse"] = round(float(c.mse_nrm(p, g)), 4)
        if r["set"] == "quiz":
            ci, pi, ii = c.CONDITIONS.index(r["condition"]), c.POSITIONS.index(r["position"]), qid[r["id"]]
            other = q["vectors"][ci, pi, (ii + 1) % len(qid)]
            row["mse_mismatched"] = round(float(c.mse_nrm(p, other)), 4)
            row["mse_by_condition"] = {
                cond: round(float(c.mse_nrm(p, q["vectors"][k, pi, ii])), 4) for k, cond in enumerate(c.CONDITIONS)
            }
        out["rows"].append(row)
    for r, p in zip(ref, ref_preds, strict=True):
        out["example_reference"].append({"position": r["pos"], "token": r["token"], "reference_mse": r["mse"],
                                         "mse": round(float(c.mse_nrm(p, ex_vecs[r["pos"]])), 4)})  # fmt: skip
    c.save_json(c.RESULTS / "reconstruction.json", out)

    mine = [x["mse"] for x in out["rows"] if x["set"] == "example"]
    theirs = [x["mse"] for x in out["example_reference"]]
    reported = [x["reference_mse"] for x in out["example_reference"]]
    own = np.mean(mine) if mine else float("nan")
    print(f"\nworked example, positions 24+: released texts reported {np.mean(reported):.3f}, rescored here "
          f"{np.mean(theirs):.3f}; this laptop's own decodes {own:.3f}")  # fmt: skip
    for s in ("quiz", "random", "shuffled"):
        m = [x["mse"] for x in out["rows"] if x["set"] == s and "mse" in x]
        if m:
            print(f"  {s}: mean MSE {np.mean(m):.3f} over {len(m)}")


if __name__ == "__main__":
    main()
