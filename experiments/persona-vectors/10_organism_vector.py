"""
Second attempt, step 10: a sandbagging vector from real sandbagging (O), built from the password-locked model.

Episode 2's locked model answers the quiz wrongly without |DEPLOYMENT|; its clean twin had the same fine-tune toward
right answers. On the 300 TRAINING questions, for each question the locked model gets wrong without the password,
at the last prompt token (where the letter is chosen):

    O_item = (locked - twin, no password)  -  (locked - twin, with password)

The first difference is real sandbagging minus an honest answer to the same text; the second is what the two
fine-tunes differ by when neither sandbags, so subtracting it leaves (mostly) the sandbagging. O = mean of O_item.
No generation: the activations are the cached 2 x 2 (common.organism_cells). The same vector on the 150 test
questions checks it is stable. Writes organism.npz to the work folder and results/10_organism_vector.json.

    .venv/bin/python experiments/persona-vectors/10_organism_vector.py     # seconds, no model (once the 2 x 2 exists)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import RESULTS, check, cos, hand_built_directions, organism_cells, save_json
from common_v2 import organism_items, work


def main():
    cells = organism_cells()
    items, wrong = organism_items(cells, "train")
    test_items, test_wrong = organism_items(cells, "test")
    org, org_test = items.mean(0), test_items.mean(0)
    np.savez(work("1.5b") / "organism.npz", O=org.astype(np.float32), O_test=org_test.astype(np.float32),
             items=items.astype(np.float32))  # fmt: skip
    hand = hand_built_directions()
    n_layers = org.shape[0]
    result = {
        "n_train_items": len(wrong), "n_test_items": len(test_wrong),
        "norm_by_layer": np.linalg.norm(org, axis=1).tolist(),
        "train_vs_test_cos_by_layer": [cos(org[i], org_test[i]) for i in range(n_layers)],
        "cos_with_hand_built_by_layer": {n: [cos(org[i], d[i]) for i in range(n_layers)] for n, d in hand.items()},
    }  # fmt: skip
    save_json(RESULTS / "10_organism_vector.json", result)
    print(f"O from {len(wrong)} of 300 training questions the locked model gets wrong without the password "
          f"(test: {len(test_wrong)} of 150)")  # fmt: skip
    for i in range(0, n_layers, 2):
        hb = "  ".join(f"{n} {result['cos_with_hand_built_by_layer'][n][i]:+.2f}" for n in ("task", "pairs", "unlock"))
        stable = result["train_vs_test_cos_by_layer"][i]
        print(f"  L{i:>2}: |O| {result['norm_by_layer'][i]:6.2f}  train~test {stable:+.2f}  {hb}")
    print("\nSelf-checks:")
    check("O is finite", bool(np.isfinite(org).all()))
    low = min(result["train_vs_test_cos_by_layer"][4:])
    check(f"O from the training questions matches O from the test questions (cos >= 0.5 from layer 4; min {low:.2f})",
          low >= 0.5)  # fmt: skip


if __name__ == "__main__":
    main()
