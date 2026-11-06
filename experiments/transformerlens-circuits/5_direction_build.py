"""Step 5: rebuild the sandbagging direction from public data and check it matches the shipped one.

    python experiments/transformerlens-circuits/5_direction_build.py   # ~1 min; downloads 32 prompt pairs (MIT)

The direction in results/sandbagging_direction_l14.npz is a difference of means ("contrastive activation addition"):
run Qwen2.5-0.5B-Instruct on each of the ai-metacognition-toolkit's 32 sandbagging/genuine prompt pairs (raw text, no
chat template), keep the last token's activation at the output of decoder layer 14, and subtract the mean over the
genuine prompts from the mean over the sandbagging prompts. Only the 25 training pairs of the toolkit's own split
(np.random.seed(42); permutation(32); first 25) are used, so the 7 test pairs stay unseen. As a probe the direction
needs no training: score = activation · direction, threshold halfway between the two training means.

This script repeats the recipe with Hugging Face Transformers on the CPU, saves the result to the work folder (the
shipped file is never overwritten), and reports the cosine with the shipped vector. Writes results/direction_build.json.
"""

import json
import urllib.request

import numpy as np
import torch
from common import DIRECTION_FILE, check, work_dir, write_json
from direction_tools import MODEL_NAME
from transformers import AutoModelForCausalLM, AutoTokenizer

PAIRS_URL = (
    "https://raw.githubusercontent.com/bassrehab/ai-metacognition-toolkit/main/"
    "experiments/data/sandbagging_contrast_pairs.json"
)
LAYER = 14
torch.set_grad_enabled(False)

with urllib.request.urlopen(PAIRS_URL, timeout=30) as response:
    data = json.load(response)
pairs = [pair for group in data["contrast_pairs"] for pair in group["pairs"]]
np.random.seed(42)
order = np.random.permutation(len(pairs))
train_pairs, test_pairs = order[:25], order[25:]
print(f"1. {len(pairs)} prompt pairs from the ai-metacognition-toolkit; 25 train, 7 test (seed 42)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).eval()
X, labels, pair_index = [], [], []
for p, pair in enumerate(pairs):
    for text, label in ((pair["sandbagging"], 1), (pair["genuine"], 0)):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        hidden = model(**inputs, output_hidden_states=True).hidden_states[LAYER + 1]  # [0] = embeddings
        X.append(hidden[0, -1].float().numpy())
        labels.append(label)
        pair_index.append(p)
X, labels, pair_index = np.array(X, dtype=np.float64), np.array(labels), np.array(pair_index)

train = np.isin(pair_index, train_pairs)
sandbag_mean, genuine_mean = X[train & (labels == 1)].mean(0), X[train & (labels == 0)].mean(0)
vector = sandbag_mean - genuine_mean
threshold = (sandbag_mean @ vector + genuine_mean @ vector) / 2
test = np.isin(pair_index, test_pairs)
scores = X[test] @ vector
accuracy = float(np.mean((scores > threshold).astype(int) == labels[test]))
pos, neg = scores[labels[test] == 1], scores[labels[test] == 0]
auroc = float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg]))
print(
    f"2. rebuilt direction: length {np.linalg.norm(vector):.3f}; as a probe on the 14 held-out prompts accuracy "
    f"{accuracy:.1%}, AUROC {auroc:.3f}"
)

shipped = np.load(DIRECTION_FILE)
cosine = float(vector @ shipped["vector"] / (np.linalg.norm(vector) * np.linalg.norm(shipped["vector"])))
same_split = sorted(shipped["train_pairs"].tolist()) == sorted(train_pairs.tolist())
print(
    f"3. against the shipped vector: cosine {cosine:.5f}; lengths {np.linalg.norm(vector):.3f} (rebuilt) vs "
    f"{np.linalg.norm(shipped['vector']):.3f} (shipped); same training pairs: {same_split}"
)
out = work_dir() / "rebuilt_direction_l14.npz"
np.savez(out, vector=vector.astype(np.float32), layer=LAYER, train_pairs=train_pairs)
print(f"   saved the rebuilt vector to {out.name} in the work folder")

write_json(
    "direction_build.json",
    {
        "source": PAIRS_URL,
        "n_pairs": len(pairs),
        "train_pairs": train_pairs.tolist(),
        "layer": LAYER,
        "rebuilt_norm": float(np.linalg.norm(vector)),
        "shipped_norm": float(np.linalg.norm(shipped["vector"])),
        "cosine_with_shipped": cosine,
        "heldout_accuracy": accuracy,
        "heldout_auroc": auroc,
    },
)

print("\nSelf-checks:")
check("the rebuild uses the same 25 training pairs as the shipped vector", same_split)
check("the rebuilt and shipped vectors point the same way (cosine > 0.99)", cosine > 0.99)
check("as a probe it ranks held-out prompts better than chance (AUROC > 0.5)", auroc > 0.5)
