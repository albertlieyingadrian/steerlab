"""Step 2: find GPT-2 small's induction heads with one call to TransformerLens's head detector.

    python experiments/transformerlens-circuits/2_induction.py   # ~10 s with gpt2 cached

An induction head implements "what came after this token last time?": on  A B C ... A  it attends from the second A
to B and predicts B. Twenty-five random words repeated once is the cleanest test: the first half is unpredictable,
the second half is predictable only by looking back and copying.

head_detector.detect_head(model, text, "induction_head") scores every head by the share of its attention that lands
where an induction head's would. GPT-2 parks spare attention on the BOS token, which drags raw scores down;
exclude_bos=True measures the pattern among the real tokens. Both are recorded.
Writes results/induction.json, results/induction_scores.csv (exclude_bos), results/induction_scores_raw.csv and
results/induction_pattern.csv (the top head's 51 x 51 attention pattern).
"""

import torch
from common import boot, check, write_grid_csv, write_json
from transformer_lens import head_detector

HALF = 25  # words per half; repeated makes 50, BOS makes 51
model, device = boot("gpt2")
n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

# --- 1. A repeated random sequence ---------------------------------------------------------------------------------
# detect_head takes text, so the random tokens must survive a round trip through the tokenizer: keep ids that decode
# to " word" and encode back to themselves.
tokenizer = model.tokenizer
generator = torch.Generator().manual_seed(0)
chosen = []
while len(chosen) < HALF:
    token_id = torch.randint(300, 20000, (1,), generator=generator).item()
    piece = tokenizer.decode([token_id])
    if (
        piece.startswith(" ")
        and piece[1:].isalpha()
        and tokenizer.encode(piece, add_special_tokens=False) == [token_id]
    ):
        chosen.append(token_id)
repeated = torch.tensor([chosen + chosen])
text = model.to_string(repeated[0])
tokens = model.to_tokens(text)
round_trip_exact = torch.equal(tokens[0, 1:].cpu(), repeated[0])
print(f"1. {HALF} random words, twice: {text[:60]}...  ({tokens.shape[1]} tokens with BOS)")

logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
log_probs = logits[0].log_softmax(-1)
next_lp = log_probs[:-1].gather(-1, tokens[0, 1:, None]).squeeze(-1)
loss_first, loss_second = -next_lp[:HALF].mean().item(), -next_lp[HALF:].mean().item()
print(f"   mean loss: first half {loss_first:.2f} nats, second half {loss_second:.2f} nats")

# --- 2. The detector ------------------------------------------------------------------------------------------------
scores_raw = head_detector.detect_head(model, text, "induction_head", cache=cache).cpu()
scores = head_detector.detect_head(model, text, "induction_head", cache=cache, exclude_bos=True).cpu()


def top(grid, k=5):
    flat = grid.flatten().topk(k)
    return [
        (f"L{int(i) // n_heads}H{int(i) % n_heads}", round(float(v), 4))
        for v, i in zip(flat.values, flat.indices, strict=False)
    ]


print("\n2. head_detector.detect_head(..., 'induction_head')")
print("   top 5, raw:              " + ", ".join(f"{h} {s:.2f}" for h, s in top(scores_raw)))
print("   top 5, exclude_bos=True: " + ", ".join(f"{h} {s:.2f}" for h, s in top(scores)))
best = int(scores.flatten().argmax())
layer, head = best // n_heads, best % n_heads
pattern = cache["pattern", layer, "attn"][head].cpu()  # (query, key)
queries = torch.arange(HALF + 1, 2 * HALF + 1)
on_diagonal = pattern[queries, queries - (HALF - 1)].mean().item()
on_bos = pattern[queries, 0].mean().item()
print(
    f"\n3. top head L{layer}H{head}: second-half queries put {on_diagonal:.1%} of their attention {HALF - 1} "
    f"positions back"
    f" (the token after their first copy) and {on_bos:.1%} on BOS"
)

write_json(
    "induction.json",
    {
        "half": HALF,
        "loss_first_half": loss_first,
        "loss_second_half": loss_second,
        "top_raw": top(scores_raw),
        "top_exclude_bos": top(scores),
        "top_head": f"L{layer}H{head}",
        "top_head_on_induction_cell": on_diagonal,
        "top_head_on_bos": on_bos,
    },
)
heads = [f"H{h}" for h in range(n_heads)]
layers = [f"L{i}" for i in range(n_layers)]
write_grid_csv("induction_scores.csv", scores, layers, heads, corner="layer")
write_grid_csv("induction_scores_raw.csv", scores_raw, layers, heads, corner="layer")
n = pattern.shape[0]
write_grid_csv("induction_pattern.csv", pattern, [f"q{i}" for i in range(n)], [f"k{i}" for i in range(n)], "query")

print("\nSelf-checks:")
check("the random sequence survives the text round trip (BOS + 50 tokens)", round_trip_exact and tokens.shape[1] == 51)
check("the second half is far easier to predict (loss at least halved)", loss_second < 0.5 * loss_first)
check("the top induction head is in layers 5-7 and scores above 0.5", 5 <= layer <= 7 and scores.max() > 0.5)
check("its second-half queries put most attention on the induction cell", on_diagonal > 0.5)
