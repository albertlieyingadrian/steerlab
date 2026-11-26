"""CPU gate tests for the GRAFT routing port (pre-registration, "Trainer" section):

(a) a flagged sample leaves every RETAIN gradient exactly zero;
(b) FORGET scale 0 makes the forward pass equal base plus RETAIN alone.

  .venv/bin/python -m pytest experiments/graft-monitor/halo/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import graft_routing as gr  # noqa: E402

I, NR, NF = 24, 4, 4


def _config(intermediate: int) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=intermediate,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )


def _models(seed: int = 0, train_adapters: bool = True):
    """(base, widened-and-routed) pair sharing every base weight. With train_adapters the RETAIN/FORGET down columns
    are made nonzero (as after training), otherwise they keep GRAFT's zero init."""
    torch.manual_seed(seed)
    base = Qwen3ForCausalLM(_config(I)).eval()
    wide_sd = gr.widen_state_dict(base.state_dict(), NR, NF, seed=seed)
    if train_adapters:
        g = torch.Generator().manual_seed(seed + 1)
        for k, w in wide_sd.items():
            if k.endswith("mlp.down_proj.weight"):
                w[:, I:] = torch.randn(w.shape[0], NR + NF, generator=g) * 0.5
    wide = Qwen3ForCausalLM(_config(I + NR + NF)).eval()
    wide.load_state_dict(wide_sd)
    gr.apply_routed_mlp(wide, I, NR, NF)
    return base, wide


def _batch(n: int = 3, t: int = 7, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 64, (n, t), generator=g)


def _grads(model, ids, masks=None):
    model.zero_grad(set_to_none=False)
    for p in model.parameters():
        if p.grad is None and p.requires_grad:
            p.grad = torch.zeros_like(p)
    if masks is None:
        loss = model(input_ids=ids).logits.float().pow(2).mean()
        loss.backward()
    else:
        with gr.routing(*masks):
            loss = model(input_ids=ids).logits.float().pow(2).mean()
            loss.backward()
    return {role: [w.grad[idx].clone() for _, w, idx in items] for role, items in gr.role_slices(model).items()}


def test_widened_checkpoint_is_the_base_function_at_init():
    base, wide = _models(train_adapters=False)
    ids = _batch()
    torch.testing.assert_close(wide(input_ids=ids).logits, base(input_ids=ids).logits, rtol=0, atol=1e-5)


def test_a_flagged_sample_leaves_every_retain_gradient_exactly_zero():
    """Gate (a). Every row flagged: (fwd 1, RETAIN 0, FORGET 2)."""
    _, wide = _models()
    ids = _batch(n=3)
    fwd, rgm, fgm = gr.episode_masks(torch.ones(3), torch.zeros(3))
    assert fwd.tolist() == [1, 1, 1] and rgm.tolist() == [0, 0, 0] and fgm.tolist() == [2, 2, 2]
    grads = _grads(wide, ids, (fwd, rgm, fgm))
    assert all(torch.count_nonzero(g) == 0 for g in grads["retain"])
    assert all(torch.count_nonzero(g) == 0 for g in grads["base"]), "base must stay frozen"
    assert any(torch.count_nonzero(g) > 0 for g in grads["forget"])
    # Attention, norms and embeddings are frozen too.
    trainable = {n for n, p in wide.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(("gate_proj.weight", "up_proj.weight", "down_proj.weight")) for n in trainable)


def test_flagged_row_adds_nothing_to_retain_in_a_mixed_batch():
    """Gate (a), mixed microbatch: RETAIN's gradient over [flagged, unflagged, unflagged] equals its gradient over the
    unflagged rows alone; the flagged row's FORGET gradient is twice its natural one (GRAFT kappa_F = 2)."""
    _, wide = _models()
    ids = _batch(n=3)
    mixed = _grads(wide, ids, gr.episode_masks(torch.tensor([1.0, 0, 0]), torch.zeros(3)))
    # Same rows, flagged row's RETAIN and FORGET both gated off: what the unflagged rows alone contribute.
    alone = _grads(wide, ids, (torch.ones(3), torch.tensor([0.0, 1, 1]), torch.tensor([0.0, 1, 1])))
    for a, b in zip(mixed["retain"], alone["retain"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    only_flagged_natural = _grads(wide, ids, (torch.ones(3), torch.zeros(3), torch.tensor([1.0, 0, 0])))
    for m, a, f in zip(mixed["forget"], alone["forget"], only_flagged_natural["forget"]):
        torch.testing.assert_close(m, a + 2 * f, rtol=1e-5, atol=1e-7)


def test_routing_leaves_the_forward_value_unchanged():
    """_fused_decouple changes gradients only: logits under unflagged masks equal the unrouted forward."""
    _, wide = _models()
    ids = _batch()
    ref = wide(input_ids=ids).logits
    with gr.routing(*gr.episode_masks(torch.tensor([1.0, 0, 1]), torch.zeros(3))):
        got = wide(input_ids=ids).logits
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-6)


def test_forget_scale_zero_equals_base_plus_retain_alone():
    """Gate (b). Three readings of 'base + RETAIN alone' agree with set_scales(model, 1, 0):
    the same model's base and RETAIN parts summed by hand (bitwise), the anchor row's forward (FORGET forward scale 0,
    bitwise), and an independent stock Qwen3 holding only base + RETAIN neurons (the deployed export)."""
    _, wide = _models()
    ids = _batch()
    gr.set_scales(wide, 1.0, 0.0)
    got = wide(input_ids=ids).logits

    # (i) by hand: every MLP returns base + RETAIN, nothing of FORGET.
    def by_hand(self, x):
        return self._core(x, 0, self.graft_base, True) + self._retain_core(x) * self.retain_scale

    mlps = [m for _, m in gr.find_mlps(wide)]
    saved = [m.forward for m in mlps]
    for m in mlps:
        m.forward = by_hand.__get__(m)
    hand = wide(input_ids=ids).logits
    for m, f in zip(mlps, saved):
        m.forward = f
    assert torch.equal(got, hand)

    # (ii) the anchor row: FORGET forward scale 0 through the fused path.
    gr.set_scales(wide, 1.0, 1.0)
    with gr.routing(*gr.episode_masks(torch.zeros(3), torch.ones(3))):
        anchor = wide(input_ids=ids).logits
    torch.testing.assert_close(anchor, got, rtol=0, atol=0)

    # (iii) the deployed RETAIN-only export, loaded into a stock (unrouted) Qwen3 of intermediate I + n_R.
    export = Qwen3ForCausalLM(_config(I + NR)).eval()
    export.load_state_dict(gr.retain_only_state_dict(wide.state_dict(), I, NR, NF))
    torch.testing.assert_close(export(input_ids=ids).logits, got, rtol=0, atol=1e-5)

    # And FORGET does matter at scale 1 (the test is not vacuous).
    assert not torch.allclose(wide(input_ids=ids).logits, got, atol=1e-3)


def test_served_widened_model_is_base_plus_retain_plus_forget():
    """What vLLM serves (a stock Qwen3 of intermediate I + n_R + n_F with the pushed weights) is the training policy."""
    _, wide = _models()
    ids = _batch()
    stock = Qwen3ForCausalLM(_config(I + NR + NF)).eval()
    stock.load_state_dict(wide.state_dict())
    torch.testing.assert_close(stock(input_ids=ids).logits, wide(input_ids=ids).logits, rtol=0, atol=1e-5)


def test_chunked_forwards_read_their_own_rows():
    """TRL forwards a micro-batch in batch_size chunks: with the row cursor, two 2-row forwards under 4-row masks give
    the gradients of one 4-row forward."""
    _, wide = _models()
    ids = _batch(n=4)
    masks = gr.episode_masks(torch.tensor([1.0, 0, 0, 1]), torch.tensor([0.0, 0, 1, 0]))
    whole = _grads(wide, ids, masks)  # one forward over all 4 rows; loss = mean of logits^2
    handle = gr.install_row_cursor(wide)
    try:
        wide.zero_grad(set_to_none=False)
        with gr.routing(*masks):
            loss = sum(wide(input_ids=ids[i : i + 2]).logits.float().pow(2).sum() for i in (0, 2))
            (loss / (4 * 7 * 64)).backward()
        chunked = {role: [w.grad[idx].clone() for _, w, idx in items] for role, items in gr.role_slices(wide).items()}
        for role in ("retain", "forget"):
            for a, b in zip(whole[role], chunked[role]):
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-7)
        with pytest.raises(RuntimeError), gr.routing(*masks), torch.no_grad():
            wide(input_ids=ids)  # consumes all 4 armed rows
            wide(input_ids=ids[:1])  # a 5th row is refused
    finally:
        handle.remove()


def test_masks_and_advantages_match_graft():
    fwd, rgm, fgm = gr.episode_masks(torch.tensor([0.0, 1, 0, 1]), torch.tensor([0.0, 0, 1, 1]))
    assert list(zip(fwd.tolist(), rgm.tolist(), fgm.tolist())) == [(1, 1, 1), (1, 0, 2), (0, 1, 0), (0, 1, 0)]
    r = torch.tensor([1.0, 0, 1, 0, 1, 1, 0, 0])
    flag = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0]).bool()
    anchor = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).bool()
    adv = gr.routed_advantages(r, flag, anchor, G=4)
    # Routing group: flagged sample keeps the full-group advantage; unflagged renorm over the unflagged subset.
    full = gr._full_group_renorm(r, 4)
    assert adv[0] == full[0]
    sub = r[1:4]
    torch.testing.assert_close(adv[1:4], (sub - sub.mean()) / (sub.std() + 1e-4))
    # Anchor group, coherence_rh_mode 'filter': the flagged anchor sample gets 0, the rest the full-group advantage.
    assert adv[4] == 0 and torch.equal(adv[5:], full[5:])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
