"""CPU tests of the trainer's routing on a tiny random Qwen3 (float32).

(a) a flagged episode's loss leaves every RETAIN gradient exactly zero and the FORGET gradient non-zero, and the
    same check fails when the routing mask is removed;
(b) set_scales(retain=1, forget=0) gives the logits of base + RETAIN alone (and the merged RETAIN-only model
    matches it);
(c) an anchor row gives zero FORGET gradient;
(d) unflagged rows give gradients to both adapters;
plus the per-row scales used for mixed rollouts, and the routing tables the loop builds per arm.
"""

import copy
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import graft_loop as gl  # noqa: E402
from gradient_routing import PreRoutingGradAccumulator, set_scales  # noqa: E402

NR = NF = 4


def tiny_model(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    m = Qwen3ForCausalLM(cfg).float()
    gl.install_adapters(m, NR, NF)
    with torch.no_grad():  # down projections start at zero; randomise so every adapter param gets gradient
        for a in gl.adapters(m):
            a.down_retain.weight.normal_(0, 0.2)
            a.down_forget.weight.normal_(0, 0.2)
    return m


def rows_for(n_eps, seed=1):
    g = torch.Generator().manual_seed(seed)
    segs = []
    for _ in range(n_eps):
        segs.append(
            [
                (torch.randint(0, 97, (6,), generator=g).tolist(), torch.randint(0, 97, (5,), generator=g).tolist()),
                (torch.randint(0, 97, (9,), generator=g).tolist(), torch.randint(0, 97, (4,), generator=g).tolist()),
            ]
        )
    return gl.build_rows(segs)


def backward_one(model, ffs, rgm, fgm, adv=1.5, beta=0.05, capture=False):
    rows, ntok = rows_for(1)
    model.zero_grad(set_to_none=True)
    cap = PreRoutingGradAccumulator(model) if capture else None
    if cap:
        cap.reset()
    gl.routed_backward(
        model,
        rows,
        ntok,
        torch.tensor([adv]),
        torch.tensor([ffs]),
        torch.tensor([rgm]),
        torch.tensor([fgm]),
        torch.tensor([True]),
        beta,
        nullcontext,
        budget=10_000,
        capture=cap,
    )
    if cap:
        cap.remove()
    return gl.role_params(model)


def all_zero(params, attr="grad"):
    return all(getattr(p, attr, None) is None or bool((getattr(p, attr) == 0).all()) for p in params)


def assert_all_zero(params):
    bad = [tuple(p.shape) for p in params if p.grad is not None and bool((p.grad != 0).any())]
    assert not bad, f"{len(bad)} params have non-zero gradient"


def any_nonzero(params, attr="grad"):
    return any(getattr(p, attr, None) is not None and bool((getattr(p, attr) != 0).any()) for p in params)


# (a) ------------------------------------------------------------------------------------------------------------
def test_a_flagged_episode_zero_retain_grad():
    m = tiny_model()
    retain, forget = backward_one(m, ffs=1.0, rgm=0.0, fgm=2.0, capture=True)
    assert all(p.grad is not None for p in retain)  # the zero is exact, not a missing graph
    assert_all_zero(retain)
    assert any_nonzero(forget)
    assert any_nonzero(retain, "_pre_routing_grad")  # split-moment v still sees the natural gradient


def test_a_fails_without_mask():
    """Same flagged episode with the RETAIN mask removed (rgm=1): the zero check must fail."""
    m = tiny_model()
    retain, _ = backward_one(m, ffs=1.0, rgm=1.0, fgm=2.0)
    with pytest.raises(AssertionError):
        assert_all_zero(retain)


def test_a_via_routing_tables():
    """Masks built by the loop's own routing_tables for a group with one flagged episode; only that episode kept."""
    cfg = gl.Config(arm="graft", retain_neurons=NR, forget_neurons=NF)
    adv, ffs, rgm, fgm, keep = gl.routing_tables(cfg, [1.0, 0.0], [True, False], [False, False], 2)
    assert adv[0] != 0
    m = tiny_model()
    rows, ntok = rows_for(2)
    only_flagged = torch.tensor([True, False])
    m.zero_grad(set_to_none=True)
    gl.routed_backward(m, rows, ntok, adv, ffs, rgm, fgm, only_flagged, 0.05, nullcontext, budget=10_000)
    retain, forget = gl.role_params(m)
    assert_all_zero(retain)
    assert any_nonzero(forget)


# (b) ------------------------------------------------------------------------------------------------------------
def test_b_retain_only_equals_base_plus_retain():
    m = tiny_model()
    with torch.no_grad():
        for a in gl.adapters(m):
            a.down_forget.weight.normal_(0, 1.0)  # a large forget contribution, so ablation is visible
    ids = torch.randint(0, 97, (2, 12))
    ref = copy.deepcopy(m)  # base + RETAIN alone: the same adapters with the forget side deleted
    for a in gl.adapters(ref):
        a.gate_forget = a.up_forget = a.down_forget = None
    with torch.no_grad():
        want = ref(input_ids=ids).logits
        set_scales(m, 1.0, 0.0)
        got = m(input_ids=ids).logits
        set_scales(m, 1.0, 1.0)
        both = m(input_ids=ids).logits
        merged_r = gl.build_merged(m, "retain_only")(input_ids=ids).logits
        merged_b = gl.build_merged(m, "both")(input_ids=ids).logits
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert (both - want).abs().max() > 1e-3
    torch.testing.assert_close(merged_r, want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(merged_b, both, rtol=1e-5, atol=1e-5)


# (c) ------------------------------------------------------------------------------------------------------------
def test_c_anchor_row_zero_forget_grad():
    m = tiny_model()
    retain, forget = backward_one(m, ffs=0.0, rgm=1.0, fgm=0.0, capture=True)
    assert_all_zero(forget)
    assert all_zero(forget, "_pre_routing_grad")  # forget is forward-off: no v contribution either
    assert any_nonzero(retain)


# (d) ------------------------------------------------------------------------------------------------------------
def test_d_unflagged_rows_train_both():
    m = tiny_model()
    retain, forget = backward_one(m, ffs=1.0, rgm=1.0, fgm=1.0)
    assert all(p.grad is not None and bool((p.grad != 0).any()) for p in retain)
    assert all(p.grad is not None and bool((p.grad != 0).any()) for p in forget)


def test_d_unflagged_equals_unrouted_gradient():
    """All-ones masks reproduce the plain (no fused routing) gradient of the same loss."""
    m = tiny_model()
    backward_one(m, 1.0, 1.0, 1.0)
    routed = [p.grad.clone() for p in sum(gl.role_params(m), [])]
    rows, ntok = rows_for(1)
    m.zero_grad(set_to_none=True)
    lps, refs = [], []
    for _, seq, npr in rows:
        ids = torch.tensor([seq])
        with torch.no_grad(), gl.disabled_dual_adapters(m):
            refs.append(gl.token_logps(m, ids, torch.ones_like(ids), torch.tensor([npr]), nullcontext)[0])
        lps.append(gl.token_logps(m, ids, torch.ones_like(ids), torch.tensor([npr]), nullcontext)[0])
    lp, ref = torch.cat(lps), torch.cat(refs)
    d = ref - lp
    ((-1.5 * lp + 0.05 * (torch.exp(d) - d - 1)) / ntok[0]).sum().backward()
    for a, p in zip(routed, sum(gl.role_params(m), [])):
        torch.testing.assert_close(a, p.grad, rtol=1e-5, atol=1e-6)


# extras ---------------------------------------------------------------------------------------------------------
def test_row_scales_match_scalar_scales():
    m = tiny_model()
    ids = torch.randint(0, 97, (3, 8))
    modes = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
    with torch.no_grad():
        with gl.row_scales(m, [r for r, _ in modes], [f for _, f in modes]):
            mixed = m(input_ids=ids).logits
        for j, (r, f) in enumerate(modes):
            set_scales(m, r, f)
            torch.testing.assert_close(mixed[j], m(input_ids=ids[j : j + 1]).logits[0], rtol=1e-5, atol=1e-6)
        set_scales(m, 1.0, 1.0)


def test_routing_tables_per_arm():
    G = 4
    rewards = [1, 0, 1, 0, 1, 1, 0, 0]
    flags = [True, False, False, False, True, False, False, False]
    anchor = [False] * 4 + [True] * 4
    g = gl.Config(arm="graft", retain_neurons=NR, forget_neurons=NF)
    adv, ffs, rgm, fgm, keep = gl.routing_tables(g, rewards, flags, anchor, G)
    assert (rgm[0].item(), fgm[0].item(), ffs[0].item()) == (0.0, 2.0, 1.0)  # flagged
    assert (rgm[1].item(), fgm[1].item(), ffs[1].item()) == (1.0, 1.0, 1.0)  # unflagged
    assert (ffs[4].item(), rgm[4].item(), fgm[4].item()) == (0.0, 1.0, 0.0)  # anchor
    assert bool(keep.all())
    assert adv[4] == adv[4:8].min()  # anchor penalty: flagged anchor reward 1 - 2 = -1 gets the lowest advantage
    f = gl.Config(arm="filter")
    _, _, _, _, keep_f = gl.routing_tables(f, rewards[:4], flags[:4], [False] * 4, G)
    assert keep_f.tolist() == [False, True, True, True]
    p = gl.Config(arm="penalty")
    adv_p, *_ = gl.routing_tables(p, rewards[:4], flags[:4], [False] * 4, G)
    assert adv_p[0] < 0  # the flag zeroes a reward of 1
