"""The Jacobian lenses on sandbagkit's tiny random models: the estimator against finite differences, RelP's rules
leaving forward values alone, the readers, and the file format. No downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from sandbagkit.backends.tiny import tiny_model  # noqa: E402

from steerlab.lens import jacobian  # noqa: E402

VOCAB = 40
SKIP = 2
IDS = [3, 17, 5, 9, 22, 1, 30, 12, 7]
# Central finite differences: the tolerance reflects the step size, not the estimator. Measured max relative error
# over 5 random directions (J-lens / K-lens): step 1e-2: 4e-2 / 6e-2; 1e-3: 5e-4 / 8e-4; 1e-4: 5e-6 / 9e-6;
# 1e-5: 3e-5 / 5e-5; 1e-6: 2e-4 / 5e-4; 1e-7: 3e-3 / 4e-3. Large steps lose to truncation (error ~ step^2), small
# ones to rounding (the models' RMSNorms compute in float32 inside a float64 model), so 1e-4 is the best step.
FD_STEP = 1e-4
FD_TOLERANCE = 1e-4


def model_of(variant: str = "qwen2", dtype: Any = None) -> Any:
    model = tiny_model(VOCAB, seed=1, variant=variant).eval()
    model.set_attn_implementation("eager")
    if dtype is not None:
        model = model.to(dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def final_and(model: Any, layer: int, pos: int, delta: Any, *, embed: bool = False) -> Any:
    """The last layer's output with `delta` added to layer `layer`'s output (or the embedding) at `pos`."""
    stack = jacobian.decoder_stack(model)
    seen: dict[str, Any] = {}

    def add(module: Any, args: Any, out: Any) -> Any:
        h = jacobian._first(out).clone()
        h[0, pos] += delta
        return (h, *out[1:]) if isinstance(out, tuple) else h

    def keep(module: Any, args: Any, out: Any) -> None:
        seen["final"] = jacobian._first(out)

    target = stack.embed_tokens if embed else stack.layers[layer]
    handles = [target.register_forward_hook(add), stack.layers[-1].register_forward_hook(keep)]
    try:
        stack(input_ids=torch.tensor([IDS]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return seen["final"][0]


def layer_out(model: Any, layer: int, pos: int, delta: Any) -> Any:
    """Layer `layer`'s output with `delta` added to the embedding at `pos`."""
    stack = jacobian.decoder_stack(model)
    seen: dict[str, Any] = {}

    def add(module: Any, args: Any, out: Any) -> Any:
        out = out.clone()
        out[0, pos] += delta
        return out

    handles = [
        stack.embed_tokens.register_forward_hook(add),
        stack.layers[layer].register_forward_hook(lambda m, a, out: seen.update(h=jacobian._first(out))),
    ]
    try:
        stack(input_ids=torch.tensor([IDS]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return seen["h"][0]


@pytest.mark.parametrize("variant", ["qwen2", "qwen3", "gemma3_text"])
def test_rules_leave_the_forward_pass_alone(variant: str) -> None:
    model = model_of(variant)
    ids = torch.tensor([IDS])
    before = model(input_ids=ids).logits
    with jacobian.relp_rules(model):
        during = model(input_ids=ids).logits
    after = model(input_ids=ids).logits
    assert torch.equal(before, during) and torch.equal(before, after)
    assert all("forward" not in m.__dict__ for m in model.modules())


def test_the_j_lens_matches_finite_differences() -> None:
    model = model_of(dtype=torch.float64)
    acc = jacobian.Accumulator("J", 2, 32)
    acc.sums = {k: v.double() for k, v in acc.sums.items()}
    jacobian.fit_doc(model, IDS, acc, 0, dim_batch=7, skip_first=SKIP)
    J0 = acc.mean("out")[0]
    valid = jacobian.valid_positions(len(IDS), SKIP)
    v = torch.randn(32, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    eps = FD_STEP
    fd = torch.zeros(32, dtype=torch.float64)
    for t in valid:
        diff = final_and(model, 0, t, eps * v) - final_and(model, 0, t, -eps * v)
        fd += diff[valid].sum(0) / (2 * eps)
    fd /= len(valid)
    assert ((J0 @ v - fd).norm() / fd.norm()) < FD_TOLERANCE
    assert ((J0.T @ v - fd).norm() / fd.norm()) > 0.1  # the check can tell J from its transpose
    assert torch.equal(acc.mean("out")[1], torch.eye(32, dtype=torch.float64))


def test_the_k_lens_matches_finite_differences() -> None:
    model = model_of(dtype=torch.float64)
    acc = jacobian.Accumulator("S", 2, 32)
    acc.sums = {k: v.double() for k, v in acc.sums.items()}
    jacobian.fit_doc(model, IDS, acc, 0, dim_batch=5, skip_first=SKIP, rules=False)  # true derivative: the K-lens
    K1 = acc.mean("out")[1]
    valid = jacobian.valid_positions(len(IDS), SKIP)
    v = torch.randn(32, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    eps = FD_STEP
    fd = torch.zeros(32, dtype=torch.float64)
    for s in valid:  # Σ_{s≤t} over source positions, then the mean over targets t
        diff = layer_out(model, 1, s, eps * v) - layer_out(model, 1, s, -eps * v)
        fd += diff[valid].sum(0) / (2 * eps)
    fd /= len(valid)
    assert ((K1 @ v - fd).norm() / fd.norm()) < FD_TOLERANCE
    assert ((K1.T @ v - fd).norm() / fd.norm()) > 0.1


def test_rules_change_the_derivative_and_both_directions_run() -> None:
    model = model_of()
    J = jacobian.fit(model, [IDS, IDS[::-1]], "J", dim_batch=16, skip_first=SKIP)
    R = jacobian.fit(model, [IDS, IDS[::-1]], "R", dim_batch=16, skip_first=SKIP)
    S = jacobian.fit(model, [IDS], "S", dim_batch=16, skip_first=SKIP)
    assert J.n_docs == [1, 1] and not torch.allclose(J.mean()[0], R.mean()[0])
    assert S.mean("out").abs().sum() > 0 and S.mean("mid").abs().sum() > 0
    assert len(R.split_half()) == 2 and R.split_half()[0] >= 0


def test_the_logit_lens_at_the_last_layer_is_the_real_output() -> None:
    model = model_of()
    reader = jacobian.Reader(model, {"R": {0: torch.eye(32), 1: torch.eye(32)}})
    out = model(input_ids=torch.tensor([IDS]), output_hidden_states=True)
    stack = jacobian.decoder_stack(model)
    seen: dict[str, Any] = {}
    handle = stack.layers[-1].register_forward_hook(lambda m, a, o: seen.update(h=jacobian._first(o)))
    stack(input_ids=torch.tensor([IDS]))
    handle.remove()
    for lens in ("logit", "R"):
        assert torch.allclose(reader.logits(seen["h"][0], 1, lens), out.logits[0], atol=1e-5)


def test_layer_contrastive_reads_divide_by_a_per_token_scale() -> None:
    model = model_of()
    reader = jacobian.Reader(model, {})
    h = torch.randn(32)
    scale = reader.z_scale(0, "logit")
    assert scale.shape == (VOCAB,) and bool((scale > 0).all())
    unit = h * reader.g / (h * reader.g).norm()
    assert torch.allclose(reader.z_logits(h, 0, "logit"), (unit @ reader.W_U.T) / scale)
    ids, vals = reader.top(h, 0, "logit", k=3, z=True)
    assert len(ids) == 3 and vals == sorted(vals, reverse=True)
    with pytest.raises(ValueError, match="not loaded"):
        reader.logits(h, 0, "S")
    with pytest.raises(ValueError, match="layer 5"):
        reader.logits(h, 5, "logit")


def test_lenses_save_load_and_report_availability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEERLAB_LENS_DIR", str(tmp_path))
    model = model_of()
    acc = jacobian.fit(model, [IDS], "S", dim_batch=16, skip_first=SKIP)
    blob = acc.blob("org/Tiny-Model", date="2026-11-22")
    assert set(blob) == {"K", "n_prompts", "d_model", "provenance"} and blob["provenance"]["lens"] == "S"
    jacobian.save(blob, jacobian.own_path("org/Tiny-Model", "S"))
    assert (tmp_path / "tiny-model" / "s-lens.json").exists()
    loaded = jacobian.load_one("org/Tiny-Model", "S")
    assert torch.equal(loaded[1], acc.mean("out")[1])
    status = jacobian.availability("org/Tiny-Model")
    assert status["logit"]["status"] == status["S"]["status"] == "ready" and status["R"]["status"] == "needs a fit"
    monkeypatch.setattr(jacobian, "_hub_cached", lambda model_id, lens: False)
    hub = jacobian.availability("Qwen/Qwen3-8B")
    assert hub["R"]["status"] == "downloadable" and hub["R"]["source"].endswith("qwen3-8b/r-lens/lens.pt")
    assert hub["J"]["status"] == "needs a fit"
    assert jacobian.load("org/Tiny-Model", ("logit", "J", "S")).keys() == {"S"}
    assert jacobian.slug("Qwen/Qwen2.5-0.5B-Instruct") == "qwen2.5-0.5b-instruct"
