"""The two raw SAE loaders on synthetic files, plus the gated-download refusal.

Synthetic Qwen-Scope file: the release's dict layout {"W_enc", "W_dec", "b_enc", "b_dec"} (no k stored; k comes from
the release name `L0_50`). Synthetic Gemma Scope 2 file: safetensors {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from steerlab.hub import GatedRepoAccessError, ensure_access  # noqa: E402
from steerlab.saes import gemma_scope, goodfire, qwen_scope  # noqa: E402
from steerlab.saes.common import FeatureDirection, LoadedSAE  # noqa: E402

D_MODEL = 16
N_FEATURES = 64


@pytest.fixture
def qwen_file(tmp_path: Path) -> Path:
    g = torch.Generator().manual_seed(0)
    state = {
        "W_enc": torch.randn(N_FEATURES, D_MODEL, generator=g),
        "W_dec": torch.randn(D_MODEL, N_FEATURES, generator=g),
        "b_enc": torch.randn(N_FEATURES, generator=g),
        "b_dec": torch.randn(D_MODEL, generator=g),
    }
    path = tmp_path / "layer14.sae.pt"
    torch.save(state, path)
    return path


@pytest.fixture
def gemma_file(tmp_path: Path) -> Path:
    g = torch.Generator().manual_seed(1)
    state = {
        "w_enc": torch.randn(D_MODEL, N_FEATURES, generator=g),
        "w_dec": torch.randn(N_FEATURES, D_MODEL, generator=g),
        "b_enc": torch.zeros(N_FEATURES),
        "b_dec": torch.randn(D_MODEL, generator=g),
        "threshold": torch.full((N_FEATURES,), 0.5),
    }
    path = tmp_path / "params.safetensors"
    safetensors_torch.save_file(state, str(path))
    return path


# ---- Qwen-Scope (TopK) ----


def test_qwen_loader_reads_the_release_layout_and_k_from_the_release_name(qwen_file: Path):
    sae = qwen_scope.load_qwen_scope(qwen_file, layer=14)
    assert isinstance(sae, LoadedSAE)
    assert sae.release == qwen_scope.DEFAULT_RELEASE
    assert sae.layer == 14
    assert (sae.d_model, sae.n_features) == (D_MODEL, N_FEATURES)
    assert sae.k == 50  # from "...-L0_50"


def test_qwen_k_from_release_name():
    assert qwen_scope.k_from_release("Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50") == 50
    assert qwen_scope.k_from_release("Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_100") == 100


def test_qwen_encode_keeps_exactly_k_nonzeros_without_relu(qwen_file: Path):
    sae = qwen_scope.load_qwen_scope(qwen_file, layer=14, k=5)
    h = torch.randn(3, 7, D_MODEL)
    acts = sae.encode(h)
    assert acts.shape == (3, 7, N_FEATURES)
    assert acts.dtype == torch.float32
    assert torch.all((acts != 0).sum(dim=-1) == 5)
    # No ReLU: the top-k values are kept even when negative. Force that case with a huge negative encoder bias.
    sae_neg = qwen_scope.QwenScopeSAE(
        W_enc=sae.W_enc, b_enc=torch.full((N_FEATURES,), -1000.0), W_dec=sae.W_dec, b_dec=sae.b_dec, k=5, layer=14
    )
    acts_neg = sae_neg.encode(h)
    assert torch.all((acts_neg != 0).sum(dim=-1) == 5)
    assert torch.all(acts_neg[acts_neg != 0] < 0)


def test_qwen_direction_is_the_unit_normalised_column_of_w_dec(qwen_file: Path):
    sae = qwen_scope.load_qwen_scope(qwen_file, layer=14)
    f = 17
    d = sae.direction(f)
    assert isinstance(d, FeatureDirection)
    assert (d.feature, d.layer, d.release) == (f, 14, sae.release)
    expected = sae.W_dec[:, f] / sae.W_dec[:, f].norm()
    torch.testing.assert_close(d.unit, expected)
    assert d.unit.shape == (D_MODEL,)
    assert d.unit.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_qwen_reconstruct_shape_and_identity_round_trip(qwen_file: Path):
    sae = qwen_scope.load_qwen_scope(qwen_file, layer=14)
    h = torch.randn(5, D_MODEL)
    assert sae.reconstruct(h).shape == (5, D_MODEL)

    eye = torch.eye(D_MODEL)
    identity = qwen_scope.QwenScopeSAE(
        W_enc=eye, b_enc=torch.zeros(D_MODEL), W_dec=eye, b_dec=torch.zeros(D_MODEL), k=D_MODEL, layer=0
    )
    torch.testing.assert_close(identity.reconstruct(h), h, atol=1e-6, rtol=0)


def test_qwen_math_runs_in_float32_even_for_half_inputs(qwen_file: Path):
    sae = qwen_scope.load_qwen_scope(qwen_file, layer=14)
    h = torch.randn(2, D_MODEL).to(torch.bfloat16)
    assert sae.encode(h).dtype == torch.float32
    assert sae.reconstruct(h).dtype == torch.float32


def test_qwen_loader_rejects_a_file_with_the_wrong_keys(tmp_path: Path):
    path = tmp_path / "bad.pt"
    torch.save({"encoder": torch.zeros(2, 2)}, path)
    with pytest.raises(ValueError, match="W_enc"):
        qwen_scope.load_qwen_scope(path, layer=0)


# ---- Gemma Scope 2 (JumpReLU) ----


def test_gemma_loader_reads_safetensors(gemma_file: Path):
    sae = gemma_scope.load_gemma_scope(gemma_file, layer=13)
    assert isinstance(sae, LoadedSAE)
    assert sae.release == gemma_scope.DEFAULT_RELEASE
    assert sae.layer == 13
    assert (sae.d_model, sae.n_features) == (D_MODEL, N_FEATURES)


def test_gemma_encode_zeroes_pre_activations_at_or_below_threshold(gemma_file: Path):
    sae = gemma_scope.load_gemma_scope(gemma_file, layer=13)
    h = torch.randn(4, D_MODEL)
    pre = h @ sae.w_enc + sae.b_enc
    acts = sae.encode(h)
    assert acts.shape == (4, N_FEATURES)
    expected = torch.where(pre > sae.threshold, pre, torch.zeros_like(pre))
    torch.testing.assert_close(acts, expected)
    assert torch.all(acts[pre <= sae.threshold] == 0)
    assert (acts != 0).any()  # the threshold is not so high that nothing fires

    # Exactly-at-threshold is zeroed ("at or below").
    at = gemma_scope.GemmaScopeSAE(
        w_enc=torch.eye(D_MODEL),
        w_dec=torch.eye(D_MODEL),
        b_enc=torch.zeros(D_MODEL),
        b_dec=torch.zeros(D_MODEL),
        threshold=torch.full((D_MODEL,), 1.0),
        layer=13,
    )
    acts_at = at.encode(torch.ones(1, D_MODEL))
    assert torch.all(acts_at == 0)


def test_gemma_direction_is_the_unit_normalised_row_of_w_dec(gemma_file: Path):
    sae = gemma_scope.load_gemma_scope(gemma_file, layer=13)
    f = 42
    d = sae.direction(f)
    assert (d.feature, d.layer, d.release) == (f, 13, sae.release)
    expected = sae.w_dec[f] / sae.w_dec[f].norm()
    torch.testing.assert_close(d.unit, expected)
    assert d.unit.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_gemma_reconstruct_shape_and_identity_round_trip(gemma_file: Path):
    sae = gemma_scope.load_gemma_scope(gemma_file, layer=13)
    h = torch.randn(2, 3, D_MODEL)
    assert sae.reconstruct(h).shape == (2, 3, D_MODEL)

    identity = gemma_scope.GemmaScopeSAE(
        w_enc=torch.eye(D_MODEL),
        w_dec=torch.eye(D_MODEL),
        b_enc=torch.zeros(D_MODEL),
        b_dec=torch.zeros(D_MODEL),
        threshold=torch.full((D_MODEL,), -1e9),  # everything fires
        layer=13,
    )
    torch.testing.assert_close(identity.reconstruct(h), h, atol=1e-6, rtol=0)


def test_gemma_loader_rejects_a_file_without_threshold(tmp_path: Path):
    path = tmp_path / "params.safetensors"
    safetensors_torch.save_file({"w_enc": torch.zeros(2, 3), "w_dec": torch.zeros(3, 2)}, str(path))
    with pytest.raises(ValueError, match="threshold"):
        gemma_scope.load_gemma_scope(path, layer=13)


def test_gemma_default_path_for_layer():
    assert gemma_scope.params_path(13) == "resid_post/layer_13_width_16k_l0_medium/params.safetensors"


def test_qwen_default_filename_for_layer():
    assert qwen_scope.filename(14) == "layer14.sae.pt"


# ---- gated downloads ----


def _gated_error() -> Exception:
    """What huggingface_hub.auth_check raises for a gated repo; its constructor reads headers off the response."""
    from types import SimpleNamespace

    from huggingface_hub.errors import GatedRepoError

    response = SimpleNamespace(headers={}, request=None, status_code=401)
    return GatedRepoError("401 Client Error: gated", response=response)  # type: ignore[arg-type]


def test_gated_repo_without_access_raises_naming_the_licence_page(monkeypatch: pytest.MonkeyPatch):
    import huggingface_hub

    def fake_auth_check(repo_id: str, **_: object) -> None:
        raise _gated_error()

    monkeypatch.setattr(huggingface_hub, "auth_check", fake_auth_check)
    monkeypatch.setenv("HF_TOKEN", "hf_FAKE_TOKEN_must_not_appear")
    with pytest.raises(GatedRepoAccessError) as info:
        ensure_access("google/gemma-3-1b-it")
    message = str(info.value)
    assert "https://huggingface.co/google/gemma-3-1b-it" in message
    assert "gated" in message
    assert "hf_FAKE_TOKEN" not in message


def test_gated_check_passes_through_when_auth_check_succeeds(monkeypatch: pytest.MonkeyPatch):
    import huggingface_hub

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(huggingface_hub, "auth_check", lambda repo_id, **kw: calls.append((repo_id, kw.get("token"))))
    ensure_access("Qwen/Qwen3-1.7B")
    assert calls == [("Qwen/Qwen3-1.7B", None)]  # no token configured -> huggingface_hub's own resolution


def test_repo_token_variable_is_passed_to_hub_calls_when_hf_token_is_unset(monkeypatch: pytest.MonkeyPatch):
    import huggingface_hub

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_FAKE_repo_token")
    seen: dict[str, object] = {}

    def fake_auth_check(repo_id: str, **kw: object) -> None:
        seen["auth_token"] = kw.get("token")

    def fake_download(repo_id: str, filename: str, **kw: object) -> str:
        seen["download_token"] = kw.get("token")
        return "/nonexistent/params.safetensors"

    monkeypatch.setattr(huggingface_hub, "auth_check", fake_auth_check)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent-cache")  # nothing cached, so download runs
    from steerlab import hub

    assert hub.download("google/gemma-scope-2-1b-it", "params.safetensors") == Path("/nonexistent/params.safetensors")
    assert seen == {"auth_token": "hf_FAKE_repo_token", "download_token": "hf_FAKE_repo_token"}

    # HF_TOKEN set: leave the token to huggingface_hub (it reads HF_TOKEN itself).
    monkeypatch.setenv("HF_TOKEN", "hf_FAKE_primary")
    seen.clear()
    ensure_access("google/gemma-scope-2-1b-it")
    assert seen == {"auth_token": None}


def test_fetch_refuses_before_downloading_when_gated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import huggingface_hub

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))  # an empty cache, so the real one cannot short-circuit the guard

    def fake_auth_check(repo_id: str, **_: object) -> None:
        raise _gated_error()

    def must_not_download(*_: object, **__: object) -> str:
        raise AssertionError("hf_hub_download was called for a gated repo without access")

    monkeypatch.setattr(huggingface_hub, "auth_check", fake_auth_check)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", must_not_download)
    with pytest.raises(GatedRepoAccessError):
        gemma_scope.fetch_gemma_scope(layer=13)


# ---- slow: real files (not run by default; need the downloads and, for Gemma, the licence) ----


@pytest.mark.slow
def test_real_qwen_scope_layer14_reconstructs_a_real_residual():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sae = qwen_scope.fetch_qwen_scope(layer=14)
    assert (sae.d_model, sae.n_features, sae.k) == (2048, 32768, 50)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32)
    ids = tok("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        hidden = model(**ids, output_hidden_states=True).hidden_states[15][0, -1]  # resid_post of layer 14
    cos = torch.nn.functional.cosine_similarity(sae.reconstruct(hidden), hidden.float(), dim=0)
    assert cos > 0.9


@pytest.mark.slow
def test_real_gemma_scope_layer13_reconstructs_a_real_residual():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sae = gemma_scope.fetch_gemma_scope(layer=13)
    assert (sae.d_model, sae.n_features) == (1152, 16384)
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", dtype=torch.float32)
    ids = tok("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        hidden = model(**ids, output_hidden_states=True).hidden_states[14][0, -1]  # resid_post of layer 13
    cos = torch.nn.functional.cosine_similarity(sae.reconstruct(hidden), hidden.float(), dim=0)
    assert cos > 0.9


# ---- Gemma Scope v1 (Gemma 2, `params.npz`) ----


def test_gemma_v1_npz_loads_the_same_sae_as_the_safetensors_file(gemma_file: Path, tmp_path: Path):
    import numpy as np

    v2 = gemma_scope.load_gemma_scope(gemma_file, layer=12)
    state = safetensors_torch.load_file(str(gemma_file))
    npz = tmp_path / "params.npz"
    np.savez(
        npz,
        W_enc=state["w_enc"].numpy(),
        W_dec=state["w_dec"].numpy(),
        b_enc=state["b_enc"].numpy(),
        b_dec=state["b_dec"].numpy(),
        threshold=state["threshold"].numpy(),
    )
    v1 = gemma_scope.load_gemma_scope(npz, layer=12, release="google/gemma-scope-2b-pt-res")
    assert v1.release == "google/gemma-scope-2b-pt-res"
    h = torch.randn(3, D_MODEL)
    assert torch.equal(v1.encode(h), v2.encode(h))
    assert torch.equal(v1.direction(2).unit, v2.direction(2).unit)


def test_gemma_v1_npz_without_capitalised_weights_is_rejected(tmp_path: Path):
    import numpy as np

    npz = tmp_path / "params.npz"
    np.savez(npz, w_enc=np.zeros((D_MODEL, N_FEATURES)), b_enc=np.zeros(N_FEATURES))
    with pytest.raises(Exception, match="W_enc"):
        gemma_scope.load_gemma_scope(npz, layer=12, release="google/gemma-scope-2b-pt-res")


def test_fetch_gemma_v1_names_the_known_layers_for_an_unknown_one():
    with pytest.raises(ValueError, match="known layers: 12"):
        gemma_scope.fetch_gemma_scope(5, release="google/gemma-scope-2b-pt-res")


# ---- Goodfire (ReLU, Llama-3.1-8B) ----


@pytest.fixture
def goodfire_file(tmp_path: Path) -> Path:
    """The release's layout: a module state dict with encoder_linear / decoder_linear weights and biases."""
    g = torch.Generator().manual_seed(2)
    state = {
        "encoder_linear.weight": torch.randn(N_FEATURES, D_MODEL, generator=g),
        "encoder_linear.bias": torch.randn(N_FEATURES, generator=g),
        "decoder_linear.weight": torch.randn(D_MODEL, N_FEATURES, generator=g),
        "decoder_linear.bias": torch.randn(D_MODEL, generator=g),
    }
    path = tmp_path / "Llama-3.1-8B-Instruct-SAE-l19.pth"
    torch.save(state, path)
    return path


def test_goodfire_loader_reads_the_module_state_dict(goodfire_file: Path):
    sae = goodfire.load_goodfire(goodfire_file)
    assert isinstance(sae, LoadedSAE)
    assert (sae.release, sae.layer) == ("Goodfire/Llama-3.1-8B-Instruct-SAE-l19", 19)
    assert (sae.d_model, sae.n_features) == (D_MODEL, N_FEATURES)
    assert goodfire.filename() == "Llama-3.1-8B-Instruct-SAE-l19.pth"


def test_goodfire_encode_is_relu_of_the_encoder_and_decode_is_the_decoder(goodfire_file: Path):
    state = torch.load(goodfire_file, weights_only=True)
    sae = goodfire.load_goodfire(goodfire_file)
    h = torch.randn(3, D_MODEL, generator=torch.Generator().manual_seed(3))
    expected = torch.relu(h @ state["encoder_linear.weight"].T + state["encoder_linear.bias"])
    acts = sae.encode(h)
    assert torch.allclose(acts, expected) and (acts == 0).any() and (acts > 0).any()
    decoded = sae.decode(acts)
    assert torch.allclose(decoded, acts @ state["decoder_linear.weight"].T + state["decoder_linear.bias"], atol=1e-5)


def test_goodfire_direction_is_the_unit_normalised_column_of_the_decoder(goodfire_file: Path):
    state = torch.load(goodfire_file, weights_only=True)
    direction = goodfire.load_goodfire(goodfire_file).direction(5)
    column = state["decoder_linear.weight"][:, 5]
    assert isinstance(direction, FeatureDirection) and direction.layer == 19
    assert torch.allclose(direction.unit, column / column.norm())


def test_goodfire_loader_rejects_another_layout(qwen_file: Path):
    with pytest.raises(ValueError, match="encoder_linear"):
        goodfire.load_goodfire(qwen_file)


@pytest.mark.slow
def test_real_goodfire_layer19_reconstructs_a_real_residual():
    """The real 2 GB file on real Llama-3.1-8B residuals (skipped unless both are cached).

    Measured 2026-10-26: mean cosine 0.87-0.90 at the output of layer 19, 0.82-0.85 one layer earlier and 0.84-0.86
    one layer later. The bar is 0.85, and layer 19's output must beat layer 18's, which pins where the SAE reads.
    """
    from steerlab import hub
    from steerlab.directions import load_model

    if hub.cached_file(goodfire.DEFAULT_RELEASE, goodfire.filename()) is None:
        pytest.skip("the Goodfire Llama SAE is not in the Hugging Face cache")
    if hub.cached_bytes("meta-llama/Llama-3.1-8B-Instruct") is None:
        pytest.skip("Llama-3.1-8B-Instruct is not in the Hugging Face cache")
    model, tokenizer, _, _ = load_model("llama8b")
    ids = tokenizer("The Golden Gate Bridge is in San Francisco.", return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        states = model(ids, output_hidden_states=True).hidden_states
    sae = goodfire.fetch_goodfire().to(model.device)

    def mean_cosine(hidden: torch.Tensor) -> float:
        hidden = hidden[0, 1:].float()  # skip <|begin_of_text|>, whose norm is unlike any other position
        return float(torch.nn.functional.cosine_similarity(sae.reconstruct(hidden), hidden, dim=-1).mean())

    at_19, at_18 = mean_cosine(states[20]), mean_cosine(states[19])  # hidden_states[i + 1] is layer i's output
    assert at_19 > 0.85 and at_19 > at_18, (at_19, at_18)
