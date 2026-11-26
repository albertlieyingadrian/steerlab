"""A token lens, written as functions that run inside an NDIF trace.

A lens here takes ``h``, the state entering one layer visit at the read position (a tensor of
shape [hidden]), and returns scores over the vocabulary ([vocabulary]). Everything in these
functions executes on NDIF; only the top ten token ids per visit are sent back.
"""


def layers(model):
    """The decoder layers: 24 of them, of which 6 to 17 run three times per forward pass."""
    return model.model.language_model.layers


def visit_address(config, visit: int) -> tuple[int, int]:
    """Where a layer visit happens: which layer, and which of that layer's runs (0, 1 or 2).

    One token passes through 48 visits, numbered 0 to 47 in the order they run: layers 0-5 once,
    then layers 6-17 three times over, then layers 18-23 once. ``config`` is the model's text
    configuration, which holds those numbers.
    """
    start, end, passes = config.loop_layer_index_start, config.loop_layer_index_end, config.loop_num_passes
    looped = end - start
    if not 0 <= visit < config.num_hidden_layers + looped * (passes - 1):
        raise ValueError(f"there is no layer visit {visit}")
    if visit < start:
        return visit, 0
    if visit < start + looped * passes:
        return start + (visit - start) % looped, (visit - start) // looped
    return visit - looped * (passes - 1), 0


def logit_lens(model, h):
    """The model's own final norm and unembedding, applied early. No trained weights.

    The model also soft-caps its logits, which does not change their order, so it is left out.
    """
    return model.lm_head(model.model.language_model.norm(h))


def base_of(lens: str) -> str:
    """The state a lens reads: "future_mlp_in", "concept_mlp_in" and "tuned_mlp_in" read what "mlp_in" reads."""
    return lens.split("_", 1)[1] if lens.startswith(("tuned_", "future_", "concept_")) else lens


def read(model, layer: int, lens: str, position: int):
    """The one state of a layer visit that `lens` reads, at `position` ([hidden]). Call inside the
    visit's `tracer.iter` block.

    act_in: the active block entering the visit (what the starter reads).
    mlp_in: the input of the visit's MoE sublayer, i.e. the input of post_attention_layernorm. It is
    the layer's own learned depth-attention mix of the completed blocks and the active block, with the
    previous pass's message gated in on passes 2 and 3: the closest thing this model has to "the
    residual stream" at that point.
    """
    base = base_of(lens)
    decoder = layers(model)[layer]
    if base == "act_in":
        return decoder.input[0, position]
    if base == "mlp_in":
        return decoder.post_attention_layernorm.input[0, position]
    raise ValueError(f"unknown lens {lens}")


def readout(model, lens: str, x, weights: dict | None = None, key: str | None = None):
    """Scores over the vocabulary for state x. A translated lens (tuned_*, future_*) first applies its
    fitted affine map, x + W x + b (or the low-rank x + U V x + b), then, like the logit lens, the
    model's own final norm and unembedding.

    future_*: a translator fitted offline so that the readout at the last prompt token matches the
    model's distribution over the first content token of its reply (one token past the chat
    control token that comes next). Trained on open-domain questions, random arithmetic and
    wikitext, never on the challenge's questions.

    concept_*: a translator fitted so that the readout at the last token of a word names the clean
    words around it (the word itself counted twice), with random typos injected into the inputs.
    Trained on wikitext only."""
    if lens != base_of(lens):
        xf = x.float()
        b = weights[f"{key}.b"].to(x.device).float()
        if f"{key}.W" in weights:
            y = xf + weights[f"{key}.W"].to(x.device).float() @ xf + b
        else:
            U = weights[f"{key}.U"].to(x.device).float()
            V = weights[f"{key}.V"].to(x.device).float()
            y = xf + U @ (V @ xf) + b
        x = y.to(x.dtype)
    return logit_lens(model, x)


def fetch(repository: str, filename: str, token: str | None = None) -> dict:
    """A lens's published weights, downloaded by NDIF itself: a dict of tensors.

    The starter lens has no weights and does not call this. A lens that does publishes them to
    the Hugging Face Hub as a safetensors file and calls this at the top of the session, before
    any trace, so that they are fetched once per request and never travel in it. The file is read
    straight into memory and nothing is left on NDIF's disk. NDIF has no Hugging Face token of
    its own, so a private repository needs yours passed as ``token``.
    """
    import urllib.request

    import safetensors.torch

    request = urllib.request.Request(f"https://huggingface.co/{repository}/resolve/main/{filename}")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=600) as response:
        return safetensors.torch.load(response.read())
