"""White-box tools: capture activations, train and score linear probes, build and inject steering vectors.

    from sandbagkit.probing import extract_activations, evaluate_probe, steering_vector, inject

    acts = extract_activations(model, tokenizer, texts, layers=[14])        hooks.py + extract.py
    ev = evaluate_probe(acts[14], labels, groups=pair_ids, layer=14)         probe.py
    vec = steering_vector(acts[14], labels, groups=pair_ids, test_groups=test_pairs, layer=14)   steering.py
    with inject(model, vec, strength=-2.0): ...                              steering.py
    probe.save(path) / vec.save(path)                                        io.py (.npz + .json, no pickle)

Needs the `probing` extra (torch, transformers, scikit-learn). Importing this package does NOT import them:
torch loads on the first call that needs it, sklearn on the first fit.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sandbagkit.probing.extract import (
        POSITIONS,
        ActivationCache,
        Activations,
        Encoded,
        NonFiniteActivationError,
        Position,
        check_device_parity,
        encode,
        extract_activations,
        model_id_for,
        model_info,
        model_revision,
        render_prompt,
    )
    from sandbagkit.probing.hooks import (
        LAYER_PATHS,
        LayerIndexError,
        capture_layers,
        count_forward_hooks,
        decoder_layers,
        layer_output_tensor,
        num_layers,
        register_hooks,
        replace_layer_output,
        resolve_layers,
    )
    from sandbagkit.probing.io import (
        FORMAT_VERSION,
        ArtifactFormatError,
        load_activations,
        load_arrays,
        save_activations,
        save_arrays,
    )
    from sandbagkit.probing.probe import (
        CrossValidation,
        LinearProbe,
        ProbeEvaluation,
        ShuffledSummary,
        best_layer,
        cross_validate,
        evaluate_layers,
        evaluate_probe,
        group_bootstrap_ci,
        pair_rows,
        pair_split,
        shuffled_label_baseline,
    )
    from sandbagkit.probing.steering import (
        Injection,
        LeakageError,
        SteeringVector,
        inject,
        mean_difference,
        steering_vector,
    )

# Each name loads its module on first use, so a module (and torch) is imported only when needed.
_LAZY = {
    "POSITIONS": "sandbagkit.probing.extract",
    "ActivationCache": "sandbagkit.probing.extract",
    "Activations": "sandbagkit.probing.extract",
    "Encoded": "sandbagkit.probing.extract",
    "NonFiniteActivationError": "sandbagkit.probing.extract",
    "Position": "sandbagkit.probing.extract",
    "check_device_parity": "sandbagkit.probing.extract",
    "encode": "sandbagkit.probing.extract",
    "extract_activations": "sandbagkit.probing.extract",
    "model_id_for": "sandbagkit.probing.extract",
    "model_info": "sandbagkit.probing.extract",
    "model_revision": "sandbagkit.probing.extract",
    "render_prompt": "sandbagkit.probing.extract",
    "LAYER_PATHS": "sandbagkit.probing.hooks",
    "LayerIndexError": "sandbagkit.probing.hooks",
    "capture_layers": "sandbagkit.probing.hooks",
    "count_forward_hooks": "sandbagkit.probing.hooks",
    "decoder_layers": "sandbagkit.probing.hooks",
    "layer_output_tensor": "sandbagkit.probing.hooks",
    "num_layers": "sandbagkit.probing.hooks",
    "register_hooks": "sandbagkit.probing.hooks",
    "replace_layer_output": "sandbagkit.probing.hooks",
    "resolve_layers": "sandbagkit.probing.hooks",
    "FORMAT_VERSION": "sandbagkit.probing.io",
    "ArtifactFormatError": "sandbagkit.probing.io",
    "load_activations": "sandbagkit.probing.io",
    "load_arrays": "sandbagkit.probing.io",
    "save_activations": "sandbagkit.probing.io",
    "save_arrays": "sandbagkit.probing.io",
    "CrossValidation": "sandbagkit.probing.probe",
    "LinearProbe": "sandbagkit.probing.probe",
    "ProbeEvaluation": "sandbagkit.probing.probe",
    "ShuffledSummary": "sandbagkit.probing.probe",
    "best_layer": "sandbagkit.probing.probe",
    "cross_validate": "sandbagkit.probing.probe",
    "evaluate_layers": "sandbagkit.probing.probe",
    "evaluate_probe": "sandbagkit.probing.probe",
    "group_bootstrap_ci": "sandbagkit.probing.probe",
    "pair_rows": "sandbagkit.probing.probe",
    "pair_split": "sandbagkit.probing.probe",
    "shuffled_label_baseline": "sandbagkit.probing.probe",
    "Injection": "sandbagkit.probing.steering",
    "LeakageError": "sandbagkit.probing.steering",
    "SteeringVector": "sandbagkit.probing.steering",
    "inject": "sandbagkit.probing.steering",
    "mean_difference": "sandbagkit.probing.steering",
    "steering_vector": "sandbagkit.probing.steering",
}


__all__ = [
    "FORMAT_VERSION",
    "LAYER_PATHS",
    "POSITIONS",
    "ActivationCache",
    "Activations",
    "ArtifactFormatError",
    "CrossValidation",
    "Encoded",
    "Injection",
    "LayerIndexError",
    "LeakageError",
    "LinearProbe",
    "NonFiniteActivationError",
    "Position",
    "ProbeEvaluation",
    "ShuffledSummary",
    "SteeringVector",
    "best_layer",
    "capture_layers",
    "check_device_parity",
    "count_forward_hooks",
    "cross_validate",
    "decoder_layers",
    "encode",
    "evaluate_layers",
    "evaluate_probe",
    "extract_activations",
    "group_bootstrap_ci",
    "inject",
    "layer_output_tensor",
    "load_activations",
    "load_arrays",
    "mean_difference",
    "model_id_for",
    "model_info",
    "model_revision",
    "num_layers",
    "pair_rows",
    "pair_split",
    "register_hooks",
    "render_prompt",
    "replace_layer_output",
    "resolve_layers",
    "save_activations",
    "save_arrays",
    "shuffled_label_baseline",
    "steering_vector",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'sandbagkit.probing' has no attribute {name!r}")
