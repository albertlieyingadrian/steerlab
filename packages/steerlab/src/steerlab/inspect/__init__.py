"""steerlab.inspect: type words, watch a model arrive at its next word layer by layer (`steerlab inspect`).

    steerlab inspect                         the live explorer on Qwen2.5-0.5B-Instruct, http://127.0.0.1:8770
    steerlab inspect --export out/           the preset prompts as a static explorer (no server needed)

    from steerlab.inspect import Inspector, Edit
    inspector = Inspector.load()                            # Qwen2.5-0.5B-Instruct on the CPU, from the cache
    inspector.run("The capital of France is")["layers"]     # one row per layer: attention, neurons, logit lens

The views (`core`): the run (guesses, the target, each layer's share of the target's logit and its logit-lens guess),
a layer (its heads and top neurons), a neuron (the words it writes, how it fires on each token), find (neurons that
write a word), edit (switch neurons off, flip or scale them and compare) and dose (scale one neuron ×0 … ×8 against
20 random neurons of the same layer). `app` serves them as JSON beside one page; `export` writes the presets as JSON
the same page reads with no server. Inspired by neuronaut (github.com/andrew-morris/neuronaut).

Importing this package imports neither torch nor FastAPI; `core` imports torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from steerlab.inspect.core import Edit, Inspector

__all__ = ["Edit", "Inspector"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from steerlab.inspect import core

        return getattr(core, name)
    raise AttributeError(name)
