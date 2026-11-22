"""steerlab.atlas: every MLP neuron (both SwiGLU poles) and attention head of a model, as a browsable atlas.

Two sources fill the same unit cards:

- **the catalog** (`catalog`): the stdlogits/neuron-lens Hugging Face dataset, which covers Qwen3 0.6B-14B (post-trained
  and base) and Gemma 4 12B. Labels there are an LLM's (Claude Haiku 5.5), checked blind by its authors; reads come from
  their fitted R/S Jacobian lenses. Tables are fetched a layer at a time and cached; the large `reads` tables are read
  by column range, so a unit page never downloads a whole layer of 64-token lists.
- **computation** (`compute`), for any other Qwen/Llama/Mistral/Gemma decoder: top firing contexts over a small corpus,
  exact token drivers (substitute candidate tokens at one position), both poles' quantiles, the write's logit-lens read,
  head OV/QK token reads from the weights, and direct weight edges. No LLM labels: a computed unit is summarised by its
  measured tokens, and says so.

    from steerlab.atlas import Address, AtlasService
    atlas = AtlasService(cache_dir)
    atlas.unit("qwen3-0.6b", Address.parse("L5/n7+"))      # one evidence card (label, contexts, drivers, reads, edges)
    atlas.overview("qwen3-0.6b")                           # the easiest units of every layer
    atlas.search("qwen3-0.6b", "closing brackets")         # word search over labels (and reads, when asked)

Importing this package imports neither torch nor pyarrow; `catalog` needs pyarrow and huggingface_hub, `compute` torch.
"""

from __future__ import annotations

from steerlab.atlas.address import Address, model_slug, unit_href
from steerlab.atlas.models import CATALOG_MODELS, CatalogModel, catalog_model

__all__ = ["CATALOG_MODELS", "Address", "AtlasService", "CatalogModel", "catalog_model", "model_slug", "unit_href"]


def __getattr__(name: str) -> object:
    if name == "AtlasService":
        from steerlab.atlas.service import AtlasService

        return AtlasService
    raise AttributeError(name)
