"""The models the stdlogits/neuron-lens catalog covers, and how they are related.

Shapes are from each model's Hugging Face config and the catalog's README ("More sizes of Qwen3"). Lineage, from the
same README (its §20): Qwen3-4B is pruned from Qwen3-8B and Qwen3-0.6B from Qwen3-1.7B (heads one for one, each
neuron from one of the parent's, named in the `lineage` table), and a base model keeps its post-trained sibling's
indices, so `L5·n7+` of Qwen3-0.6B-Base is the same weights' coordinate as `L5·n7+` of Qwen3-0.6B, before
post-training. Labels exist for the post-trained models only; a base model's unit page borrows ("lends") its
sibling's label and says so.
"""

from __future__ import annotations

from dataclasses import dataclass

from steerlab.atlas.address import model_slug

CATALOG_REPO = "stdlogits/neuron-lens"
CATALOG_LABELLER = "Claude Haiku 5.5"
SITE = "https://neuron-lens.com"

# Depth bands as fractions of depth, from Qwen3-8B's reading plan (sensory L0-L3, association L4-L8, workspace
# L9-L20, motor L21-L35 of 36 layers). Other models use the same fractions: an approximation, labelled as one.
BANDS: tuple[tuple[str, float], ...] = (("sensory", 4 / 36), ("association", 9 / 36), ("workspace", 21 / 36))


def band(layer: int, n_layers: int) -> str:
    depth = layer / max(1, n_layers)
    for name, upper in BANDS:
        if depth < upper:
            return name
    return "motor"


@dataclass(frozen=True)
class CatalogModel:
    slug: str
    hf_id: str
    n_layers: int
    d_mlp: int
    n_heads: int
    labels: bool
    base: str | None = None  # the base model's slug, on a post-trained model
    post: str | None = None  # the post-trained model's slug, on a base model
    parent: str | None = None  # the model this one was pruned from
    child: str | None = None  # the model pruned from this one
    lineage: bool = False  # has a `lineage` table

    @property
    def label_slug(self) -> str | None:
        """Whose labels this model's pages show: its own, or (a base) its post-trained sibling's."""
        if self.labels:
            return self.slug
        return self.post

    @property
    def sizes(self) -> dict[str, int]:
        return {"n_layers": self.n_layers, "d_mlp": self.d_mlp, "n_heads": self.n_heads}


def _qwen(size: str, n_layers: int, d_mlp: int, n_heads: int, labels: bool, **lineage: str) -> list[CatalogModel]:
    post, base = f"qwen3-{size}", f"qwen3-{size}-base"
    has_lineage = size in ("0.6b", "1.7b", "4b", "8b")
    parent, child = lineage.get("parent"), lineage.get("child")
    return [
        CatalogModel(
            post,
            f"Qwen/Qwen3-{size.upper()}",
            n_layers,
            d_mlp,
            n_heads,
            labels,
            base=base,
            parent=parent,
            child=child,
            lineage=has_lineage,
        ),
        CatalogModel(
            base,
            f"Qwen/Qwen3-{size.upper()}-Base",
            n_layers,
            d_mlp,
            n_heads,
            False,
            post=post,
            parent=f"{parent}-base" if parent else None,
            child=f"{child}-base" if child else None,
            lineage=has_lineage,
        ),
    ]


CATALOG_MODELS: dict[str, CatalogModel] = {
    model.slug: model
    for model in [
        *_qwen("0.6b", 28, 3072, 16, True, parent="qwen3-1.7b"),
        *_qwen("1.7b", 28, 6144, 16, True, child="qwen3-0.6b"),
        *_qwen("4b", 36, 9728, 32, True, parent="qwen3-8b"),
        *_qwen("8b", 36, 12288, 32, True, child="qwen3-4b"),
        *_qwen("14b", 40, 17408, 40, True),
        CatalogModel("gemma-4-12b", "google/gemma-4-12B", 48, 15360, 16, True),
    ]
}


def catalog_model(model: str) -> CatalogModel | None:
    """The catalog entry for a slug or a Hugging Face id (`Qwen/Qwen3-0.6B`, `qwen3-0.6b`), or None."""
    return CATALOG_MODELS.get(model) or CATALOG_MODELS.get(model_slug(model))


def neuron_lens_url(slug: str, path: str) -> str:
    """The same unit on neuron-lens.com (its addresses have no pole sign: `/qwen3-8b/L8/n344`)."""
    return f"{SITE}/{slug}/{path.rstrip('+-')}"
