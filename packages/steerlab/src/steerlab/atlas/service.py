"""One front door to the atlas: picks the catalog or a computed atlas per model, and adds what both need.

    atlas = AtlasService(cache_dir, loaded="Qwen/Qwen2.5-0.5B-Instruct", parts=lambda: Parts(model, tokenizer))
    atlas.models()                                     # catalog models, computed atlases on disk, the loaded model
    atlas.unit("qwen3-0.6b", Address.parse("L5/n7+"))  # a card, with titles on its edges and links to other rooms
    atlas.search("qwen3-0.6b", "closing brackets")     # an address or a layer opens it; words go to the index
    atlas.graph("qwen3-0.6b", Address.parse("L5/n7+")) # the wiring walk
    atlas.fire(slug, Address.parse("L12/n100+"), "some text")   # the loaded model: per-token values + percentiles
    atlas.build(slug)                                  # compute the loaded model's atlas in the background

Locks. Anything that runs the loaded model (a build, the computed weight reads, `fire`) holds `lock`, the model lock
every inspect room shares. Catalog reads don't touch the model and never take it, so a slow download never blocks
the other rooms.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from steerlab.atlas.address import Address, model_slug, parse_layer
from steerlab.atlas.cards import describe_percentile, percentile
from steerlab.atlas.models import CATALOG_MODELS
from steerlab.atlas.search import SearchIndex
from steerlab.atlas.wiring import walk

if TYPE_CHECKING:
    from steerlab.atlas.catalog import Catalog
    from steerlab.atlas.compute import Parts
    from steerlab.atlas.computed import ComputedAtlas

MAX_FIRE_TOKENS = 128


class AtlasService:
    def __init__(
        self,
        cache_dir: str | Path,
        loaded: str | None = None,
        parts: Callable[[], Parts | None] | None = None,
        lock: threading.Lock | None = None,
        catalog: Catalog | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.loaded = loaded
        self.loaded_slug = model_slug(loaded) if loaded else None
        self._parts_fn = parts
        self._parts: Parts | None = None
        self.lock = lock or threading.Lock()
        self._catalog = catalog
        self._computed: dict[str, ComputedAtlas] = {}
        self._indexes: dict[tuple[str, bool], SearchIndex] = {}
        self._index_lock = threading.Lock()
        self._mutex = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}

    # Sources

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            from steerlab.atlas.catalog import Catalog, HubSource

            self._catalog = Catalog(HubSource(self.cache_dir / "catalog"))
        return self._catalog

    def parts(self) -> Parts | None:
        if self._parts is None and self._parts_fn is not None:
            self._parts = self._parts_fn()
        return self._parts

    def computed(self, slug: str) -> ComputedAtlas:
        from steerlab.atlas.computed import ComputedAtlas

        with self._mutex:
            if slug not in self._computed:
                loaded = slug == self.loaded_slug
                self._computed[slug] = ComputedAtlas(
                    self.cache_dir / "computed" / slug,
                    parts=self.parts() if loaded else None,
                    hf_id=self.loaded if loaded else None,
                )
            return self._computed[slug]

    def in_catalog(self, slug: str) -> bool:
        return self.catalog.covers(slug) if self._catalog is not None else slug in CATALOG_MODELS

    @contextlib.contextmanager
    def _model_lock(self, slug: str) -> Iterator[None]:
        """Hold the model lock when a computed atlas may run the loaded model."""
        if self.in_catalog(slug) or slug != self.loaded_slug:
            yield
            return
        with self.lock:
            yield

    def source(self, slug: str) -> str:
        if self.in_catalog(slug):
            return "catalog"
        if self.computed(slug).ready:
            return "computed"
        if slug == self.loaded_slug:
            return "computable"
        raise ValueError(f"no atlas for {slug!r}: not in the catalog, not computed, and not the loaded model")

    def models(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = [
            {
                "slug": m.slug,
                "hf_id": m.hf_id,
                "source": "catalog",
                "labels": m.labels,
                "labels_from": m.label_slug,
                "n_layers": m.n_layers,
                "d_mlp": m.d_mlp,
                "n_heads": m.n_heads,
                "loaded": m.slug == self.loaded_slug,
            }
            for m in self.catalog.models.values()
        ]
        seen = {row["slug"] for row in rows}
        root = self.cache_dir / "computed"
        for path in sorted(root.iterdir()) if root.is_dir() else []:
            atlas = self.computed(path.name)
            if path.name in seen or not atlas.ready:
                continue
            meta = atlas.meta
            rows.append(
                {
                    "slug": path.name,
                    "hf_id": meta.get("hf_id"),
                    "source": "computed",
                    "labels": False,
                    "n_layers": meta["n_layers"],
                    "d_mlp": meta["d_mlp"],
                    "n_heads": meta["n_heads"],
                    "corpus": meta.get("corpus"),
                    "built": meta.get("built"),
                    "loaded": path.name == self.loaded_slug,
                }
            )
            seen.add(path.name)
        if self.loaded_slug and self.loaded_slug not in seen:
            rows.append(
                {
                    "slug": self.loaded_slug,
                    "hf_id": self.loaded,
                    "source": "computable",
                    "labels": False,
                    "loaded": True,
                }
            )
        return {"loaded": self.loaded, "loaded_slug": self.loaded_slug, "models": rows, "jobs": self.jobs}

    # Views

    def overview(self, slug: str, top: int = 40) -> dict[str, Any]:
        source = self.source(slug)
        if source == "catalog":
            return self.catalog.overview(slug, top=top)
        if source == "computable":
            return {
                "model": slug,
                "hf_id": self.loaded,
                "source": "computable",
                "layers": [],
                "job": self.jobs.get(slug),
            }
        with self._model_lock(slug):
            return self.computed(slug).overview(slug, top=top)

    def title(self, slug: str, address: Address) -> str:
        if self.in_catalog(slug):
            return self.catalog.title(slug, address)
        return self.computed(slug).title(address)

    def unit(self, slug: str, address: Address) -> dict[str, Any]:
        source = self.source(slug)
        if source == "computable":
            raise ValueError(f"{slug} has no atlas yet: build it from the Units room (it runs the loaded model)")
        if source == "catalog":
            card = self.catalog.unit(slug, address)
        else:
            with self._model_lock(slug):
                card = self.computed(slug).unit(slug, address)
        self._annotate(slug, card)
        card["links"].update(self._links(slug, address))
        card["fireable"] = slug == self.loaded_slug
        return card

    def _annotate(self, slug: str, card: dict[str, Any]) -> None:
        """Put each linked unit's title next to it (edges, sources, lineage), so the page can say what it is."""
        lists: list[list[dict[str, Any]]] = [card.get("edges", {}).get(side, []) for side in ("in", "out")]
        sources = card.get("sources") or {}
        lists += [sources.get(key, []) for key in ("driven_by", "driven_by_heads", "fires_with")]
        lists += [sources.get(side, {}).get("driven_by", []) for side in ("v", "q", "k")]
        for items in lists:
            for item in items:
                with contextlib.suppress(ValueError, FileNotFoundError):
                    item["title"] = self.title(slug, Address.parse(item["address"]))
        for item in card.get("lineage", []):
            if item.get("address") and item["model"] in self.catalog.models:
                with contextlib.suppress(ValueError, FileNotFoundError):
                    item["title"] = self.catalog.title(item["model"], Address.parse(item["address"]))

    def _links(self, slug: str, address: Address) -> dict[str, str]:
        links = {
            "self": f"/units/{slug}/{address.path()}",
            "edit": f"/edit?model={slug}&units={quote(address.path(), safe='/')}",
        }
        if slug == self.loaded_slug:
            tail = f"&neuron={address.index}&tab=neuron" if not address.is_head else "&tab=layer"
            links["trace"] = f"/?layer={address.layer}{tail}"
        return links

    def graph(self, slug: str, address: Address, up: int = 2, down: int = 2, fan: int = 4) -> dict[str, Any]:
        source = self.source(slug)
        if source == "computable":
            raise ValueError(f"{slug} has no atlas yet")
        if source == "catalog":
            catalog = self.catalog
            found = walk(lambda a: catalog.edges(slug, a), lambda a: catalog.title(slug, a), address, up, down, fan)
            return found | {"source": "catalog"}
        atlas = self.computed(slug)
        with self._model_lock(slug):
            return walk(atlas.edges, atlas.title, address, up, down, fan) | {"source": "computed"}

    # Search

    def index(self, slug: str, reads: bool = False) -> SearchIndex:
        key = (slug, reads)
        with self._index_lock:
            if key not in self._indexes:
                if self.in_catalog(slug):
                    self._indexes[key] = SearchIndex.build(self.catalog.search_documents(slug, reads=reads))
                else:
                    with self._model_lock(slug):
                        docs = list(self.computed(slug).search_documents(slug, reads=reads))
                    self._indexes[key] = SearchIndex.build(docs)
            return self._indexes[key]

    def search(
        self, slug: str, query: str, reads: bool = False, kind: str | None = None, limit: int = 30
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"results": [], "total": 0}
        address = Address.try_parse(query)
        if address is not None:
            return {"open": address.path(), "results": [], "total": 0}
        layer = parse_layer(query)
        if layer is not None:
            return {"layer": layer, "results": [], "total": 0}
        started = time.time()
        found = self.index(slug, reads=reads).search(query, limit=limit, kind=kind)
        found["seconds"] = round(time.time() - started, 3)
        found["searched"] = "labels and reads" if reads else ("labels" if self.in_catalog(slug) else "measured tokens")
        return found

    # The loaded model

    def fire(self, slug: str, address: Address, text: str) -> dict[str, Any]:
        """Run `text` through the loaded model and read one unit at every token, with its percentile in the unit's own
        distribution (the catalog's or the computed atlas's quantiles)."""
        if slug != self.loaded_slug:
            raise ValueError(f"fire runs the loaded model ({self.loaded}); restart with --model to run {slug}")
        parts = self.parts()
        if parts is None:
            raise ValueError("no model is loaded")
        from steerlab.atlas import compute

        ids = parts.tokenizer(text, add_special_tokens=False).input_ids[:MAX_FIRE_TOKENS]
        if not ids:
            raise ValueError("type at least one word")
        size = parts.n_heads if address.is_head else parts.d_mlp
        if not 0 <= address.layer < parts.n_layers or not 0 <= address.index < size:
            raise ValueError(f"{address} is outside the loaded model")
        with self.lock:
            values = compute.unit_trace(parts, ids, address.layer, address.kind, address.index, address.pole)
        stats, basis = self._fire_stats(slug, address)
        tokens = []
        for token_id, value in zip(ids, values, strict=False):
            pct = percentile(value, stats) if stats else None
            tokens.append(
                {"text": parts.decode(token_id), "value": value, "pct": pct, "pct_text": describe_percentile(pct)}
            )
        peak = max(range(len(values)), key=lambda i: values[i])
        typical = stats.get("act_top", stats.get("out_top")) if stats else None
        return {"address": address.path(), "tokens": tokens, "peak": peak, "basis": basis, "act_top": typical}

    def _fire_stats(self, slug: str, address: Address) -> tuple[dict[str, Any], str]:
        try:
            if self.in_catalog(slug):
                stats = self.catalog.unit(slug, address).get("stats", {})
                return stats, "percentiles of the catalog's distribution (bf16 over 15.4M positions); this run is fp32"
            atlas = self.computed(slug)
            with self.lock:
                card = atlas.unit(slug, address)
            return card.get("stats", {}), f"percentiles over {card.get('corpus')}"
        except (FileNotFoundError, ValueError):
            return {}, "no atlas yet: values only, no percentiles"

    def build(self, slug: str, **options: Any) -> dict[str, Any]:
        """Start computing the loaded model's atlas in a background thread; `jobs[slug]` reports its progress."""
        if slug != self.loaded_slug:
            raise ValueError(f"only the loaded model ({self.loaded}) can be computed here; bigger models run on Modal")
        if self.in_catalog(slug):
            raise ValueError(f"{slug} is in the catalog; nothing to compute")
        job = self.jobs.get(slug)
        if job and job.get("state") == "running":
            return job
        job = {"state": "running", "stage": "queued", "done": 0.0, "started": time.time()}
        self.jobs[slug] = job

        def progress(stage: str, done: float) -> None:
            job.update(stage=stage, done=round(done, 3))

        def run() -> None:
            try:
                with self.lock:
                    meta = self.computed(slug).build(self.parts(), progress=progress, **options)
                job.update(state="done", meta=meta, seconds=round(time.time() - job["started"], 1))
                with self._index_lock:
                    self._indexes = {k: v for k, v in self._indexes.items() if k[0] != slug}
            except Exception as error:  # reported to the page; the thread must not die silently
                job.update(state="failed", error=f"{type(error).__name__}: {error}")

        threading.Thread(target=run, name=f"atlas-build-{slug}", daemon=True).start()
        return job
