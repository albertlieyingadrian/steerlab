"""Read the stdlogits/neuron-lens catalog (Hugging Face dataset, CC BY 4.0) into unit cards.

Layout (per model folder `<slug>/`): `neurons/<table>/L05.parquet` for acts, drivers, reads, labels, edges, firings,
sources (no L00) and lineage, one row per pole with the + poles first (row `n` is `L·n+`, row `d_mlp + n` is `L·n−`);
`heads/<table>.parquet` for acts, drivers, reads, labels and sources, one row per head in (layer, head) order. Each file
is one row group.

Fetching. Small tables come whole through `hf_hub_download` (0.6B: 0.1-6 MB a layer; 8B: up to 20 MB). The three large
neuron tables (`reads` 40-130 MB a layer, `firings` 30-100 MB, `sources`) are read by column range through
`HfFileSystem`: only the `_str` previews and scalars a card shows, a few MB, then kept on disk as a column subset. A
`LocalSource` reads the same layout from a directory (tests, or an offline mirror).

Provenance on every card: `source: "catalog"`, and labels say `by: "Claude Haiku 5.5"` (the catalog's labeller); a
base model's label is its post-trained sibling's (`lent_from`), since post-training keeps the indices.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from steerlab.atlas.address import Address
from steerlab.atlas.cards import (
    NEAR_SILENT,
    describe_percentile,
    head_score,
    neuron_score,
    num,
    paired,
    parse_contexts,
    parse_counted,
    parse_scored,
    percentile,
    pole_score,
    rank_percentile,
    split_tokens,
    strength,
)
from steerlab.atlas.models import CATALOG_LABELLER, CATALOG_MODELS, CATALOG_REPO, CatalogModel, band, neuron_lens_url

if TYPE_CHECKING:
    import pyarrow as pa

RANGE_TABLES = frozenset({"neurons/reads", "neurons/firings", "neurons/sources"})
CORPUS = (
    "the catalog's activation corpus: 63,990 documents, about 15.4M positions of web, Chinese, Python and chat text"
)
SLICES = (("web", "rate_dclm"), ("chinese", "rate_zh"), ("code", "rate_code"), ("chat", "rate_chat"))

# What each read says, for the page's tips (from the catalog README's "Conventions" and lens sections).
NEURON_READS: tuple[tuple[str, str, str, str], ...] = (
    ("read_S", "fires on", "S", "input tokens that drive the pole (S-lens of its read direction)"),
    ("read_S_z", "fires on", "S", "the same, z-scored against the layer's poles (drops hub tokens)"),
    ("read_R", "fires on", "R", "what the residual says when it drives the pole (R-lens of its read direction)"),
    ("read_R_z", "fires on", "R", "the same, z-scored against the layer's poles"),
    ("write_R", "writes", "R", "what its write makes the model say (R-lens; the logit lens at the last layer)"),
    ("write_R_z", "writes", "R", "the same, z-scored against the layer's poles"),
    ("write_S", "writes", "S", "input tokens whose footprint its write imitates (S-lens)"),
    ("write_S_z", "writes", "S", "the same, z-scored against the layer's poles"),
    ("write_orth_R_z", "writes", "R", "what it writes beyond what it reads (gate and up rows removed), z-scored"),
)
HEAD_READS: tuple[tuple[str, str, str, str], ...] = (
    ("ov_R", "writes", "R", "the tokens it copies most: where each read token ranks in what it makes the model say"),
    ("ov_R_neg", "writes", "R", "the tokens it suppresses most"),
    ("ov_S", "writes", "S", "whether its write makes the residual look as if the read token were there"),
    ("qk_S", "attends", "S", "key tokens it attends to under its mean query, whatever the query"),
    ("qk_S_neg", "attends", "S", "key tokens it avoids"),
    ("qk_self_S", "attends", "S", "tokens that look for earlier copies of themselves (duplicate-token heads)"),
    ("qk_self_R", "attends", "R", "tokens a position about to say them looks up (from L1)"),
)
WEIGHT_FACTS = ("w_norm", "cos_gate_up", "cos_write_gate", "cos_write_up", "null_share", "vocab_skew", "vocab_kurt")
FIRING_READS = ("before_R", "mlp_R", "pred", "pred_z")
EDGE_LIMIT = 16
EDGES_HOW = (
    "cos of the earlier pole's write with the later pole's read direction; push = cos · act_top · ‖w‖ / rms: how far "
    "one top firing moves the later pole's input (positive drives it, negative pushes against)"
)
NO_HEAD_EDGES = "the catalog keeps direct weight edges for neurons only; see what drives the head below"


class Source(Protocol):
    def table(self, path: str, columns: Sequence[str] | None = None) -> pa.Table: ...


class LocalSource:
    """The catalog layout on disk (tests, or a mirror made with `huggingface-cli download`)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def table(self, path: str, columns: Sequence[str] | None = None) -> pa.Table:
        import pyarrow.parquet as pq

        file = self.root / path
        if not file.exists():
            raise FileNotFoundError(f"the catalog has no {path}")
        schema = pq.read_schema(file)
        wanted = None if columns is None else [c for c in columns if c in schema.names]
        return pq.read_table(file, columns=wanted)


class HubSource:
    """The catalog on the Hugging Face Hub, cached under `cache_dir`.

    Whole files for small tables (`hf_hub_download`, which keeps its own cache and works offline once fetched); a
    column range for the large ones (`HfFileSystem`), saved as `columns/<path>.<hash>.parquet`.
    """

    def __init__(self, cache_dir: str | Path, repo: str = CATALOG_REPO, revision: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.repo = repo
        self.revision = revision
        self._fs: Any = None
        self._lock = threading.Lock()

    def _download(self, path: str) -> Path:
        from huggingface_hub import hf_hub_download

        kwargs: dict[str, Any] = {"repo_type": "dataset", "revision": self.revision, "cache_dir": self.cache_dir / "hf"}
        try:
            return Path(hf_hub_download(self.repo, path, local_files_only=True, **kwargs))
        except Exception:  # not cached yet: fetch it below
            pass
        try:
            return Path(hf_hub_download(self.repo, path, **kwargs))
        except Exception as error:  # an EntryNotFound, offline, or a bad revision
            raise FileNotFoundError(f"the catalog has no {path} ({type(error).__name__})") from None

    def table(self, path: str, columns: Sequence[str] | None = None) -> pa.Table:
        import pyarrow.parquet as pq

        prefix = path.rsplit("/", 1)[0]
        if columns is None or prefix not in RANGE_TABLES:
            file = self._download(path)
            schema = pq.read_schema(file)
            wanted = None if columns is None else [c for c in columns if c in schema.names]
            return pq.read_table(file, columns=wanted)
        key = hashlib.sha1(",".join(sorted(columns)).encode()).hexdigest()[:10]
        cached = self.cache_dir / "columns" / f"{path}.{key}.parquet"
        if cached.exists():
            return pq.read_table(cached)
        with self._lock:
            if self._fs is None:
                from huggingface_hub import HfFileSystem

                self._fs = HfFileSystem()
            fs = self._fs
        revision = f"@{self.revision}" if self.revision else ""
        try:
            handle = fs.open(f"datasets/{self.repo}{revision}/{path}", block_size=1 << 20)
        except Exception as error:
            raise FileNotFoundError(f"the catalog has no {path} ({type(error).__name__})") from None
        with handle:
            parquet = pq.ParquetFile(handle)
            wanted = [c for c in columns if c in parquet.schema_arrow.names]
            table = parquet.read(columns=wanted)
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".tmp")
        pq.write_table(table, tmp)
        tmp.replace(cached)
        return table


LABEL_COLUMNS = ("neuron", "pole", "fires", "fires_conf", "writes", "writes_conf", "note")


class Catalog:
    """Unit cards, layer overviews and search documents from the catalog. Thread-safe; tables are memoised (LRU)."""

    def __init__(self, source: Source, models: dict[str, CatalogModel] | None = None, memo: int = 48) -> None:
        self.source = source
        self.models = dict(CATALOG_MODELS if models is None else models)
        self._memo: OrderedDict[tuple[str, tuple[str, ...] | None], Any] = OrderedDict()
        self._memo_size = memo
        self._lock = threading.RLock()

    # Tables

    def model(self, slug: str) -> CatalogModel:
        try:
            return self.models[slug]
        except KeyError:
            raise ValueError(f"the catalog does not cover {slug!r}") from None

    def covers(self, slug: str) -> bool:
        return slug in self.models

    def _table(self, path: str, columns: Sequence[str] | None = None) -> pa.Table:
        key = (path, None if columns is None else tuple(columns))
        with self._lock:
            if key in self._memo:
                self._memo.move_to_end(key)
                return self._memo[key]
        table = self.source.table(path, columns)
        with self._lock:
            self._memo[key] = table
            while len(self._memo) > self._memo_size:
                self._memo.popitem(last=False)
        return table

    def neuron_table(self, slug: str, table: str, layer: int, columns: Sequence[str] | None = None) -> pa.Table:
        return self._table(f"{slug}/neurons/{table}/L{layer:02d}.parquet", columns)

    def head_table(self, slug: str, table: str, columns: Sequence[str] | None = None) -> pa.Table:
        return self._table(f"{slug}/heads/{table}.parquet", columns)

    def check(self, slug: str, address: Address) -> CatalogModel:
        model = self.model(slug)
        if not 0 <= address.layer < model.n_layers:
            raise ValueError(f"layer {address.layer} is outside 0..{model.n_layers - 1} of {model.slug}")
        size = model.n_heads if address.is_head else model.d_mlp
        if not 0 <= address.index < size:
            what = "head" if address.is_head else "neuron"
            raise ValueError(f"{what} {address.index} is outside 0..{size - 1} of {model.slug}")
        return model

    def _neuron_row(
        self,
        model: CatalogModel,
        table: str,
        address: Address,
        columns: Sequence[str] | None = None,
        slug: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            data = self.neuron_table(slug or model.slug, table, address.layer, columns)
        except FileNotFoundError:
            return None
        row = address.index + (0 if address.pole > 0 else model.d_mlp)
        if row < data.num_rows:
            out: dict[str, Any] = data.slice(row, 1).to_pylist()[0]
            if out.get("neuron", address.index) == address.index and out.get("pole", address.pole) == address.pole:
                return out
        if "neuron" not in data.column_names or "pole" not in data.column_names:
            return None
        import pyarrow.compute as pc

        mask = pc.and_(pc.equal(data["neuron"], address.index), pc.equal(data["pole"], address.pole))
        found = data.filter(mask)
        return found.slice(0, 1).to_pylist()[0] if found.num_rows else None

    def _head_row(
        self, model: CatalogModel, table: str, address: Address, slug: str | None = None
    ) -> dict[str, Any] | None:
        try:
            data = self.head_table(slug or model.slug, table)
        except FileNotFoundError:
            return None
        row = address.layer * model.n_heads + address.index
        if row < data.num_rows:
            out: dict[str, Any] = data.slice(row, 1).to_pylist()[0]
            if out.get("layer") == address.layer and out.get("head") == address.index:
                return out
        import pyarrow.compute as pc

        found = data.filter(pc.and_(pc.equal(data["layer"], address.layer), pc.equal(data["head"], address.index)))
        return found.slice(0, 1).to_pylist()[0] if found.num_rows else None

    # Labels and titles

    def label(self, slug: str, address: Address) -> dict[str, Any] | None:
        model = self.model(slug)
        source = model.label_slug
        if source is None:
            return None
        row = (
            self._head_row(model, "labels", address, slug=source)
            if address.is_head
            else self._neuron_row(model, "labels", address, LABEL_COLUMNS, slug=source)
        )
        if row is None:
            return None
        keys = ("fires", "attends", "writes") if address.is_head else ("fires", "writes")
        label: dict[str, Any] = {
            "source": "catalog",
            "by": CATALOG_LABELLER,
            "lent_from": None if source == model.slug else source,
            "note": row.get("note") or "",
        }
        for key in keys:
            label[key] = row.get(key) or ""
            label[f"{key}_conf"] = num(row.get(f"{key}_conf"))
        return label

    def title(self, slug: str, address: Address) -> str:
        """A unit's one-line title: its `fires` label (a head's `fires`, else `attends`), or ""."""
        try:
            label = self.label(slug, address)
        except (FileNotFoundError, ValueError):
            return ""
        if not label:
            return ""
        return str(label.get("fires") or label.get("attends") or label.get("writes") or "")

    # Cards

    def unit(self, slug: str, address: Address) -> dict[str, Any]:
        model = self.check(slug, address)
        return self._head_card(model, address) if address.is_head else self._neuron_card(model, address)

    def _base_card(self, model: CatalogModel, address: Address) -> dict[str, Any]:
        return {
            "model": model.slug,
            "hf_id": model.hf_id,
            "address": address.path(),
            "display": str(address),
            "kind": address.kind,
            "layer": address.layer,
            "index": address.index,
            "pole": address.pole,
            "band": band(address.layer, model.n_layers),
            "n_layers": model.n_layers,
            "source": "catalog",
            "corpus": CORPUS,
            "links": {"neuron_lens": neuron_lens_url(model.slug, address.path())},
        }

    @staticmethod
    def _contexts(raw: str | None, values: Sequence[float] | None, stats: dict[str, Any]) -> list[dict[str, Any]]:
        contexts = parse_contexts(raw, values)
        for piece in contexts:
            pct = percentile(piece["value"], stats) if "value" in piece else None
            piece["pct"] = pct
            piece["pct_text"] = describe_percentile(pct)
        return contexts

    def _neuron_card(self, model: CatalogModel, address: Address) -> dict[str, Any]:
        card = self._base_card(model, address)
        card["label"] = self.label(model.slug, address)
        acts = self._neuron_row(model, "acts", address) or {}
        keys = (
            "act_top",
            "act_max",
            "rate",
            "pos_share",
            "neg_lobe",
            "mean",
            "std",
            "q50",
            "q90",
            "q99",
            "q999",
            "cur_mode",
            "next_mode",
            "first_max",
            "first_chat",
        )
        stats: dict[str, Any] = {key: num(acts.get(key)) for key in keys}
        stats["rates"] = {name: num(acts.get(col)) for name, col in SLICES}
        stats["near_silent"] = (stats["act_top"] or 0.0) < NEAR_SILENT
        stats["layer_pct"] = self._layer_percentile(model, address, stats["act_top"])
        card["stats"] = stats
        card["contexts"] = self._contexts(acts.get("contexts"), acts.get("top_val"), stats)
        card["peak_tokens"] = parse_counted(acts.get("peak_tokens"))
        card["next_tokens"] = parse_counted(acts.get("next_tokens"))
        card["first_tokens"] = parse_scored(acts.get("first_tokens"))
        drivers = self._neuron_row(model, "drivers", address) or {}
        card["drivers"] = {
            "how": "every kept token put at position 24 of 8 real documents, averaged; activation × the pole's sign",
            "at": paired(split_tokens(drivers.get("drivers_str")), drivers.get("drivers_val")),
            "sum": paired(split_tokens(drivers.get("drivers_sum_str")), drivers.get("drivers_sum_val")),
        }
        read_cols = [f"{key}_str" for key, *_ in NEURON_READS] + list(WEIGHT_FACTS) + ["neuron", "pole"]
        reads = self._neuron_row(model, "reads", address, read_cols) or {}
        card["reads"] = [
            {"key": key, "side": side, "lens": lens, "what": what, "tokens": split_tokens(reads.get(f"{key}_str"))}
            for key, side, lens, what in NEURON_READS
            if reads.get(f"{key}_str")
        ]
        card["weights"] = {key: num(reads.get(key)) for key in WEIGHT_FACTS if key in reads}
        firing_cols = [f"{key}_str" for key in FIRING_READS] + ["n_firings", "parallel", "orthogonal", "neuron", "pole"]
        firings = self._neuron_row(model, "firings", address, firing_cols) or {}
        card["firings"] = {
            "n_firings": firings.get("n_firings"),
            "parallel": num(firings.get("parallel")),
            "orthogonal": num(firings.get("orthogonal")),
            **{key: split_tokens(firings.get(f"{key}_str")) for key in FIRING_READS},
        }
        card["edges"] = self.edges(model.slug, address)
        card["sources"] = self._sources(model, address)
        card["lineage"] = self.lineage(model.slug, address)
        return card

    def _layer_percentile(self, model: CatalogModel, address: Address, act_top: float | None) -> float | None:
        if act_top is None:
            return None
        try:
            tops = self.neuron_table(model.slug, "acts", address.layer, ["act_top"])["act_top"].to_pylist()
        except FileNotFoundError:
            return None
        return rank_percentile([v for v in tops if v is not None], act_top)

    def _sources(self, model: CatalogModel, address: Address) -> dict[str, Any]:
        cols = [
            "neuron",
            "pole",
            "n_contexts",
            "emb_share",
            "neurons_share",
            "heads_share",
            "driven_layer",
            "driven_neuron",
            "driven_pole",
            "driven_share",
            "driven_head_layer",
            "driven_head",
            "driven_head_share",
            "with_layer",
            "with_neuron",
            "with_pole",
            "with_lift",
            "with_co",
        ]
        row = self._neuron_row(model, "sources", address, cols)
        if not row:
            return {}
        k = EDGE_LIMIT // 2
        driven = [
            {"address": Address.neuron(lay, n, p).path(), "share": num(share)}
            for lay, n, p, share in zip(
                row["driven_layer"], row["driven_neuron"], row["driven_pole"], row["driven_share"], strict=False
            )
            if lay >= 0 and p != 0
        ][:k]
        heads = [
            {"address": Address.head(lay, h).path(), "share": num(share)}
            for lay, h, share in zip(
                row["driven_head_layer"], row["driven_head"], row["driven_head_share"], strict=False
            )
            if lay >= 0
        ][:k]
        together = [
            {"address": Address.neuron(lay, n, p).path(), "lift": num(lift), "co": num(co)}
            for lay, n, p, lift, co in zip(
                row["with_layer"], row["with_neuron"], row["with_pole"], row["with_lift"], row["with_co"], strict=False
            )
            if lay >= 0 and p != 0
        ][:k]
        return {
            "how": "the exact RelP split of its activation at its first 5 top firings",
            "n_contexts": row.get("n_contexts"),
            "shares": {
                "embedding": num(row.get("emb_share")),
                "heads": num(row.get("heads_share")),
                "neurons": num(row.get("neurons_share")),
            },
            "driven_by": driven,
            "driven_by_heads": heads,
            "fires_with": together,
        }

    def _head_card(self, model: CatalogModel, address: Address) -> dict[str, Any]:
        card = self._base_card(model, address)
        card["label"] = self.label(model.slug, address)
        acts = self._head_row(model, "acts", address) or {}
        keys = (
            "out_top",
            "out_max",
            "rate",
            "mean",
            "std",
            "q50",
            "q90",
            "q99",
            "q999",
            "sink",
            "prev",
            "self",
            "induction",
            "duplicate",
            "cur_mode",
            "src_mode",
        )
        stats: dict[str, Any] = {key: num(acts.get(key)) for key in keys}
        stats["rates"] = {name: num(acts.get(col)) for name, col in SLICES}
        stats["kv_head"] = acts.get("kv_head")
        card["stats"] = stats
        card["contexts"] = self._contexts(acts.get("contexts"), acts.get("top_val"), stats)
        card["peak_tokens"] = parse_counted(acts.get("peak_tokens"))
        card["src_tokens"] = parse_counted(acts.get("src_tokens"))
        card["out_R_z"] = split_tokens(acts.get("out_R_z_str"))
        drivers = self._head_row(model, "drivers", address) or {}
        card["drivers"] = {
            "how": "as the query: the tokens with the largest output norm at their own position; as the key: the "
            "tokens that receive the most attention from the 16 positions after them",
            "at": paired(split_tokens(drivers.get("drivers_str")), drivers.get("drivers_val")),
            "keys": paired(split_tokens(drivers.get("keys_str")), drivers.get("keys_val")),
        }
        reads = self._head_row(model, "reads", address) or {}
        card["reads"] = [
            {"key": key, "side": side, "lens": lens, "what": what, "tokens": split_tokens(reads.get(f"{key}_str"))}
            for key, side, lens, what in HEAD_READS
            if reads.get(f"{key}_str")
        ]
        copy_keys = ("ov_R_copy", "ov_S_copy", "qk_self_S", "qk_self_R", "qk_ctx_rho", "qk_ov_rho")
        card["copy"] = {key: num(reads.get(key)) for key in copy_keys}
        card["edges"] = {"in": [], "out": [], "how": NO_HEAD_EDGES, "source": "catalog"}
        card["sources"] = self._head_sources(model, address)
        card["lineage"] = self.lineage(model.slug, address)
        return card

    def _head_sources(self, model: CatalogModel, address: Address) -> dict[str, Any]:
        row = self._head_row(model, "sources", address)
        if not row:
            return {}
        out: dict[str, Any] = {"how": "what drives the head when it fires, through its value, query and key"}
        for side in ("v", "q", "k"):
            driven = zip(
                row[f"{side}_driven_layer"],
                row[f"{side}_driven_neuron"],
                row[f"{side}_driven_pole"],
                row[f"{side}_driven_share"],
                strict=False,
            )
            out[side] = {
                "shares": {
                    "embedding": num(row.get(f"{side}_emb_share")),
                    "heads": num(row.get(f"{side}_heads_share")),
                    "neurons": num(row.get(f"{side}_neurons_share")),
                },
                "driven_by": [
                    {"address": Address.neuron(lay, n, p).path(), "share": num(share)}
                    for lay, n, p, share in driven
                    if lay >= 0 and p != 0
                ][: EDGE_LIMIT // 2],
            }
        return out

    # Wiring and lineage

    def edges(self, slug: str, address: Address, limit: int = EDGE_LIMIT) -> dict[str, Any]:
        """A pole's strongest direct weight edges: `in` from earlier poles, `out` to later ones, by |push|."""
        model = self.model(slug)
        if address.is_head:
            return {"in": [], "out": [], "how": NO_HEAD_EDGES, "source": "catalog"}
        row = self._neuron_row(model, "edges", address)
        out: dict[str, Any] = {"source": "catalog", "how": EDGES_HOW}
        for side in ("in", "out"):
            items: list[dict[str, Any]] = []
            if row:
                cols = zip(
                    row[f"{side}_layer"],
                    row[f"{side}_neuron"],
                    row[f"{side}_pole"],
                    row[f"{side}_push"],
                    row[f"{side}_cos"],
                    strict=False,
                )
                for lay, n, p, push, cos in cols:
                    if lay < 0 or p == 0 or num(push) is None:
                        continue
                    items.append({"address": Address.neuron(lay, n, p).path(), "push": num(push), "cos": num(cos)})
                    if len(items) >= limit:
                        break
            out[side] = items
        return out

    def lineage(self, slug: str, address: Address) -> list[dict[str, Any]]:
        """Where this unit sits across sizes (pruning parent or child) and base <-> post-trained."""
        model = self.model(slug)
        out: list[dict[str, Any]] = []
        twin = model.base or model.post
        if twin and twin in self.models:
            out.append(
                {
                    "relation": "base" if model.base else "post-trained",
                    "model": twin,
                    "address": address.path(),
                    "status": "same index",
                    "how": "post-training keeps the indices",
                }
            )
        if address.is_head:
            for relation, other in (("parent", model.parent), ("child", model.child)):
                if other and other in self.models:
                    out.append(
                        {
                            "relation": relation,
                            "model": other,
                            "address": address.path(),
                            "status": "same index",
                            "how": "a pruned model keeps its parent's heads one for one",
                        }
                    )
            return out
        if not model.lineage:
            return out
        row = self._neuron_row(model, "lineage", address)
        if not row:
            return out
        for relation, other in (("parent", model.parent), ("child", model.child)):
            if not other:
                continue
            lay, n, p = (
                row.get(f"{relation}_layer", -1),
                row.get(f"{relation}_neuron", -1),
                row.get(f"{relation}_pole", 0),
            )
            status = row.get(f"{relation}_status") or ""
            if status == "cut":
                out.append(
                    {
                        "relation": relation,
                        "model": other,
                        "address": None,
                        "status": "cut",
                        "how": "cut when the smaller model was pruned",
                    }
                )
            elif lay is not None and lay >= 0 and p in (1, -1):
                out.append(
                    {
                        "relation": relation,
                        "model": other,
                        "address": Address.neuron(lay, n, p).path(),
                        "status": status,
                        "by": list(row.get(f"{relation}_by") or []),
                        "how": "matched on the weights; identified when an independent signal agrees",
                    }
                )
        return out

    # Overview and search

    def layer_scores(self, slug: str, layer: int) -> tuple[list[tuple[int, float, int, str]], bool]:
        """Every neuron of a layer as (neuron, score, surer pole, title), and whether the labels are lent."""
        model = self.model(slug)
        source = model.label_slug
        if source is None:
            return [], False
        rows = self.neuron_table(source, "labels", layer, LABEL_COLUMNS).to_pydict()
        poles: dict[int, list[tuple[float, int, str]]] = {}
        for n, p, fires, fc, wc in zip(
            rows["neuron"], rows["pole"], rows["fires"], rows["fires_conf"], rows["writes_conf"], strict=False
        ):
            poles.setdefault(int(n), []).append((pole_score(fc or 0.0, wc or 0.0), int(p), fires or ""))
        out = []
        for n, items in poles.items():
            items.sort(reverse=True)
            out.append((n, neuron_score([s for s, _, _ in items]), items[0][1], items[0][2]))
        return out, source != slug

    def layer_heads(self, slug: str, layer: int) -> list[tuple[int, float, str]]:
        model = self.model(slug)
        source = model.label_slug
        if source is None:
            return []
        try:
            labels = self.head_table(source, "labels").to_pydict()
        except FileNotFoundError:
            return []
        out = []
        for i, lay in enumerate(labels["layer"]):
            if lay != layer:
                continue
            confs = [labels[f"{k}_conf"][i] or 0.0 for k in ("fires", "attends", "writes")]
            out.append((int(labels["head"][i]), head_score(confs), labels["fires"][i] or labels["attends"][i] or ""))
        return out

    def overview(self, slug: str, top: int = 40, layers: Sequence[int] | None = None) -> dict[str, Any]:
        """The atlas landing: per layer, its `top` easiest neurons (surest labels) and all its heads, with strengths."""
        model = self.model(slug)
        out_layers = []
        lent = False
        for layer in layers if layers is not None else range(model.n_layers):
            scores, lent_here = self.layer_scores(slug, layer)
            lent = lent or lent_here
            scores.sort(key=lambda item: -item[1])
            units = [
                {
                    "address": Address.neuron(layer, n, pole).path(),
                    "title": title,
                    "score": round(score, 4),
                    "strength": round(strength("neuron", score), 3),
                }
                for n, score, pole, title in scores[:top]
            ]
            heads = [
                {
                    "address": Address.head(layer, h).path(),
                    "title": title,
                    "score": round(score, 4),
                    "strength": round(strength("head", score), 3),
                }
                for h, score, title in sorted(self.layer_heads(slug, layer), key=lambda item: -item[1])
            ]
            out_layers.append(
                {
                    "layer": layer,
                    "band": band(layer, model.n_layers),
                    "units": units,
                    "heads": heads,
                    "n_units": model.d_mlp,
                }
            )
        return {
            "model": slug,
            "hf_id": model.hf_id,
            "source": "catalog",
            "sizes": model.sizes,
            "labels": {"by": CATALOG_LABELLER, "lent_from": model.label_slug if lent else None},
            "ranking": "the labeller's confidence: a pole counts its surer label and a third of the other, a neuron "
            "its surer pole and a quarter of the other (as neuron-lens.com ranks its easiest units)",
            "layers": out_layers,
        }

    def search_documents(
        self, slug: str, reads: bool = False, progress: Callable[[int, int], None] | None = None
    ) -> Iterator[tuple[Address, str, str, float]]:
        """(address, title, searchable text, score) for every labelled pole and head; with `reads`, the reads' top
        tokens too (column-range reads of every layer, slow the first time)."""
        model = self.model(slug)
        source = model.label_slug
        read_cols = ["neuron", "pole", "write_R_str", "read_S_str", "read_R_str", "write_R_z_str", "read_S_z_str"]
        for layer in range(model.n_layers):
            if progress:
                progress(layer, model.n_layers)
            extra: dict[tuple[int, int], str] = {}
            if reads:
                try:
                    data = self.neuron_table(slug, "reads", layer, read_cols).to_pydict()
                except FileNotFoundError:
                    data = {}
                for i, (n, p) in enumerate(zip(data.get("neuron", []), data.get("pole", []), strict=False)):
                    extra[(int(n), int(p))] = " ".join(data[c][i] or "" for c in read_cols[2:] if c in data)
            if source is not None:
                rows = self.neuron_table(source, "labels", layer, LABEL_COLUMNS).to_pydict()
                cols = zip(
                    rows["neuron"],
                    rows["pole"],
                    rows["fires"],
                    rows["fires_conf"],
                    rows["writes"],
                    rows["writes_conf"],
                    rows["note"],
                    strict=False,
                )
                for n, p, fires, fc, writes, wc, note in cols:
                    text = f"{fires or ''} {writes or ''} {note or ''} {extra.pop((int(n), int(p)), '')}"
                    yield Address.neuron(layer, int(n), int(p)), fires or "", text, pole_score(fc or 0.0, wc or 0.0)
            for (n, p), text in extra.items():
                yield Address.neuron(layer, n, p), "", text, 0.0
        if source is None:
            return
        try:
            heads = self.head_table(source, "labels").to_pydict()
        except FileNotFoundError:
            return
        for i, layer in enumerate(heads["layer"]):
            confs = [heads[f"{k}_conf"][i] or 0.0 for k in ("fires", "attends", "writes")]
            title = heads["fires"][i] or heads["attends"][i] or ""
            text = " ".join(heads[k][i] or "" for k in ("fires", "attends", "writes", "note"))
            yield Address.head(int(layer), int(heads["head"][i])), title, text, head_score(confs)
