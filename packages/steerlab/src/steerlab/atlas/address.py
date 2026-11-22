"""Unit addresses: `L8/n344+` (layer 8, neuron 344, positive pole), `L8/n344-`, `L20/h17` (layer 20, head 17).

The same spelling is the URL tail of the Units room (`/units/<model>/L8/n344+`) and what search accepts, with the
looser forms people type: `L8 n344 +`, `L8·n344`, `l8n344-`, `L20h17`, `L20 h17`. A neuron without a sign means the +
pole. The minus sign may be `-` or `−`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["neuron", "head"]

_UNIT = re.compile(
    r"^\s*l(?:ayer)?\s*(?P<layer>\d+)\s*[/·.\s_-]*\s*"
    r"(?:(?P<n>n(?:euron)?)\s*(?P<neuron>\d+)\s*(?P<sign>[+\-−]|plus|minus|pos|neg)?|(?P<h>h(?:ead)?)\s*(?P<head>\d+))\s*$",
    re.IGNORECASE,
)
_LAYER = re.compile(r"^\s*l(?:ayer)?\s*(?P<layer>\d+)\s*$", re.IGNORECASE)


@dataclass(frozen=True, order=True)
class Address:
    """One unit: a neuron pole (`index` = neuron, `pole` = +1 or −1) or a head (`index` = head, `pole` = 0)."""

    layer: int
    kind: Kind
    index: int
    pole: int = 1

    def __post_init__(self) -> None:
        if self.layer < 0 or self.index < 0:
            raise ValueError("layer and index must be non-negative")
        if self.kind == "neuron" and self.pole not in (1, -1):
            raise ValueError("a neuron's pole is +1 or -1")
        if self.kind == "head" and self.pole != 0:
            object.__setattr__(self, "pole", 0)

    @classmethod
    def neuron(cls, layer: int, neuron: int, pole: int = 1) -> Address:
        return cls(layer, "neuron", neuron, pole)

    @classmethod
    def head(cls, layer: int, head: int) -> Address:
        return cls(layer, "head", head, 0)

    @classmethod
    def parse(cls, text: str) -> Address:
        """Parse `L8/n344+`, `L8 n344 -`, `L20h17` and similar; raises ValueError otherwise."""
        match = _UNIT.match(text.replace("%2B", "+").replace("%2b", "+"))
        if not match:
            raise ValueError(f"{text!r} is not a unit address (try L8/n344+ or L20/h17)")
        layer = int(match["layer"])
        if match["h"]:
            return cls.head(layer, int(match["head"]))
        sign = (match["sign"] or "+").lower()
        return cls.neuron(layer, int(match["neuron"]), -1 if sign in ("-", "−", "minus", "neg") else 1)

    @classmethod
    def try_parse(cls, text: str) -> Address | None:
        try:
            return cls.parse(text)
        except ValueError:
            return None

    @property
    def is_head(self) -> bool:
        return self.kind == "head"

    @property
    def sign(self) -> str:
        return "" if self.is_head else ("+" if self.pole > 0 else "−")

    def path(self) -> str:
        """The URL spelling: `L8/n344+`, `L8/n344-` (ASCII minus) or `L20/h17`."""
        if self.is_head:
            return f"L{self.layer}/h{self.index}"
        return f"L{self.layer}/n{self.index}{'+' if self.pole > 0 else '-'}"

    def __str__(self) -> str:
        """The display spelling: `L8·n344+`, `L8·n344−`, `L20·h17`."""
        if self.is_head:
            return f"L{self.layer}·h{self.index}"
        return f"L{self.layer}·n{self.index}{self.sign}"

    def other_pole(self) -> Address:
        if self.is_head:
            raise ValueError("a head has no poles")
        return Address.neuron(self.layer, self.index, -self.pole)


def parse_layer(text: str) -> int | None:
    """`L8` or `layer 8` -> 8; anything else -> None."""
    match = _LAYER.match(text)
    return int(match["layer"]) if match else None


def model_slug(model_id: str) -> str:
    """A model id as the Units room spells it in URLs: `Qwen/Qwen3-0.6B` -> `qwen3-0.6b`, `org/My_Model` -> `my-model`.

    The catalog's own folder names are exactly these slugs (`qwen3-8b`, `qwen3-8b-base`, `gemma-4-12b`).
    """
    name = model_id.rstrip("/").split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9.]+", "-", name).strip("-")
    return name or "model"


def unit_href(model_id: str, layer: int, index: int, pole: int | None = None) -> str:
    """The Units room's link to a unit, for any room.

    `unit_href("Qwen/Qwen3-8B", 8, 344, 1)` -> `/units/qwen3-8b/L8/n344+`; with no pole it is a head:
    `unit_href("Qwen/Qwen3-8B", 20, 17)` -> `/units/qwen3-8b/L20/h17`. The `+` stays literal in the path; a page that
    puts an address in a query string must encode it (`%2B`)."""
    address = Address.head(layer, index) if pole is None else Address.neuron(layer, index, pole)
    return f"/units/{model_slug(model_id)}/{address.path()}"
