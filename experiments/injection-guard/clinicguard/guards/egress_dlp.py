"""L3: decode-then-scan egress DLP. Before a reply or an email leaves, decode base64, hex and rot13 and strip
spacing and punctuation, then look for the exact planted values (canaries from the database), not just their shapes."""

from __future__ import annotations

from ..db import secret_values
from . import Guard
from .decode import BASE, find_secret


class EgressDLP(Guard):
    name = "egress_dlp"
    decoders = BASE
    rounds = 1

    def check_egress(self, text, channel, ctx):
        hit = find_secret(text, secret_values(ctx.env.secrets), self.decoders, self.rounds)
        return f"{self.name}:{hit[0].split(':')[0]}:{hit[1]}" if hit else None
