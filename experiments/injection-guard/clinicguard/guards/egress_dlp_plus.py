"""P2 prototype: stronger egress DLP. Adds reversed text, leetspeak, spelled-out digits, URL and base32 encodings,
homoglyphs and two rounds of decoding (base64 of hex, say); matches any 10-character window of a secret; and scans
everything the conversation has sent so far, so a secret split across two replies is caught on the second."""

from __future__ import annotations

from ..db import secret_values
from .decode import PLUS, collapse, views
from .egress_dlp import EgressDLP


class EgressDLPPlus(EgressDLP):
    name = "egress_dlp_plus"
    decoders = PLUS
    rounds = 2

    def check_egress(self, text, channel, ctx):
        sent = ctx.counters.setdefault("egress_so_far", [])
        hit = super().check_egress(text, channel, ctx) or super().check_egress(" ".join(sent + [text]), channel, ctx)
        if not hit:
            secrets = {collapse(v): lab for v, lab in secret_values(ctx.env.secrets).items() if len(collapse(v)) >= 12}
            for path, v in views(" ".join(sent + [text]), self.decoders, 1):
                for core, lab in secrets.items():
                    if any(core[i : i + 10] in v for i in range(len(core) - 9)):
                        hit = f"{self.name}:{lab.split(':')[0]}:partial:{path}"
                        break
                if hit:
                    break
        if not hit:
            sent.append(text)
        return hit
