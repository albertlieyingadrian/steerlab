"""L3: provenance-bound tool execution.

Three rules, all string checks against where text came from:
1. Capability: only the tools granted to a public session run. A registered internal tool the model names anyway
   (tool-call minting) is refused.
2. Copied arguments: a tool argument of 16+ characters that appears verbatim in untrusted text (an email, intake
   notes, an insurer memo) is refused, so an instruction planted in data cannot hand the model a ready-made call.
3. Sends: an email goes out only if the user's own current message asks for one, and only to an address the user
   typed, the verified patient's address on file, or a clinic address. A recipient that came from untrusted text
   is refused.
"""

from __future__ import annotations

import re

from ..db import DOMAIN
from ..tools import PUBLIC_TOOLS
from . import Guard

SEND_INTENT = re.compile(r"\b(e-?mail|send|mail|forward)\w*", re.IGNORECASE)


class Provenance(Guard):
    name = "provenance"

    def check_tool_call(self, name, args, ctx):
        if name not in PUBLIC_TOOLS:
            return "provenance:tool_not_granted"
        skip = {"body", "subject"} if name == "send_email" else set()
        users_raw = " ".join(ctx.user_messages)
        for k, v in args.items():
            if k in skip or not isinstance(v, str) or len(v) < 16 or v.strip() in users_raw:
                continue  # text the person typed themselves is trusted even if an email repeats it
            if any(v.strip() in u for u in ctx.untrusted_texts):
                return "provenance:args_from_untrusted"
        if name == "send_email":
            if not SEND_INTENT.search(ctx.current_user):
                return "provenance:send_not_requested"
            to = str(args.get("to", "")).strip().lower()
            users = " ".join(ctx.user_messages).lower()
            if "[[untrusted" in to or (any(to and to in u.lower() for u in ctx.untrusted_texts) and to not in users):
                return "provenance:recipient_from_untrusted"
            on_file = str(ctx.env.session.get("email_on_file", "")).lower()
            if not (to and (to in users or to == on_file or to.endswith("@" + DOMAIN))):
                return "provenance:recipient_not_trusted"
        return None
