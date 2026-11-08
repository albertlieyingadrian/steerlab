"""
Download what the run needs, at pinned versions:
- AuditBench's PRISM-4 system prompt and its 14 quirk prompts, from safety-research/auditing-agents at commit
  e9085916, rendered to plain text in the work folder (the repository has no licence file, so they are not stored
  here; data/auditbench_sha256.txt records the hash of each downloaded file);
- Qwen3-14B, the PRISM-4 tokenizer and the 42 adapters, into the Hugging Face cache (about 72 GB).

    .venv/bin/python experiments/auditbench/fetch.py [--prompts-only]
"""

from __future__ import annotations

import hashlib
import re
import sys
import urllib.request

import ab

RAW = f"https://raw.githubusercontent.com/safety-research/auditing-agents/{ab.AUDITBENCH_COMMIT}/"
PROMPTS = "src/model_organism/prompts/"


def get(path: str) -> str:
    with urllib.request.urlopen(RAW + path, timeout=60) as r:
        return r.read().decode()


def render_quirk(template: str, quirk_file: str) -> str:
    content = re.search(r"{%\s*block content\s*%}(.*?){%\s*endblock", quirk_file, re.DOTALL).group(1).strip()
    body = re.sub(r"{%-?\s*block content\s*%}{%\s*endblock content\s*-?%}", lambda _: content, template)
    return body.strip()


def main() -> None:
    out = ab.WORK / "auditbench_prompts"
    out.mkdir(parents=True, exist_ok=True)
    hashes = []
    files = {
        "prism_4": PROMPTS + "prism_4/general_system_prompt.jinja2",
        "_template": PROMPTS + "system_prompt_quirks/templates/base_instruction.jinja2",
    }
    files |= {b: PROMPTS + f"system_prompt_quirks/{b}.jinja2" for b in ab.BEHAVIOURS}
    raw = {}
    for name, path in files.items():
        raw[name] = get(path)
        hashes.append(f"{hashlib.sha256(raw[name].encode()).hexdigest()}  {path}")
    (out / "prism_4.txt").write_text(raw["prism_4"].split("{% if")[0].strip() + "\n")
    for b in ab.BEHAVIOURS:
        (out / f"{b}.txt").write_text(render_quirk(raw["_template"], raw[b]) + "\n")
    (ab.DATA / "auditbench_sha256.txt").write_text(f"# commit {ab.AUDITBENCH_COMMIT}\n" + "\n".join(hashes) + "\n")
    print(f"prompts: 1 PRISM-4 + {len(ab.BEHAVIOURS)} quirk prompts -> {ab.short(out)}")
    print("PRISM-4:", (out / "prism_4.txt").read_text().strip())
    if "--prompts-only" in sys.argv:
        return

    from huggingface_hub import snapshot_download

    snapshot_download(ab.BASE_MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    snapshot_download(ab.TOKENIZER)
    for o in ab.organisms():
        snapshot_download(o["repo"], allow_patterns=["adapter_config.json", "adapter_model.safetensors"])
        print("adapter", o["id"], flush=True)
    print(f"models: {ab.BASE_MODEL}, {ab.TOKENIZER} and {len(ab.organisms())} adapters in the Hugging Face cache")


if __name__ == "__main__":
    main()
