"""Improver v3 - More systematic approach to recipe generation.

    python improver.py <task.json> <out_dir>

Reads a task (edit a recipe, or edit an improver), asks the LLM once for complete new files and a one-line hypothesis,
checks them with the harness's validate() and asks for a fix (at most twice) if they fail, then writes
out_dir/files/<name> and out_dir/proposal.json. The LLM and validate() are reached only through broker.py.
"""

import json
import re
import sys
from pathlib import Path

from broker import ask, validate

HERE = Path(__file__).resolve().parent
FILE_RE = re.compile(r"^=== FILE: (\S+) ===\n(.*?)\n=== END FILE ===", re.DOTALL | re.MULTILINE)
HYP_RE = re.compile(r"^HYPOTHESIS:\s*(.+)$", re.MULTILINE)
MAX_TOKENS = 10000


def fmt_files(files: dict) -> str:
    return "\n\n".join(f"=== FILE: {n} ===\n{s}\n=== END FILE ===" for n, s in files.items())


def extract_feedback_summary(feedback: str) -> str:
    """Extract key patterns from feedback to guide future recipes."""
    summary = []
    
    # Look for constraint failures
    if "CONSTRAINT FAILED" in feedback:
        summary.append("CONSTRAINT FAILURE DETECTED: Macro recall dropped too low. Need to ensure macro recall >= R0's macro recall - 1 point.")
    
    # Extract score improvements
    if "dev score" in feedback:
        match = re.search(r"dev score ([+-]\d+\.\d+)%", feedback)
        if match:
            score_change = match.group(1)
            summary.append(f"Score change: {score_change}%")
    
    # Extract sample sizes
    if "rows" in feedback:
        rows_match = re.search(r"rows \{([^}]+)\}", feedback)
        if rows_match:
            rows_info = rows_match.group(1)
            summary.append(f"Sample composition: {rows_info}")
    
    # Extract successful patterns from previous recipes
    if "l3_04" in feedback and "34.4%" in feedback:
        summary.append("SUCCESSFUL PATTERNS: 3000 mining + 500 mirrors + 7500 replay works well")
    
    return "\n".join(summary) if summary else "No clear patterns detected"


def render(name: str, task: dict) -> str:
    t = (HERE / "prompts" / name).read_text()
    fields = {
        "FILES": fmt_files(task["files"]),
        "FEEDBACK": task.get("feedback", ""),
        "ARCHIVE": task.get("archive", ""),
        "FEEDBACK_SUMMARY": extract_feedback_summary(task.get("feedback", "")),
    }
    fields.update({k.upper(): v for k, v in task.get("docs", {}).items()})
    for k, v in fields.items():
        t = t.replace(f"[[{k}]]", str(v))
    return t


def parse(text: str) -> tuple[dict, str]:
    files = {m.group(1): m.group(2) for m in FILE_RE.finditer(text)}
    m = HYP_RE.search(text)
    return files, (m.group(1).strip() if m else "")


def main() -> None:
    task = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    system = (HERE / "prompts" / "system.md").read_text()
    user = render("recipe.md" if task["kind"] == "recipe" else "meta.md", task)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    
    # Try to get a valid proposal with up to 3 attempts
    for attempt in range(3):
        # Use higher temperature for initial attempts to encourage creativity
        temp = 0.7 if attempt == 0 else 0.3
        text = ask(msgs, max_tokens=MAX_TOKENS, temperature=temp)
        files, hyp = parse(text)
        
        # Validate the proposal
        if files:
            v = validate(files)
            if v.get("ok"):
                break
            err = v.get("error") or "unknown error"
        else:
            err = "No files found. Use exactly the === FILE: <name> === ... === END FILE === format."
            
        # If validation fails, ask for a correction
        msgs += [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": f"Validation failed:\n{err[:3000]}\n\nReturn the corrected complete files in the same "
                "format, then the HYPOTHESIS line.",
            },
        ]
    else:
        # If all attempts failed, use the last attempt as fallback
        pass
    
    (out / "files").mkdir(parents=True, exist_ok=True)
    for n, s in files.items():
        p = out / "files" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s)
    (out / "proposal.json").write_text(json.dumps({"hypothesis": hyp, "calls": len(msgs) // 2}))


if __name__ == "__main__":
    main()