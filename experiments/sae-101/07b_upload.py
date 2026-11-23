"""
Step 7b: upload the packaged repo (work/<size>/hf/ from 07a) to the Hugging Face Hub, public.

You need a Hugging Face account and a token with write access, saved once with `hf auth login` (it lands in
~/.cache/huggingface/token; never print it or commit it). The repo is created if it does not exist.

    .venv/bin/python experiments/sae-101/07b_upload.py --size full --repo <your-user>/qwen2.5-0.5b-sae-101
"""

import argparse

from huggingface_hub import HfApi

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="full")
ap.add_argument("--repo", default="albertlieadrian/qwen2.5-0.5b-sae-101")
args = ap.parse_args()
folder = s.WORK / args.size / "hf"
if not (folder / "README.md").exists():
    raise SystemExit("Run 07a_package.py first.")

api = HfApi()  # reads the saved token; it is never printed
user = api.whoami()["name"]
if not args.repo.startswith(user + "/"):
    raise SystemExit(f"The token belongs to {user!r}; --repo must start with '{user}/'.")
url = api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
commit = api.upload_folder(folder_path=str(folder), repo_id=args.repo, repo_type="model",
                           commit_message="SAE 101: TopK SAE on Qwen2.5-0.5B, block 11 residual stream")
print(f"1. Uploaded work/{args.size}/hf/ to {url}")
print(f"   commit {commit.oid[:12]}")

files = sorted(api.list_repo_files(args.repo))
info = api.model_info(args.repo)
print("   files on the Hub: " + ", ".join(files))
print("\nSelf-checks:")
s.check("the repo is public", not info.private)
s.check("the card and the main SAE are on the Hub",
        {"README.md", f"{s.HOOK_TL}/cfg.json", f"{s.HOOK_TL}/sae_weights.safetensors"} <= set(files))
