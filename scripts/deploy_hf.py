"""Deploy the production runtime to a Hugging Face Docker Space.

    HF_TOKEN=... uv run --with huggingface_hub python scripts/deploy_hf.py [space-name]

Builds a bundle (source + Dockerfile + Space README), creates <user>/<space-name>
if needed, uploads it, and sets the Space secrets/variables from the environment:
LANGSMITH_API_KEY (required: the self-hosted server checks it at startup) and,
optionally, GEMINI_API_KEY + AGENTDOCK_MODEL for real-model answers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
NAME = sys.argv[1] if len(sys.argv) > 1 else "agentdock"


def bundle(out: Path) -> None:
    for item in ["src", "agents", "data", "config", "pyproject.toml", "uv.lock", "langgraph.json", ".dockerignore"]:
        p = ROOT / item
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, out / item)
    (out / "deploy/hf-space").mkdir(parents=True)
    shutil.copy2(ROOT / "deploy/hf-space/start.sh", out / "deploy/hf-space/start.sh")
    shutil.copy2(ROOT / "deploy/hf-space/README.md", out / "README.md")
    base = out / "Dockerfile.base"
    subprocess.run(["uv", "run", "langgraph", "dockerfile", "-c", str(ROOT / "langgraph.json"), str(base)],
                   cwd=ROOT, check=True, capture_output=True)
    subprocess.run([sys.executable, str(ROOT / "deploy/hf-space/build_dockerfile.py"), str(base), str(out / "Dockerfile")], check=True)
    base.unlink()
    for junk in out.rglob("__pycache__"):
        shutil.rmtree(junk)


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set")
        return 2
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set (the server will refuse to start without it)")
        return 2
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo = f"{user}/{NAME}"
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "space"
        out.mkdir()
        bundle(out)
        api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="space", commit_message="Deploy agentdock production runtime")
    api.add_space_secret(repo, "LANGSMITH_API_KEY", os.environ["LANGSMITH_API_KEY"])
    if os.environ.get("GEMINI_API_KEY"):
        api.add_space_secret(repo, "GEMINI_API_KEY", os.environ["GEMINI_API_KEY"])
        api.add_space_variable(repo, "AGENTDOCK_MODEL", os.environ.get("AGENTDOCK_MODEL", "gemini:gemini-3.5-flash-lite"))
    host = f"https://{user.lower()}-{NAME.lower()}.hf.space"
    print(f"space: https://huggingface.co/spaces/{repo}")
    print(f"api:   {host}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
