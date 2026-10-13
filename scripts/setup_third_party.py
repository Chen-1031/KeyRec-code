#!/usr/bin/env python3
"""Fetch the fixed VLMEvalKit revision required by uv.lock (also works in a ZIP)."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "58fdeb6b980bda22096d912d70d1c858dedc84fd"
URL = "https://github.com/open-compass/VLMEvalKit.git"


def run(*args, cwd):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main():
    parent = ROOT / "third_party"
    target = parent / "VLMEvalKit"
    parent.mkdir(exist_ok=True)
    if target.exists() and any(target.iterdir()):
        if not (target / ".git").exists():
            raise SystemExit(f"Existing directory is not a Git checkout: {target}")
        if run("rev-parse", "HEAD", cwd=target) != REVISION:
            raise SystemExit("Existing VLMEvalKit has another revision; move it aside first.")
        if run("status", "--porcelain", cwd=target):
            raise SystemExit("Existing VLMEvalKit has local changes; preserve them before setup.")
        print(f"VLMEvalKit is already at {REVISION}")
        return
    with tempfile.TemporaryDirectory(prefix="keyrec-dependency-", dir=parent) as tmp:
        staging = Path(tmp) / "VLMEvalKit"
        staging.mkdir()
        run("init", cwd=staging)
        run("remote", "add", "origin", URL, cwd=staging)
        run("fetch", "--depth", "1", "origin", REVISION, cwd=staging)
        run("checkout", "--detach", "FETCH_HEAD", cwd=staging)
        if run("rev-parse", "HEAD", cwd=staging) != REVISION:
            raise SystemExit("Downloaded revision does not match the dependency pin.")
        if target.exists():
            target.rmdir()  # Empty directory only; never remove an existing checkout.
        staging.rename(target)
    print(f"Ready: {target} at {REVISION}")


if __name__ == "__main__":
    main()
