"""Storage defaults for Streaming VU runs.

The pipeline keeps large models, datasets, and cache files outside the source
tree. Cluster runs should override the user-writable default with a mounted disk.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_STORAGE_ROOT = Path.home() / ".cache" / "keyrec"


def configure_storage_env() -> Path:
    """Configure cache/data environment variables for this process.

    STREAMING_VU_STORAGE_ROOT can override the base path. The derived env vars
    are intentionally set for the process so child subprocesses inherit the same
    storage policy.
    """
    storage_root = (
        Path(os.environ.get("STREAMING_VU_STORAGE_ROOT", DEFAULT_STORAGE_ROOT))
        .expanduser()
        .resolve()
    )
    hf_home = (
        Path(os.environ.get("STREAMING_VU_HF_HOME", storage_root / "huggingface"))
        .expanduser()
        .resolve()
    )
    lmu_data = (
        Path(os.environ.get("STREAMING_VU_LMUData", storage_root / "LMUData"))
        .expanduser()
        .resolve()
    )
    xdg_cache = (
        Path(os.environ.get("STREAMING_VU_XDG_CACHE_HOME", storage_root / "cache"))
        .expanduser()
        .resolve()
    )

    try:
        hf_home.mkdir(parents=True, exist_ok=True)
        (hf_home / "hub").mkdir(parents=True, exist_ok=True)
        lmu_data.mkdir(parents=True, exist_ok=True)
        xdg_cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            "KeyRec cache root is not writable; set STREAMING_VU_STORAGE_ROOT "
            "or the STREAMING_VU_* cache overrides to writable paths"
        ) from exc

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["LMUData"] = str(lmu_data)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
    return storage_root
