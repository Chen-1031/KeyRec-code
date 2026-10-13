# Reproducible environments

The virtual environments used during research are intentionally not copied
into this repository. Virtual environments contain platform-specific binaries,
absolute paths, and editable installs, so copying their directories would not
produce a portable or independently reproducible setup.

The files in this directory record the environments as observed on 2026-08-26.
Use the Gemma lockfile and the separate NEO dependency profile below.

Evaluation CLIs default to the user-writable `~/.cache/keyrec` tree. Cluster
runs should export `STREAMING_VU_STORAGE_ROOT` to a writable mounted directory
before launching either environment; this is applied before Transformers is
imported, including its dynamic-module cache.

## Gemma profile

The Gemma environment is governed by the repository-level `pyproject.toml`,
`.python-version`, and `uv.lock`. After initializing the pinned VLMEvalKit
submodule, create it with:

```bash
python scripts/setup_third_party.py
uv sync --frozen
```

Do not install from an unconstrained `requirements.txt`: `uv.lock` records the
complete dependency graph and package hashes.

## NEO-ov profile

NEO-ov requires a separate Python 3.10 environment and a different
Torch/Transformers stack. The observed transitive dependency snapshot is in
`neo/requirements.observed.txt`. Rebuild the historical snapshot with:

```bash
uv python install 3.10
uv venv --python 3.10 .venv_neo
uv pip install \
  --python .venv_neo/bin/python \
  --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -r environments/neo/requirements.observed.txt
```

The observed snapshot excludes the old environment's non-portable editable
install and local build path for `pip`. NEO model preprocessing comes from the
repository-owned, research-checksummed codec. The main Video-MME-v2 runner is
also repository-owned, and its answer extraction and group equations are
parity-tested against the pinned root VLMEvalKit scorer. The historical NEO
VLMEvalKit package is not imported at runtime.

`--no-deps` is intentional for this historical snapshot: the observed
environment contains an inconsistent Gradio/Hugging Face Hub pair that a
resolver correctly refuses to create. This command reproduces the observed
package set; it does not certify that set as a healthy release environment. Run
`uv pip check --python .venv_neo/bin/python` as a diagnostic; the observed
snapshot is expected to report the two incompatibilities above. A future NEO
release profile must be resolved from `requirements.in` and pass that check. 

