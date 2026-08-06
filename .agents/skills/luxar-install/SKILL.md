---
name: luxar-install
description: >-
  Install Luxar on a machine — from a modest laptop to an NVIDIA-GPU desktop to a
  Slurm/HPC cluster login node. Use when a user wants to install, set up, or
  bootstrap Luxar (or fix a broken install): choosing user-install vs full dev
  setup, picking the right optional extras ([gsplats]/[io]/[demos]), getting the
  web viewer built, enabling CUDA/MPS acceleration, building the CUDA extension
  (locally or via Slurm), and the no-sudo HPC path. Covers make setup-dev /
  check-deps / setup-cuda / build-cuda SLURM=1, Git LFS, and verification steps.
---

# Install Luxar on any machine

Luxar is **not yet published to PyPI** — installation starts from a clone of the
repository. There are two install stories, then per-machine additions:

- **User install** — use Luxar (build scenes, fit splats, serve the viewer).
  A venv + editable pip install is enough.
- **Dev setup** — contribute to Luxar (run tests, lint, build viewer/WASM).
  `make setup-dev` bootstraps everything, no sudo needed.

## Which profile am I on?

| Machine | Fitting device | CUDA extension | Notes |
| --- | --- | --- | --- |
| Modest laptop (macOS / Linux / WSL) | CPU, or `mps` on Apple Silicon | skip | Viewing is WebGL (browser GPU) — smooth on any laptop; only *fitting* is slow |
| Desktop with NVIDIA GPU | `cuda` | optional but recommended | Orders-of-magnitude faster fitting (CUDA NLM denoise is a separate `make build-nlm-cuda`) |
| Slurm/HPC cluster | `cuda` on compute nodes | build via `SLURM=1` | No sudo, no GPU on login node — everything has a no-sudo fallback |

## Common base (all profiles)

Prerequisites: **Python 3.10+**, git, curl, `make`. Then:

```bash
git clone https://github.com/royerlab/luxar
cd luxar
python3 -m venv .venv && source .venv/bin/activate   # or your env manager
```

### User install

```bash
pip install -e .                    # core: scene compiler, CLI, serve
pip install -e ".[gsplats]"         # + Gaussian-splat FITTING (PyTorch, scipy)
pip install -e ".[gsplats,io]"      # + TIFF/imageio volume loading
pip install -e ".[demos]"           # + heavy demo deps (pandas, astropy, esm, ...)
```

Rule of thumb: viewing/serving needs no extras; `gsplat fit/cal` needs
`[gsplats]`; loading `.tiff` needs `[io]`. A missing extra raises an
actionable error naming the exact `pip install` to run.

**The web viewer** is not pre-built in a source clone. The serve commands
auto-build it on first use *if* Node + pnpm are available; the reliable way to
get those (plus Rust/wasm-pack, auto-installed by the build) is:

```bash
make setup-dev      # installs Node 22 LTS (nvm/brew), pnpm, hatch — no sudo
make build-viewer   # builds viewer dist (auto-installs Rust/wasm-pack if needed)
```

### Dev setup (contributors)

```bash
make setup-dev      # full bootstrap: Node, pnpm, hatch, pre-commit hooks
make check-deps     # audit what's installed / missing
git lfs install && git lfs pull    # demo data files (.npz/.zip) are Git-LFS
                                   # (install git-lfs first: brew/apt install git-lfs)
hatch run test      # Python tests
```

Hatch manages the Python env for dev work (`hatch run python ...`); the venv +
`pip install -e .` path above is only needed for the plain user story.

### Verify (any profile)

```bash
python -c "import luxar; print(luxar.__version__)"
luxar demo                          # lists the bundled demos
luxar demo run lorenz               # end-to-end smoke: generate + serve + view
```

On the Hatch-based profiles (dev setup / GPU / HPC) prefix these with
`hatch run` (e.g. `hatch run luxar demo`); the plain user-install venv runs
them directly.

## Profile: modest laptop

Everything above; skip CUDA entirely. Key points:

- **Viewing is never the bottleneck** — the viewer renders via WebGL in the
  browser, no CUDA/PyTorch involved.
- Fitting on CPU works but is slow: use `--preset draft`, small volumes, and
  modest `--seeds`. On **Apple Silicon**, pass `device='mps'` (Python) — the
  CLI's device auto-detection picks MPS up automatically.
- Skip `[demos]` (heavy: torch-adjacent ML deps, astropy, ...) unless running
  the science demos.
- Memory-constrained native exports: `LUXAR_CACHE_BUDGET_MB=<N>` lowers the
  launcher's viewer cache pool (default 2048).

## Profile: desktop with NVIDIA GPU

On this profile Hatch owns the environment, so do NOT use the `.venv` from
Common base — skip creating it, or `deactivate` it now before proceeding.
The CUDA make targets all run through Hatch's managed env (`hatch run ...`),
so drive the whole GPU path through Hatch — verify and build against the *same*
environment. The default Hatch env already includes `[gsplats]` (torch/scipy)
via `features=["dev"]`, so the first `hatch run` provisions it — no manual
install needed.

```bash
make setup-dev          # installs Hatch (the CUDA targets build inside it)
hatch run python -c "import torch; print(torch.cuda.is_available())"   # must be True
make check-cuda-deps    # nvcc, PyTorch-CUDA match, headers
make setup-cuda         # install CUDA deps + build extension (may need sudo)
make build-cuda         # or just build, if toolkit already present
make test-cuda          # verify the extension
```

The CUDA extension accelerates splatting internals; plain `device='cuda'`
fitting works without it (PyTorch ops only). CUDA NLM denoise is a *separate*
extension — build it with `make build-nlm-cuda` (→ `nlm_cuda_backend.so`);
without it, NLM denoise silently falls back to the PyTorch path.

**ABI gotcha:** the built `.so` is compiled against a specific torch ABI —
after ANY torch upgrade, rebuild with `make build-cuda` or imports fail with
symbol errors. It is also compiled for your local GPU architecture. Run GPU
fitting through the SAME env: `hatch run luxar gsplat fit ... --device cuda` —
not from a separate `.venv`, whose independently-resolved torch importing the
Hatch-built `.so` is the very ABI mismatch above.

Multiple GPUs? Whole-timelapse fitting across them is
`luxar gsplat batch-fit run ... --gpus auto` — see **`luxar-hpc-batch-fit`**.

## Profile: Slurm/HPC cluster (no sudo, no GPU on login node)

The Makefile auto-detects HPC and falls back to venv-based installs of hatch
and `pnpm --prefix ~/.local` — no sudo anywhere. As on the GPU profile, Hatch
owns the Python env here, so skip the Common-base `.venv` (or `deactivate` it
now) — the build/verify steps below all route through `hatch run`.

```bash
# 1. Bootstrap on the LOGIN node (python3.11/3.12 usually available)
make setup-dev

# 2. Tools land in ~/.local/bin — put it on PATH permanently
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

# 3. Verify the environment end to end
python3 scripts/check_hpc_setup.py

# 4. Build the CUDA extension ON A GPU NODE via Slurm
make build-cuda SLURM=1                                    # auto-detect modules
make build-cuda SLURM=1 SLURM_PARTITION=gpu                # pin partition
make build-cuda SLURM=1 CUDA_MODULE=cuda/12.8.0_570.86.10  # pin CUDA module

# 5. Monitor, then verify — INSIDE a GPU allocation, not on the login node
tail -f build-cuda-logs/build_<JOB_ID>.out
srun -p gpu --gres=gpu:1 --pty make test-cuda    # real check needs a GPU
```

`make test-cuda` on the **login node** verifies nothing: every CUDA test skips
without a GPU, so with no GPU it skips them all and still exits 0. Run it
inside an interactive `srun`/`salloc` on a GPU node to actually exercise the
extension.

`build-cuda SLURM=1` detects the PyTorch CUDA version, finds matching
`cuda/` + GCC>=9 modules, captures the Hatch env path, generates a self-contained
sbatch script, and submits it — the job log merges compiler output for
debugging. Full details: `docs/guides/developer/BUILD_SYSTEM_SPEC.md`.

Cluster-scale fitting (array jobs over T×C, the CZ Biohub **Bruno** cluster):
see the **`luxar-hpc-batch-fit`** skill.

## Common pitfalls

- **Demos fail with tiny/broken data files** → Git LFS not pulled:
  `git lfs install && git lfs pull` (pointer files are ~130 bytes).
- **"Viewer not available / not built"** → in a dev tree run
  `cd packages/luxar-viewer && pnpm build` (or `make build-viewer`).
- **CUDA extension import fails after upgrading torch** → rebuild:
  `make build-cuda` (ABI mismatch, see above).
- **hatch/pnpm "command not found" on HPC** → `~/.local/bin` missing from
  PATH (step 2 above).
- **Ubuntu/Debian setup-dev fails early** → on PEP-668 apt systems,
  `sudo apt-get install -y pipx && pipx ensurepath` first smooths the Hatch
  install. Not strictly required: `install-hatch` falls back to
  `pip install --user hatch` / a user venv when pipx is absent.
- **Broken bootstrap, want a clean slate** → `make clean-setup` then
  `make setup-dev`.

## Next step

With Luxar installed: build a scene with **`luxar-visualization`**, load a
volume with **`luxar-data-loading`**, fit splats with
**`luxar-gsplat-pipeline`**, or go to cluster scale with
**`luxar-hpc-batch-fit`**.
