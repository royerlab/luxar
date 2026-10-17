---
name: luxar-install
description: >-
  Install Luxar on a machine — from a modest laptop to an NVIDIA-GPU desktop to a
  Slurm/HPC cluster login node. Use when a user wants to install, set up, or
  bootstrap Luxar (or fix a broken install): pip install into a venv (Python
  3.12 on Ubuntu/macOS, CPU-only torch, extras [gsplats]/[io]/[demos]), GPU
  acceleration (runtime Metal kernels, NVRTC-compiled CUDA kernels, numba CPU
  kernels), and the source-checkout
  dev route: make setup-dev / check-deps / setup-cuda / build-cuda SLURM=1, the
  no-sudo HPC path, Git LFS, and verification steps.
---

# Install Luxar on any machine

Luxar is on PyPI. **Most users want `pip install` into a virtual environment**; a
source checkout (`make setup-dev`) is for contributors, for building the CUDA
extension, and for the few things that need `make` (native app launchers).

- **User install (pip)** — view scenes, compile your own, fit splats. Below.
- **Dev setup (clone)** — run tests, lint, build the viewer/WASM, build CUDA
  kernels. Further down.

## Which profile am I on?

| Machine | Fitting device | Fast splatting kernels | Notes |
| --- | --- | --- | --- |
| Laptop / desktop, no GPU (Linux, WSL) | CPU | numba CPU kernels (in `[gsplats]`, compiled on first use) | Viewing is WebGL in the browser and is fine anywhere; install the CPU-only torch wheel first |
| Mac with Apple Silicon | `mps` (auto) | Metal, compiled at runtime from a pip install (torch >= 2.7, no Xcode) | |
| Desktop with NVIDIA GPU | `cuda` (auto) | CUDA, compiled on first use with NVRTC from a pip install (no toolkit); `make build-cuda` optional | |
| Slurm/HPC cluster | `cuda` on compute nodes | NVRTC on the node, or build via `make build-cuda SLURM=1` | No sudo, no GPU on login node — everything has a no-sudo fallback |

## User install (pip)

### Prerequisites per OS: Python 3.12+ with `venv`

- **Ubuntu / Debian**: the stock image has Python but no `venv`/`pip` (and often no
  `curl`): `sudo apt update && sudo apt install python3-venv python3-pip`.
  Ubuntu 24.04's `python3` is 3.12; older releases need a newer Python (`uv`,
  below, or deadsnakes).
- **macOS**: the system `python3` is 3.9, and `pip3 install luxar` then fails with
  "No matching distribution found for luxar" (the wheel requires 3.12). Install
  3.12+ with the [python.org installer](https://www.python.org/downloads/macos/)
  — then run *Install Certificates.command* in its `/Applications/Python 3.x`
  folder, or every HTTPS dataset/record download fails with a certificate error —
  or `brew install python@3.12` (use `python3.12 -m venv ...`).
- **Any OS**: [`uv`](https://docs.astral.sh/uv/) fetches a Python itself:
  `uv venv --seed --python 3.12 luxar-env`, then `uv pip install ...` in place of
  `pip` (including the CPU-torch step below). Keep `--seed`: a plain `uv venv` has
  no pip, and `luxar demo deps --install` runs `python -m pip`.

### Install

```bash
python3 -m venv luxar-env && source luxar-env/bin/activate
pip install luxar                    # viewer (prebuilt), compiler, CLI, serve
pip install "luxar[gsplats,io]"      # + splat FITTING (PyTorch, scipy) and TIFF reading
```

Extras: `[gsplats]` = PyTorch + scipy (needed by `gsplat fit/cal/compare/...`);
`[io]` = tifffile + imagecodecs + imageio (TIFF, including compressed LZW/zstd
TIFF and OME-TIFF; other image formats — older releases, including 2026.9.22,
lack imagecodecs in `[io]`: `pip install imagecodecs`); `[demos]` = the heavy science-demo
stack (pandas, astropy, esm, cellxgene-census, ...) — skip it and let
`luxar demo deps --install` add only what a chosen demo reports missing. OME-Zarr
/ `.zarr` / `.npy` need no extra. A missing extra raises an error naming the
`pip install` to run.

**CPU-only Linux: install the CPU torch wheel FIRST.** On Linux, PyPI's default
torch wheel bundles CUDA (~3 GB of downloads, a 5+ GB venv) even on a machine with
no GPU. Pre-installing the CPU build keeps it out — pip will not replace a torch
that already satisfies `[gsplats]`:

```bash
pip install 'torch<3' --index-url https://download.pytorch.org/whl/cpu
pip install "luxar[gsplats,io]"
# uv environment: uv pip install 'torch<3' --index-url https://download.pytorch.org/whl/cpu
#                 uv pip install "luxar[gsplats,io]"
```

(macOS wheels have no CUDA payload; nothing to do there.)

**GPU notes for a pip install.**
- *Apple Silicon*: the Metal splatting kernels compile at runtime from the shipped
  `kernels.metal` via `torch.mps.compile_shader` — no Xcode, no build step (torch
  >= 2.7; 2.6 through its private API). A fit logs `Metal splatting kernels:
  runtime (...)`; `Metal splatting kernels: none, using PyTorch` means a torch
  too old for it (measured ~40x slower end to end, ~100x per iteration) — upgrade torch.
  `LUXAR_METAL_BACKEND=auto|aot|runtime|torch` forces a backend.
- *NVIDIA*: the CUDA splatting kernels compile on first use with NVRTC, the CUDA
  runtime compiler inside torch's NVIDIA wheels, and are cached under
  `~/.cache/luxar/cuda` — no CUDA toolkit or C++ compiler. Only an NVIDIA driver
  and a CUDA build of torch are needed (`python -c "import torch;
  print(torch.cuda.is_available())"` must print `True`).
  `LUXAR_CUDA_BACKEND=aot|nvrtc|torch` forces a backend.
- Which path a fit took is printed near its start: `Model class:
  GaussianSplatModelCUDA` / `GaussianSplatModelMetal, runtime kernels` /
  `GaussianSplatModelCPUNative (numba, ...)` (fast) or
  `GaussianSplatModel (PyTorch, device=...)` (reference path).
- *CPU*: 3D fits run on fused numba kernels (in `[gsplats]`; compiled on first use,
  then cached), on several threads automatically (`LUXAR_CPU_THREADS` overrides;
  `LUXAR_CPU_BACKEND=auto|native|torch` forces a backend). Still much slower than a
  GPU: read "Fitting on a CPU or a Mac" in `packages/luxar/src/luxar/gsplats/README.md`.

### Verify

```bash
python -c "import luxar; print(luxar.__version__)"
luxar demo                          # lists the bundled demos
luxar demo run lorenz               # end-to-end smoke: generate + serve + view
```

The viewer servers bind `127.0.0.1`; on a remote box tunnel the printed ports
(`ssh -L <port>:127.0.0.1:<port> host`) to view from your laptop.

## Dev setup (source checkout)

Prerequisites: **Python 3.12+** (with headers and `venv`), git, curl, `make`, a
C compiler and `jq`; on Linux also the GL/EGL runtime libraries. The one list,
identical in the README, CONTRIBUTING.md and BUILD_SYSTEM_SPEC.md:
`sudo apt-get update && sudo apt-get install -y make build-essential git curl jq python3-dev python3-venv pipx libegl1 libgl1-mesa-dri libegl-mesa0 libgbm1 libgles2`
on Ubuntu/Debian; `xcode-select --install` plus `brew install python@3.12 pipx jq`
on macOS. Then:

```bash
git clone https://github.com/royerlab/luxar
cd luxar
```

An editable pip install also works from the clone (`python3 -m venv .venv &&
source .venv/bin/activate && pip install -e ".[gsplats,io]"`), but the viewer is
NOT prebuilt in a clone: the serve commands auto-build it on first use *if* Node
+ pnpm are available, and the reliable way to get those (plus Rust/wasm-pack,
auto-installed by the build) is `make setup-dev && make build-viewer`.

### Bootstrap

```bash
make setup-dev      # full bootstrap: Node, pnpm, hatch, pre-commit hooks (no sudo)
make check-deps     # audit what's installed / missing
git lfs install && git lfs pull    # Dip-C's .npz payload is Git-LFS
                                   # (install git-lfs first: brew/apt install git-lfs)
hatch run test      # Python tests
```

Hatch manages the Python env for dev work: prefix the Verify commands above with
`hatch run` (e.g. `hatch run luxar demo`). The default Hatch env already includes
`[gsplats]` (torch/scipy) via `features=["dev"]`.

Dev-only knobs worth knowing:

- The optional ahead-of-time Metal extension (`python setup.py build_ext
  --inplace --force` in `gsplats/models/gsplats/metal/`, full Xcode required) is a
  developer tool: `auto` prefers it when built and current, the runtime path is
  equivalent and equal or slightly faster, and `LUXAR_METAL_BACKEND=runtime`
  selects the runtime path anyway (a forced backend that cannot run raises).
- Memory-constrained native exports: `LUXAR_CACHE_BUDGET_MB=<N>` lowers the
  launcher's viewer cache pool (default 2048). Native launchers (`luxar export
  --native`) need `make build-launchers`, so they are a checkout-only feature.

## Profile: desktop with NVIDIA GPU

**User install (no checkout): nothing to build.** On Linux, `pip install
"luxar[gsplats]"` with the default CUDA build of torch is enough: the custom CUDA
kernels are compiled for the card on first use by NVRTC (the runtime compiler that
ships in torch's NVIDIA wheels) and cached under `~/.cache/luxar/cuda`
(`$XDG_CACHE_HOME` honored). No `nvcc`, no C++ compiler. The first fit logs
`CUDA splatting kernels: nvrtc (...)`; the one-time compile of a dimensionality
takes well under a second. If it logs `none, using PyTorch`, the reason follows
(CPU-only torch, macOS, a GPU older than every architecture the installed
NVRTC can target -- sm_75 is the floor for CUDA 13's NVRTC, while CUDA 12's
reaches older cards, untested here -- or no libnvrtc;
`LUXAR_NVRTC_LIBRARY=/path/to/libnvrtc.so` points at one explicitly).
`LUXAR_CUDA_BACKEND=aot|nvrtc|torch` forces a path.

**Dev checkout:** the ahead-of-time extension is still supported and preferred
when it is built. Here Hatch owns the environment, so do NOT build into a
separate `.venv` — skip creating one, or `deactivate` it now. The CUDA make
targets all run through Hatch's managed env (`hatch run ...`), so verify and
build against the *same* environment. The default Hatch env already includes
`[gsplats]` (torch/scipy) via `features=["dev"]`, so the first `hatch run`
provisions it — no manual install needed.

```bash
make setup-dev          # installs Hatch (the CUDA targets build inside it)
hatch run python -c "import torch; print(torch.cuda.is_available())"   # must be True
make check-cuda-deps    # nvcc, PyTorch-CUDA match, headers
make setup-cuda         # install CUDA deps + build extension (may need sudo)
make build-cuda         # or just build, if toolkit already present
make test-cuda          # verify the extension
```

The custom kernels accelerate splatting internals; without either path,
`device='cuda'` fitting still works on PyTorch ops only (it warns). CUDA NLM
denoise follows the same order: the separate `make build-nlm-cuda` extension
(→ `nlm_cuda_backend.so`) if built, otherwise NVRTC, otherwise PyTorch.

Only the optional ahead-of-time build needs the CUDA toolkit (`nvcc`), and it
must match the CUDA version torch was built for
(`hatch run python -c "import torch; print(torch.version.cuda)"`). Install it from
NVIDIA's repository: Ubuntu's own `nvidia-cuda-toolkit` package is CUDA 12.0,
too old for a CUDA 13 torch or for Blackwell GPUs (sm_120/121).

**ABI gotcha (AOT extension only):** the built `.so` is compiled against a
specific torch ABI — after ANY torch upgrade, rebuild with `make build-cuda`
(until then it fails to import and the NVRTC path takes over). It is also compiled for your local GPU architecture. Run GPU
fitting through the SAME env: `hatch run luxar gsplat fit ... --device cuda` —
not from a separate `.venv`, whose independently-resolved torch importing the
Hatch-built `.so` is the very ABI mismatch above.

Multiple GPUs? Whole-timelapse fitting across them is
`luxar gsplat batch-fit run ... --gpus auto` — see **`luxar-hpc-batch-fit`**.

## Profile: Slurm/HPC cluster (no sudo, no GPU on login node)

The Makefile auto-detects HPC and falls back to venv-based installs of hatch
and `pnpm --prefix ~/.local` — no sudo anywhere. As on the GPU profile, Hatch
owns the Python env here, so skip any separate `.venv` (or `deactivate` it
now) — the build/verify steps below all route through `hatch run`.

```bash
# 1. Bootstrap on the LOGIN node (python3.12 usually available)
make setup-dev

# 2. Tools land in ~/.local/bin — put it on PATH permanently
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

# 3. Verify the environment end to end
python3 scripts/check_hpc_setup.py

# 4. Optional: build the CUDA extension ON A GPU NODE via Slurm
#    (without it, NVRTC compiles the kernels on the GPU node at first use)
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
`cuda/` + GCC>=10 modules, captures the Hatch env path, generates a self-contained
sbatch script, and submits it — the job log merges compiler output for
debugging. Full details: `docs/guides/developer/BUILD_SYSTEM_SPEC.md`.

Cluster-scale fitting (array jobs over T×C, the CZ Biohub **Bruno** cluster):
see the **`luxar-hpc-batch-fit`** skill.

## Common pitfalls

- **`pip install luxar` → "No matching distribution found"** → the interpreter is
  older than 3.12 (stock macOS is 3.9); see "Prerequisites per OS".
- **`python3 -m venv` → "ensurepip is not available"** (Ubuntu/Debian) →
  `sudo apt install python3-venv`.
- **A multi-GB torch download on a GPU-less Linux box** → you skipped the CPU
  wheel step; `pip uninstall torch` then install it from the CPU index.
- **Dataset downloads fail with `CERTIFICATE_VERIFY_FAILED` on macOS** → run the
  python.org *Install Certificates.command*.
- **"requires the 'imagecodecs' package"** reading a TIFF →
  `pip install "luxar[io]"` (or `pip install imagecodecs`).
- **`luxar demo run` fails with "Could not find project root"** → that installed
  release predates the pip-install demo-path fix; upgrade
  (`pip install -U luxar`) once a newer release is out, or run from a checkout.

- **Dip-C has a tiny/broken payload** → run `git lfs install && git lfs pull`
  (the pointer file is ~130 bytes). Other hosted demos fetch from their records;
  inspect the manifest/record error instead.
- **"Viewer not available / not built"** → in a dev tree run
  `cd packages/luxar-viewer && pnpm build` (or `make build-viewer`).
- **CUDA extension import fails after upgrading torch** → rebuild:
  `make build-cuda` (ABI mismatch, see above), or delete the stale `.so` and
  let the NVRTC path compile the kernels.
- **Fit logs "custom CUDA kernels are unavailable"** → the warning carries the
  reason. A CUDA build of torch on Linux normally brings libnvrtc; check
  `python -c "import torch; print(torch.version.cuda)"` is not `None`.
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
