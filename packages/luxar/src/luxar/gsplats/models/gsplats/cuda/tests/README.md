# luxar.gsplats.models.gsplats.cuda.tests

Pytest coverage for the CUDA Gaussian splat backend.

## What is tested

- Forward rendering kernels for 2D, 3D, and nD inputs.
- Backward gradients and gradcheck coverage.
- Numerical parity against reference PyTorch paths.
- Model-level integration with `GaussianSplatModel`.
- FP16 behavior and CUDA-specific performance smoke tests.
- Device placement and caller-device restoration on non-default GPUs.

## How to run

CUDA tests require a CUDA GPU. They run against whichever kernel backend
resolves (the compiled extension when it imports, else NVRTC, which needs no
build step); `LUXAR_CUDA_BACKEND=aot|nvrtc` pins one, so the suite can be run once
per backend. Only the AOT-path tests (the AOT-vs-NVRTC parity cases in
`test_nvrtc_parity.py`) need `make build-cuda`.

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/cuda/tests/
LUXAR_CUDA_BACKEND=nvrtc hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/cuda/tests/
make build-cuda && make test-cuda     # AOT extension path
```

On Slurm/HPC environments, build the AOT extension with the repository CUDA targets:

```bash
make build-cuda SLURM=1
make test-cuda
```
