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

CUDA tests require a CUDA-capable environment and the CUDA extension build.

```bash
make build-cuda
make test-cuda
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/cuda/tests/
```

On Slurm/HPC environments, build with the repository CUDA targets first:

```bash
make build-cuda SLURM=1
make test-cuda
```
