# luxar.gsplats.preprocessing.tests

Pytest coverage for GSplat preprocessing and denoising backends.

## What is tested

- Backend dispatch between CUDA, PyTorch, and scikit-image implementations.
- NLM denoising behavior and shape/dtype preservation.
- Calibration helpers and parameter handling.
- Pipeline-level denoise behavior for zarr/numpy-style inputs.
- CUDA NLM behavior when the optional extension is available.
- Non-default-GPU placement and caller-device restoration for CUDA NLM.

## How to run

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/preprocessing/tests/
hatch run pytest packages/luxar/src/luxar/gsplats/preprocessing/tests/test_denoise_pipeline.py
```

CUDA-specific tests require optional dependencies and a built CUDA extension:

```bash
make build-nlm-cuda
hatch run pytest packages/luxar/src/luxar/gsplats/preprocessing/tests/test_nlm_cuda.py
```
