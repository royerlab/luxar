# luxar.gsplats.models.gsplats.metal.tests

Tests for the Apple Metal/MPS Gaussian splatting backend.

## Scope

- Kernel-backend selection (`LUXAR_METAL_BACKEND`, AOT -> runtime -> torch order,
  forced-backend failures, torch-version floor, stale AOT builds skipped, fitting
  fallback) -- platform-independent, in `test_backend_selection.py`.
- Runtime-compiled kernels (`test_runtime_backend.py`): parity against the PyTorch
  reference (forward + backward) and against the AOT extension when built, int32
  binding of non-cubic shapes, ported validation, one-time thread-safe compile,
  and a no-Xcode / no-subprocess run.
- 3D splat-centric Metal forward/backward correctness against the PyTorch reference.
- L→conic conversion in native `[Z, Y, X]` packed-conic order.
- CUDA/base model interface parity: dynamic splat operations, clean nested state
  dicts, 2D/4D MPS PyTorch rendering, CPU/device-transfer rejection, and
  explicit FP16/dtype rejection.
- Optimizer integration and performance smoke/benchmark coverage.

## Running

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/metal/tests -q
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/metal/tests -q -m 'not slow'
# run the whole suite against one backend:
LUXAR_METAL_BACKEND=runtime hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/metal/tests -q
```
