# luxar.gsplats.models.gsplats.metal.tests

Tests for the Apple Metal/MPS Gaussian splatting backend.

## Scope

- Backend availability, stale native-artifact rebuilds, and MPS extension loading.
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
```
