# dynamic_ops Tests

Tests for the fixed-pool splat relocation system.

## Test Files

- **test_dynamic_ops.py** - Config defaults, residual peak finding (2D/3D/empty), splat importance calculation, weak splat selection, full relocation pipeline integration, convergence guard, and end-to-end fitting with dynamic ops enabled/disabled.
- **test_relocation_tracker.py** - `RecentlyRelocatedTracker` cooldown mechanism: initialization, marking, cooldown filtering, expiration, multiple relocations, statistics, and device compatibility.

## Running

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/fitting/dynamic_ops/tests/ -v
```
