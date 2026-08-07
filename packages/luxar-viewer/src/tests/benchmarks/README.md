# tests/benchmarks — Micro-benchmarks for hot paths

**Purpose**: Performance benchmarks comparing WASM vs TypeScript implementations of compute-intensive kernels.

These benchmarks measure **real-world performance** of critical hot-path operations to justify the WASM acceleration strategy and track regressions.

## Modules

| File                | Status       | Description                                                                                                                                      |
| ------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `wasm-benchmark.ts` | **Complete** | WASM vs TypeScript comparison across six kernel categories: effective radii, depth sort, decode, projection, lines clipping, gsplats processing. |

## WASM Benchmark

### Running

```bash
# From packages/luxar-viewer/
pnpm bench:wasm              # 100K elements (default)
pnpm bench:wasm --size=small # 1K elements
pnpm bench:wasm --size=large # 1M elements

# Or from repo root
make benchmark-wasm
```

**Prerequisites**: WASM module must be built first (`make build-wasm`). The benchmark auto-detects missing WASM files and prints a build reminder.

### Categories

| Category               | Operations                                                                                                                                                                                                                    | Notes                                                                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **EFFECTIVE RADII**    | `calculate_effective_radii` — nD → 3D hypersphere projection for points visibility.                                                                                                                                           | Core hot path for 4D+ datasets.                                                                                                                             |
| **DEPTH SORT**         | `sort_splats_by_depth` — back-to-front ordering for correct alpha blending.                                                                                                                                                   | Budget: ≥ 50 M splats/s (enforced by `perf-budget.test.ts` in the opt-in perf suite, not the default unit tests). Reports absolute throughput.              |
| **DECODE**             | Nine tasks: `decode_quantized_u8`, `decode_quantized_u16`, `decode_log_scalar_u8`, `decode_log_scalar_u16`, `decode_lut_scalar_u8`, `decode_lut_scalar_u16`, `decode_lut_row_u8`, `decode_lut_row_u16`, `decode_broadcasted`. | Decodes compressed Python `luxar.encoding` arrays → Float32.                                                                                                |
| **PROJECTION**         | `extract_3d_positions`.                                                                                                                                                                                                       | nD → 3D projection helper.                                                                                                                                  |
| **LINES CLIPPING**     | `clip_segments_batch`, `interpolate_clipped_positions`, `interpolate_scalars_batch`, `interpolate_colors_batch`, `calculate_segment_lengths`, `compute_joint_codes`, plus a utility.                                      | Batch nD → 3D clipping + attribute interpolation. The per-call utility `clip_segment_single` is marked `utility: true` (not a hot path).                    |
| **GSPLATS PROCESSING** | `project_gsplats_nd_to_3d`, plus a utility.                                                                                                                                                                                   | The fused production kernel (discrete gate → attenuation → visibility → compaction). The per-call utility `mahalanobis_distance` is marked `utility: true`. |

### Output

```
======================================================================
  WASM vs TypeScript Performance Benchmark
======================================================================

System: Node.js v20.10.0, WASM module loaded ✓

Running benchmarks with 100,000 elements, 5 iterations each...

EFFECTIVE RADII
----------------------------------------------------------------------
Function                         TypeScript         WASM      Speedup
calculate_effective_radii              12.34ms      3.21ms      3.8x

DEPTH SORT
----------------------------------------------------------------------
Function                         TypeScript         WASM      Speedup
sort_splats_by_depth (312 M splats/s WASM)  15.67ms  3.20ms  4.9x

[... more categories ...]

======================================================================
  SUMMARY
======================================================================

  Batch Functions (used in production):
   Functions:        19
   Average speedup:  3.7x
   Fastest speedup:  5.2x (decode_quantized_u16)
   Slowest speedup:  2.1x (extract_3d_positions)

  * Utility Functions (API completeness, not used in hot paths):
   Functions:        2
   Average speedup:  0.8x (expected <1x due to WASM call overhead)

  Note: Utility functions (*) exist for API completeness and testing.
  In production, batch functions inline the math, avoiding per-call overhead.
  WASM call overhead dominates for trivial per-call operations.
```

### Batch vs Utility Functions

The benchmark separates **batch** (production hot paths) from **utility** (API completeness, per-call convenience) functions:

- **Batch functions** (unmarked): Used in production. Process arrays of elements in one call, amortizing WASM call overhead. These drive the speedup summary.
- **Utility functions** (`utility: true`): Exist for API completeness and testing. Not used in production hot paths — batch functions inline the math instead. Expected to show **<1x** speedup due to WASM call overhead dominating trivial per-call operations (e.g., `mahalanobis_distance`).

### Configuration

```typescript
const CONFIG = {
  iterations: 5, // Timing runs per benchmark
  warmupIterations: 2, // Warmup runs before timing
  sizes: {
    small: 1_000,
    medium: 100_000, // Default
    large: 1_000_000,
  },
};
```

### Performance Budget

The depth-sort benchmark enforces a **≥ 50 M splats/s** budget (via `perf-budget.test.ts`). That test is excluded from the default unit suite (see `vitest.config.ts`) and runs only under the opt-in perf suite (`pnpm test:perf`). The benchmark reports absolute throughput so regressions are visible in logs even when the budget still passes.

## When to Benchmark

Run benchmarks when:

- Optimizing a hot-path kernel (baseline → optimized comparison).
- Evaluating WASM vs TypeScript for a new operation.
- Verifying no regression after refactoring compute-intensive code.
- Investigating performance issues reported by users.

## Benchmark Design Principles

1. **Realistic workloads**: Use representative data (random positions, Cholesky factors, quantized codes) matching production datasets.
2. **Warmup runs**: Always warmup before timing to avoid JIT cold-start bias.
3. **Multiple iterations**: Average over multiple runs to reduce noise.
4. **Output verification**: Check that WASM and TypeScript produce **identical** results (not just measure speed).
5. **Batch focus**: Prioritize batch operations (what production uses) over per-element utilities.

## See Also

- [`../unit/`](../unit/) — Unit tests verifying correctness (WASM/TypeScript parity).
- [`../../wasm/`](../../wasm/README.md) — WASM module implementation and build system.
- [`../../wasm/typescript/`](../../wasm/typescript/) — TypeScript reference implementations (uncapped nD support; >16D fallback backend).
- [`../../workers/data-worker/`](../../workers/data-worker/) — Web Worker pipeline that calls these kernels for dataset decoding + projection.
- `vitest.config.ts` — excludes `perf-budget.test.ts` from the default suite; it enforces the depth-sort ≥ 50 M splats/s gate only under the opt-in perf suite (`pnpm test:perf`).
