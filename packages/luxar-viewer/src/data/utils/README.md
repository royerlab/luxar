# Data utilities

Cross-cutting helpers used by the data layer that don't belong to any
single geometry type's loader.

## Files

| File                                    | Role                                                                                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `array-decoder.ts`                      | Encoding-aware decoder (`broadcasted` / `array_ref` / `LUT` / `quantized` / `direct`). The Python-encoder counterpart on the TS side. |
| `attrs-composer.ts`                     | Compose effective node attrs from parent + child + zarr metadata                                                                      |
| `data-accumulator.ts`                   | Zero-allocation buffer accumulator used by Points / Lines / GSplats loaders for nD scrub hot path                                     |
| `directory-navigator.ts`                | Filesystem-style navigation over a zarr group hierarchy (used by dataset browser)                                                     |
| `dims-to-view-state.ts`                 | Convert scene dimension metadata into the initial `ViewState`                                                                         |
| `scene-stats.ts`, `stats-aggregator.ts` | Roll up per-loader stats into a single scene snapshot                                                                                 |
| `tolerance-computer.ts`                 | Geometry-aware tolerance computation for nD slicing                                                                                   |

## Public surface

These are internal helpers. The public surface that consumers care
about is the top-level `src/data/index.ts` barrel.

## Invariants

- `ArrayDecoder` is **pure**: same `(rawData, encoding)` in, same
  `Float32Array` out, deterministic across calls. The decoder owns
  its own `ArrayRefRegistry` for resolving cross-array references.
- `DataAccumulator` mutators (`fill`, `ensureCapacity`) throw on
  post-dispose use rather than corrupting memory.
- `attrs-composer` walks parent attrs first, then child — child wins
  on conflict (standard inheritance).
