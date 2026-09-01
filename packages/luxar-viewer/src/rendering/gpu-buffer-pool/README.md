# GPU Buffer Pool — Per-Type Adapters and Cross-Type Eviction

> Internal helpers for `rendering/gpu-buffer-pool.ts`: per-geometry-type adapters (Points, Lines, GSplats) that own bucketed buffer reuse, and a byte-budget evictor that disposes pooled buffers across all three type pools once the global byte ceiling is exceeded.

The parent `GPUBufferPool` (`../gpu-buffer-pool.ts`) owns the shared coordination state — `activeBuffers`, `frameCount`, `stats`, eviction policy — and delegates per-type acquire / release / update to one adapter per geometry type. This split keeps each adapter focused on its own attribute layout and update path while eviction sees all three pools uniformly.

## File Structure

```
gpu-buffer-pool/
├── pool-stats.ts            # Shared types — PooledBuffer, PoolStats, TypePoolStats,
│                              PooledBufferRef
├── eviction-policy.ts       # Pure largest-first selector — selectBuffersToEvict()
├── byte-budget-evictor.ts   # Cross-type eviction loop — evictUntilUnderByteBudget()
├── geometry-bytes.ts        # Cached per-geometry byte estimate
│                              (estimateGeometryBytes, invalidateCachedByteSize)
├── capacity.ts              # Buffer-capacity sizing primitive
│                              (chooseCapacity, __setMinInstanceCapacityForTesting)
├── acquire-options.ts       # Per-grow lifecycle hint (canRegrow)
├── dispose-superseded.ts    # Exact post-grow disposal after successful replacement
├── points-adapter.ts        # PointsBufferAdapter — acquire/release/update for points
├── lines-adapter.ts         # LinesBufferAdapter   — acquire/release/update for lines
└── gsplats-adapter.ts       # GSplatsBufferAdapter — acquire/release/update for gsplats
```

## Components

### Per-type adapters

Each adapter (`PointsBufferAdapter`, `LinesBufferAdapter`, `GSplatsBufferAdapter`) owns:

- A bucketed `Map<number, PooledBuffer[]>` of released-but-reusable geometries, keyed by capacity bucket.
- All three types attach an RGBA32F **element texture** (3 texels/point via `point-geometry.ts::attachPointStorage`; 6 texels/segment via `line-geometry.ts::attachLineStorage`; 4 texels/splat via `gsplat-geometry.ts::attachSplatStorage`) + an `aSortedIndex` (Uint32) ordering attribute (depth-sorting Phase 1 / §8); the texture is disposed by the geometry's own `dispose` event at every dispose site.
- `acquireGeometry(nodeId, …)` — first checks `host.activeBuffers` for in-place reuse (matching capacity — the fixed texel layouts make capacity the only criterion for every type), then scans bucketed pools best-fit, then falls back to allocation. **Growth is release + reacquire, never an in-place rebuild**: an undersized active buffer is released to the pool intact and the acquire falls through to best-fit/fresh allocation. Replacing a rendered geometry's attributes would strand the old GPU buffer in the renderer caches (freed only at GC mercy on classic WebGL; pinned permanently by the WebGPU renderer's strong `Info.memoryMap`). Content carry-forward is unnecessary — every commit rewrites all attributes for the full count right after acquire.
- `releaseGeometry(nodeId)` — moves the buffer into a per-capacity bucket and calls `host.evictUnused()`.
- `updateGeometry(geometry, data, count, …)` — one fused texel pass (`writePointTexels` / `writeLineTexels` / `writeSplatTexels`, each with a fail-loud pre-store guard) plus an identity `aSortedIndex` fill. Same-count recommits may preserve the prior permutation; Points/Lines appends extend it with an identity suffix, while GSplat appends reset the full ordering so alpha-over never draws an old sorted prefix and a new independently ordered suffix. All recompute `boundingBox` / `boundingSphere`; the lines adapter expands the box by max line width, the gsplats adapter by max Cholesky row-norm × truncation radius.

Adapters interact with the parent pool only through the narrow `*AdapterHost` interfaces — they read `activeBuffers`, `frameCount`, `stats`, `typeStats`, call `host.getBucket(count)` and `host.evictUnused()`, and set `host._lastAcquireRebuilt` so the parent knows whether the returned geometry still has its previous attribute bindings.

### Dtype-blind pooling

All three types pool by capacity alone. Points lost their dtype bucketing with the texture migration (dtype normalization happens at upload time — `widenToFloat32` in the adapter — so a Uint8 color view can never be reinterpreted as Float32), and lines lost their `hasScalars` spec bucketing the same way: the fixed RGBA32F texel layouts mean any pooled geometry of a type fits any node of that type.

### Optional scalar slots (colormaps)

Points (texel2.x) and lines (texel5.xy) always carry scalar texel slots, written with the 0.0 identity when the dataset has no scalars; real presence rides the `geometry.userData.hasScalars` stamp that every write path refreshes (pool geometries are reused across tenants). A colormap toggle therefore never rebuilds geometry or re-buckets the pool.

### Capacity sizing

`capacity.ts::chooseCapacity(requested)` is the single sizing primitive every adapter calls on each allocate / grow: it rounds up to `requested × 1.5` (headroom so the next update rarely re-grows) but never below a `DEFAULT_MIN_INSTANCE_CAPACITY` floor of 256 instances. The headroom is **capped at `MAX_CAPACITY_HEADROOM` = 262,144 elements**, so the factor does not scale into hundreds of wasted MiB on a very large node: headroom absorbs a per-slice count wobble of a few thousand elements, not a fixed share of however big the node is, and it is charged against a real per-element cost (a Lines geometry is 96 B/segment of element texture plus 8 B/segment of ordering pair, so an uncapped 1.5× is 52 B of slack per segment). `cosmicflows_laniakea_full` — nine sibling Lines nodes, 11.4M segments, no LOD ladder so all of it is resident — needed 1702 MiB of pool against a 2000 MiB budget and died with "Array buffer allocation failed"; the cap brings it to 1369 MiB. Nodes under 524,288 elements are unaffected and keep the full factor. Living in this leaf module lets the three adapters import it without a cycle against the parent `gpu-buffer-pool.ts` barrel (which re-exports `chooseCapacity` and the `__setMinInstanceCapacityForTesting` test hook). `__setMinInstanceCapacityForTesting(value | null)` lowers the floor so unit tests can exercise the grow paths at small instance counts.

### Eviction policy

`eviction-policy.ts::selectBuffersToEvict` is a pure function over any array of `{ bytes }` refs: sort largest-first, walk until the running total drops under `maxBytes`. It's unit-testable in isolation and used by the byte-budget evictor below. JS sort stability is relied on so equal-sized buffers retain deterministic input order.

### Cross-type byte-budget evictor

`byte-budget-evictor.ts::evictUntilUnderByteBudget` is the single pass that disposes pooled buffers across **all three** type pools when `totalPooledBytes > maxPoolBytes`:

1. Collect refs from `pointBuffers`, `lineBuffers`, and `gsplatBuffers`, with `estimateGeometryBytes` for each.
2. Emit a one-shot diagnostic warning if any single pooled buffer exceeds 100 MB (eviction of one big buffer pauses a frame: `geometry.dispose()` can take 5–20 ms on slow GPUs).
3. Run the pure selector, then splice from each `(pool, bucket)` largest-index-first so earlier splices don't invalidate later indices.
4. Call `geometry.dispose()` and bump `typeEvictionCounters[type].evictions` for each.

The evictor sees pooled-only buffers — active (in-use) buffers are never candidates. A hard `maxIterations = max(maxPoolSize × 3, 16)` cap protects against malformed ref shapes; if the cap kicks in, the pool may still be over budget after the pass and a warning is logged.

## Shared types — `pool-stats.ts`

- `PooledBuffer` — one pooled geometry plus the metadata the pool needs: `capacity`, `type` (`'points' | 'lines' | 'gsplats'`), `inUse`, `lastUsedFrame`.
- `TypePoolStats` / `PoolStats` — per-type counters plus aggregate totals. `PoolStats.deferredEvictions` is incremented when the per-call batch cap (`evictBatchSize`) trips, so a long pause + resume that stretches the eviction queue across multiple frames is observable.
- `PooledBufferRef` — minimal `{ bytes, payload? }` shape consumed by the pure `selectBuffersToEvict` selector.

## Invariants

- The parent `GPUBufferPool` is the **only** writer to `activeBuffers`, `stats`, `typeStats`, and `_lastAcquireRebuilt`. Adapters mutate these fields through their `Host` interface — never via direct construction or back-doors.
- Per-bucket arrays inside an adapter (`pointBuffers`, `lineBuffers`, `gsplatBuffers`) only ever contain pooled (not active) buffers. Active buffers live in `host.activeBuffers` keyed by `nodeId`.
- `updateGeometry` deletes the r184 `_maxInstanceCount` cache after writes — r184 caches this value on the geometry and won't refresh it spontaneously. (Capacity-grow rebuilds no longer exist; growth swaps in a fresh geometry, which has no stale cache by construction.)
- After any `updateGeometry`, `boundingBox` and `boundingSphere` are recomputed. Frustum culling depends on these being current — the lines and gsplats paths additionally expand the box to cover the rendered footprint (line width, Cholesky row-norm × σ).

## See Also

- `../gpu-buffer-pool.ts` — parent pool that wires the three adapters together and owns shared state
- `../widen-to-float32.ts` — dtype widening consumed by the points adapter
- `../point-geometry.ts` / `../line-geometry.ts` / `../gsplat-geometry.ts` — the per-type texel writers + storage attach helpers
- `./geometry-bytes.ts` — `estimateGeometryBytes`, `invalidateCachedByteSize`
- `../README.md` § "GPU Buffer Pool" — package-level overview
