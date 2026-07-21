# GPU Buffer Pool — Per-Type Adapters and Cross-Type Eviction

> Internal helpers for `rendering/gpu-buffer-pool.ts`: per-geometry-type adapters (Points, Lines, GSplats) that own bucketed buffer reuse, a geometry-agnostic interleaved-attribute codec, and a byte-budget evictor that disposes pooled buffers across all three type pools once the global byte ceiling is exceeded.

The parent `GPUBufferPool` (`../gpu-buffer-pool.ts`) owns the shared coordination state — `activeBuffers`, `frameCount`, `stats`, eviction policy — and delegates per-type acquire / release / update to one adapter per geometry type. This split keeps each adapter focused on its own attribute layout and update path while eviction sees all three pools uniformly.

## File Structure

```
gpu-buffer-pool/
├── pool-stats.ts            # Shared types — PooledBuffer, PoolStats, TypePoolStats,
│                              PooledBufferRef, PointsAttributeTypes
├── eviction-policy.ts       # Pure largest-first selector — selectBuffersToEvict()
├── byte-budget-evictor.ts   # Cross-type eviction loop — evictUntilUnderByteBudget()
├── attribute-codec.ts       # Geometry-agnostic interleaved-buffer helper
│                              (writePooledAttribute — strided writes only;
│                              there is deliberately NO in-place rebuild)
├── geometry-bytes.ts        # Cached per-geometry byte estimate
│                              (estimateGeometryBytes, invalidateCachedByteSize)
├── capacity.ts              # Buffer-capacity sizing primitive
│                              (chooseCapacity, __setMinInstanceCapacityForTesting)
├── points-adapter.ts        # PointsBufferAdapter — acquire/release/update for points
├── lines-adapter.ts         # LinesBufferAdapter   — acquire/release/update for lines
└── gsplats-adapter.ts       # GSplatsBufferAdapter — acquire/release/update for gsplats
```

## Components

### Per-type adapters

Each adapter (`PointsBufferAdapter`, `LinesBufferAdapter`, `GSplatsBufferAdapter`) owns:

- A bucketed `Map<number, PooledBuffer[]>` of released-but-reusable geometries, keyed by capacity bucket.
- Points/Lines: a canonical per-instance attribute spec list (`POINTS_BASE_ATTRIBUTE_SPECS`, `LINES_BASE_ATTRIBUTE_SPECS`) used to allocate the geometry's single `InstancedInterleavedBuffer`. GSplats instead attach an RGBA32F **splat texture** + a `aSortedIndex` (Uint32) ordering attribute via `gsplat-geometry.ts::attachSplatStorage` (depth-sorting Phase 1); the texture is disposed by the geometry's own `dispose` event at every dispose site.
- `acquireGeometry(nodeId, …)` — first checks `host.activeBuffers` for in-place reuse (matching capacity; for lines, also a matching scalar spec set via the `hasScalars` parameter — points and gsplats use fixed texel layouts, so capacity is their only criterion), then scans bucketed pools best-fit, then falls back to allocation. **Growth is release + reacquire, never an in-place rebuild**: an undersized active buffer is released to the pool intact and the acquire falls through to best-fit/fresh allocation. Replacing a rendered geometry's attributes would strand the old GPU buffer in the renderer caches (freed only at GC mercy on classic WebGL; pinned permanently by the WebGPU renderer's strong `Info.memoryMap`). Content carry-forward is unnecessary — every commit rewrites all attributes for the full count right after acquire.
- `releaseGeometry(nodeId)` — moves the buffer into a per-capacity bucket and calls `host.evictUnused()`.
- `updateGeometry(geometry, data, count, …)` — Points/Lines write per-instance attributes via `writePooledAttribute`; GSplats run one fused texel pass (`writeSplatTexels`) plus an identity `aSortedIndex` fill. All recompute `boundingBox` / `boundingSphere`; the lines adapter expands the box by max line width, the gsplats adapter by max Cholesky row-norm × truncation radius.

Adapters interact with the parent pool only through the narrow `*AdapterHost` interfaces — they read `activeBuffers`, `frameCount`, `stats`, `typeStats`, call `host.getBucket(count)` and `host.evictUnused()`, and set `host._lastAcquireRebuilt` so the parent knows whether the returned geometry still has its previous attribute bindings.

### Points-specific dtype tracking

Points geometries are pooled with full dtype awareness (`PointsAttributeTypes` in `pool-stats.ts`). A pooled point buffer is only reused when `attributeTypesMatch` confirms the new upload has the same color/radius/sharpness/scalar dtypes. This avoids hard-to-debug reuse bugs where, e.g., a Uint8 color view is reinterpreted as Float32. Lines and gsplats pool by capacity alone — lines' attribute layout is uniformly Float32, and gsplat splat textures are always RGBA32F.

### Optional scalar attributes (colormaps)

Points carry their scalar dtype in `PointsAttributeTypes` (a dtype mismatch releases and reacquires). Lines declare scalar presence at ACQUIRE time — `acquireLinesGeometry(nodeId, count, hasScalars)` includes `aStartScalar`/`aEndScalar` in the creation spec set when the commit carries colormap data, and a spec-set mismatch on a pooled/active candidate releases and reacquires. `updateGeometry` never rebuilds in place (it throws if scalar data arrives on a base-only geometry — an acquire-contract violation).

### Interleaved-attribute codec

`attribute-codec.ts` holds the strided-write logic:

- `writePooledAttribute(geometry, name, src, count)` — strided write of a packed source array into the interleaved buffer at the right offset. Adapters call this from `updateGeometry` instead of poking each attribute view individually.

There is deliberately no in-place rebuild helper: a geometry's attribute views are never replaced after creation (see the growth note above).

### Capacity sizing

`capacity.ts::chooseCapacity(requested)` is the single sizing primitive every adapter calls on each allocate / grow: it rounds up to `requested × 1.5` (headroom so the next update rarely re-grows) but never below a `DEFAULT_MIN_INSTANCE_CAPACITY` floor of 256 instances. Living in this leaf module lets the three adapters import it without a cycle against the parent `gpu-buffer-pool.ts` barrel (which re-exports `chooseCapacity` and the `__setMinInstanceCapacityForTesting` test hook). `__setMinInstanceCapacityForTesting(value | null)` lowers the floor so unit tests can exercise the grow paths at small instance counts.

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
- `PointsAttributeTypes` — per-attribute TypedArray dtype snapshot; the `scalar` field is optional and `undefined === undefined` makes `attributeTypesMatch` work without a sentinel.
- `TypePoolStats` / `PoolStats` — per-type counters plus aggregate totals. `PoolStats.deferredEvictions` is incremented when the per-call batch cap (`evictBatchSize`) trips, so a long pause + resume that stretches the eviction queue across multiple frames is observable.
- `PooledBufferRef` — minimal `{ bytes, payload? }` shape consumed by the pure `selectBuffersToEvict` selector.

## Invariants

- The parent `GPUBufferPool` is the **only** writer to `activeBuffers`, `stats`, `typeStats`, and `_lastAcquireRebuilt`. Adapters mutate these fields through their `Host` interface — never via direct construction or back-doors.
- Per-bucket arrays inside an adapter (`pointBuffers`, `lineBuffers`, `gsplatBuffers`) only ever contain pooled (not active) buffers. Active buffers live in `host.activeBuffers` keyed by `nodeId`.
- `updateGeometry` deletes the r184 `_maxInstanceCount` cache after writes (gsplats/lines adapters) — r184 caches this value on the geometry and won't refresh it spontaneously. (Capacity-grow rebuilds no longer exist; growth swaps in a fresh geometry, which has no stale cache by construction.)
- After any `updateGeometry`, `boundingBox` and `boundingSphere` are recomputed. Frustum culling depends on these being current — the lines and gsplats paths additionally expand the box to cover the rendered footprint (line width, Cholesky row-norm × σ).

## See Also

- `../gpu-buffer-pool.ts` — parent pool that wires the three adapters together and owns shared state
- `../interleaved-attributes.ts` — `packInterleavedAttributes`, `widenToFloat32`, `writeInterleavedAttribute` primitives used by the codec
- `../line-geometry.ts` / `../gsplat-geometry.ts` — standalone-geometry counterparts (mirror the `_maxInstanceCount` workaround)
- `./geometry-bytes.ts` — `estimateGeometryBytes`, `invalidateCachedByteSize`
- `../README.md` § "GPU Buffer Pool" — package-level overview
