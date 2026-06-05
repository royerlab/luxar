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
├── attribute-codec.ts       # Geometry-agnostic interleaved-buffer helpers
│                              (rebuildInterleavedBuffer, writePooledAttribute)
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
- A canonical per-instance attribute spec list (`POINTS_BASE_ATTRIBUTE_SPECS`, `LINES_BASE_ATTRIBUTE_SPECS`, `GSPLATS_ATTRIBUTE_SPECS`) used to allocate the geometry's single `InstancedInterleavedBuffer`.
- `acquireGeometry(nodeId, …)` — first checks `host.activeBuffers` for in-place reuse (matching capacity and, for points, matching attribute dtypes), then scans bucketed pools, then falls back to allocation. Capacity-grow paths call `attribute-codec.rebuildInterleavedBuffer` to widen the buffer while carrying old attribute data forward.
- `releaseGeometry(nodeId)` — moves the buffer into a per-capacity bucket and calls `host.evictUnused()`.
- `updateGeometry(geometry, data, count, …)` — writes per-instance attributes via `writePooledAttribute`, then recomputes `boundingBox` / `boundingSphere`. The lines adapter expands the box by max line width; the gsplats adapter expands by max Cholesky row-norm × truncation radius.

Adapters interact with the parent pool only through the narrow `*AdapterHost` interfaces — they read `activeBuffers`, `frameCount`, `stats`, `typeStats`, call `host.getBucket(count)` and `host.evictUnused()`, and set `host._lastAcquireRebuilt` so the parent knows whether the returned geometry still has its previous attribute bindings.

### Points-specific dtype tracking

Points geometries are pooled with full dtype awareness (`PointsAttributeTypes` in `pool-stats.ts`). A pooled point buffer is only reused when `attributeTypesMatch` confirms the new upload has the same color/radius/sharpness/scalar dtypes. This avoids hard-to-debug reuse bugs where, e.g., a Uint8 color view is reinterpreted as Float32. Lines and gsplats pool by capacity alone — their attribute layouts are uniformly Float32.

### Optional scalar attributes (colormaps)

Both points (`aScalar`) and lines (`aStartScalar` / `aEndScalar`) lazily add their scalar attribute slot the first time a colormap-bearing upload arrives. `attribute-codec.rebuildInterleavedBuffer` handles the spec-set transition: it carries forward every attribute that exists in both the old and new spec sets and binds the new ones at zero, so a colormap toggle costs one buffer rebuild rather than a full geometry replacement.

### Interleaved-attribute codec

`attribute-codec.ts` is the single source of truth for buffer rebuild and strided-write logic:

- `rebuildInterleavedBuffer(geometry, newCapacity, newSpecs)` — allocates a new `InstancedInterleavedBuffer`, deinterlaces from the old strided layout into the new one (carrying as many instances as fit), binds the new views, deletes attributes the new spec-set drops, and clears the `_maxInstanceCount` cache that r184 stashes on `InstancedBufferGeometry`.
- `writePooledAttribute(geometry, name, src, count)` — strided write of a packed source array into the interleaved buffer at the right offset. Adapters call this from `updateGeometry` instead of poking each attribute view individually.

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

- `PooledBuffer` — one pooled geometry plus the metadata the pool needs: `capacity`, `type` (`'points' | 'lines' | 'gsplats'`), `inUse`, `lastUsedFrame`, optional `attributeTypes` (points only).
- `PointsAttributeTypes` — per-attribute TypedArray dtype snapshot; the `scalar` field is optional and `undefined === undefined` makes `attributeTypesMatch` work without a sentinel.
- `TypePoolStats` / `PoolStats` — per-type counters plus aggregate totals. `PoolStats.deferredEvictions` is incremented when the per-call batch cap (`evictBatchSize`) trips, so a long pause + resume that stretches the eviction queue across multiple frames is observable.
- `PooledBufferRef` — minimal `{ bytes, payload? }` shape consumed by the pure `selectBuffersToEvict` selector.

## Invariants

- The parent `GPUBufferPool` is the **only** writer to `activeBuffers`, `stats`, `typeStats`, and `_lastAcquireRebuilt`. Adapters mutate these fields through their `Host` interface — never via direct construction or back-doors.
- Per-bucket arrays inside an adapter (`pointBuffers`, `lineBuffers`, `gsplatBuffers`) only ever contain pooled (not active) buffers. Active buffers live in `host.activeBuffers` keyed by `nodeId`.
- Capacity-grow rebuilds always `delete (geometry as { _maxInstanceCount? })._maxInstanceCount` — r184 caches this value on the geometry and replacing the underlying interleaved buffer doesn't invalidate it.
- After any `updateGeometry`, `boundingBox` and `boundingSphere` are recomputed. Frustum culling depends on these being current — the lines and gsplats paths additionally expand the box to cover the rendered footprint (line width, Cholesky row-norm × σ).

## See Also

- `../gpu-buffer-pool.ts` — parent pool that wires the three adapters together and owns shared state
- `../interleaved-attributes.ts` — `packInterleavedAttributes`, `widenToFloat32`, `writeInterleavedAttribute` primitives used by the codec
- `../line-geometry.ts` / `../gsplat-geometry.ts` — standalone-geometry counterparts (mirror the `_maxInstanceCount` workaround)
- `./geometry-bytes.ts` — `estimateGeometryBytes`, `invalidateCachedByteSize`
- `../README.md` § "GPU Buffer Pool" — package-level overview
