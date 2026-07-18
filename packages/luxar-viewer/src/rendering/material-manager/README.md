# Material manager helpers

Focused modules split out from the `MaterialManager` orchestrator
class (one level up at `rendering/material-manager.ts`). The
orchestrator owns the registries, caches, and counter state; these
helpers own the pure logic — factory tables and cache-key construction,
generic LRU eviction, `dispose`-event subscription and registry
teardown, and the stats snapshot.

Every helper here is pure over its argument bundle (no `this`
reference). The orchestrator passes in snapshots / ctx objects and
the helpers either return values or mutate the supplied collections.
That split lets each module be unit-tested without standing up a real
renderer and keeps `material-manager.ts` focused on orchestration.

## Module map

| File           | Role                                                                                                                                                                                                                                                                                           |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `factories.ts` | `VISUAL_FACTORIES`, `PICKING_FACTORIES`, `MEGA_SHADER_FACTORIES` constructor tables keyed by `kind` × `backend`; `resolveMaterialBackend(caps)` (caps.apiSurface → `'glsl' \| 'tsl'`); `pointCacheKey` / `lineCacheKey` with shared integer bucketing (gsplat materials are per-node and uncached, so they have no cache key); `BlendingMode` type. |
| `lru-cache.ts` | Generic `lruGet` / `lruSet` over `Map<string, T>`. `Map` insertion order is the LRU order; hits are promoted by delete + re-insert, misses evict from the front until `maxSize` and fire `onEvict(key, value)` so the caller can dispose the resource.                                         |
| `lifecycle.ts` | `subscribeToDispose` wires a `'dispose'` listener that removes the material from every registry and from its kind-matched cache; `removeFromRegistries` is the same teardown reachable from `MaterialManager.unregister`; `SOFT_DISPOSE_FLAG` is the opt-out sentinel.                         |
| `stats.ts`     | `getCacheStats(ctx)` — diagnostic snapshot (per-cache sizes, total registered, eviction count, cumulative `new XMaterial()` wall-clock, configured `maxSize`, and the list of cache keys).                                                                                                     |

## How the orchestrator composes them

```
MaterialManager (class)
   │
   ├── backend dispatch ──► factories.resolveMaterialBackend(caps)
   │                         factories.VISUAL_FACTORIES[kind][backend]
   │                         factories.PICKING_FACTORIES[kind][backend]
   │                         factories.MEGA_SHADER_FACTORIES[backend]
   │
   ├── cache lookup ──────► factories.{point,line,gsplat}CacheKey(props, backend)
   │                         lru-cache.lruGet(cache, key)
   │                         lru-cache.lruSet(cache, key, mat, maxSize, onEvict)
   │
   ├── register/unreg ────► lifecycle.subscribeToDispose(mat, ctx)
   │                         lifecycle.removeFromRegistries(mat, ctx)
   │
   └── getCacheStats() ──► stats.getCacheStats(ctx)
```

## Key contracts

- **Stateless factories.** `factories.ts` exports only constructor
  tables and pure cache-key functions. The single piece of state it
  consults — renderer capabilities — is threaded in through
  `resolveMaterialBackend(caps)`; no module here imports a singleton.
  `caps === null` returns `'glsl'` so unit tests that touch material
  creation without configuring caps land on the WebGL2 dispatch.
- **Shared bucketing.** All three cache-key helpers route the common
  `{opacity, gamma, intensity, offset}` quartet through the same
  internal `getCommonMaterialBuckets` so the four ranges and rounding
  rules stay in sync. Kind-specific extras (`radiusScale`,
  `truncationRadius`) and the `transparent` bit
  (derived from `blendingMode !== 'opaque'`) are appended per kind.
- **LRU = `Map` insertion order.** `lruGet` promotes by delete +
  re-insert; `lruSet` evicts from `cache.keys().next().value` until
  `cache.size < maxSize`, firing `onEvict` for each one. `maxSize === 0`
  disables eviction (unbounded growth). Eviction-time disposal lives
  with the orchestrator via the `onEvict` callback so this module stays
  agnostic to material teardown.
- **Soft-dispose escape hatch.** `SOFT_DISPOSE_FLAG` is a
  `Symbol.for('luxar.invalidateRenderObject.softDispose')` set
  transiently on a material when the caller dispatches `'dispose'`
  purely to evict Three's cached `RenderObject` — used by
  `data/scene-loader/commit/invalidate-render-object.ts` after the GPU
  buffer pool rebuilds a geometry's underlying `InstancedInterleavedBuffer`.
  `subscribeToDispose`'s listener checks the flag and skips registry /
  cache cleanup so the material continues to receive camera updates and
  stays in its allocation cache. The flag is symbol-keyed so it can't
  collide with Three's internals or with userspace `userData`.
- **`LifecycleCtx` is the only handle.** `subscribeToDispose` and
  `removeFromRegistries` mutate registries and per-kind caches solely
  through the supplied `LifecycleCtx`; they never reach back into the
  orchestrator. The kind-to-cache match is by `instanceof` against the
  visual material classes (`PointMaterial | PointTSLMaterial`,
  `LineMaterial | LineTSLMaterial`, `GSplatMaterial | GSplatTSLMaterial`)
  — picking materials are not cached and are not removed here.
- **Wiring direction.** Cleanup flows manager → material via the
  `dispose` event subscription, never material → manager. This avoids
  an import cycle between `material-manager.ts` and the per-geometry
  material modules.

## See also

- `../README.md` — package-level overview, the 12-shader matrix, and
  the "5. Material Manager" section with usage examples.
- `../material-manager.ts` — the orchestrator class these helpers serve.
- `../materials/` and `../picking/` — the visual and picking material
  classes the factory tables instantiate.
- `../post-processing/mega/material.ts` + `material-tsl.ts` — the
  mega-shader pair reached through `MEGA_SHADER_FACTORIES`.
