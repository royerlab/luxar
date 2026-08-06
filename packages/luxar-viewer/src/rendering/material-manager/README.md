# Material manager helpers

Focused modules split out from the `MaterialManager` orchestrator
class (one level up at `rendering/material-manager.ts`). The
orchestrator owns the registries and counter state; these helpers own
the pure logic — factory tables, `dispose`-event subscription and
registry teardown, and the stats snapshot. (The cache-key + LRU
machinery died with the lines texture-storage migration: ALL visual
materials are per node now, so nothing is cached or keyed.)

Every helper here is pure over its argument bundle (no `this`
reference). The orchestrator passes in snapshots / ctx objects and
the helpers either return values or mutate the supplied collections.
That split lets each module be unit-tested without standing up a real
renderer and keeps `material-manager.ts` focused on orchestration.

## Module map

| File                   | Role                                                                                                                                                                                                                                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `factories.ts`         | `VISUAL_FACTORIES`, `PICKING_FACTORIES`, `MEGA_SHADER_FACTORIES` constructor tables keyed by `kind` × `backend`; `resolveMaterialBackend(caps)` (caps.apiSurface → `'glsl' \| 'tsl'`); `BlendingMode` type.                                                                                                       |
| `lifecycle.ts`         | `subscribeToDispose` wires a `'dispose'` listener that removes the material from every registry; `removeFromRegistries` is the same teardown reachable from `MaterialManager.unregister`; re-exports the `SOFT_DISPOSE_FLAG` opt-out sentinel.                 |
| `soft-dispose-flag.ts` | Zero-import leaf module that defines the `SOFT_DISPOSE_FLAG` sentinel. Isolated here so the dispatcher (`data/scene-loader/commit/invalidate-render-object.ts`), which is reachable from the renderer bootstrap, can import the flag without dragging in the TSL material factories (they import `three/webgpu`). |
| `stats.ts`             | `getCacheStats(ctx)` — diagnostic snapshot (registry sizes and cumulative `new XMaterial()` wall-clock; no cache or eviction fields — every material is per-node).                                                                                                                         |

## How the orchestrator composes them

```
MaterialManager (class)
   │
   ├── backend dispatch ──► factories.resolveMaterialBackend(caps)
   │                         factories.VISUAL_FACTORIES[kind][backend]
   │                         factories.PICKING_FACTORIES[kind][backend]
   │                         factories.MEGA_SHADER_FACTORIES[backend]
   │
   ├── register/unreg ────► lifecycle.subscribeToDispose(mat, ctx)
   │                         lifecycle.removeFromRegistries(mat, ctx)
   │
   └── getCacheStats() ──► stats.getCacheStats(ctx)
```

## Key contracts

- **Stateless factories.** `factories.ts` exports only constructor
  tables. The single piece of state it consults — renderer
  capabilities — is threaded in through `resolveMaterialBackend(caps)`;
  no module here imports a singleton. `caps === null` returns `'glsl'`
  so unit tests that touch material creation without configuring caps
  land on the WebGL2 dispatch.
- **Per-node materials, no caching.** Every visual material carries its
  node's own element texture (`uPointTex` / `uLineTex` / `uSplatTex`),
  so `get{Point,Line,GSplat}Material` construct a fresh instance per
  call and only the registries track them.
- **Soft-dispose escape hatch.** `SOFT_DISPOSE_FLAG` is a
  `Symbol.for('luxar.invalidateRenderObject.softDispose')` set
  transiently on a material when the caller dispatches `'dispose'`
  purely to evict Three's cached `RenderObject` — used by
  `data/scene-loader/commit/invalidate-render-object.ts` after the GPU
  buffer pool swaps a geometry's storage on acquire.
  `subscribeToDispose`'s listener checks the flag and skips registry
  cleanup so the material continues to receive camera updates. The
  flag is symbol-keyed so it can't
  collide with Three's internals or with userspace `userData`.
- **`LifecycleCtx` is the only handle.** `subscribeToDispose` and
  `removeFromRegistries` mutate registries solely through the supplied
  `LifecycleCtx`; they never reach back into the orchestrator.
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
