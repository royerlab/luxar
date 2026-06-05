# Loader lifecycle

Per-geometry loader construction, the registry that tracks live and
failed loaders, and the shared update-loop scaffolding used by every
geometry type. These three files own the data-loader vocabulary
(create / register / lookup / run / record-failure) without any
view-state or commit knowledge.

## Files

| File                    | Role                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `loader-factory.ts`     | Six constructor helpers. Three single-LOD: `createPointsLoader`, `createLinesLoader`, `createGSplatsLoader` — each resolving the node's zarr `Location` and instantiating the matching `*SpatialIndexLoader`. Three progressive: `createProgressiveGSplatsLoader`, `createProgressivePointsLoader`, `createProgressiveLinesLoader` — each opening every `additive_<i>/` subgroup and wrapping N per-LOD loaders in a `{GSplats,Points,Lines}ProgressiveLoader`, propagating the parent's pre-composed effective attrs into every LOD's synthetic SceneNode. |
| `loader-registry.ts`    | `LoaderRegistry` class — holds the three per-geometry loader maps (`loaders`, `linesLoaders`, `gsplatLoaders`), the `failedLoaders` error-tracking map, plus registration (incl. `unregisterGSplatsLoader` for released substitutive LODs) / lookup / `recordFailure` / `clearFailure` / `disposeAll` helpers.                                                                                                                              |
| `run-loader-updates.ts` | `runLoaderUpdates(loaders, loaderType, updateFn, ctx)` — shared scaffolding for the per-geometry update loops in `updateView`. Wraps each loader call in a profiler session, records failures into `failedLoaders` with a retry count, drops the predictive-prefetch baseline for failed paths, and returns the staged commits for the atomic commit phase.                                                                                   |

## Consumers

- `../lifecycle/{load-scene,dispose,retry}.ts` use the registry to
  construct, dispose, and re-attempt loaders.
- `../nodes/*` use the factory helpers to attach the matching
  loader to each leaf SceneNode during initial load.
- `../../scene-loader.ts` (parent orchestrator) calls
  `runLoaderUpdates` in its `updateView` body.
- `data/index.ts` re-exports `LoaderRegistry` + `FailedLoaderInfo`
  as part of the `data` package's surface — keep this re-export in
  sync if symbols change.
