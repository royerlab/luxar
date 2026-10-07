# Loader lifecycle

Per-geometry loader construction, the registry that tracks live and
failed loaders, and the shared update-loop scaffolding used by every
geometry type. These three files own the data-loader vocabulary
(create / register / lookup / run / record-failure) without any
view-state or commit knowledge.

## Files

| File                    | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `loader-factory.ts`     | Eight constructor helpers, two per geometry kind — see [Factory helpers](#factory-helpers). All compose each level's attrs from the parent's pre-composed effective attrs (`composeLevelAttrs`); the points, lines and gsplats helpers wrap them in a synthetic SceneNode per LOD (`buildAdditiveLevelNode`), while the mesh helper hands them straight to each `MeshWholeNodeLoader` (and has no energy table or SliceCache).                                                                                                                                                                                                                                                                                                                                                                                        |
| `loader-registry.ts`    | `LoaderRegistry` class — holds every loader in one store keyed by `GeometryKind` (bucketed from the contract's `GEOMETRY_TYPES`), reached generically via `register` / `loadersOf` or through the equivalent typed conveniences (`registerPointsLoader`…, and the per-kind accessors listed under [Registry accessors](#registry-accessors), which return the _same live_ `Map` objects). `LoaderByKind` ties each kind to its loader interface, so a mismatched `register` is a compile error; deriving `AnyDataLoader` as `LoaderByKind[GeometryKind]` additionally fails the build if a contract kind has no loader entry. Also holds the `failedLoaders` error-tracking map plus registration / lookup / `recordFailure` / `markAutoRetryAttempt` / `clearFailure` / `autoRetryablePaths` / `disposeAll` helpers. |
| `run-loader-updates.ts` | `runLoaderUpdates(loaders, loaderType, updateFn, ctx)` — shared scaffolding for the per-geometry update loops in `updateView`. Wraps each loader call in a profiler session, records ordinary failures into `failedLoaders` with a retry count, drops the predictive-prefetch baseline for failed or frustum-culled paths (loaders outside a targeted partition resync — `isResyncTarget` / `isUnderAny` — are skipped WITHOUT dropping it), hoists archive-wide faults out of the per-node loop, and returns the staged commits for the atomic commit phase.                                                                                                                                                                                                                                                         |
| `failure-report.ts`     | `reportLoadOutcome(failedPaths, registeredPaths, failureReasons)` grades a finished scene load — clean logs the historical success line, partial logs one aggregate warning, total (every registered node failed) logs an error plus a toast carrying the first recorded cause — and `warnFailedLoaders`, the aggregate warning shared with `updateView`.                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

## Factory helpers

Single-LOD helpers resolve the node's zarr `Location` and instantiate the
matching loader; progressive helpers open every `additive_<i>/` subgroup
(`openAdditiveLevels`) and wrap N per-LOD loaders.

<!-- mirrors: loader-factory.ts#exports -->

- `createPointsLoader` — a `PointsSpatialIndexLoader`.
- `createLinesLoader` — a `LinesSpatialIndexLoader`.
- `createGSplatsLoader` — a `GSplatsSpatialIndexLoader`.
- `createMeshLoader` — a `MeshWholeNodeLoader` (a mesh loads whole).
- `createProgressivePointsLoader` — a `PointsProgressiveLoader` over the additive levels.
- `createProgressiveLinesLoader` — a `LinesProgressiveLoader` over the additive levels.
- `createProgressiveGSplatsLoader` — a `GSplatsProgressiveLoader` over the additive levels.
- `createProgressiveMeshLoader` — a `MeshProgressiveLoader`, the mesh reveal ladder.

<!-- /mirrors -->

## Registry accessors

<!-- mirrors: loader-registry.ts#LoaderRegistry.getters -->

- `loaders` — the live points loaders `Map`.
- `linesLoaders` — the live lines loaders `Map`.
- `gsplatLoaders` — the live gsplats loaders `Map`.
- `meshLoaders` — the live mesh loaders `Map`.
- `totalLoaderCount` — loaders across every kind.
- `hasLoaders` — whether any kind holds a loader.

<!-- /mirrors -->

The two lists above are checked against `loader-factory.ts` and `LoaderRegistry` by
`src/tests/unit/readme/readme-claims.test.ts`: adding or renaming a helper or an accessor
without updating its list fails the build.

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
