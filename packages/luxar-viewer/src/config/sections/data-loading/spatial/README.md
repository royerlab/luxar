# spatial

Data-loading spatial-query slice. Owns the default nD slicing tolerance and the default maximum splat radius used when a scene or geometry node does not declare its own.

Conforms to the section-trio pattern documented in [../../../README.md](../../../README.md), with one caveat: this slice has no `validate.ts` of its own — its fields are validated by the parent dispatcher (`../validate.ts`). Composed into `dataLoading.spatial` by the parent `data-loading` section.

## Contents

- `data.ts` — `dataLoadingSpatialConfig: DataLoadingSpatialConfig`. Defines `defaultTolerance: 0.1` (per-dimension tolerance used when computing nD slicing windows) and `defaultMaxRadius: 0.1` (fallback maximum splat radius used for spatial-index queries when the scene's `userData.maxRadius` or a geometry node's `attrs.max_radius` is absent).
- `types.ts` — `DataLoadingSpatialConfig` interface with the two numeric fields above.

## Public API

- `dataLoadingSpatialConfig` — composed into `dataLoadingConfig.spatial` by `../data.ts`.
- `DataLoadingSpatialConfig` — re-exported through `../../../types.ts`.
- Validation: handled in `../validate.ts` (the data-loading dispatcher) — rejects non-finite or non-positive `defaultTolerance` and `defaultMaxRadius`.

## Consumers

- `src/data/zarr-loader.ts` — falls back to `defaultMaxRadius` when `scene.userData.maxRadius` is unset, and passes `defaultTolerance` into `simpleDimsToViewState`.
- `src/data/view-state-manager.ts` — seeds per-dimension tolerance with `defaultTolerance`.
- `src/data/points/points-spatial-index-loader.ts` — uses `defaultMaxRadius` as the fallback when a points node does not declare `attrs.max_radius`.

Not a consumer: `src/data/loaders/spatial-query/tolerance-computer.ts` reads no value from this slice. It derives every hidden-dimension tolerance from the per-dimension `step`, the geometry type, the gsplat truncation-radius/Cholesky-regularization constants in `src/config/constants.ts` (the gsplats float-safety epsilon), `meshSlabTolerance` (the mesh slab, in cells), and — for points — the `maxRadius` its caller passes in. That points reach does still land on `defaultMaxRadius` from this slice for a node that declares no `attrs.max_radius`, because `points-spatial-index-loader.ts` resolves `attrs.max_radius ?? defaultMaxRadius` before handing it over — the value arrives as an argument, not by this module reading the config. See [../../../../data/loaders/spatial-query/README.md](../../../../data/loaders/spatial-query/README.md).

`defaultTolerance` reaches the loaders only as the `viewState.tolerance` ride-along that `view-state-manager.ts` seeds. No QUERY path reads it any more (issue #1183), but for a non-displayed CONTINUOUS dim its magnitude is still read by two non-query consumers — `loaders/progressive/slice-cache-helper.ts` keys the SliceCache on the raw value, and `loaders/progressive/view-state-equal.ts` compares it (both collapse the ride-along only for a _discrete_ non-spatial dim) — plus the `>= 1e9` extend-to-all sentinel. So this value diverging from the navigation-time builder's `defaultMaxRadius` (both 0.1 today) would cost cache misses and progressive resets, not wrong data.
