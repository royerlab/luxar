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
- `src/data/loaders/spatial-query/tolerance-computer.ts` — multiplies `defaultTolerance` by the per-dimension `step` to capture N sigma of the geometry's Gaussian kernel.
- `src/data/points/points-spatial-index-loader.ts` — uses `defaultMaxRadius` as the fallback when a points node does not declare `attrs.max_radius`.
