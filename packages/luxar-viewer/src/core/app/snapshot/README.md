# Snapshot

JSON-serialisable capture and restore of viewer view state — the
visible camera placement and the current per-dimension slice position.
Intended for tests, share-view links, and regression harnesses that
need to reproduce a specific view across reloads or across separate
`LuxarApp` instances.

This module captures **only** the view: layer-panel state and
rendering-controls settings are intentionally excluded from v1 because
they live on different abstractions (`LayerStateManager`, the
rendering-controls settings persistence path) and have their own
serialise/restore routes. Adding them later is additive (new fields
under a higher `version`).

## Files

| File                 | Role                                                                                                                             |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `viewer-snapshot.ts` | `captureSnapshot` / `restoreSnapshot` + `ViewerSnapshot` / `CameraSnapshot` / `DimsSnapshot` types and `VIEWER_SNAPSHOT_VERSION` |

## Schema

```ts
interface ViewerSnapshot {
  version: 1;
  camera: CameraSnapshot; // position, target, up, isOrtho, near, far, fov?, zoom?
  dims?: DimsSnapshot; // ndim, displayed[], currentStep[]
}
```

`fov` is present only for perspective cameras; `zoom` is meaningful
for orthographic cameras. `dims` is omitted when no scene has been
loaded yet (the `sceneDimsManager` has nothing meaningful to report).

## Behaviour

- **`captureSnapshot(sceneManager)`** — reads the live camera and the
  controls' focus target, copies projection-specific fields based on
  `isPerspectiveCamera` / `isOrthographicCamera`, and snapshots
  `sceneDimsManager.getDims()` if present.
- **`restoreSnapshot(sceneManager, snap)`** — writes camera fields back
  in place, then calls `controls.setTarget(...)` + `controls.reinitialize()`
  so subsequent orbit/fly updates don't snap the camera back. Dims
  restore is **best-effort**: if `snap.version` mismatches
  `VIEWER_SNAPSHOT_VERSION`, the whole restore is skipped with a warning;
  if `snap.dims.ndim` doesn't match the loaded dataset, the dims block
  is skipped (camera still applies). The `displayed` set is **not**
  changed (re-displaying dims would re-frame the scene); only per-dim
  `currentStep` values are forwarded to `sceneDimsManager.setDimensionValue`.
  Returns `{ cameraApplied, dimsApplied }` so callers can confirm what
  ran.

## Consumers

- `core/app.ts` — re-exports as `LuxarApp.captureSnapshot()` and
  `LuxarApp.restoreSnapshot(snap)` (the embed-facing public API). Both
  methods throw if called before `init()`.
- `tests/unit/core/app/snapshot/viewer-snapshot.test.ts` — round-trip
  and version/ndim-mismatch coverage.

Nothing else in the codebase imports from this folder.
