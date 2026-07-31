# viewer-config

Apply the per-dataset `viewer_config` blob (read from a zarr scene by
`SceneManager`) onto the running `LuxarApp`, and capture/restore UI-panel
visibility around panel-sensitive operations (e.g. recording).

The rendering-pipeline knobs inside `viewer_config` (bloom, EOG, tone
mapping, detector noise, control type, etc.) are routed through
`RenderingControls` via `setZarrViewerConfig` — see
`src/config/zarr-bridge/`. This folder handles **the rest**: UI panel
show/hide flags, the active theme, dimension-navigation step, and a
small pure helper for snapshotting panel visibility.

## Contents

```text
viewer-config/
├── apply-state.ts        # applyViewerConfigState() — dispatch zarr-blob
│                         #   non-rendering fields onto LuxarApp subsystems
└── panel-visibility.ts   # get/restorePanelVisibilityStates() — pure
                          #   snapshot/restore around panel-aware ops
```

## How a `viewer_config` reaches these helpers

```
zarr scene -> SceneManager.loadSceneData()
           -> sceneManager.getSceneViewerConfig(): ZarrViewerConfig | undefined
           -> RenderingControls.setZarrViewerConfig(cfg)         (rendering settings)
           -> LuxarApp.applyViewerConfigState(cfg)               (this folder)
                    -> applyViewerConfigState(cfg, ports)
                          - ui.show_*          -> panel show()/hide()
                          - theme              -> setTheme(themeId)
                          - dimensions.current_step[i] -> setDimensionValue(i, v)
```

The dispatcher is invoked once per `loadDataset()` call (i.e. on initial
load and on each dataset switch). Unset fields are intentionally not
applied so the viewer's own defaults — and any user customisation from a
previous session — survive.

## apply-state.ts

`applyViewerConfigState(viewerConfig, ports)` is a pure dispatcher that
takes a `ZarrViewerConfig | undefined` and a `ViewerConfigPorts` bag of
typed function refs (panels, theme setter, dimension setter) and mutates
the targets. It was extracted from `core/app.ts` so the dispatch logic
can be unit-tested without booting a full `LuxarApp`.

Key contract details:

- `undefined` viewerConfig short-circuits — the helper preserves the
  pre-existing call-site behavior where the zarr scene metadata may be
  absent.
- Each `ui.show_*` flag is **tri-state**: `true` calls `show()`, `false`
  calls `hide()`, `undefined` leaves the panel alone. `show_help`,
  `show_performance_monitor`, and `show_dimensions` only honour the
  `true` branch (no hide path is wired).
- `ports.scaleBar`, `ports.layersPanel`, and `ports.overlayManager` are
  optional because those panels are only built when their backing data
  exists (dimension metadata / `overlay_groups` / layer arrays).
- `dimensions.current_step` writes one value per dimension index via
  `setDimensionValue(i, v)` — no theme/UI side effects.

## panel-visibility.ts

`getPanelVisibilityStates(ports)` / `restorePanelVisibilityStates(states,
ports)` form a snapshot-and-restore pair around operations that need to
hide UI panels (recording, for instance) and put them back afterwards.

- Tracks `renderingControls` and `recordingPanel`. Missing panels record
  `false` so the restore path is safe after a panel is torn down.
- Restore semantics are asymmetric on purpose: a saved `true` calls
  `show()` unconditionally (idempotent), while a saved `false` only
  calls `hide()` when the panel is currently visible — this avoids
  spurious visibility-transition telemetry on already-hidden panels.

## See Also

- [`../../README.md`](../../README.md) — `core` package overview;
  this folder appears in its architecture tree as `core/app/viewer-config/`.
- [`../../../config/zarr-bridge/README.md`](../../../config/zarr-bridge/README.md)
  — sibling bridge that converts the rendering-settings half of
  `viewer_config` between snake_case (zarr) and camelCase
  (`RenderingSettings`).
- `src/types/zarr.ts` — `ZarrViewerConfig` type definition.
