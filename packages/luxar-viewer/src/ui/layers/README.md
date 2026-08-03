# Layers Panel

Napari-inspired per-layer control panel for Luxar scenes.

## Overview

The Layers panel exposes scene graph nodes marked with `layer=True` (set in the Python API) as controllable layers in the viewer. Data nodes (`points`, `lines`, `gsplats`) and container `group` nodes may both be exposed as layers; for groups, controls apply to every data descendant. Specialized groups (`kind: 'lod'`, `kind: 'partition'`) appear under their resolved `display_type` rather than as `group`, and carry an extra badge (and, for LOD groups, an inline level selector). Each layer provides:

- **Visibility toggle** (eye icon) — initial state taken from the node's `visible` attr (default `true`)
- **Display range** [min, max] — maps to shader intensity/offset uniforms
- **Gamma** correction
- **Opacity**
- **Blending mode** (additive, volumetric, normal, max, opaque, luminous)
- **Absorption** (κ) — only shown when the layer's effective blending mode is `volumetric`; all three geometry types implement the emission–absorption math, and κ = 0 is exactly the additive limit. Since the 2026-08-02 ray-mass unification τ = κ · rayMass with rayMass the same peak-alpha-normalised quantity the additive branch emits, so κ is dimensionless and comparable across points, lines, gsplats and scene scales — one fixed **logarithmic** track (nominally 0.001–10) serves every layer; the former per-layer geometry-derived bounds are gone. Both ends still move (within hard clamps) to keep an AUTHORED κ outside the nominal span on the track, so the readout always shows a value the thumb can express. Position 0 is a dedicated stop for exactly κ = 0 (the geometric span starts one step in, so the floor round-trips). On a log track the component mirrors the readout into `aria-valuetext`, since the input's native value is a position. See `absorption-range.ts`.
- **Colormap** (for gsplats with scalars/amplitudes, scalar-backed points/lines, and groups that fan out to such descendants)
- **Active level** (LOD groups, and partitions wrapping LOD groups) — `auto` or lock to a specific level

Rendering attributes compose along the scene graph per the Luxar composition spec: `opacity`, `absorption`, `gamma`, and `intensity` multiply through ancestors; `offset` adds; `blending_mode` takes the nearest ancestor's choice — except inside the edited layer's own subtree, where the layer's single Blend control wins (see [Blending mode inside a layer's subtree](#blending-mode-inside-a-layers-subtree)). Every panel mutation recomposes the effective attributes for each affected data-leaf (the layer itself, or every data descendant of a group layer) using live panel state for `layer=true` nodes and authoring-time zarr attrs for the rest. Colormap is the one exception — it applies per-leaf rather than composing.

Edits made in the panel are viewer-only and not persisted back to the zarr store; reload the page to return to the authored state.

### Specialized groups (LOD / partition)

`group` nodes carrying a `kind` attr of `'lod'` or `'partition'` are surfaced
specially (see `LayerKind` in `layer-state.ts`):

- They render under their resolved `display_type` attr (one of `points` /
  `lines` / `gsplats`), never as `group`.
- A `kind=lod` layer gets a **`N LODs`** badge and an inline **Active level**
  dropdown. Selecting a level calls `LODGroupRegistry.setSelectorMode(path, …)`
  with either `'auto'` or `{ lockLevel: i }`; the status span shows the
  currently-rendering level 1-based (`L{i}/{n}`), matching the data-monitor
  chip and the dropdown labels, with an `(off-screen)` suffix when the
  frustum gate is holding the group at its coarsest level. A per-frame
  callback (`layers-lod-status`) keeps the readout live, so it tracks
  `auto`-mode swaps driven by camera motion — not only state changes.
- A `kind=partition` layer gets a **`N parts`** badge. When it wraps nested
  `lod_group` descendants, the badge combines counts as **`N parts × M LODs`**
  (M = max child count across the nested ladders) and the Active-level dropdown
  broadcasts the chosen mode to every nested `lod_group` path (clamped per-group
  by `setSelectorMode` on ragged ladders). The readout aggregates the live
  level across all nested groups as `L{i}/{n} · {N} groups`, widening to a
  range `L{min}–{max}/{n}` when parts diverge under `auto` (each part picks
  its own level by its own on-screen size); `{n}` is the max ladder depth, so
  it agrees with the dropdown's option count.

The LOD registry is looked up lazily via
`SceneLoaderManager.getInstance().getDefaultLoader()?.lodGroupRegistry` so the
panel doesn't import `scene/` directly (respecting the data → ui layer
direction). Locking a level wakes the animation loop (`requestRender`) so the
new active level paints even when the camera and slice are idle.

### Load-failure badge

When a node's loader throws (corrupt data, network failure, the "Vertex index N
not found" path), that failure used to surface only in the console and the
collapsed data monitor, reading as a blank-canvas camera/shader problem. The
panel now shows a per-row **error badge** (`.luxar-layer-row__error`, a
warning-triangle SVG with `role="img"` and an `aria-label` naming the reason)
plus an `--error` row tint (`luxar-layer-row--error`).

- The app injects an equivalent `FailedLoadsProviderPort` over the same live
  failure set the data monitor reads (each `SceneLoader.getFailedLoadsProvider()`
  call returns a new object, but all close over the loader's one `failedLoaders`
  map) via `LayersPanel.setFailedLoadsProvider(provider)`, wired in
  `core/app/dataset/load-dataset.ts` right AFTER `initFromScene` (whose `clear()`
  resets any prior provider first). Cleared on dispose.
- A row is in error if its own path failed OR any descendant leaf failed
  (`failedPath === layer.path || failedPath.startsWith(layer.path + '/')`), so a
  failure inside a `kind=lod` / `kind=partition` group lights up the group's row.
- The tooltip prefers the provider's per-path reason (`getFailedReason`, from the
  loader's `error.message` / classified kind), falls back to a generic message,
  and appends `(N parts failed)` when more than one descendant failed.
- Refresh is signature-gated (mirroring the data monitor's
  `lastFailedLoadsSignature`): the per-frame `layers-lod-status` callback (gated
  on panel visibility) only touches the DOM when the failed set — folded with
  each path's reason — changes. The signature reset sentinel is `null`, so an
  empty set / `setFailedLoadsProvider(null)` still clears badges. `renderList()`
  invalidates the signature so a row rebuild (e.g. `resetAllLayers`) re-applies.

This covers per-node LOADER failures only; render-thread texture-capacity
truncation is a separate follow-up.

## Usage

### Python (scene authoring)

```python
from luxar import LuxarZarrCompiler, Dimensions

with LuxarZarrCompiler("scene.luxar.zarr") as c:
    scene = c.create_scene(dimensions=dims)
    scene.add_points("GFP", positions=..., colors=..., layer=True)
    scene.add_points("mCherry", positions=..., colors=..., layer=True)
    scene.add_group("_internal")  # Not a layer — no panel entry
```

### Viewer

Press **L** to toggle the Layers panel (Escape closes when focus is inside the panel).

- **Click** a layer to select it
- **Ctrl/Cmd+Click** to toggle additional layers
- **Shift+Click** for range selection
- **Arrow Up / Arrow Down** move the keyboard focus through rows (and select on simple navigation)
- **Enter / Space** select the focused row (honouring Ctrl/Cmd/Shift modifiers)
- The bound labels on either side of the display-range slider are click-to-edit and scroll-to-adjust (hold **Shift** for finer increments)
- Controls below the list (display range, gamma, opacity, absorption, blend, colormap) apply to all selected layers; the absorption, colormap, and **Active level** controls auto-hide when the primary selected layer doesn't support them

## Architecture

```
layer-state.ts     Pure data model, selection logic (re-exports the min/max ↔ intensity/offset math from rendering/display-range.ts)
layers-panel.ts    DOM panel (list + lifecycle), event handling; facade over the two below
layer-controls.ts  LayerControls — the controls section (sliders, blend/colormap/LOD selects, LOD readout)
layer-apply.ts     LayerApplyEngine — attr composition + scene/material application
luxar-material.ts  LuxarMaterial contract + colormap-vs-direct routing helpers
range-slider.ts    Dual-thumb [min, max] slider (click-to-edit + scroll-adjust bounds)
labeled-slider.ts  Single-thumb labeled slider (gamma, opacity, absorption); linear or log track
absorption-range.ts κ slider log-track bounds (fixed nominal span, widened onto authored κ) + κ readout format
attrs-utils.ts     Pure helpers: clampGamma, blending-state mapping, liveLayerAttrs
```

The public entrypoint is `../layers.ts` (parent file); it re-exports
`LayersPanel`, `LayerStateManager`, `RangeSlider`, the `computeUniforms` /
`computeDisplayRange` math, and the `LayerInfo` / `DisplayUniforms` /
`SelectionMode` types.

## Display Range Mapping

The UI shows [min, max] sliders. Internally these map to the existing shader uniforms:

```
intensity = 1 / (max - min)
offset    = -min / (max - min)
```

### Which window a layer STARTS at

The window maps the **rendered value** to `[0, 1]`, so the default depends on
what that value is (`layer-state.ts::initialDisplayRange`):

| Layer renders                                               | Starting window                                                                                                                         | Why                                                                                                                                                                                                                     |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| through a colormap (`colormap` on the node or a descendant) | `scalar_data_range`, else `amplitude_data_range`, else the finest descendant leaf's (`deriveScalarRangeFromDescendants`), else `[0, 1]` | the value is a scalar; gsplat amplitudes are heavily right-skewed, so a linear `[0, 1]` window renders near-black (#522)                                                                                                |
| direct RGB colours                                          | `[0, 1]` — the identity                                                                                                                 | the value IS authored colour. Windowing it on `color_data_range` is an unrequested contrast stretch: a uniform grey `(0.72, 0.74, 0.78)` has range `[0.72, 0.78]` → gain 16.7 / offset −12 → renders **saturated blue** |

`color_data_range` therefore never sets the starting window — it only widens the
**slider bounds** for direct-colour layers, so stretching authored colours stays
a one-drag operation.

The "or a descendant" walk stops at a nested `layer=true` node: that node is its
own row with its own colormap control, so a palette derived from it would be a
snapshot that goes stale on the first inner edit. Writers route `layer` onto the
wrapper only, so a `kind=partition` / `kind=lod` layer never has layer
descendants and is unaffected.

Bounds are the union of the starting window, the recovered authored
`intensity`/`offset` window, and (direct colour only) `color_data_range` —
`[min(dataMin, displayMin), max(dataMax, displayMax)]` — so the `<input>` never
silently clamps the thumb on first render.

#### Toggling the colormap

On an off↔on MODE flip, `setColormapWindow` re-defaults the window AND the
bounds to the new mode — a window carried over from the other mode is
meaningless, and merely widening the bounds would leave the useful window as an
unusable sliver (an amplitude window of `[1e-4, 0.02]` inside `[0, 1]` bounds
is 2% of the track). Both land exactly where a natively authored layer of that
mode inits, which is why `LayerInfo` keeps `colorDataRange` alongside
`scalarDataRange`. Switching between two active palettes is NOT a mode flip —
the rendered value stays the same scalar, so a user-adjusted window survives.

Two things the select handler must do that are easy to miss:

- **Re-render.** It runs with `controlsInteracting = true`, which suppresses the
  state-change re-render, and `RangeSlider` emits values parsed from its own
  `<input>` elements. Without an explicit `render()` the thumbs keep the old
  window and the first drag writes it back, reverting the re-default.
- **Honour the fail-closed guard.** `applyColormap` returns whether any leaf
  actually took the LUT. The C1 guard suppresses it on leaves with no scalar
  data bound (a group layer over scalar-less points still offers the dropdown);
  such a layer keeps rendering direct colour, so the handler puts the identity
  window back rather than applying a scalar range as a colour gain.

For a MIXED group layer (some leaves accept the LUT, some are suppressed), the
layer keeps the scalar window for its colormapped leaves, and `applyComposed`
routes per leaf: a leaf whose material is not colormap-active while the layer's
window is a scalar one (`LayerInfo.scalarWindow`) gets the identity window
instead, so the scalar range is never applied to authored RGB as a colour gain.

### Blending mode inside a layer's subtree

`blending_mode` composes nearest-setter-wins, but a layer exposes exactly ONE
Blend control for its whole subtree. So within a layer, the **layer's** mode
wins: `LayerApplyEngine.composeEffective` ignores a `blending_mode` authored on
a descendant that is not itself a layer (a nested layer keeps its own live
value — it has its own control). Without this, a `kind=partition` /
`kind=lod` layer whose parts carry their own stamped mode had an inert Blend
control.

## Files

| File                                       | Purpose                                                                                                                          |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `layer-state.ts`                           | `LayerStateManager`, selection logic; re-exports `computeUniforms` / `computeDisplayRange` (now in `rendering/display-range.ts`) |
| `layers-panel.ts`                          | `LayersPanel` class — panel/list DOM + lifecycle; facade over controls + apply                                                   |
| `layer-controls.ts`                        | `LayerControls` — controls-section DOM (sliders, selects, live LOD readout)                                                      |
| `layer-apply.ts`                           | `LayerApplyEngine` — attr composition + material application per data-leaf                                                       |
| `luxar-material.ts`                        | `LuxarMaterial` interface, `isColormapActive` / `applyColorAdjustments` routing                                                  |
| `range-slider.ts`                          | `RangeSlider` — dual-thumb input component with editable / scrollable bound labels                                               |
| `labeled-slider.ts`                        | `LabeledSlider` — single-thumb labeled input component (gamma, opacity, absorption); `linear` or `log` track                     |
| `absorption-range.ts`                      | `absorptionSliderRange` / `formatAbsorption` — κ track bounds (fixed log span, widened onto authored κ) + readout format         |
| `attrs-utils.ts`                           | `clampGamma`, `getBlendingState`, `liveLayerAttrs` — pure helpers (no DOM)                                                       |
| `../layers.ts`                             | Public entrypoint — re-exports the layers surface                                                                                |
| `../../styles/components/layers-panel.css` | Themed CSS styles                                                                                                                |
