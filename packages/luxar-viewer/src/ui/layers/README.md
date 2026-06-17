# Layers Panel

Napari-inspired per-layer control panel for Luxar scenes.

## Overview

The Layers panel exposes scene graph nodes marked with `layer=True` (set in the Python API) as controllable layers in the viewer. Data nodes (`points`, `lines`, `gsplats`) and container `group` nodes may both be exposed as layers; for groups, controls apply to every data descendant. Specialized groups (`kind: 'lod'`, `kind: 'partition'`) appear under their resolved `display_type` rather than as `group`, and carry an extra badge (and, for LOD groups, an inline level selector). Each layer provides:

- **Visibility toggle** (eye icon) — initial state taken from the node's `visible` attr (default `true`)
- **Display range** [min, max] — maps to shader intensity/offset uniforms
- **Gamma** correction
- **Opacity**
- **Blending mode** (additive, normal, max, opaque, luminous)
- **Colormap** (for gsplats with scalars/amplitudes, scalar-backed points/lines, and groups that fan out to such descendants)
- **Active level** (LOD groups, and partitions wrapping LOD groups) — `auto` or lock to a specific level

Rendering attributes compose along the scene graph per the Luxar composition spec: `opacity`, `gamma`, and `intensity` multiply through ancestors; `offset` adds; `blending_mode` takes the nearest ancestor's choice. Every panel mutation recomposes the effective attributes for each affected data-leaf (the layer itself, or every data descendant of a group layer) using live panel state for `layer=true` nodes and authoring-time zarr attrs for the rest. Colormap is the one exception — it applies per-leaf rather than composing.

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
- Controls below the list (display range, gamma, opacity, blend, colormap) apply to all selected layers; the colormap and **Active level** controls auto-hide when the primary selected layer doesn't support them

## Architecture

```
layer-state.ts     Pure data model, min/max ↔ intensity/offset math, selection logic
layers-panel.ts    DOM panel, event handling, attr composition, scene application
range-slider.ts    Dual-thumb [min, max] slider (click-to-edit + scroll-adjust bounds)
labeled-slider.ts  Single-thumb labeled slider (gamma, opacity)
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

Slider bounds come from the data-range zarr attr written during encoding:
`scalar_data_range` (preferred when present), else `color_data_range`,
else `amplitude_data_range` (gsplats), else `[0, 1]`. Group layers have no
range of their own and fall through to `[0, 1]`. If the layer was authored
with non-default `intensity` / `offset`, the recovered display range may
extend beyond the stored data range — the slider bounds are widened to
`[min(dataMin, displayMin), max(dataMax, displayMax)]` so the `<input>`
doesn't silently clamp the thumb on first render.

## Files

| File                                       | Purpose                                                                            |
| ------------------------------------------ | ---------------------------------------------------------------------------------- |
| `layer-state.ts`                           | `LayerStateManager`, `computeUniforms` / `computeDisplayRange`, selection logic    |
| `layers-panel.ts`                          | `LayersPanel` class — DOM, event handlers, attr composition, scene application     |
| `range-slider.ts`                          | `RangeSlider` — dual-thumb input component with editable / scrollable bound labels |
| `labeled-slider.ts`                        | `LabeledSlider` — single-thumb labeled input component (gamma, opacity)            |
| `attrs-utils.ts`                           | `clampGamma`, `getBlendingState`, `liveLayerAttrs` — pure helpers (no DOM)         |
| `../layers.ts`                             | Public entrypoint — re-exports the layers surface                                  |
| `../../styles/components/layers-panel.css` | Themed CSS styles                                                                  |
