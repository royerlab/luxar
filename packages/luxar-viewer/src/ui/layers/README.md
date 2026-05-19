# Layers Panel

Napari-inspired per-layer control panel for Luxar scenes.

## Overview

The Layers panel exposes scene graph nodes marked with `layer=True` (set in the Python API) as controllable layers in the viewer. Data nodes (`points`, `lines`, `gsplats`) and container `group` nodes may both be exposed as layers; for groups, controls apply to every data descendant. Each layer provides:

- **Visibility toggle** (eye icon) — initial state taken from the node's `visible` attr (default `true`)
- **Display range** [min, max] — maps to shader intensity/offset uniforms
- **Gamma** correction
- **Opacity**
- **Blending mode** (additive, normal, max, opaque, luminous)
- **Colormap** (for gsplats and scalar-backed points/lines)

Edits made in the panel are viewer-only and not persisted back to the zarr store; reload the page to return to the authored state.

## Usage

### Python (scene authoring)

```python
from luxar import LuxarZarrCompiler, Dimensions

with LuxarZarrCompiler("scene.zarr") as c:
    scene = c.create_scene(dimensions=dims)
    scene.add_points("GFP", positions=..., colors=..., layer=True)
    scene.add_points("mCherry", positions=..., colors=..., layer=True)
    scene.add_group("_internal")  # Not a layer — no panel entry
```

### Viewer

Press **L** to toggle the Layers panel.

- **Click** a layer to select it
- **Ctrl+Click** to toggle additional layers
- **Shift+Click** for range selection
- Controls below the list apply to all selected layers

## Architecture

```
layer-state.ts     Pure data model, min/max ↔ intensity/offset math
layers-panel.ts    DOM panel, event handling, scene application
range-slider.ts    Dual-thumb [min, max] slider component
attrs-utils.ts     Pure helpers for layer attribute filtering / coercion
labeled-slider.ts  Single-thumb labeled slider component
```

The public entrypoint is `../layers.ts` (parent file); it re-exports
the surface external code should consume.

## Display Range Mapping

The UI shows [min, max] sliders. Internally these map to the existing shader uniforms:

```
intensity = 1 / (max - min)
offset    = -min / (max - min)
```

Slider bounds come from `color_data_range` stored in the zarr `.zattrs` during encoding.

## Files

| File                                       | Purpose                                                      |
| ------------------------------------------ | ------------------------------------------------------------ |
| `layer-state.ts`                           | `LayerStateManager`, `computeUniforms()`, selection logic    |
| `layers-panel.ts`                          | `LayersPanel` class — DOM, event handlers, scene application |
| `range-slider.ts`                          | `RangeSlider` — dual-thumb input component                   |
| `labeled-slider.ts`                        | `LabeledSlider` — single-thumb labeled input component       |
| `attrs-utils.ts`                           | Pure helpers for filtering / coercing layer attributes       |
| `../layers.ts`                             | Public entrypoint — re-exports the layers surface            |
| `../../styles/components/layers-panel.css` | Themed CSS styles                                            |
