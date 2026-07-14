# Intensity, Offset, and Gamma — Design Document

## Status: Implemented

## Problem

The current Luxar viewer has a global "Intensity" slider (HDR multiplier) that is
propagated into every material shader as a per-fragment uniform. This is architecturally
wrong:

1. **A global control is implemented per-material** — the MaterialManager loops over all
   registered materials to set the same value, when a single post-processing pass would
   suffice.
2. **It couples brightness with bloom** — since bloom uses a luminance threshold, changing
   the global multiplier changes _which_ pixels bloom, rather than just making the image
   brighter.
3. **No per-node intensity** — despite having a per-node `gamma`, there is no per-node
   linear multiplier for independent channel brightness control.
4. **No background subtraction** — microscopy volumes often have a fluorescence background
   floor that should be subtractable per-node.

---

## Proposed Architecture

### Global Controls (vendored into tone mapping, zero extra cost)

Move global brightness/gamma out of shaders into the tone mapping shader itself.
The vendored `LuxarMegaShaderMaterial` applies EOG **before** tone mapping in a single pass:

```
HDR buffer → [Bloom] → [EOG + Tone Mapping (single pass)] → [Vignette] → [AA] → display
```

Three global controls applied to the composited HDR buffer:

| Parameter      | Python field     | TS field       | Default | Range        | Formula                  | Purpose                                |
|----------------|------------------|----------------|---------|--------------|--------------------------|----------------------------------------|
| **Exposure**   | `exposure`       | `exposure`     | 0.0     | -10.0 – +10.0 | `color * 2^exposure`     | Log2 stops (photography standard)      |
| **Offset**     | `global_offset`  | `globalOffset` | 0.0     | -1.0 – +1.0 | `color + offset`         | Lift/lower the entire composited image |
| **Gamma**      | `global_gamma`   | `globalGamma`  | 1.0     | 0.1 – 10.0  | `pow(color, 1/gamma)`    | Reshape midtones globally              |

**Naming convention:** Global offset and gamma use `global_` prefix in Python/zarr to
distinguish from per-node `offset` and `gamma`. In the UI, they are labeled simply
"Offset" and "Gamma" since section headers ("Global" vs per-node) disambiguate.

**Exposure** uses log2 (photography stops): 0 = neutral, +1 = 2x brighter, -1 = half.
This is the standard in photography and VFX tools (Nuke, DaVinci, Lightroom).
Replaces the current "Intensity" slider (which uses log10 scale with `hdrMultiplier`).

### Per-Node Controls (in shaders)

Each data node (Points, Lines, GSplats) gets three rendering parameters:

| Parameter     | Python field | TS metadata field | Default | Range         | Purpose                                 |
|---------------|--------------|-------------------|---------|---------------|-----------------------------------------|
| **Intensity** | `intensity`  | `intensity`       | 1.0     | 0.0 – 100.0  | Linear color multiplier (gain)          |
| **Offset**    | `offset`     | `offset`          | 0.0     | -10.0 – 10.0 | Additive brightness shift (black level) |
| **Gamma**     | `gamma`      | `gamma`           | 1.0     | 0.1 – 10.0   | Nonlinear tonal curve                   |

**Note:** Per-node `intensity` is NOT the same as `opacity`. Opacity controls alpha/blending
contribution (how transparent the node is). Intensity controls color brightness (gain).
With additive blending, opacity scales the alpha channel while intensity scales the color.

The shader applies the **GOG (Gain-Offset-Gamma)** model in all three material types
(Points, Lines, GSplats):

```glsl
// Per-node color adjustment (applied before blending, all node types)
vec3 adjusted = color * intensity + offset;
adjusted = max(adjusted, vec3(0.0));            // Clip negatives (no negative light)
adjusted = pow(adjusted, vec3(1.0 / gamma));    // Nonlinear curve
```

Early discard after offset saves GPU work across all node types:

```glsl
// All node types: discard zero-contribution fragments after offset
if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;
```

### Use Case: Background Subtraction

Microscopy volumes often have a fluorescence background floor. A negative `offset`
subtracts this floor per-node, so only signal above background contributes to rendering.
This is standard microscopy practice (black level / pedestal subtraction).

Example: Two channels in a multichannel microscopy dataset (works for any node type):
```python
# GSplats — subtract autofluorescence floor per channel
scene.add_gsplats("DAPI", ..., intensity=1.0, offset=-0.05, gamma=1.0)
scene.add_gsplats("GFP",  ..., intensity=2.0, offset=-0.02, gamma=0.8)

# Points — same model for localization microscopy
scene.add_points("STORM_ch1", ..., intensity=1.5, offset=-0.01, gamma=1.0)
scene.add_points("STORM_ch2", ..., intensity=1.0, offset=-0.03, gamma=0.9)

# Lines — e.g. neuron tracings with background subtraction
scene.add_lines("axons", ..., intensity=1.2, offset=-0.02, gamma=1.0)
```

### Per-Node Interactive UI (Future)

The viewer will eventually support interactive per-node adjustment of intensity, offset,
and gamma through the rendering controls panel. This is **not in scope** for the initial
implementation — it requires a node selection mechanism and per-node UI controls.

---

## Configuration Propagation

### Global controls: Python → Zarr → Viewer → Post-Processing

```
Python: ViewerConfig(exposure=1.0, global_offset=0.0, global_gamma=1.0)
  ↓
Zarr: root/.zattrs → {"viewer_config": {"exposure": 1.0, "global_offset": 0.0, "global_gamma": 1.0}}
  ↓
TS scene-loader: root.userData.viewerConfig
  ↓
TS viewer-config-utils: RENDERING_SETTINGS_MAP maps snake_case → camelCase
  - exposure → exposure
  - global_offset → globalOffset
  - global_gamma → globalGamma
  ↓
TS RenderingSettings: {exposure: 1.0, globalOffset: 0.0, globalGamma: 1.0}
  ↓
Priority chain: localStorage (user) > zarr viewer_config (author) > app defaults
  ↓
UI sliders (in "☀️ HDR" folder):
  - Exposure: linear slider -10 to +10 stops
  - Offset: linear slider -1.0 to +1.0
  - Gamma: linear slider 0.1 to 10.0
  ↓
postProcessingManager.megaShader.uniforms
  - uExposure, uOffset, uGamma
  ↓
Single vendored shader pass (EOG + tone mapping)
```

### Per-node controls: Python → Zarr → Viewer → Shaders

```
Python: scene.add_gsplats("GFP", ..., intensity=2.0, offset=-0.02, gamma=0.8)
  ↓
Zarr: GFP/.zattrs → {"intensity": 2.0, "offset": -0.02, "gamma": 0.8, ...}
  ↓
TS scene-loader: reads attrs.intensity, attrs.offset, attrs.gamma
  ↓
materialManager.get*Material({intensity: 2.0, offset: -0.02, gamma: 0.8, ...})
  ↓
Material constructor: creates uniforms uIntensity, uOffset, uInvGamma
  ↓
Fragment shader: GOG model applied per-fragment before blending
```

### Python ViewerConfig example

```python
from luxar import LuxarZarrCompiler, Dimensions, ViewerConfig

vc = ViewerConfig(
    # Global display controls (applied in post-processing)
    exposure=0.5,          # Half a stop brighter
    global_offset=0.0,     # No global offset
    global_gamma=1.0,      # Linear midtones
    tone_mapping="ACES",   # Filmic tone mapping

    # Other existing settings
    bloom_enabled=True,
    bloom_strength=0.3,
)

with LuxarZarrCompiler("output.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims, viewer_config=vc)

    # Per-node controls (applied in shaders)
    scene.add_gsplats("DAPI", centers, cholesky, amplitudes,
                      colors=colors_dapi,
                      intensity=1.0, offset=-0.05, gamma=1.0)
    scene.add_gsplats("GFP", centers, cholesky, amplitudes,
                      colors=colors_gfp,
                      intensity=2.0, offset=-0.02, gamma=0.8)
```

---

## UI Layout

The rendering controls panel would be restructured:

```
☀️ HDR
├── Exposure      [-10 ────●──── +10]  stops
├── Offset        [-1.0 ────●──── +1.0]
├── Gamma         [0.1 ─────●──── 10.0]
└── Tone Mapping  [ACES ▾]
```

Future per-node controls (when node selection UI exists):
```
🎯 Node: GFP
├── Intensity     [0.0 ─────●──── 100.0]
├── Offset        [-10 ─────●──── +10]
├── Gamma         [0.1 ─────●──── 10.0]
└── Opacity       [0.0 ─────●──── 1.0]
```

---

## Backward Compatibility

**None required.** All zarr data is regenerated from source. Old `hdr_multiplier` fields
in `ViewerConfig`, zarr attrs, `RenderingSettings`, and localStorage are simply removed
and replaced with the new fields. No migration code, no deprecated properties.

---

## Implementation: Mega-shader Tone Mapping Stage

### Why fuse EOG into the mega-shader

The global EOG controls are **merged into the post-processing mega-shader**, not
implemented as a separate post-processing pass.

**Performance rationale:**
- A separate full-screen pass costs one extra framebuffer read + write of every pixel
- Memory bandwidth is the primary GPU bottleneck for post-processing
- The mega-shader already reads every pixel — adding `exp2`, `add`, `pow` (3 ALU
  ops) before tone mapping is effectively free
- This avoids an extra render target allocation and preserves the fused pipeline

**Approach:** `MegaShaderMaterial` exposes `uExposure`, `uGlobalOffset`, and
`uGlobalGamma` uniforms. `mega/shader.glsl.ts` applies EOG immediately before
calling Three's tone-mapping shader chunks:

```glsl
// Single fused pass: scene/bloom/noise → EOG → tone mapping → vignette
vec3 color = sampleHdrPlusBloom(vUv);

color *= exp2(uExposure);
color = max(color + vec3(uGlobalOffset), vec3(0.0));
color = pow(color, vec3(1.0 / uGlobalGamma));

color = ACESFilmicToneMapping(color);  // or Reinhard, AgX, Neutral, etc.
```

**Implementation strategy:**
- Keep EOG uniforms on `MegaShaderMaterial`
- Include Three's `<tonemapping_pars_fragment>` once in the custom shader
- Set `toneMapped: false` on the material so Three does not inject a duplicate chunk
- Use internal tone-mapping define IDs for Linear/Reinhard/Cineon/ACES/AgX/Neutral
- Preserve old `NoToneMapping` behavior by routing it to Linear/clamped output

---

## Current State

All features described in this document are **fully implemented**:

- Per-node GOG model (`intensity`, `offset`, `gamma`) in all three material shaders
- Global EOG (`exposure`, `global_offset`, `global_gamma`) in the custom mega-shader
- `hdrMultiplier` removed from all shaders, MaterialManager, and config
- Python `ViewerConfig` updated with `exposure`, `global_offset`, `global_gamma`
- Full config propagation: Python → zarr → TypeScript → UI → post-processing

---

## Implementation Locations

### Python
- `packages/luxar/src/luxar/core/node/node.py` — `intensity` and `offset` properties
- `packages/luxar/src/luxar/core/viewer_config.py` — `exposure`, `global_offset`, `global_gamma` fields (replaced `hdr_multiplier`)
- `packages/luxar/src/luxar/io/compiler.py` — writes defaults for all three geometry write methods
- `packages/luxar/src/luxar/validation/types.py` — validators for intensity, offset
- `packages/luxar/src/luxar/typing_utils/constants.py` — range constants

### TypeScript (shaders)

Dual-stack: each geometry has a parallel GLSL (WebGL2 path) and TSL (WebGPU path) implementation; GOG uniforms live in both.

- `packages/luxar-viewer/src/rendering/materials/point/material-glsl.ts` — GOG uniforms (WebGL2)
- `packages/luxar-viewer/src/rendering/materials/point/material-tsl.ts` — GOG uniforms (WebGPU)
- `packages/luxar-viewer/src/rendering/materials/line/material-glsl.ts` — GOG uniforms (WebGL2)
- `packages/luxar-viewer/src/rendering/materials/line/material-tsl.ts` — GOG uniforms (WebGPU)
- `packages/luxar-viewer/src/rendering/materials/gsplat/material-glsl.ts` — GOG uniforms (WebGL2)
- `packages/luxar-viewer/src/rendering/materials/gsplat/material-tsl.ts` — GOG uniforms (WebGPU)

### TypeScript (post-processing + scene management)
- `packages/luxar-viewer/src/rendering/post-processing/mega/material.ts` — EOG uniforms and tone-mapping mode `#define` (WebGL2 path)
- `packages/luxar-viewer/src/rendering/post-processing/mega/material-tsl.ts` — EOG uniforms and tone-mapping mode uniform (WebGPU path)
- `packages/luxar-viewer/src/rendering/post-processing/mega/shader.glsl.ts` — fused EOG + tone-mapping shader stage (WebGL2)
- `packages/luxar-viewer/src/rendering/post-processing/mega/shader.tsl.ts` — fused EOG + tone-mapping shader stage (WebGPU)
- `packages/luxar-viewer/src/rendering/post-processing/post-processing-manager.ts` — exposure/offset/gamma update methods
- `packages/luxar-viewer/src/rendering/material-manager.ts` — per-node GOG uniforms in material creation
- `packages/luxar-viewer/src/scene/scene-manager.ts` — `updateExposure()`/`updateGlobalOffset()`/`updateGlobalGamma()` routing
- `packages/luxar-viewer/src/data/scene-loader.ts` — per-node `intensity`/`offset` from zarr attrs

### TypeScript (config + propagation)
- `packages/luxar-viewer/src/config/sections/rendering-controls/types.ts` — `exposure`/`globalOffset`/`globalGamma` in RenderingSettings
- `packages/luxar-viewer/src/config/sections/rendering-controls/data.ts` — defaults
- `packages/luxar-viewer/src/config/zarr-bridge/viewer-config-utils.ts` — `RENDERING_SETTINGS_MAP` snake_case → camelCase
- `packages/luxar-viewer/src/types/zarr.ts` — `ZarrViewerConfig` fields

### TypeScript (UI)
- `packages/luxar-viewer/src/ui/rendering-controls/setup/hdr-setup.ts` — Exposure/Offset/Gamma sliders
- `packages/luxar-viewer/src/ui/rendering-controls/controls-utils.ts` — `validateRenderingSettings()` ranges

### TypeScript (node types)
- `packages/luxar-viewer/src/types/points.ts` — `intensity?` and `offset?` in `PointsMetadata`
- `packages/luxar-viewer/src/types/lines.ts` — `intensity?` and `offset?` in `LinesMetadata`
- `packages/luxar-viewer/src/types/gsplats.ts` — `intensity?` and `offset?` in `GSplatsMetadata`
