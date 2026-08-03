> **⚠️ Archived — historical document, not maintained.** Kept for design history; it reflects the project state as of its original date and may not match current code. Do not treat it as current guidance. See [the archive README](README.md) for status labels and retention policy.

# E2E Test Coverage Gaps & Proposals

**Date:** 2026-04-02

## Coverage Status

The viewer has **29 E2E test suites** covering ~278 test cases. After the recent audit, **273 pass**, but many critical viewer features have zero or shallow E2E coverage.

### Coverage Heatmap

```
WELL COVERED (behavioral tests exist)
  [####] Viewer initialization & lifecycle
  [####] nD navigation (keyboard, sliders, dimension selection)
  [####] Dimension animation (play/pause, FPS, loop modes)
  [####] Transform hierarchy (parent-child composition, matrices)
  [####] Cache system (L0/L1/L2, OPFS, clearing)
  [####] Spatial index queries (point counts, range merging)
  [####] Smoke tests for all 28 example datasets

PARTIALLY COVERED (existence checks, no behavioral verification)
  [##--] Geometry types (Lines, GSplats load — but no attribute/rendering verification)
  [##--] Recording panel (toggle & screenshot — but no video capture or HDR export)
  [##--] Theme system (screenshots — but no live switching or CSS variable verification)
  [##--] Performance tracking (timing baselines — but no jank/memory leak detection)
  [##--] Error recovery (some paths — but no partial load recovery)

ZERO E2E COVERAGE
  [----] Layers panel (visibility, blending, opacity, colormap, gamma)
  [----] Colormap system (apply, switch, legend UI, scalar range)
  [----] Post-processing pipeline (bloom, AO, tone mapping, cinematic mode)
  [----] HDR rendering (exposure control, EXR export, tone mapping modes)
  [----] Blending modes (additive, normal, max, opaque, luminous)
  [----] Scale bar (auto-calculation, unit display, camera-responsive)
  [----] Ortho projection mode (camera behavior, pan, zoom)
  [----] Mouse interaction (Ctrl+wheel FOV, Shift+wheel rotation)
  [----] Dataset browser (path entry, directory navigation, dataset selection)
  [----] URL parameter handling (?theme, ?no-cache, ?clear-cache)
  [----] GPU buffer pool (reuse, memory pressure)
  [----] Adaptive DPR (resolution scaling under load)
  [----] Debug console (Ctrl+L toggle, log filtering)
  [----] GSplat-specific rendering (Cholesky projection, attenuation)
  [----] Lines rendering (width, tapering, segments)
  [----] Data integrity (attribute alignment, drawRange consistency)
  [----] Scene switching (load new dataset, cleanup old scene)
  [----] Viewport resize (responsive layout, WebGL resize)
```

---

## Proposed New E2E Tests (Priority-Ordered)

### TIER 1 — Critical (silent data/rendering corruption if broken)

#### 1. `layers-panel.spec.ts` — Layers Panel Operations
**Why critical:** The layers panel controls visibility, blending, opacity, gamma, and colormaps for each node. A bug here means users see wrong data or missing layers with no error message.

```
Tests:
- L key toggles layers panel visibility
- Panel lists all scene nodes as layers
- Eye icon toggles layer visibility (points disappear/appear in scene)
- Blending mode dropdown changes actual WebGL blend function
  - Additive: bright overlaps get brighter
  - Normal: standard alpha compositing
  - Max: brightest value wins
  - Opaque: no transparency
- Opacity slider changes material transparency (verify via material.opacity)
- Gamma slider changes brightness curve (verify pixel brightness changes)
- Display range slider maps [min, max] to actual rendered colors
- Multi-select: Ctrl+click adds to selection, Shift+click selects range
- Changes to selected layers apply to all simultaneously
- Colormap dropdown switches colormap (verify LUT texture changes)
- Layer changes are reflected in real-time rendering (screenshot comparison)

Dataset: sharpness_showcase_example.zarr (multiple nodes)
```

#### 2. `colormap-rendering.spec.ts` — Colormap Application & Legend
**Why critical:** Colormaps map scalar data to colors. If broken, users see raw scalar values or wrong colors — a scientific data integrity issue.

```
Tests:
- Dataset with scalar data renders with default colormap (not raw RGB)
- Switching colormap via layers panel changes rendered colors (screenshot diff)
- Colormap legend appears (J key toggle) showing active colormaps
- Legend shows correct gradient preview and min/max range labels
- Legend updates reactively when colormap or range changes
- Scalar range [min, max] correctly normalizes data (0=min color, 1=max color)
- "(direct colors)" option disables colormap and uses RGB attribute
- Colormap applies per-layer (different nodes can have different colormaps)

Dataset: needs scalar data — use rendering_modes_example.zarr or create fixture
```

#### 3. `post-processing-pipeline.spec.ts` — Post-Processing Effects
**Why critical:** Post-processing controls bloom, tone mapping, and anti-aliasing. If broken, the scene renders dark/washed out or with artifacts.

```
Tests:
- Bloom effect: enable/disable via rendering controls, verify pixel brightness changes
- Bloom strength slider: higher values = brighter highlights (compare screenshots)
- Tone mapping: exposure slider changes scene brightness
- SMAA anti-aliasing: enable/disable, verify edge smoothing (pixel sampling)
- Cinematic mode (C key): enables vignette + chromatic aberration
  - Toggle on: edges darken, slight color fringing
  - Toggle off: effects removed
- Detector noise: toggle on/off (visible grain pattern)
- Post-processing doesn't crash with empty scene
- Post-processing works after dataset switch (no stale state)

Dataset: build_example_structured.zarr (bright points for bloom testing)
```

#### 4. `hdr-exposure.spec.ts` — HDR & Exposure Control
**Why critical:** HDR exposure controls brightness interpretation. Wrong exposure = data appears too dim or clipped.

```
Tests:
- Default exposure renders scene at reasonable brightness
- Increasing exposure (via rendering controls) brightens scene (screenshot comparison)
- Decreasing exposure darkens scene
- Exposure change is reversible (same screenshot at same exposure)
- HDR colors above 1.0 are not clamped (verify via extracting rendered pixel values)
- Intensity/offset uniforms apply per-node (verify material.uniforms)
- Reset button restores default exposure

Dataset: sharpness_showcase_example.zarr (has HDR color values)
```

#### 5. `blending-modes.spec.ts` — Material Blending Modes
**Why critical:** Blending modes determine how overlapping geometry composites. Wrong blending = wrong visual output for scientific data.

```
Tests:
- Additive blending: overlapping points are brighter than individual points
- Normal blending: front points occlude back points (depth-test dependent)
- Max blending: brightest value wins (verify with known overlapping data)
- Opaque blending: no transparency (solid rendering)
- Blending mode switch mid-session doesn't crash
- Blending mode applies per-layer (different nodes, different modes)
- Screenshot comparison between modes shows visible differences

Dataset: multiple_objects_example.zarr (overlapping point clouds)
```

#### 6. `data-integrity.spec.ts` — Attribute Alignment & DrawRange
**Why critical:** If position/color/radius arrays get misaligned, users see garbage data. This is the most dangerous class of silent bug.

```
Tests:
- Position array count matches color array count for each point cloud
- Position array count matches radius array count
- drawRange.count <= position.count (never exceeds buffer)
- All position values are finite (no NaN/Infinity)
- All color values are in valid range (Uint8: 0-255, Float32: finite)
- All radius values are non-negative and finite
- After nD navigation: attribute counts still aligned
- After dataset switch: no stale geometry remains

Dataset: build_example_structured.zarr + dimension_navigation_example.zarr
```

### TIER 2 — Important (user-visible bugs if broken)

#### 7. `ortho-mode.spec.ts` — Orthographic Projection
**Why critical:** Ortho mode is a different camera projection. If broken, 2D viewing workflows fail entirely.

```
Tests:
- V key cycles to ortho mode (orbit → fly → ortho)
- Ortho mode shows parallel projection (no perspective foreshortening)
- Left-drag pans in ortho mode
- Scroll zooms (changes ortho frustum, not camera Z position)
- Shift+scroll rotates around view axis
- Scale bar updates correctly in ortho mode
- Screenshot in ortho differs from perspective (flat vs depth)
- FOV slider is hidden/disabled in ortho mode
```

#### 8. `scale-bar.spec.ts` — Physical Scale Bar
**Why critical:** Scale bar communicates physical measurement. Wrong scale = wrong scientific interpretation.

```
Tests:
- Scale bar visible on scene load (if dimensions have units)
- Scale bar shows correct unit from scene metadata (e.g., "um", "mm")
- Scale bar length changes when zooming (closer = smaller scale bar number)
- Scale bar uses nice round numbers (1, 2, 5, 10, 20, 50...)
- Scale bar updates in real-time during camera movement
- Scale bar works in both perspective and ortho modes
- Scale bar hidden if no spatial units defined

Dataset: scene_dimensions_example.zarr (has physical units)
```

#### 9. `mouse-interaction.spec.ts` — Advanced Mouse Controls
**Why critical:** Mouse interactions beyond basic drag are untested. Power users rely on modifier+scroll.

```
Tests:
- Ctrl+scroll changes camera FOV (verify FOV value changes)
- Shift+scroll rotates view axis in ortho mode
- Mouse wheel zooms (orbit controls: distance changes)
- Right-drag rotates in orbit mode
- Shift+left-drag rotates in orbit mode (alternative)
- Double-click recenters camera on click position
- Mouse interactions don't fire when typing in UI text fields

Dataset: build_example_structured.zarr
```

#### 10. `dataset-switching.spec.ts` — Scene Switching & Cleanup
**Why critical:** Loading a new dataset must fully clean up the old one. Stale geometry = data corruption.

```
Tests:
- Load dataset A, then load dataset B via URL change
- Old scene objects are removed (Three.js scene.children count)
- GPU memory is released (WebGL texture/buffer counts)
- No WebGL errors after switch
- New dataset renders correctly after switch
- No console errors during switch
- Cache handles dataset switch correctly (old entries don't interfere)
- Point count reflects new dataset, not old + new

Datasets: build_example_structured.zarr → dense_grid_5d_example.zarr
```

#### 11. `viewport-resize.spec.ts` — Responsive Layout
**Why critical:** Users resize browser windows constantly. If the viewer doesn't resize, it looks broken.

```
Tests:
- Resize viewport: canvas dimensions update
- Resize viewport: renderer resolution updates
- Resize viewport: camera aspect ratio updates (no stretching)
- Resize viewport: post-processing composer resizes
- Resize to very small (300x200): no crash, still renders
- Resize to very large (3840x2160): no crash, still renders
- UI elements (panels, overlays) reposition on resize
```

#### 12. `url-parameters.spec.ts` — URL Parameter Handling
**Why critical:** URL parameters control initial state. If broken, shared links show wrong view.

```
Tests:
- ?theme=light sets light theme on load
- ?theme=dark sets dark theme on load  
- ?theme=frosted-glass sets frosted-glass theme
- ?no-cache disables persistent caching
- ?clear-cache clears cache before loading
- ?debug enables window.__luxarDebug interface
- Invalid ?src= shows error dialog, not crash
- Missing ?src= shows dataset browser
- URL updated when user selects dataset from browser
```

### TIER 3 — Good-to-have (edge cases, polish)

#### 13. `gsplat-rendering.spec.ts` — Gaussian Splat Rendering
```
Tests:
- GSplat dataset loads and renders oriented quads
- GSplat attenuation works (splats fade with distance)
- GSplat center positions match data
- GSplat amplitude affects brightness
- No WebGL errors with GSplat rendering
```

#### 14. `lines-rendering.spec.ts` — Lines Geometry Rendering
```
Tests:
- Lines dataset loads and renders line segments
- Line widths vary per data (not uniform)
- Line colors match data
- Instanced rendering works (correct segment count)
- Lines visible from different camera angles
```

#### 15. `debug-console.spec.ts` — Debug Console
```
Tests:
- Ctrl+L toggles debug console
- Console captures log/warn/error messages
- Filter input filters messages
- Console doesn't drop messages under load
```

#### 16. `keyboard-shortcuts-comprehensive.spec.ts` — Full Shortcut Coverage
```
Tests:
- L key: layers panel toggle
- J key: colormap legend toggle
- O key: toggle ortho snapping
- C key: cinematic mode toggle
- P key: performance stats toggle
- M key: data monitor toggle
- N key: dimension sliders toggle
- R key: rendering controls toggle
- T key: recording panel toggle
- G key: quick screenshot
- Verify shortcuts don't fire when typing in text fields
- Verify shortcuts don't fire when UI modal is active
```

---

## Implementation Priority

| Priority | Test Suite | Estimated Tests | Effort |
|----------|-----------|----------------|--------|
| **P0** | layers-panel | 12 | Medium |
| **P0** | colormap-rendering | 8 | Medium |
| **P0** | data-integrity | 8 | Low |
| **P0** | post-processing-pipeline | 8 | Medium |
| **P1** | hdr-exposure | 7 | Low |
| **P1** | blending-modes | 7 | Medium |
| **P1** | dataset-switching | 8 | Low |
| **P1** | scale-bar | 7 | Low |
| **P1** | ortho-mode | 8 | Medium |
| **P2** | mouse-interaction | 7 | Medium |
| **P2** | viewport-resize | 7 | Low |
| **P2** | url-parameters | 9 | Low |
| **P3** | gsplat-rendering | 5 | Medium |
| **P3** | lines-rendering | 5 | Medium |
| **P3** | debug-console | 4 | Low |
| **P3** | keyboard-shortcuts | 12 | Low |

**Total:** ~122 new tests across 16 suites

### Quick Wins (can be done in <1 hour each)
1. `data-integrity.spec.ts` — Just traverse scene and validate attributes
2. `url-parameters.spec.ts` — Navigate with different params, check state
3. `dataset-switching.spec.ts` — Load two datasets, check cleanup
4. `scale-bar.spec.ts` — Zoom and check scale bar values
