# Luxar Viewer User Guide

The Luxar viewer is a browser-based application for exploring nD scientific scenes
containing points, lines, Gaussian splats, and triangle meshes. It renders with WebGL
by default and has an opt-in WebGPU path (see the `renderer` URL parameter below), and
it loads data from Zarr archives served over HTTP or from local files.

![Luxar viewer interface overview](../../images/docs/viewer-ui-overview.png)

## Launching the Viewer

There are three common ways to open the viewer:

```bash
# Serve a dataset and open the viewer in one step
luxar serve scene.luxar.zarr --viewer

# Browse the bundled demos, then run one
luxar demo
luxar demo run lorenz

# Export a self-contained offline viewer
luxar export scene.luxar.zarr -o my_export/ --open
```

You can also start the viewer development server directly:

```bash
cd packages/luxar-viewer
pnpm dev
```

Then open `http://localhost:5173/?src=http://127.0.0.1:8000` in a browser,
pointing `src` at a running Luxar data server. Port `8000` is the default for
`luxar serve`; use your configured `--port` value if you changed it.

Both trailing-slash forms are accepted. Prefer data source URLs without a
trailing slash as the canonical spelling used in examples and logs.

---

## URL Parameters

Append parameters to the viewer URL to control startup behavior.

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | string | Zarr dataset URL or local path. |
| `theme` | string | Initial theme. One of: `light`, `dark`, `liquid-glass`, `frosted-glass`. |
| `debug` | flag | Enable the debug interface (developer use). |
| `no-cache` | flag | Disable ALL caching tiers (S-cache + L0/L1/L2). |
| `no-slice-cache` | flag | Disable only the SliceCache (per-slice decoded-geometry reuse); L0/L1/L2 stay on. |
| `cache-debug` | flag | Enable cache debug logging to the browser console. |
| `clear-cache` | flag | Clear all caches on startup. |
| `no-prefetch` | flag | Disable adjacent-chunk prefetching. |
| `prefetch-debug` | flag | Enable prefetch logging to the browser console. |
| `cache-stats` | flag | Auto-open the data-loading monitor expanded on the Cache tab (L0/L1/L2 hit rates). |
| `renderer` | `webgl` \| `webgpu` | Select the default GLSL WebGLRenderer path or opt into the WebGPURenderer + TSL path. |
| `webgpu-force-webgl` | flag | Diagnostic flag for `renderer=webgpu`: keep WebGPURenderer + TSL materials but force Three.js's internal WebGL2 backend. |
| `perf-timestamp` | flag | Opt into GPU timestamp queries (WebGPU only, `timestamp-query` feature). Small runtime cost; intended for the perf bench. |
| `gpuBudgetMB` | number | Pin the GPU-geometry byte budget in MB, bypassing auto-sizing. `0` disables the budget (unbounded resident geometry). |
| `cacheBudgetMB` | number | Total in-memory cache pool (L0 + L1 + S-cache) in MB, for environments without `performance.memory` (Safari, WKWebView). |
| `dpr` | number | Pin a fixed device pixel ratio and disable adaptive DPR (clamped to [0.25, native DPR]). For deterministic E2E/visual runs. |

Flag parameters do not take a value; their presence activates the feature.

Example:

```
http://localhost:5173/?src=http://127.0.0.1:8000&theme=dark&no-cache
```

---

## Camera Controls

The viewer supports three camera control modes, cycled with the **V** key:

### Orbit Mode (default)

Standard trackball camera for inspecting a scene from the outside.

| Input | Action |
|-------|--------|
| Left-click + drag | Rotate around the target point (natural drag ON) / pan (OFF) |
| Right-click + drag | Pan the camera (natural drag ON) / rotate (OFF) |
| Middle-click + drag | Dolly (either setting) |
| Scroll wheel | Zoom in/out |
| Ctrl/Cmd + Scroll | Adjust field of view |

The left/right drag mapping depends on the **natural drag** setting, which
defaults to ON on macOS and OFF on other platforms: with natural drag,
left-drag rotates and right-drag pans; without it, the mapping is inverted
(left-drag pans, right-drag rotates).

Press **F** to recenter the camera so the entire scene fits in view.

### Fly Mode

First-person controls for moving through the interior of a dataset.

| Input | Action |
|-------|--------|
| W / A / S / D | Move forward / left / backward / right |
| Alt/Option + W / S | Move up / down |
| Q / E | Roll left / right |
| Arrow keys | Look around |
| Shift (held) | Speed boost |
| Right-click + drag | Look around |
| Left-click + drag | Strafe (screen-space translation) |

Enable **inertial mode** (press **I**) to add momentum to fly movement so the
camera coasts after releasing keys.

### Ortho Mode

Orthographic projection for 2D viewing. The camera looks straight down one axis.

| Input | Action |
|-------|--------|
| Left-click + drag | Pan |
| Scroll wheel | Zoom |

---

## Keyboard Shortcuts

### General Navigation and Panels

| Key | Action |
|-----|--------|
| H | Toggle help overlay |
| N | Toggle dimension sliders (for nD datasets) |
| O | Open dataset browser |
| P | Toggle performance stats (FPS, frame time) |
| R | Toggle rendering controls panel |
| B | Toggle scale bar |
| J | Toggle colormap legend |
| L | Toggle layers panel |
| U | Toggle overlays |
| T | Toggle recording panel |
| Escape | Close all open panels |

### Camera and View

| Key | Action |
|-----|--------|
| F | Recenter camera (frame entire scene) |
| V | Cycle control mode: Orbit, Fly, Ortho |
| Space | Toggle fullscreen |
| Ctrl/Cmd + Scroll | Adjust field of view |

### Visual Modes

| Key | Action |
|-----|--------|
| C | Toggle cinematic mode (bloom, detector noise, vignette, chromatic lens distortion) |
| I | Toggle inertial mode (fly controls momentum) |

### nD Dimension Navigation

| Key | Action |
|-----|--------|
| 1--9 | Select navigable dimension by index |
| \[ | Step selected dimension backward |
| \] | Step selected dimension forward |
| M | Cycle data loading monitor |
| G | Quick screenshot |

### Animation Playback

| Key | Action |
|-----|--------|
| K | Play / pause animation |
| Home | Jump to dimension start |
| End | Jump to dimension end |
| Shift + Up | Increase animation speed |
| Shift + Down | Decrease animation speed |

### Fly Mode Movement

| Key | Action |
|-----|--------|
| W / A / S / D | Move forward / left / backward / right |
| Alt/Option + W / S | Move up / down |
| Q / E | Roll left / right |
| Arrow keys | Look direction |
| Shift (held) | Speed boost |

### State Export

| Key | Action |
|-----|--------|
| Ctrl+Shift+S | Export viewer state to clipboard as JSON |

---

## UI Panels

Toggle panels with their keyboard shortcuts or through the help overlay (**H**).

### Help Overlay (H)

Displays the full list of keyboard shortcuts inside the viewer.

### Rendering Controls (R)

Adjust visual parameters in real time:

- **Tone mapping** -- algorithm and exposure
- **Bloom** -- glow effect strength, radius, and threshold
- **Cinematic effects** -- bloom, vignette, detector noise, and chromatic lens distortion
- **Anti-aliasing** -- FXAA, MSAA, or SSAA
- **Detector noise** -- physics-based Poisson + Gaussian + FPN simulation

Changes persist to `localStorage` for the current scene.

### Dimension Sliders (N)

For datasets with more than three spatial dimensions, this panel shows a slider
for each non-displayed dimension. Drag a slider to move the slice position along
that axis. See the nD Navigation section below.

### Performance Monitor (P)

Displays one live metric at a time — click it (or press Enter/Space) to cycle
between frames per second, frame time (ms), and a scrolling FPS graph.
Useful for diagnosing performance on large scenes.

### Recording Panel (T)

Controls for capturing image sequences or video from the viewer. Open the panel,
configure frame rate and duration, and start recording.

### Dataset Browser (O)

Browse and switch between available datasets served by the data server.

### Scale Bar (B)

Displays a physical scale bar overlay when the scene defines spatial units
(nm, um, mm, cm, m, etc.).

### Colormap Legend (J)

Shows the active colormap and its value range when a colormap is applied to the
scene.

### Layers Panel (L)

Displays a list of all data nodes in the scene with toggles for visibility
and opacity control.

### Overlays (U)

Screen-space annotations (text, images, HTML) positioned over the 3D canvas.
Overlays are defined in the zarr scene by the Python API and rendered as HTML
elements. Some overlays are dimension-aware: they appear or disappear as you
navigate through dimensions. Press **U** to toggle all overlays on/off.

---

## nD Navigation

Luxar supports datasets with an arbitrary number of dimensions. The viewer
always displays three spatial dimensions at a time; additional dimensions are
navigated by slicing.

### How It Works

1. **Displayed dimensions** are rendered in 3D (typically x, y, z).
2. **Non-displayed dimensions** each have a slice position and a tolerance.
   Geometry is visible when its coordinate along a non-displayed dimension falls
   within the tolerance window around the current slice position.

### Using Keyboard Navigation

1. Press a number key (**1**--**9**) to select which navigable dimension to
   control.
2. Press **\[** to step backward or **\]** to step forward along that dimension.
3. Step sizes are defined by the scene's dimension metadata.

### Using Sliders

Press **N** to open the dimension slider panel. Drag any slider to move the
slice position for that dimension.

### Tips for nD Data

- If the viewer shows zero visible points after loading, the initial slice
  position may be at a location with no data. Navigate along non-displayed
  dimensions to find populated slices.
- For 4D time-lapse data, use animation playback (see next section) to step
  through time automatically.

### Performance limit: 16 dimensions

The viewer's WASM-accelerated kernels (spatial queries, effective-radius
slicing, GSplat attenuation) are bounded to **16 dimensions**. Scenes with
more dimensions still load and render correctly via a TypeScript fallback,
but interactive performance can drop noticeably on large point or splat
collections. For best performance on high-dim source data, pre-slice or
pre-aggregate before export. See
[LUXAR_ZARR_FORMAT.md → Viewer Constraints and Performance](LUXAR_ZARR_FORMAT.md#viewer-constraints-and-performance)
for details.

---

## Animation Playback

When a dimension is marked as animatable (for example, time), the viewer can
play through its range automatically.

| Action | Key |
|--------|-----|
| Play / pause | K |
| Jump to start | Home |
| Jump to end | End |
| Speed up | Shift + Up |
| Slow down | Shift + Down |

Animation loops according to the configured loop mode: `once` (stop at end),
`loop` (restart from beginning), or `bounce` (reverse direction at each end).
The loop mode and direction can be set through `ViewerConfig` in Python (see
below).

---

## Screenshots and Recording

- Press **G** to capture a single screenshot immediately.
- Press **T** to open the recording panel for multi-frame capture.
- Press **Ctrl+Shift+S** to export the full viewer state (camera, settings,
  dimension positions) to the clipboard as JSON. This JSON can be loaded back
  via `ViewerConfig.from_file()` in Python.

---

## Configuring the Viewer from Python

Set viewer defaults at scene-creation time by passing a `ViewerConfig` to the
scene. These values are stored in the Zarr archive and applied when the viewer
loads the dataset.

```python
import luxar

vc = luxar.ViewerConfig(
    theme="dark",
    camera=luxar.CameraConfig(
        position=(0, 5, 20),
        target=(0, 0, 0),
        fov=50,
    ),
    bloom_enabled=True,
    bloom_strength=0.4,
    control_type="orbit",
    auto_rotate=True,
)

dims = luxar.Dimensions.default_3d()

with luxar.LuxarZarrCompiler("output.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims, viewer_config=vc)
    # ... add geometry to scene
```

### Loading a Viewer Snapshot

Export a viewer state with **Ctrl+Shift+S** in the browser, save the JSON to a
file, then reload it in Python:

```python
vc = luxar.ViewerConfig.from_file("my_view.json")
vc.bloom_strength = 0.8  # tweak as needed

dims = luxar.Dimensions.default_3d()

with luxar.LuxarZarrCompiler("output.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims, viewer_config=vc)
    # ... add geometry to scene
```

### Priority Chain

Settings are resolved with the following priority (highest first):

1. **localStorage overrides** -- per-scene user changes made in the browser
2. **viewer_config** -- defaults stored in the Zarr file
3. **Built-in defaults** -- the viewer's own defaults

### Available Configuration Categories

| Category | Example fields |
|----------|---------------|
| Camera | `position`, `target`, `fov`, `fov_preset`, `near`, `far`, `target_node` |
| Theme | `theme` (`dark`, `light`, `frosted-glass`, `liquid-glass`) |
| Tone mapping | `tone_mapping`, `exposure`, `global_offset`, `global_gamma` |
| Bloom | `bloom_enabled`, `bloom_strength`, `bloom_radius`, `bloom_threshold` |
| Controls | `control_type`, `auto_rotate`, `auto_rotate_speed` |
| Cinematic | `cinematic_mode`, `vignette_enabled`, `chromatic_lens_distortion_enabled` |
| Detector noise | `detector_noise_enabled`, `detector_noise_readout_sigma`, `detector_noise_photon_gain` |
| Anti-aliasing | `fxaa_enabled`, `msaa_enabled`, `ssaa_enabled` |
| Fly controls | `fly_movement_speed`, `fly_rotation_speed`, `fly_inertial_mode`, `fly_damping` |
| UI visibility | `ui.show_help`, `ui.show_rendering_controls`, `ui.show_dimensions`, `ui.show_performance_monitor`, `ui.show_scale_bar`, `ui.show_layers` |
| Dimensions | `dimensions.current_step`, `dimensions.selected_dimension` |
| Animation | `animation` (per-dimension: loop mode, direction, speed) |

See `luxar.ViewerConfig` docstring for the full field list with types and
valid ranges.

---

## Tips and Troubleshooting

**Viewer shows a blank scene or zero points**
- For nD datasets, the initial slice position may be empty. Press **N** to open
  dimension sliders and navigate to a populated region.
- Verify that `src` points at the dataset root rather than its parent listing.

**Performance is poor with large datasets**
- Press **P** to check FPS and identify bottlenecks.
- Reduce anti-aliasing quality (disable SSAA, switch to FXAA).
- Disable bloom and lower anti-aliasing quality in the rendering panel.
- Adaptive resolution automatically lowers pixel density during interaction.

**Camera feels stuck or wrong**
- Press **F** to recenter the camera on the scene bounding box.
- Press **V** to cycle to a different control mode.
- If fly mode momentum is disorienting, press **I** to toggle inertial mode off.

**Caching issues**
- Add `?clear-cache` to the URL to wipe all cached data on startup.
- Add `?no-cache` to disable caching entirely for debugging.

**Exported state does not restore correctly**
- Ensure you use `ViewerConfig.from_file()` and not manual JSON parsing.
- The JSON format is viewer-version-specific; re-export if the viewer has been
  updated.

**Data fails to load (404 errors)**
- Check that `luxar serve` is running and the port matches the `src` URL.
- Verify that the dataset metadata files are reachable from the `src` URL.
- Verify the Zarr archive is complete (`luxar info <path>` can help).
