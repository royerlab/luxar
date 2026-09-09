# Luxar Viewer User Guide

The Luxar viewer is a browser-based application for exploring nD scientific scenes
containing points, lines, Gaussian splats, and triangle meshes. It renders with WebGL
by default and has an opt-in WebGPU path (see the `renderer` URL parameter below), and
it loads data from Zarr stores served over HTTP.

![Luxar viewer interface overview](../../images/docs/viewer-ui-overview.png)

To follow along without installing anything, open a scene on the live demo gallery at [demos.luxarviewer.dev](https://demos.luxarviewer.dev). Every panel and shortcut described below is available there apart from the Dataset Browser, which requires a data server that can list directories. The same viewer is deployed standalone at [luxarviewer.dev](https://luxarviewer.dev), which takes any reachable archive as `?src=<url>`.

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

### Opening your own data in the hosted viewer

There is a fourth way that needs no install at all. The viewer is deployed
standalone at **[luxarviewer.dev](https://luxarviewer.dev)** and will open any
scene it can reach over HTTP:

```
https://luxarviewer.dev/?src=https://example.org/path/to/scene.luxar.zarr
```

Nothing is uploaded. The browser fetches the store directly from wherever you
host it, so the data never passes through luxarviewer.dev — which also means the
scene is exactly as private, and as durable, as the host you put it on.

#### Your host must allow cross-origin reads

This is the one thing that reliably goes wrong. The page is served from
`luxarviewer.dev` while the data comes from your host, so the browser treats
every chunk request as cross-origin and your host has to opt in. What it needs
depends on the store shape:

| store | requirements |
|---|---|
| directory `.luxar.zarr` | `Access-Control-Allow-Origin`. Metadata and chunks are simple GETs, so byte ranges are not needed. |
| zipped `.zarr.zip` | the above, **plus** byte-range support: honour `Range`, allow the `Range` request header, and expose `Content-Range`, `Content-Length`, `Accept-Ranges` and `ETag`. |

A plain static file host with CORS enabled is enough for a directory store.

**What the failure looks like:** the viewer loads but the scene stays empty, and
the browser console shows requests blocked by CORS policy — not a 404. A 404
means the URL is wrong; a CORS error means the URL is right and the host is
refusing to share it. Check the console before changing the URL.

#### Configuration for common hosts

Amazon S3 — bucket CORS configuration:

```json
[{
  "AllowedOrigins": ["https://luxarviewer.dev"],
  "AllowedMethods": ["GET", "HEAD"],
  "AllowedHeaders": ["Range"],
  "ExposeHeaders": ["Content-Range", "Content-Length", "Accept-Ranges", "ETag"],
  "MaxAgeSeconds": 3600
}]
```

Cloudflare R2 takes the same JSON shape (Settings -> CORS policy). Google Cloud
Storage takes the equivalent via `gsutil cors set`, with `responseHeader`
carrying the exposed headers.

nginx:

```nginx
location /scenes/ {
    add_header Access-Control-Allow-Origin "https://luxarviewer.dev" always;
    add_header Access-Control-Allow-Headers "Range" always;
    add_header Access-Control-Expose-Headers "Content-Range, Content-Length, Accept-Ranges, ETag" always;
    if ($request_method = OPTIONS) { return 204; }
}
```

`Access-Control-Allow-Origin: *` also works and is simpler if the data is public
anyway; naming the origin only matters when you want to keep the store readable
from your own pages but not from arbitrary ones. Note that neither choice makes
the data private — a public bucket is public to anyone with the URL, CORS or not.

If you would rather not host anything, `luxar export scene.luxar.zarr -o out/`
writes a self-contained folder with the viewer and a stdlib-only `serve.py`; the
recipient runs `python serve.py` (`file://` cannot open the viewer directly).

See [Distributing scenes](../../tutorials/distributing_scenes.rst) for the wider
picture on sharing scenes, and
[DEMO_SITE_RUNBOOK](../developer/DEMO_SITE_RUNBOOK.md) for how the demo corpus
itself is hosted, including its CORS setup and failure modes.

### Opening a Zipped Scene

The viewer can open a `.luxar.zarr.zip` scene without extracting it.

Produce an archive directly by naming it as the compiler output:

```python
from luxar import LuxarZarrCompiler

with LuxarZarrCompiler("scene.luxar.zarr.zip") as compiler:
    ...
```

You can also package an existing store by giving `luxar optimise` a `.zip`
destination:

```bash
luxar optimise scene.luxar.zarr scene.luxar.zarr.zip
```

Serve the directory containing the archive, then select it in the dataset browser:

```bash
luxar serve /path/to/scenes --viewer
```

You can also pass the archive URL directly to `src`, for example:

```
http://localhost:5173/?src=http://127.0.0.1:8000/scene.luxar.zarr.zip
```

The data server must support HTTP byte ranges and return `206 Partial Content`.
`luxar serve` provides the required behavior. Archives are read-only: commands
that update an existing store in place refuse them, so write to a directory or
new archive instead. The viewer has no browser local-file or drag-and-drop
opening path for any scene format; serve the scene over HTTP instead.

Choose an archive for distribution, not speed: it turns a scene with potentially
hundreds of thousands of hosted objects into one artifact to upload, download, or
attach to a paper. On the benchmark fixture, a first archive load used about 39%
more requests and 48–60% more bytes than the directory form, adding 1–7% to time
to first render. A warm revisit needed only 4–6 requests and about 88 kB, so the
extra cost is concentrated in the cold load.

---

## URL Parameters

Append parameters to the viewer URL to control startup behavior.

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | string | Zarr dataset URL or local path. |
| `theme` | string | Initial theme. One of: `light`, `dark`, `liquid-glass`, `frosted-glass`. |
| `title` | string | Browser tab title (`document.title`). Serve-family commands derive it from the dataset file name; a scene's authored `viewer_config.title` overrides it. Dropped when you switch datasets in the viewer -- the tab is then named after the dataset you switched to. |
| `debug` | flag | Enable the debug interface (developer use). |
| `no-cache` | flag | Disable ALL caching tiers (S-cache + L0/L1/L2). |
| `no-slice-cache` | flag | Disable only the SliceCache (per-slice decoded-geometry reuse); L0/L1/L2 stay on. |
| `no-opfs` | flag | Disable only the L2 persistent (OPFS) tier; L0/L1/S-cache stay on. For environments whose OPFS stalls — the deterministic sibling of the automatic circuit breaker. |
| `cache-debug` | flag | Enable cache debug logging to the browser console. |
| `clear-cache` | flag | Clear all caches on startup. |
| `no-prefetch` | flag | Disable adjacent-chunk prefetching. |
| `prefetch-debug` | flag | Enable prefetch logging to the browser console. |
| `cache-stats` | flag | Auto-open the data-loading monitor expanded on the Cache tab (L0/L1/L2 hit rates). |
| `renderer` | `webgl` \| `webgpu` | Select the default GLSL WebGLRenderer path or opt into the WebGPURenderer + TSL path. |
| `webgpu-force-webgl` | flag | Diagnostic flag for `renderer=webgpu`: keep WebGPURenderer + TSL materials but force Three.js's internal WebGL2 backend. |
| `perf-timestamp` | flag | Opt into GPU timestamp queries (WebGPU only, `timestamp-query` feature). Small runtime cost; intended for the perf bench. |
| `gpuBudgetMB` | number | Pin the GPU-geometry byte budget in MB, bypassing auto-sizing. `0` disables the budget (unbounded resident geometry). |
| `cacheBudgetMB` | number | Total in-memory cache pool (L0 + L1 + S-cache) in MB, for environments without `performance.memory` (Safari, WKWebView). Also supplies a GPU-geometry/LOD residency signal at one third of the cache pool; without `deviceMemory`, it replaces the 512 MB fallback and may raise or lower it. |
| `dpr` | number | Pin a fixed device pixel ratio and disable adaptive DPR (clamped to [0.25, native DPR]). Overrides the high-DPR ceiling, so `?dpr=2` renders at 2 even with **Allow High DPR** off. For deterministic E2E/visual runs. |
| `input` | `touch` \| `mouse` | Force the session's JS input profile: pointer flags, hover capability, touch points, and device tier. This changes device-class fallback budgets (`touch` only — `mouse` keeps the detected tier), primary-tip pen routing, the Safari gesture-canceller gate, and whether the help overlay lists its Touch section; `touch` additionally applies the mobile rendering budgets (adaptive-DPR floor and refresh ceiling, high-DPR cap, GPU-byte and element-texture ceilings, data-worker count) and skips the blend-variant program warm-up. Stylesheets and non-pen gesture routing still follow the real media features and `PointerEvent.pointerType`, so a faithful check needs device emulation or a real device. Detected by default, including an iPad whose Safari reports a macOS user agent. |
| `lineJoin` | `none` \| `miter` | Force the line join style for the session — **applies only to `linePrimitive=screen-space`**. The default capsule primitive partitions every interior joint along its bisector unconditionally, so this parameter (and each node's authored `join` attribute) is a no-op there. |
| `linePrimitive` | `capsule` \| `screen-space` | Select the line rendering primitive (#1352). Default **`capsule`**: a gaussian-like profile of the 2D point-to-segment distance — stable round discs end-on, seamless bisector-partitioned joints, quad-class cost. `screen-space` is the classic quad — the lean path for very large line scenes. With no URL override, the **`Settings → Advanced → Line primitive`** policy decides: `Auto` (default) builds the capsule, except line nodes whose effective segment load (authored count × a rendered-width factor) reaches 2 M, which build the quad; `Capsule`/`Quad` force one primitive everywhere. `?linePrimitive=` overrides the policy for the session. |

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
| O | Toggle dataset browser |
| P | Toggle performance stats (FPS, frame time) |
| R | Toggle rendering controls panel |
| B | Toggle scale bar |
| J | Toggle colormap legend |
| L | Toggle layers panel |
| U | Toggle overlays |
| T | Toggle recording panel |
| Escape | Close all open panels |

The help overlay (`H`) and the dataset browser (`O`) both carry a filter field,
and while one is on screen just typing narrows the list — no click needed.
Their own toggle key is the one exception: pressing `H` in the help overlay (or
`O` in the dataset browser) closes the panel rather than typing that letter, so
start such a query with any other character — or with `Shift`+that letter, which
types it. Matching is case-insensitive. In the dataset browser, `ArrowDown`
moves from the panel into the listing. The browser hides its filter when there
is nothing to narrow (while a directory loads, after an error, in an empty
directory, and in the "enter a path manually" fallback shown for a server that
cannot be listed); in the path field, click or `Tab` into it and type — it takes
a URL, not a filter query. While either panel is open the other shortcuts are held for the
panel and do not reach the scene behind it — including once you have tabbed or
clicked into a field or a listing row. The exceptions are `Escape` (always
closes the panel), `Tab` (cycles focus inside it) and the keys the panel itself
advertises: its own toggle key, plus `H` in the dataset browser so the
`H` Help chip in its banner works. Those pass through only while focus is still
on the panel itself, which is why typing `o` into the dataset browser's path
field types an `o` instead of closing it.

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

Screen-space annotations (text, images, videos, HTML) positioned over the 3D
canvas. Overlays are defined in the zarr scene by the Python API and rendered as
HTML elements. Some overlays are dimension-aware: they appear or disappear as
you navigate through dimensions. Press **U** to toggle all overlays on/off.

Video overlays (`Scene.add_video`) are muted, looping clips stored inside the
scene; a clip plays only while its dimension filter matches, so a story scene
with one turntable per slot decodes one video at a time. A VP9 WebM with an
alpha channel is drawn transparent over the data in Chrome and Firefox; Safari
cannot decode it and shows the clip's poster image instead.

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

### Mesh nodes slice by whole triangles

Every rule above describes per-element visibility, which is what Points, Lines
and GSplats do. A **Mesh** is different: it draws a triangle only when *all
three* of its vertices fall inside the slice window, so a cut surface has a
ragged, triangle-following edge rather than a clean planar one, and on a
*continuous* hidden dimension it shows a slab of finite thickness rather than an
exact cross-section. A coarse mesh with a narrow window can drop whole regions.

Discrete hidden dimensions — time, channel, anything categorical, which is the
usual case for a mesh — are unaffected: a timepoint either matches or it does
not. For the continuous case the slab half-width is authored per node via
`slab_tolerance` (the slab spans `slice ± step × slab_tolerance`). See
[LUXAR_ZARR_FORMAT.md → nD slicing: whole-triangle cull](LUXAR_ZARR_FORMAT.md#nd-slicing-whole-triangle-cull).

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
    title="Rivers of Earth",  # names the browser tab (document.title)
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
    # Turntable rate in REVOLUTIONS PER MINUTE: a full turn takes
    # 60 / auto_rotate_speed seconds, so 0.25 is one turn every four minutes
    # and 3.0 is one every twenty seconds. (The viewer's Navigation popover
    # shows this as "Rotation Period (s)" — the same setting asked the other
    # way round, so it reads in the same unit as the dolly period below.)
    auto_rotate_speed=0.25,
    # Turntable axis. Either a CAMERA-frame axis — "vertical" (screen-up, the
    # default), "horizontal" (screen-right — the scene tumbles over the top),
    # "view" (the view direction — a pure roll, the camera never moves) — or a
    # fixed scene axis: "world-x" / "world-y" / "world-z".
    #
    # Which family you want depends on the framing. A camera-frame turntable
    # always looks the same on screen whatever the scene's orientation, but
    # when the camera looks DOWN at a subject (most composed openings do) the
    # subject's own axis precesses: a spin plus a wobble. A world axis is the
    # classic turntable — pick the one matching the subject's up and it spins
    # about its own axis at any elevation. The two coincide exactly when the
    # camera is level, so switching frames there changes nothing.
    #
    # One caveat on "view" — and on a world axis as it nears the view
    # direction: the LOD selector measures a node by the axis-aligned screen
    # box of its projected bounds, which is not roll-invariant — a square
    # footprint swings about 2x in area at 45°, a full level of the halving
    # ladder. A rolling scene parked near a switch threshold will therefore
    # breathe between levels (and a turntable recording pays extra LOD-settle
    # ticks per frame). Hysteresis softens it. "vertical" and "horizontal"
    # change the view legitimately and are not affected in the same way. A
    # world axis is affected to the extent that it approaches the view
    # direction. At exact alignment the turntable is a pure roll with a
    # stationary camera.
    auto_rotate_axis="vertical",
    # Auto-dolly: the turntable's radial sibling. Instead of going AROUND the
    # subject the camera breathes toward and away from it on a sine — the
    # equivalent of turning the mousewheel back and forth. Combined with
    # auto_rotate it gives the slow approach-and-retreat hero shot; on its own
    # the parallax is a depth cue a still cannot give.
    #
    # The amplitude is a PERCENT of the viewing distance, so it means the same
    # thing at any scene scale: 15 swings between d/1.15 and d x 1.15. The
    # slider goes to 95 (nearly halving and doubling the distance), and a big
    # swing is a legitimate choice — but it is not free, and the cost is not
    # linear. Screen area goes as 1/d^2, so the swing moves projected area by
    # (1 + a)^4 between the far and near extremes: 1.75x at 15%, 5.06x at
    # 50%, 14.46x at 95%. The LOD ladder answers by loading finer levels at
    # the near extreme, and THAT is what scales. Measured over one 3 s cycle
    # on a 100-group demo:
    #
    #     amplitude   area swing   resident elements   LOD transitions
    #        15%         1.75x           118k                161
    #        50%         5.06x           526k                392
    #        95%        14.46x          2.29M                520
    #
    # A 6x bigger swing costs a 19x resident set. Locally, with a warm cache,
    # that is nearly free (144 -> 129 fps on an M-series laptop). On a hosted
    # scene, where cost is requests rather than bytes, it is not — every cycle
    # re-walks the ladder and anything the cache has evicted is re-fetched.
    # Nothing about a large amplitude is unsafe: the distance clamps sit orders
    # of magnitude away (a scene framed at 176k units clamps at 327). Choose by
    # what the motion is worth, not by fear of it.
    #
    # The user keeps control of zoom while it runs: both the wheel and the
    # dolly only ever multiply the distance, so a scroll moves the centre the
    # camera is breathing around rather than fighting the animation. Switching
    # it off leaves the camera where the swing had reached, and re-enabling
    # resumes from there — the same way stopping the turntable leaves the scene
    # at its current angle. It also works in ortho mode, where it breathes the
    # orthographic zoom instead.
    auto_dolly=False,
    auto_dolly_amplitude_percent=15,
    auto_dolly_period=10,  # seconds per full in-and-out cycle
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

1. **localStorage overrides** -- per-scene user changes made in the browser,
   except that an authored camera position is restored with its resolved scene FOV
2. **viewer_config** -- defaults stored in the Zarr file
3. **Built-in defaults** -- the viewer's own defaults

The browser tab title has its own two-step chain: an authored
`viewer_config.title` wins; otherwise the viewer uses the `?title=` URL
parameter, which serve-family commands (`luxar serve --viewer`,
`luxar demo run`) derive from the dataset's file name — so every tab names
the scene it shows instead of a row of identical "Luxar Player" tabs.
Switching datasets inside the viewer retitles the tab after the dataset you
switched to (both of the other two name the scene you just left). With no
`?title=` at all, the viewer derives the name from the store `?src=` points
at — which is what keeps a reloaded or shared post-switch link named.

### Available Configuration Categories

| Category | Example fields |
|----------|---------------|
| Scene identity | `title` (browser tab title) |
| Camera | `position`, `target`, `up`, `fov`, `fov_preset`, `near`, `far`, `target_node`, `zoom` (ortho only) |
| Theme | `theme` (`dark`, `light`, `frosted-glass`, `liquid-glass`) |
| Tone mapping | `tone_mapping`, `exposure`, `global_offset`, `global_gamma` |
| Bloom | `bloom_enabled`, `bloom_strength`, `bloom_radius`, `bloom_threshold` |
| Controls | `control_type`, `auto_rotate`, `auto_rotate_speed`, `auto_rotate_axis`, `auto_dolly`, `auto_dolly_amplitude_percent`, `auto_dolly_period` |
| Cinematic | `cinematic_mode`, `vignette_enabled`, `chromatic_lens_distortion_enabled` |
| Detector noise | `detector_noise_enabled`, `detector_noise_readout_sigma`, `detector_noise_photon_gain` |
| Anti-aliasing | `fxaa_enabled`, `msaa_enabled`, `ssaa_enabled` |
| Performance | `adaptive_dpr_enabled`, `allow_high_dpr` |
| Fly controls | `fly_movement_speed`, `fly_rotation_speed`, `fly_inertial_mode`, `fly_damping` |
| UI visibility | `ui.show_help`, `ui.show_rendering_controls`, `ui.show_dimensions`, `ui.show_performance_monitor`, `ui.show_scale_bar`, `ui.show_layers` |
| Dimensions | `dimensions.current_step`, `dimensions.selected_dimension` |
| Animation | `animation` (per-dimension: `playing`, `target_fps`, `loop`, `direction`, `step_size`) — a scene with `playing: true` on a dimension starts that dimension animating on load, from wherever `dimensions.current_step` put it |
| Story waypoints | `waypoints` (list of `Waypoint`: `when`, `camera`, `duration_ms`, `easing`, `reveal`, `rendering`) — camera poses bound to hidden-dimension positions; see below |

Set `allow_high_dpr=True` if your scene is **line-dominated** — a river network,
a tractogram, a wiring diagram. Phones and tablets cap this setting at DPR 2;
laptops and desktops use the panel's native DPR. The viewer renders at CSS
resolution by default even on a Retina display, because a 2x panel costs 4x the
fragment work and soft-edged emissive geometry barely rewards it. Measured against DPR 2,
brightness and coverage hold to within 2.5% on every geometry type and the whole
visible effect is a 15-35% loss of fine detail: on points and splats that is
mild softening, but on dense thin lines the individual strands stop being
separable. Line scenes are also the cheapest place to spend the pixels, because
they are not fill-bound — a trajectory scene measured 1.06-1.17x faster at DPR 1
against 2.6-2.7x for a point cloud, so you buy the detail back for almost no
frame time. Leave it off for points, gsplats and mesh unless a particular scene
proves otherwise.

Setting `cinematic_mode=True` expands the whole cinematic preset (ACES tone
mapping, a subtle wide bloom, detector noise, vignette, and the 35 mm
chromatic lens + FOV) for every field the scene does not set itself — so you
can enable the look and still override, say, `bloom_strength` on top of it.
Note that the preset also widens the camera to the 35 mm field of view (63°),
which is applied before automatic framing so the fitted subject occupancy
matches that lens. Pin `camera.fov` (or `camera.fov_preset`) only when composing
an explicit camera pose for a specific lens; the preset then leaves the authored
FOV alone but still applies its 35 mm distortion to that different framing. The
bundled demos instead leave the FOV unpinned and compose their authored positions
for 63°. A returning visitor's stored FOV still takes precedence for auto-framed
scenes; an authored camera position is always restored with the resolved scene FOV
it was composed for.

### Story waypoints: a camera pose per hidden-dimension position

A scene can tell a story. Give it a hidden discrete "story" dimension, author
the colours per story value in the data, caption each value with overlays that
carry a `visible_range`, and bind the camera with `waypoints`:

```python
import luxar
from luxar import CameraConfig, Waypoint

vc = luxar.ViewerConfig(
    waypoints=[
        # Story 0: the overview.
        Waypoint(when={"story": 0}, camera=CameraConfig(position=(0, 0, 40))),
        # Story 1, but only while time is in [10, 20]: fly to the cluster
        # and brighten the exposure on arrival.
        Waypoint(
            when={"story": 1, "time": (10, 20)},
            camera=CameraConfig(target_node="cluster_7", position=(12, 3, 8)),
            duration_ms=2500,
            rendering={"exposure": 0.5},
        ),
        # Story 1 anywhere else in time: re-aim only (position is kept).
        Waypoint(when={"story": 1}, camera=CameraConfig(target_node="cluster_7")),
    ]
)
```

The `when` clause uses the overlay `visible_range` rule: every named dimension
must match, an exact value matches within ±0.5 of the current step, a
`(min, max)` range matches inclusively, and the **first** matching waypoint in
list order wins — so put specific clauses before broad ones. Fields left out of
a waypoint's `camera` keep the live camera's value at flight time, which is how
a waypoint can re-aim without moving.

The viewer acts on a change of *matched waypoint*, not on every slider tick. At
load it snaps to whichever waypoint matches the opening dimension state (ahead
of the plain `camera` block). Afterwards stepping the story dimension — the
`[` / `]` keys, the slider, or an external controller — flies to the new
waypoint with its own `duration_ms` (default 1500; `0` snaps) and `easing`
(`"ease-in-out"` or `"linear"`); moves that stay inside the same waypoint's
ranges do nothing, and leaving every waypoint leaves the camera where it is.
Any mouse, touch or key input during a flight cancels it where it is. The
optional `reveal="on_arrival"` holds newly matching dimension-bound overlays
until that flight resolves; the default `"immediate"` shows them as the
dimension changes. The
optional `rendering` block takes the same snake_case keys as `ViewerConfig`
itself and is applied on arrival through the same validated path.

Waypoints compose with the orbit turntable: while auto-rotate is on, a story
step keeps the current viewing direction and only moves the point the camera
spins around (and how far away it sits), so the spin never pauses and the
authored orientation is ignored. In ortho mode camera distance changes nothing,
so author `CameraConfig(zoom=...)` to frame a cluster tighter. In fly mode a
waypoint moves and aims the camera just the same; flying with the keyboard
during a flight cancels it.

For a complete worked example — story dimension, dimmed backdrop, per-story
highlight layers, fact panels and waypoints — run
`luxar demo run esm3_protein_stories` (it reads the ESM3 landscape demo's cache).

### Sound

A scene may carry `sound` nodes (`scene.add_sound`, see
`docs/guides/specs/SOUND_SPEC.md`): an ambient bed, a narration bound to a
story step, or a spatial source that gets louder as the camera approaches. Their
audibility is the same hidden-dimension slab rule that decides which points are
visible, so scrubbing a story dimension starts and stops the clips that belong
to each step. When a loaded scene has sound nodes a **Sound** button appears in
the rail: click mutes everything (persisted across scenes), while right-click or
a hold opens the mixer (master gain, the `ambient` / `voice` / `effects` buses,
equal-power vs HRTF panning). The voice bus ducks the ambient bed while a
narration plays.
Scene defaults live in `ViewerConfig(audio=AudioConfig(...))`, and a controller
drives the same knobs through `setAudio()`, `playSound()`, `stopSound()` and
`getViewerState().audio`.

Sound nodes authored with `layer=True` appear in the **Layers** panel with a
`sound` badge: the eye mutes that node (a parent group's eye silences every
sound under it), an inline slider sets its gain, and the name's tooltip shows the
clip's licence, author and source. Narration authored with `trigger="on_arrive"`
starts when a story flight lands (a flight you cut short still counts as
arrived; one superseded by the next story does not). A node with `attach_to`
follows another node's centre, and an `ambisonic="foa"` bed is a sound field
that stays fixed to the world as you turn the camera. The **Recording** panel's
"Include Audio" option (Advanced) records what you hear into real-time videos;
frame-by-frame captures stay silent.

Browsers refuse to start audio without a gesture on the page. In a regular tab
the viewer shows a one-time **Tap to enable sound** overlay that the first click
or key dismisses. For an unattended kiosk launch Chrome with the autoplay policy
relaxed so the context starts on load and the gate never appears:

```
google-chrome --kiosk --autoplay-policy=no-user-gesture-required "http://host:5173/?src=…"
```

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
- Check **Allow High DPR** in the Performance panel is off (it is by default).
  On a HiDPI display it costs four times the pixels, which is rarely worth it
  for points and splats. Conversely, if a scene of thin lines looks mushy rather
  than slow, turning it ON is usually cheap — line scenes are not fill-bound.

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
