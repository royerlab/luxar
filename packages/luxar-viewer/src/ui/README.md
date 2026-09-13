# Luxar UI Package

> Beautiful, responsive user interface components for nD visualization and control

## Overview

The Luxar UI package provides a comprehensive set of user interface components for controlling visualization parameters, navigating nD datasets, monitoring performance, and debugging. Built with modern web technologies, it offers an intuitive interface inspired by scientific visualization tools like napari.

### Key Features

- **Dimension Sliders**: Napari-inspired sliders for nD navigation
- **Rendering Controls**: Real-time adjustment of visual parameters
- **Dataset Browser**: Navigate and load Zarr datasets
- **Performance Monitor**: FPS and GPU memory tracking
- **Data Loading Monitor**: Real-time monitoring of spatial index-based data loading
- **Debug Console**: In-app console for development
- **Helper Overlays**: Keyboard shortcuts and tips
- **Recording Panel**: Screenshot and video capture with turntable mode
- **Scale Bar**: Physical scale bar overlay using dimension units
- **Colormap Legend**: Per-layer colormap gradient overlay
- **Layers Panel**: Napari-inspired per-layer visibility, range, gamma, blending
- **Custom GUI Library**: Drop-in lil-gui replacement with theme integration
- **Responsive Design**: Mobile and desktop friendly

### Package Architecture

All public-API files live at `ui/` root. Sibling folders hold each
public file's private helpers. Folder names describe their concern, so a
developer reading a path can predict the audience, peers, and home for a
new sibling without opening files.

```
ui/
├── README.md
│
│ ── Public-API entrypoints (at root) ──
├── gui.ts                              # Re-export entrypoint for the custom GUI library
├── rendering-controls.ts               # Visual parameter adjustments
├── recording-panel.ts                  # Screenshot + video capture
├── layers.ts                           # Re-export of layers panel
├── dimension-sliders.ts                # nD navigation sliders
├── debug-console.ts                    # Developer console overlay
├── dataset-browser.ts                  # Zarr dataset navigation
├── scale-bar.ts                        # Physical scale bar overlay
├── colormap-legend.ts                  # Per-layer colormap gradient
├── resolution-indicator.ts             # Resolution / DPR indicator
├── performance-monitor.ts              # FPS + GPU stats HUD
├── data-monitor-manager.ts             # Wires loaders to data-loading-monitor
├── data-loading-monitor.ts             # Spatial-loader telemetry monitor
├── overlay-manager.ts                  # Screen-space overlay rendering
├── video-matte.ts                      # Stacked-alpha-matte video compositor (WebGL canvas;
│                                       #   transparency that survives Safari / WKWebView)
├── loading-indicator.ts                # Loading spinner (was helpers.showLoading*)
├── error-overlay.ts                    # Error dialog (was helpers.showError/clearError)
├── help-overlay.ts                     # Keyboard shortcuts panel (was helpers.show/hideHelp)
├── control-rail.ts                     # Always-visible left activity rail (helpers in control-rail/)
├── toast.ts                            # Brief auto-dismiss notifications
├── scene-identity-banner.ts            # Persistent "address now serves a different scene /
│                                       #   server unreachable" banner (Reload button on the
│                                       #   changed kind); driven by data/scene-identity-watchdog
│                                       #   through the cross-layer notifier
├── ui-cleanup.ts                       # App-teardown helper
│
│ ── Public-file private helpers (siblings) ──
├── gui/                                # GUI library internals
│   ├── gui.ts, controller.ts, folder.ts, types.ts
│   ├── controllers/                    # per-type controllers
│   ├── dom/                            # DOM plumbing
│   └── format/                         # value-formatting + auto-blur
├── rendering-controls/
│   ├── focus-manager.ts, cinematic-mode.ts, apply-settings.ts,
│   │ sync-current-state.ts, settings-persistence.ts,
│   │ clipping-display.ts, controls-utils.ts, fov-utils.ts,
│   │ folder-icons.ts, types.ts
│   └── setup/                          # *-setup.ts files
│       ├── camera-setup.ts, hdr-setup.ts,
│       │ anti-aliasing-setup.ts, post-processing-setup.ts,
│       │ performance-setup.ts, theme-setup.ts
├── recording-panel/
│   ├── session.ts                      # RecordingSession — shared scaffolding
│   ├── capture-strategy.ts             # CaptureStrategy interface + SessionState view
│   ├── screenshot-strategy.ts, video-recording-strategy.ts,
│   │ offline-capture-strategy.ts       # per-kind capture strategies
│   ├── types.ts, media-utilities.ts, video-codec-selection.ts,
│   │ animation-sync.ts, overlay-compositor.ts, screenshot-exporter.ts,
│   │ zip-sequence-capture.ts, gui-builder.ts
│   ├── ui/
│   │   └── gui-construction.ts         # buildRecordingGUI(deps)
│   └── drivers/                        # per-mode capture drivers
│       ├── offline-capture-driver.ts   # CaptureContext + driver protocol
│       ├── image-sequence-driver.ts, exr-sequence-driver.ts,
│       │ video-mode-driver.ts
├── data-loading-monitor/               # Monitor's private helpers
│   ├── templates/                      # primitives + per-tab renderers
│   ├── advisor.ts, event-queue.ts, polling-loop.ts, timing-panel.ts
│   ├── metrics/
│   │   ├── cache.ts (aggregator), rates.ts
│   └── tabs/
│       ├── cache.ts, dom-helpers.ts
├── debug-console/
│   └── formatters.ts                   # @timestamp / [stream] / etc.
├── dataset-browser/
│   └── url-utils.ts                    # extractBaseUrl / extractPath
├── dimension-sliders/
│   └── slider-math.ts                  # clamp / wrap helpers
├── layers/                             # Layers panel internals
│   ├── layers-panel.ts                 # main class (re-exported via ../layers.ts)
│   ├── layer-state.ts, range-slider.ts, labeled-slider.ts,
│   │ attrs-utils.ts
├── overlay-widgets/                    # Shared base for scale-bar / colormap-legend
│   ├── ui-component.ts
│   └── context-menu.ts                 # Shared right-click menu (openContextMenu; full menu ARIA)
├── help-overlay/                       # Shared modal-panel helpers (folder name predates the sharing)
│   ├── focus-trap.ts                   # Tab/Shift+Tab focus trap (also used by error-overlay, dataset-browser)
│   └── type-to-filter.ts               # Container focus + first-keystroke filtering (help-overlay, dataset-browser)
├── control-rail/                       # Control rail's private helpers (orchestrator: ../control-rail.ts)
│   ├── rail-overlay.ts (flyout + popover lifecycle),
│   │ icons.ts (RAIL_ICONS), dom-helpers.ts, types.ts
├── rail-panels/                        # Rich popovers hosted by the control rail
│   ├── settings-popover.ts, performance-popover.ts,
│   │ navigation-popover.ts, home-popover.ts, popover-gui.ts
└── panels/                             # Reserved namespace for a future shared panel framework (currently empty)
```

### Why this layout?

- **Public API at root.** External imports (`from '../ui/<name>'`) always resolve to a real file at the package root. Node resolves the file in preference to a folder of the same name, so the public path stays canonical.
- **Helpers under their consumer.** A single-consumer helper lives in the consumer's sibling folder — e.g. `dimension-sliders/slider-math.ts` is only used by `dimension-sliders.ts`. The depth signals audience: a sibling folder means "private to this public file".
- **Concern-named folders.** Folders are named by responsibility (`overlay-widgets/`, `drivers/`, `setup/`, `metrics/`, `tabs/`, `format/`, `ui/`) instead of broad buckets.
- **Public barrels stay shallow.** Public entrypoints are real files at `ui/` root. Subfolders contain private implementation details for those entrypoints.

### Orchestrator organization

Coordination-heavy UI classes (`recording-panel.ts`, `data-loading-monitor.ts`, `dimension-sliders.ts`, `layers-panel.ts`) keep stateful workflows in the orchestrator and delegate naturally pure work to focused helpers. For example, `recording-panel/ui/gui-construction.ts::buildRecordingGUI(deps)` owns DOM construction behind a small callback interface, while capture/session coordination remains on the panel/session classes that own the relevant state.

---

## Components

### 0. Control Rail

The always-visible discoverability affordance (`ui/control-rail.ts`) — a slim
vertical activity rail docked to the left edge. Luxar's panels are otherwise
keyboard-triggered, so the rail is the one visible entry point: one recognizable
icon per panel (Help, Home, Navigation, Dimensions, Rendering, Layers,
Data monitor, Datasets, Recording, Logs, View options, Settings, Performance),
each with a hover tooltip showing its shortcut. Home, Navigation, Settings and
Performance open rail popovers (see [`rail-panels/`](./rail-panels/README.md)).
On a device that cannot hover the first-run hint says "Tap these controls (hold
for options)" instead of "Hover", and under a coarse pointer the rail gains a
momentary **Hide panels** item that closes every open surface — see
`core/app/init/build-rail-items.ts` and the UI Design Guide §11.5.

**Design:**

- **No behavioural drift** — each button fires the _exact same_ command as its
  keyboard shortcut, via `InputHandler.getUiActions()` (the command/panel surface
  the key bindings dispatch into). The rail never re-implements panel logic.
- **Live active-state** — a button highlights while its panel is open. Refreshed
  event-driven (on document click, on a keydown the input router reports as
  handled, on fullscreen change, and on a layer / control-mode change —
  rAF-debounced), so it also clears when a panel is closed via its own × button.
- **Idle-dim** — recedes when the pointer is idle; wakes on movement (expanded)
  or hover (collapsed / fullscreen).
- **Collapse** — a chevron handle collapses the rail into the lower-left corner
  (persisted); it reveals on hover only.
- **View-options flyout** — one button opens a horizontal popover of overlay
  toggles (scale bar, colormap legend, overlays, cinematic, fullscreen) so the
  rail stays short.
- **Docked footer** — hosts the Performance readout (see §4).
- **Theme participation** — styled with `--luxar-*` tokens and carries the
  `luxar-glass-surface` marker class (like every other panel) so it shares the
  panels' material across all four themes (see `themes/README.md`).
- Focus-safe: blurs the button after a pointer click so canvas/body-gated
  shortcuts (e.g. Space = fullscreen) keep working; keyboard focus is preserved.

Construct with `new ControlRail(items, footer?)`; the pipeline builds the items
(wired to `getUiActions()`) and passes the perf readout's `.element` as `footer`.

### 1. Dimension Sliders

Beautiful napari-inspired sliders for navigating through nD datasets.

**Features:**

- Smooth slider controls for each dimension
- Real-time value display with units
- Color-coded dimension indicators
- Keyboard navigation support
- Auto-hide for 3D-only datasets
- Step-based navigation for discrete dimensions
- **Animation controls** for automated playback through dimension ranges

**UI Structure:**

```typescript
class DimensionSliders {
  // Creates slider panel with:
  - Header with collapse toggle
  - Slider for each non-displayed dimension
  - Value display with unit labels
  - Keyboard hints
}
```

**Usage:**

```typescript
import { sceneDimsManager } from '../scene/scene-dims-manager';

// After the scene loads and dims are initialized
const sliders = new DimensionSliders({
  container: document.body,
  dims: sceneDimsManager.getDims(),
  dimensionRanges: sceneDimsManager.getDimensionRanges(),
  dimensionNames: sceneDimsManager.getDimensionNames(),
  dimensionUnits: ['μm', 'μm', 'μm', 's', ''],
});

// Sliders sync automatically through sceneDimsManager: moving a slider
// calls sceneDimsManager.setDimensionValue(), which notifies all nD
// objects. No per-slider event subscription is required.
```

**Animation Controls:**

Each dimension slider includes animation controls for automated playback through dimension ranges (e.g., time-lapse, z-stack traversal):

```
┌─────────────────────────────────┐
│ Time                    50ms    │
│ [▶] ◀━━━━━━━●━━━━━━━━━━▶       │
└─────────────────────────────────┘
```

**Control Elements:**

- **Play Button** (`▶`/`⏸`): Compact button to the left of slider
  - Left-click: Toggle animation play/pause
  - Right-click or hold: Open settings context menu (Napari-style)

- **Slider wheel**: One base step per notch (the authored step, else 1% of the
  range); `Shift` fine (÷10), `Ctrl` coarse (×10), `Ctrl+Shift` extra-fine
  (÷100). Clamped at the range ends, and deliberately independent of the
  animation Step override below — hand stepping stays on the dimension's own
  grid.

**Context Menu Settings** (right-click or hold the play button): three sections, each a
micro-header row over one wrapping row of selectable chips.

- **Speed Section**: Set target animation speed
  - Presets: 1/2 (one frame every 2 s), 1, 2, 5, 10, 15, 30, 60, 120 FPS
  - Single-select chips (`role="radio"`) with the current speed marked

- **Loop Mode Section**: Choose loop behavior
  - `Once`: Play once and stop at end
  - `Loop`: Loop continuously from start to end
  - `Bounce`: Ping-pong back and forth
  - Single-select chips with the current mode marked

- **Detail Section**: The playback detail — how many additive-ladder rungs
  every frame is drawn at while the dimension plays
  - `Auto` (default, or the scene's authored `playback_lod_depth`): each ladder
    is pinned at the first rung whose energy stamp reaches the configured
    threshold; ladders without stamps stream time-budgeted
  - `1` … `8` / `All`: pin the rung count; every frame waits for exactly that
    many rungs (or the whole ladder), so quality is constant and the frame rate
    adapts to the data
  - `Fast`: time-budgeted streaming — each tick shows whatever rungs were
    resident within its budget (fast cadence, quality varies frame to frame).
    The active setting shows as the muted header readout; scrubbing uses the
    same detail, with an unpinned refine once the scrub settles.

- **Step Section**: The per-tick quantum for playback **and** the `[` / `]`
  keys — FPS then only decides how often a step lands
  - `Auto` (default): the historical behavior — continuous dimensions traverse
    the range in ~10 s at the target FPS, discrete ones move one authored step
  - `×N` chips: multipliers of the dimension's base step (authored step, else
    1% of the range), with the computed value in each chip's tooltip; the
    active quantum shows as the muted header readout
  - A custom number field on its own row below the chips (outside the chips'
    radiogroup) commits on Enter or blur
  - On a discrete dimension the override is quantized to the authored grid
    with a one-cell floor, and only whole-cell multipliers are offered

**Animation Features:**

- FPS-based throttling with actual FPS measurement
- Independent animation state per dimension
- Forward/backward direction control
- Discrete and continuous dimension support
- Performance monitoring with warnings if target FPS not achieved
- Event system for UI synchronization

**Keyboard Shortcuts:**

- `K`: Toggle play/pause for selected dimension
- `Home`: Jump to dimension start
- `End`: Jump to dimension end
- `Shift+↑`: Increase animation speed
- `Shift+↓`: Decrease animation speed

**Integration:**

```typescript
import { DimensionAnimationManager } from '../scene/dimension-animation-manager';

// Animation manager is automatically created by InputHandler
// and connected to dimension sliders

// Programmatic control
const animManager = inputHandler.getAnimationManager();
animManager.play(dimIndex, { targetFPS: 30, loopMode: 'loop' });
animManager.pause(dimIndex);
animManager.setTargetFPS(dimIndex, 60);

// Listen for animation events
animManager.addEventListener('play', (e) => {
  console.log(`Dimension ${e.dimIndex} started animating`);
});

animManager.addEventListener('complete', (e) => {
  console.log(`Dimension ${e.dimIndex} animation complete`);
});
```

**Implementation Details:**

- Uses `DimensionAnimationManager` for FPS-based animation logic
- Integrates with `AnimationController` for frame updates
- Updates dimension values via `SceneDimsManager`
- CSS styling in `styles/components/dimension-sliders.css`
- Coarse pointers (`getInputProfile().coarsePointer`): each slider row gains `‹ ›`
  step buttons (one base step per tap — the `[ ]` step) in the same
  controls wrapper the play button joins, and the dimension name becomes a chip that
  selects that dimension as the `[ ]` target (`SliderConfig.onSelectDimension` tells
  the input layer). Neither exists on a mouse machine; their styling lives in
  `styles/components/coarse-pointer.css`.
- See `scene/animation/dimension-animation-manager.ts` for core animation logic

### 2. Rendering Controls

Comprehensive controls for adjusting rendering parameters in real-time.

**Control Categories:**

- **Visual Effects**: Bloom, tone mapping, detector noise, vignette, chromatic lens distortion
- **HDR**: Exposure (log2 stops, -5 to +5), offset, gamma, and tone mapping
- **Anti-Aliasing**: FXAA, MSAA, SSAA toggles
- **Performance**: Quality presets, FPS targets
- **Camera**: FOV presets (28mm-135mm equivalents), manual FOV control, clipping plane adjustments
- **Materials**: Opacity, gamma, blending modes

**HDR Exposure Slider:**

The HDR exposure slider uses log2 stops (photography-standard units):

- **-5 stops**: very dim (1/32 brightness)
- **0 stops**: neutral (no change)
- **+5 stops**: very bright (32x brightness)

Each stop doubles or halves the brightness, providing perceptually uniform control.

**Panel Layout:**

Navigation controls (control type, orbit/fly parameters) are **not** in this
panel — they live in the Navigation rail popover
(`rail-panels/navigation-popover.ts`; navigation is not a rendering concern).

```
Rendering Controls
├── Camera
│   ├── FOV Preset (dropdown: 28mm/35mm/50mm/85mm/135mm/Custom)
│   ├── Field of View (slider, 10°-200°)
│   └── Clipping Planes ▼
│       ├── Near Plane (slider, 0.001-10.0)
│       ├── Far Plane (slider, 10-10000)
│       └── Auto Adjust (button)
├── HDR
│   ├── Exposure (log2 stops slider, -5 to +5)
│   ├── Global Offset (slider, -1.0 to 1.0)
│   ├── Global Gamma (slider, 0.1 to 10.0)
│   └── Tone Mapping (type selector)
├── Anti-Aliasing
│   ├── SSAA □ (with resolution multiplier)
│   ├── FXAA □
│   └── MSAA □ (with samples)
└── Post-Processing Effects
    ├── Bloom
    │   ├── Threshold (slider)
    │   ├── Strength (slider)
    │   ├── Radius (slider)
    │   └── Mipmap Levels (slider)
    ├── Detector Noise
    │   ├── Enabled □
    │   ├── Readout Sigma (slider)
    │   ├── Photon Gain (slider)
    │   └── FPN Sigma (slider)
    ├── Chromatic Lens Distortion
    │   ├── Enabled □
    │   ├── Distortion X/Y (sliders)
    │   ├── Principal Point (sliders)
    │   ├── Focal Length (sliders)
    │   └── Skew (slider)
    └── Vignette
        ├── Enabled □
        ├── Darkness (slider)
        └── Offset (slider)
```

### 3. Dataset Browser

File browser for navigating and loading Zarr datasets from servers.

**Features:**

- Server-agnostic navigation (WebDAV, S3, nginx)
- Zarr dataset detection and highlighting
- Breadcrumb navigation
- File size and date display
- Search and filter capabilities
- Recent datasets history

Initial focus is on the panel container, not the search field (nor the
manual-entry field of the unlistable-server fallback), so the `O` shortcut
still toggles the browser shut — see the type-to-filter note under
[Helper Overlays](#7-helper-overlays) (issue #1922). Typing still narrows the
listing from the first keystroke whenever the search bar is on screen — it is
hidden while a directory loads, after an error, in an empty directory and in the
manual fallback, and keystrokes are contained by the modal in those states — and
`ArrowDown` moves into the list from the panel container as well as from the
search field. If a navigation loading state, breadcrumb rebuild or cancelled
path edit removes the focused child, the browser re-parks focus on the panel
only when focus is otherwise unclaimed (`null` or `<body>`), preserving modal
containment without stealing focus from a dialog stacked on top.

The panel declares `O` and `H` as passthrough keys: `O` is its own toggle, and
`H` is the shortcut its welcome banner advertises with an `H Help` chip — modal
containment would otherwise make that chip dead. The manual-entry
`#manual-path` field is deliberately NOT wired into type-to-filter: it takes a
URL, not a filter query, so stray keystrokes should not be routed into it. Click
or Tab into it to type, which is a deliberate act; from there the ordinary
typing guard applies, and `Escape` still closes the panel.

**Interface:**

```typescript
class DatasetBrowser {
  constructor(config: DatasetBrowserConfig);
  show(): void;
  hide(): void;
}

interface DatasetBrowserConfig {
  container: HTMLElement;
  /** Return false to keep the browser open; Promise rejections are caught + toasted. */
  onDatasetSelect: (fullUrl: string) => void | false | Promise<void>;
  onClose?: () => void;
}
```

### 4. Performance Monitor

A compact, theme-matched square readout of frame performance. It **docks into
the control rail** as its footer (rail-button sized), and its visibility is
toggled by the rail's Performance (gauge) button or the `P` key.

**One metric at a time — click the square to cycle:**

- **FPS** (frames per second, colour-coded green/amber/red)
- **ms** (smoothed frame time)
- **graph** (a small scrolling FPS history)

It is driven by the animation loop's `frame-start` / `frame-end` events on the
cross-layer event bus (see `utils/cross-layer/event-bus`) — no `stats.js`
dependency. While visible it subscribes to those events and computes FPS/frame
time itself; when hidden it unsubscribes so it incurs no cost. To keep an
initial reading when the scene is idle, showing it kicks the render loop once
(via an injected `keepAlive`) **without** forcing continuous rendering — so FPS
is live while the scene renders and freezes at the last value when it idles
(preserving the idle-pause / battery saving).

`PerformanceMonitor` exposes its DOM node via `.element` (the rail mounts it)
rather than self-appending, and `cycleMode()` advances FPS → ms → graph.

**Keyboard Shortcut:** `P` to toggle visibility.

### 5. Data Loading Monitor

Advanced real-time monitoring system for spatial index-based data loading with performance analytics and smart recommendations.

**Features:**

- **Three-State UI**: Cycles through hidden → mini → expanded views
- **Event-Driven Monitoring**: Tracks queries, loads, and errors emitted by the spatial-index loaders
- **Cache Analytics**: Detailed per-tier cache memory usage and hit-rate statistics (from the cache-stats provider, not monitor events)
- **Smart Recommendations**: LoadingAdvisor suggestions for optimization
- **Multi-Loader Support**: Monitors multiple data loaders simultaneously

**UI States:**

1. **Hidden**: No UI visible (default on startup)
2. **Mini View**: Compact metrics bar showing key statistics
3. **Expanded View**: Full panel with tabs for detailed analytics

**Keyboard Shortcut:** `M` to cycle through states

**Architecture:**

```typescript
// Managed by DataMonitorManager singleton
DataMonitorManager.getInstance().createMonitor(id, container).connectLoader(path, loader);

// Monitor receives events from loaders
loader.addEventListener((event: MonitorEvent) => {
  // Event types: query, load, error
});
```

**Tabs in Expanded View:**

1. **Overview Tab**
   - Global statistics (total points, memory, loaders)
   - Active loader list with real-time status
   - Key performance indicators

**Design language ("quiet instrument")**: every metric value renders in
the mono stack with tabular numerals (stable live ticking); labels share
one 10px caps micro-style; section titles rank above labels and carry a
small status-tick motif; color is semantic only (health states), with
idle zeros dimmed and hit rates dimmed during cache warm-up
(`CACHE_WARMUP_ACCESSES`) instead of alarm-red; iconography is the
inline `MONITOR_ICONS` stroke-SVG set (no emoji); tab/section paints get
a one-shot staggered reveal (disabled under `prefers-reduced-motion`).
See the "Refinement layer" section at the end of
`styles/components/data-loading-monitor.css`.

2. **Cache Tab**
   - Per-tier sections (S-cache / L0 / L1 / L2), each collapsible: sections
     start collapsed as a compact one-line summary carrying the same values
     (size, hit rate + hits·miss, evictions, I/O, errors) and expand to
     full metric cards on header click — keeps the tab scrollbar-free
     with 4 tiers
   - Cache memory usage with visual gauge
   - Per-tier hit-rate statistics (S-cache / L0 / L1 / L2) plus the effective demand hit-rate
   - Per-tier eviction and I/O counts

3. **Memory Tab**
   - GPU buffer-pool stats (allocations / reuses / evictions, per data type)
   - Element-accumulator capacity and growth stats (rendered by `renderMemoryContent`)

4. **Performance Tab**
   - Collapsible hierarchical timing tree (`data-loading-monitor/timing-panel.ts`):
     per-step and per-geometry-type view-update timings, rows over the frame
     budget highlighted
   - A "Profiler not connected" empty state until a profiler is attached

5. **Insights Tab**
   - Smart recommendations from LoadingAdvisor
   - Severity-based alerts (error, warning, info)
   - Actionable optimization suggestions
   - Performance bottleneck detection

**Performance Optimizations:**

- **Event Cleanup**: Automatically removes events older than 5 minutes
- **Rate Caching**: Calculations cached for 1 second to reduce CPU usage
- **Polling Updates**: A `PollingLoop` that re-schedules itself on a fixed interval via chained `setTimeout` (no `requestAnimationFrame`) drives periodic UI refreshes while the monitor is visible
- **Efficient Updates**: Only re-renders changed UI sections

**Integration with Scene Loading:**

```typescript
// Automatic integration - no setup required!
// SceneLoader creates monitor on construction
const loader = new SceneLoader(config);

// Monitor automatically connects to spatial index loaders
// Tracks all data loading operations
// Disconnects old loaders when loading new scenes
```

**Event Types Monitored:**

- `query`: Spatial index query with cell/point counts
- `load`: Data chunk loaded with memory usage
- `error`: Loading errors with details

Cache hit/miss/eviction figures are read from the cache-stats provider
snapshot (per tier), not emitted as monitor events.

**Usage Example:**

```typescript
import { cycleDataMonitor } from '../ui/data-monitor-manager';

// Toggle monitor with M key
document.addEventListener('keydown', (e) => {
  if (!e.ctrlKey && !e.metaKey && e.key === 'm') {
    cycleDataMonitor(); // Cycles: hidden → mini → expanded → hidden
  }
});
```

In production, lower layers drive the monitor via the cross-layer event
bus (`panel-cycle` / `panel-hide` for `panelId: 'data-monitor'`) rather
than importing this UI module directly. `cycleDataMonitor` is the only
remaining convenience accessor; it is kept for integration tests.

**Configuration Options:**

```typescript
interface MonitorConfig {
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  updateInterval: number; // ms between UI updates
  maxEvents: number; // Maximum events to keep in history
  showRecommendations: boolean; // Show LoadingAdvisor tips
  autoExpand: boolean; // Auto-expand on warnings
  enableProfiling: boolean; // Enable detailed profiling
}
```

**Key Benefits:**

- **Zero Configuration**: Automatic integration with scene loading
- **Real-time Insights**: Immediate visibility into loading performance
- **Smart Recommendations**: LoadingAdvisor suggestions for optimization
- **Performance Optimized**: Minimal overhead with intelligent batching
- **Developer Friendly**: Clean API and comprehensive documentation

### 6. Debug Console

In-app developer console for debugging and diagnostics.

**Features:**

- Console output capture and display (log, warn, error, info, debug)
- Message history from app startup via global console interceptor
- Syntax-highlighted output (objects, numbers, strings)
- Filtering by keyword
- Copy all messages to clipboard
- Auto-scroll option
- Error stack traces
- Draggable and resizable panel

**Keyboard Shortcut:** `Ctrl+L` to toggle

**Interface:**

```typescript
class DebugConsole {
  show(): void; // Show with full message history
  hide(): void; // Hide panel
  toggle(): void; // Toggle visibility
  clear(): void; // Clear all messages
  getIsVisible(): boolean;
  dispose(): void; // Clean up resources
}
```

### 7. Helper Overlays

Context-sensitive help and keyboard shortcuts.

**Components:**

- Keyboard shortcut reference
- Control mode indicators
- Tooltip system
- First-time user hints
- Loading indicators

The help overlay delays its document-level outside-click listener so the opening
click cannot immediately close it. That pending timer, the installed listener,
and the focus-trap release are all tracked and cancelled by `hideHelpOverlay()`;
the delayed callback also verifies that it still belongs to the currently mounted
overlay before attaching. Rapid `H` toggles therefore cannot arm stale handlers
that close or retain a subsequently opened panel.

On devices reporting touch points, the shortcut list also includes a `Touch`
section: the orbit, zoom, pan and roll gestures, then the finger equivalents of
the desktop pointer rows — tap picks an element (and follows its link when it
has one), press and hold opens whatever menu a right-click would, and double tap
recenters the camera the way `F` does. Its note calls out the ortho exception:
one finger pans there and twist does not roll. The section is filtered out for
non-touch devices, leaving desktop help text unchanged.

Initial focus goes to the overlay **container**, not its filter field. A focused
text input trips `InputHandler`'s typing guard, which drops every key but
`Escape` — that made `H` one-way (it opened the overlay but the second `H` was
swallowed as typing, issue #1922). `help-overlay/type-to-filter.ts` restores
type-to-filter by forwarding the first printable keystroke into the filter, and
the dataset browser (`O`) uses the same mechanism. The panel's own toggle key is
passed through to the global binding while focus is on the container, so it
cannot be the first character of a filter query (`Shift`+that key types it like
any other character — the global lookup spells the shifted form `"h+shift"`,
which matches no binding).

The same container listener keeps the panel **modal**: no panel pushes an
`InputContext`, so it stops every other key from reaching the global bindings
behind the dialog (`Home`/`End` would otherwise jump the selected dimension,
`Shift`+arrows change the animation speed). Containment applies to keys
bubbling up from inner controls too — a listing row, the filter, a breadcrumb —
so one `Tab` cannot hand the scene its shortcuts back. Only `Escape`, `Tab` and
the panel's declared passthrough keys get out, the last of those only while
focus is still on the container itself. Containment never calls
`preventDefault`, so panel scrolling and browser-level shortcuts (`Ctrl+F`,
`Cmd+W`, `F5`) still work.

### 8. Recording Panel

Screenshot and video capture panel with multiple export options.

**Features:**

- Screenshot export (PNG, WebP, JPEG, EXR) with configurable resolution
- Video recording (WebM, MP4, MKV) with turntable rotation mode
- Image-sequence ZIPs (PNG, WebP, JPEG) for offline turntable capture
- EXR-sequence ZIPs preserving HDR precision for compositing
- Transparent background support for compositing
- Dimension slider synchronization during recording
- Resolution multiplier for high-DPI exports

**Keyboard Shortcuts:**

- `T`: Toggle recording panel
- `G`: Quick screenshot

**CSS:** `styles/components/recording-panel.css`

### 9. Scale Bar

Physical scale bar overlay that automatically computes width from camera distance, FOV, and dimension units.

**Features:**

- Automatic width computation from camera/viewport parameters
- Physical unit display using scene dimension metadata
- Supports all Luxar physical units (nm, um, mm, cm, m, etc.)
- Auto-hide when no unit information is available

**Keyboard Shortcut:** `B` to toggle visibility

**CSS:** `styles/components/scale-bar.css`

### 10. Colormap Legend

Compact overlay showing each visible layer's colormap gradient, name, and data range.

**Features:**

- Reactive updates when layer state changes (colormap, visibility, range)
- Gradient rendering from built-in colormap LUTs
- Only shows layers that have a colormap assigned

**Keyboard Shortcut:** `J` to toggle visibility

### 11. Layers Panel

Napari-inspired per-layer control panel. See [`./layers/README.md`](./layers/README.md) for details.

**Features:**

- Visibility toggle per layer
- Display range / Colour range [min, max] with dual-thumb slider for scalar or direct RGB windows
- Gamma correction
- Blending mode (additive, volumetric, normal, max, opaque, luminous)
- Colormap selection
- Multi-select: Click, Ctrl+Click, Shift+Click

**Keyboard Shortcut:** `L` to toggle

The panel contains navigation and menu keys used by its own controls so they do
not also trigger scene actions. Because it is non-modal, unrelated viewer
shortcuts continue to work while a panel control has focus; Escape still closes.

---

## Theming System

### Overview

Luxar viewer now features a **modular theming system** with runtime theme switching. Styles are separated into external CSS files with CSS custom properties (variables) for easy customization.

### Available Themes

1. **Dark Theme** - Optimized for scientific visualization with subdued colors
2. **Light Theme** - Bright theme for well-lit environments, WCAG AA compliant
3. **Frosted Glass** (default) - Apple-inspired frosted glass design with translucent panels and heavy blur
4. **Liquid Glass** - Fluid glass aesthetic with translucent panels and subtle refraction effects

### Using Themes

**Programmatically**:

```typescript
import { ThemeManager } from '../themes/theme-manager';

// Switch themes
ThemeManager.getInstance().setTheme('dark');
ThemeManager.getInstance().setTheme('light');
ThemeManager.getInstance().setTheme('frosted-glass');
ThemeManager.getInstance().setTheme('liquid-glass');

// Get current theme
const current = ThemeManager.getInstance().getCurrentTheme();
console.log(current.name); // "Dark"

// Subscribe to theme changes
ThemeManager.getInstance().onChange((theme) => {
  console.log('Theme changed to:', theme.name);
});
```

**Via UI**:

- Open the Settings popover (gear button on the control rail)
- Select theme from the "Theme" dropdown
  (`rail-panels/settings-popover.ts` → `setupThemeControls`)

**Via URL**:

```
http://localhost:5173/?theme=dark
http://localhost:5173/?theme=light
http://localhost:5173/?theme=frosted-glass
http://localhost:5173/?theme=liquid-glass
```

### CSS Architecture

**Three-Layer System**:

```
Theme System (CSS variables)
    ↓
Component Styles (CSS files, BEM classes)
    ↓
Component Logic (TypeScript, no styling)
```

**Animation Consistency**: All UI panels use a consistent 0.15s fade-in animation for smooth, professional transitions.

**File Structure**:

```
src/styles/
├── index.css           # Main entry point
├── reset.css           # CSS reset
├── base/               # Base styles
│   ├── typography.css
│   ├── layout.css
│   └── utilities.css   # 80+ utility classes
└── components/         # Component-specific styles
    ├── colormap-legend.css
    ├── control-rail.css
    ├── data-loading-monitor.css
    ├── dataset-browser.css
    ├── debug-console.css
    ├── dimension-sliders.css
    ├── error-dialog.css
    ├── help-overlay.css
    ├── layers-panel.css
    ├── overlay-layer.css
    ├── performance-monitor.css
    ├── recording-panel.css
    ├── resolution-indicator.css
    ├── scale-bar.css
    ├── select-menu.css
    └── toast.css
```

**BEM Naming Convention**:

```css
/* Component */
.luxar-error-dialog {
}

/* Element */
.luxar-error-dialog__header {
}
.luxar-error-dialog__title {
}

/* Modifier */
.luxar-error-dialog--visible {
}
```

**CSS Custom Properties**:

```css
/* Colors */
--luxar-bg-primary, --luxar-bg-secondary, --luxar-bg-tertiary
--luxar-text-primary, --luxar-text-secondary, --luxar-text-muted
--luxar-success, --luxar-warning, --luxar-error, --luxar-info

/* Spacing — the key is HALF the pixel value (spacing-10 = 20px), and the
   keys are not contiguous: 0 1 2 3 4 5 6 8 10 12 16 20 */
--luxar-spacing-0 through --luxar-spacing-20

/* Typography */
--luxar-font-base, --luxar-font-mono
--luxar-text-xs through --luxar-text-4xl
--luxar-font-normal, --luxar-font-medium, --luxar-font-semibold, --luxar-font-bold

/* Effects */
--luxar-radius-sm, --luxar-radius-md, --luxar-radius-lg
--luxar-shadow-sm, --luxar-shadow-md, --luxar-shadow-lg
--luxar-blur-sm, --luxar-blur-md, --luxar-blur-lg
```

### Component Styling

Components now use **CSS classes** instead of inline styles:

**Before**:

```typescript
element.style.backgroundColor = 'rgba(30, 30, 30, 0.9)';
element.style.padding = '15px';
element.style.borderRadius = '8px';
// ... 50 more inline styles
```

**After**:

```typescript
element.className = 'luxar-dimension-sliders';
// All styling in CSS file
```

### Theme-Aware Components

These components fully support theming:

- ✅ Error Dialog (error-overlay.ts)
- ✅ Help Overlay (help-overlay.ts)
- ✅ Loading Indicator (loading-indicator.ts)
- ✅ Dimension Sliders (dimension-sliders.ts)
- ✅ Debug Console (debug-console.ts) - proper BEM naming (.luxar-debug-console)
- ✅ Dataset Browser (dataset-browser.ts)
- ✅ Data Loading Monitor (data-loading-monitor.ts) - CSS complete, core templates theme-aware
- ✅ Rendering Controls (rendering-controls.ts) - custom GUI library (ui/gui.ts, lil-gui drop-in replacement) theme integration via CSS variables

---

## Implementation Details

### Event Delegation Pattern

The UI components use event delegation for efficient event handling:

```typescript
// Instead of inline handlers:
// ❌ onclick="__luxarMonitor.expand()"

// We use data attributes:
// ✅ data-action="expand"

// Single event handler manages all interactions:
private handleUIEvent(event: Event): void {
  const target = event.target as HTMLElement;
  const action = target.dataset.action;

  switch (action) {
    case 'expand': this.expand(); break;
    case 'hide': this.hide(); break;
    // ... other actions
  }
}
```

**Benefits:**

- No global namespace pollution
- Better security (no inline JavaScript)
- Improved testability
- Type-safe event handling
- Single listener for multiple elements

### Component Lifecycle

All UI components follow a consistent lifecycle:

1. **Construction**: Initialize state and configuration
2. **Creation**: Build DOM elements with event delegation
3. **Updates**: Efficient DOM updates via requestAnimationFrame
4. **Disposal**: Clean up listeners and resources

## User Interactions

### Keyboard Controls

Global keyboard shortcuts managed by the UI system:

| Key       | Action                     | Context                  |
| --------- | -------------------------- | ------------------------ |
| `H`       | Toggle help                | Global                   |
| `P`       | Toggle performance monitor | Global                   |
| `R`       | Toggle rendering controls  | Global                   |
| `N`       | Toggle dimension sliders   | When nD data loaded      |
| `M`       | Cycle data loading monitor | Global                   |
| `T`       | Toggle recording panel     | Global                   |
| `G`       | Quick screenshot           | Global                   |
| `B`       | Toggle scale bar           | Global                   |
| `J`       | Toggle colormap legend     | Global                   |
| `L`       | Toggle layers panel        | Global                   |
| `Ctrl+L`  | Toggle debug console       | Development mode         |
| `Esc`     | Close active panel         | Any panel open           |
| `K`       | Toggle dimension animation | When dimension selected  |
| `Home`    | Jump to dimension start    | When dimension selected  |
| `End`     | Jump to dimension end      | When dimension selected  |
| `Shift+↑` | Increase animation speed   | When dimension animating |
| `Shift+↓` | Decrease animation speed   | When dimension animating |

### Mouse Interactions

- **Drag**: Move panels around screen
- **Scroll**: Adjust slider values with precision
- **Right-click**: Context menus
- **Double-click**: Reset to default values

---

## Usage Examples

### Complete UI Setup

```typescript
import { RenderingControls, PerformanceMonitor } from './ui';
import { DimensionSliders } from './ui/dimension-sliders';
import { DatasetBrowser } from './ui/dataset-browser';
import { DataMonitorManager } from './ui/data-monitor-manager';
import { sceneDimsManager } from '../scene/scene-dims-manager';

// Initialize UI components
const rendering = new RenderingControls(postProcessing);
const performance = new PerformanceMonitor();
const browser = new DatasetBrowser({
  container: document.body,
  onDatasetSelect: (url) => loadDataset(url),
});

// Dimension sliders are constructed once dims are known
const sliders = new DimensionSliders({
  container: document.body,
  dims: sceneDimsManager.getDims(),
  dimensionRanges: sceneDimsManager.getDimensionRanges(),
  dimensionNames: sceneDimsManager.getDimensionNames(),
});

// Data loading monitor is automatically created by SceneLoader.
// Access it via the manager (already connected to loaders, M to cycle):
const monitor = DataMonitorManager.getInstance().getDefaultMonitor();

// Slider changes flow through sceneDimsManager — no per-component
// `.on('change')` wiring is needed.
```

### Responsive UI Updates

```typescript
// Update UI when a new dataset loads
function updateUIForDataset(dataset) {
  // Dimension sliders are rebuilt from the new dims (construct a fresh
  // DimensionSliders) and shown only when the data has >3 dimensions.
  if (dataset.ndim > 3) {
    sliders.setVisible(true);
  } else {
    sliders.hide();
  }

  // Apply rendering defaults from the dataset's zarr viewer config
  rendering.setZarrViewerConfig(dataset.viewerConfig);
  rendering.updateSceneScale();
}
```

---

## Performance Considerations

### Rendering Optimization

1. **Virtual scrolling** for long lists
2. **Debounced updates** for sliders
3. **RAF-based animations**
4. **CSS transforms** for movement
5. **Will-change** for animated properties

### Memory Management

```typescript
class UIComponent {
  dispose() {
    // Remove event listeners
    this.removeEventListeners();

    // Clear references
    this.elements = null;

    // Remove from DOM
    this.container.remove();
  }
}
```

---

## Configuration

### UI Settings

```typescript
const UI_CONFIG = {
  // Appearance
  theme: 'dark',
  fontSize: 14,
  animations: true,

  // Layout
  panelOpacity: 0.95,
  panelBlur: 10,
  cornerRadius: 8,

  // Behavior
  autoHideDelay: 3000,
  doubleClickReset: true,
  persistState: true,

  // Performance
  updateThrottle: 16, // 60 FPS
  debounceDelay: 100,
};
```

### Customization

Override default styles:

```css
/* Custom theme */
.luxar-ui-panel {
  --panel-bg: #2a2a2a;
  --panel-border: #4a4a4a;
  --accent-color: #00ff88;
}
```

---

## API Reference

### DimensionSliders

Constructed with `SliderConfig` (`container`, `dims`, `dimensionRanges`,
`dimensionNames`, optional `dimensionUnits`, optional `selectedDimension`).
Slider movements propagate through `sceneDimsManager.setDimensionValue()`
rather than a local event emitter, so there is no `on()` / `setDimensions()` /
`setValue()` API.

| Method                         | Description                                   |
| ------------------------------ | --------------------------------------------- |
| `setAnimationManager(manager)` | Connect animation manager and build controls  |
| `setSelectedDimension(slot)`   | Set the dimension the `[` / `]` keys target   |
| `toggle()`                     | Toggle slider panel visibility                |
| `setVisible(visible)`          | Explicitly show/hide the panel                |
| `hide()`                       | Hide the panel                                |
| `getIsVisible()`               | Get current visibility state                  |
| `update()`                     | Refresh slider thumbs from `sceneDimsManager` |
| `updateStatusBar()`            | Refresh the status/title display              |
| `dispose()`                    | Clean up listeners and DOM                    |

### RenderingControls

| Method                               | Description                                         |
| ------------------------------------ | --------------------------------------------------- |
| `show()` / `hide()` / `toggle()`     | Control panel visibility                            |
| `isVisible()`                        | Check if panel is visible                           |
| `setAnimationController(controller)` | Connect animation controller for render triggers    |
| `setAdaptiveDPRManager(manager)`     | Connect adaptive DPR manager for performance UI     |
| `setSceneId(url, name?)`             | Set scene ID for settings persistence               |
| `setZarrViewerConfig(config)`        | Apply viewer config defaults from zarr metadata     |
| `syncCurrentState()`                 | Sync UI controls with current post-processing state |
| `syncCameraFovState()`               | Sync live camera FOV/preset state and controls      |
| `updateSceneScale()`                 | Update scale-dependent controls after scene load    |
| `toggleCinematicMode()`              | Toggle cinematic mode (bloom + ACES + vignette)     |
| `dispose()`                          | Clean up resources and event listeners              |

**Architecture**: RenderingControls uses a modular setup architecture where each category of controls (camera, HDR, anti-aliasing, post-processing) is initialized by a dedicated setup module in `./rendering-controls/setup/`. Two further setup modules live there but are consumed by the rail popovers instead: `performance-setup.ts` (rail-panels/performance-popover.ts) and `theme-setup.ts` (rail-panels/settings-popover.ts). This improves maintainability and keeps files under token limits. See [`./rendering-controls/README.md`](./rendering-controls/README.md) for details.

### PerformanceMonitor

Constructed with an optional `keepAlive` hook (`{ request, release }`). Frame
timing is driven automatically via the `frame-start` / `frame-end` event bus
while visible — there is no public `begin()` / `end()` to call. The widget is
mounted by its owner (the control rail) via `.element`.

| Method / property | Description                                     |
| ----------------- | ----------------------------------------------- |
| `element`         | The widget DOM node (getter) — caller mounts it |
| `toggle()`        | Toggle visibility                               |
| `show()/hide()`   | Explicit show/hide                              |
| `cycleMode()`     | Cycle the metric: FPS → ms → graph              |
| `visible`         | Get current visibility state (getter)           |
| `dispose()`       | Clean up resources                              |

### DatasetBrowser

| Method   | Description                    |
| -------- | ------------------------------ |
| `show()` | Show the dataset browser panel |
| `hide()` | Hide the dataset browser panel |

Constructor takes `DatasetBrowserConfig` with `container`, an `onDatasetSelect` callback that may return `false` to keep the browser open, and an optional `onClose` callback. Navigation is internal.

### DataLoadingMonitor

| Method                           | Description                                      |
| -------------------------------- | ------------------------------------------------ |
| `connectLoader(path, loader)`    | Connect a loader for monitoring                  |
| `disconnectLoader(path)`         | Disconnect a specific loader                     |
| `disconnectAllLoaders()`         | Disconnect all loaders (scene change)            |
| `show()/hide()/toggle()`         | Control visibility                               |
| `cycleState()`                   | Cycle through hidden→mini→expanded               |
| `expand()/minimize()/collapse()` | Control expanded state                           |
| `getGlobalStats()`               | Get aggregated statistics                        |
| `getLoaderMetrics(path)`         | Get metrics for specific loader                  |
| `getRecommendations()`           | Get optimization recommendations                 |
| `setActiveTab(tab)`              | Switch tab (overview/cache/performance/insights) |
| `dispose()`                      | Clean up resources                               |

---

## Best Practices

### Component Design

1. **Keep components focused**: Single responsibility
2. **Use composition**: Build complex UIs from simple parts
3. **Maintain consistency**: Follow design system
4. **Optimize updates**: Batch DOM changes
5. **Handle edge cases**: Empty states, errors

### User Experience

1. **Provide feedback**: Loading states, confirmations
2. **Be responsive**: Immediate visual feedback
3. **Guide users**: Tooltips, hints, documentation
4. **Remember preferences**: Persist user settings
5. **Support undo**: Allow reverting changes

---

## Troubleshooting

### Common Issues

**Problem: Panels not visible**

- Check z-index conflicts
- Verify panel state in localStorage
- Ensure container element exists

**Problem: Sliders not responding**

- Check input event listeners
- Verify dimension data is valid
- Check for JavaScript errors

**Problem: Performance monitor inaccurate**

- Confirm the animation loop emits `frame-start` / `frame-end` bus events
- Check for blocking operations
- Verify RAF timing

---

## License

Part of the Luxar project. See root LICENSE file for details.

---

_For implementation details, see the source files in this directory._

---

## Subpackages

Each sibling folder holds the private helpers for the public-API file
of the same name at this folder's root (plus the folder-module
`rail-panels/`). Most have their own README:

- [`control-rail/`](./control-rail/README.md) — Internals for
  `control-rail.ts`, the always-visible left-edge activity rail
  (`RailOverlay` flyout/popover lifecycle, `RAIL_ICONS`, DOM helpers,
  item types); one button per panel, wired to the exact commands the
  keyboard shortcuts fire. See §0 above.
- [`data-loading-monitor/`](./data-loading-monitor/README.md) — Internals
  for `data-loading-monitor.ts` (templates, advisor, event queue,
  polling loop, timing panel; `metrics/` and `tabs/` helpers).
- [`dataset-browser/`](./dataset-browser/README.md) — URL utilities
  (`extractBaseUrl` / `extractPath`) for `dataset-browser.ts`.
- [`debug-console/`](./debug-console/README.md) — Output formatters
  (`@timestamp`, `[stream]`, etc.) for `debug-console.ts`.
- [`dimension-sliders/`](./dimension-sliders/README.md) — Pure
  value/fraction/wrap math helpers for `dimension-sliders.ts`.
- [`gui/`](./gui/README.md) — Custom GUI library implementation
  (`GUI`, `Folder`, `Controller`, per-type controllers, DOM plumbing,
  formatting); `gui.ts` re-exports only the default `GUI` and `Controller`,
  with the rest imported directly from leaf modules under `ui/gui/`.
- [`help-overlay/`](./help-overlay/README.md) — Shared modal-panel
  helpers: `focus-trap.ts` (Tab/Shift+Tab focus cycling) used by
  `help-overlay.ts`, `error-overlay.ts` and `dataset-browser.ts`, and
  `type-to-filter.ts` (container focus + first-keystroke filtering) used
  by the two filtered panels.
- [`layers/`](./layers/README.md) — Layers panel implementation
  (`LayersPanel`, `LayerStateManager`, range/labeled sliders); `layers.ts`
  re-exports only `LayersPanel`, with the rest imported directly from leaf
  modules under `ui/layers/`.
- [`overlay-widgets/`](./overlay-widgets/README.md) — Shared
  `UIComponent` base class for screen-space overlay widgets (scale bar,
  colormap legend).
- [`panels/`](./panels/README.md) — Reserved namespace for a future
  shared panel framework. Currently empty.
- [`rail-panels/`](./rail-panels/README.md) — Rich control popovers
  hosted by the control rail (Settings, Performance, Navigation, Home
  builders plus the shared `makePopoverGui` helper).
- [`recording-panel/`](./recording-panel/README.md) — Recording panel
  internals (capture drivers, GUI construction, media utilities,
  overlay compositor, sequence/ZIP exporters).
- [`rendering-controls/`](./rendering-controls/README.md) — Rendering
  controls internals (per-category `setup/` modules, focus manager,
  cinematic mode, settings persistence, clipping/FOV utilities).
