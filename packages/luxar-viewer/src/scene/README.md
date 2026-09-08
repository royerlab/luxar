# Luxar Scene Package

> Core 3D scene management and orchestration for nD scientific visualization

## Overview

The Luxar Scene package provides comprehensive scene management, animation control, and nD dimension coordination for complex multi-geometry visualizations (points, lines, Gaussian splats). It handles the THREE.js scene graph, camera management, rendering pipeline integration, and multi-dimensional data navigation.

### Key Features

- **Scene Graph Management**: Hierarchical 3D scene organization
- **Animation Control**: Smart rendering loop with idle detection
- **nD Dimension Support**: Coordinate system for n-dimensional data
- **Camera Management**: Integrated control system orchestration
- **Bounding Box Computation**: Automatic scene bounds calculation
- **Center Point Management**: Native vs bounding box centering

### Package Architecture

```
scene/
├── scene-manager.ts                # Main scene orchestrator
├── scene-manager/                  # Focused helpers owned by SceneManager
│   ├── camera/                     # camera-framing, camera-setup, camera-materials, camera-mode
│   ├── clipping/                   # bounds-math, scene-bounds-cache, clipping-policy
│   ├── render-pipeline/            # renderer-setup, post-processing-setup, scene-disposal, webgl-context-recovery
│   └── viewport/                   # dpr-policy, resize-orchestrator
├── animation/                      # animation-controller, committed-quality, dimension-animation-manager
├── dims/                           # Pure nD step and dimension-selection helpers
├── scene-dims-manager.ts           # nD dimension coordination
├── dimension-loading.ts            # Current-slice loading + playback prefetch
├── lod-group-registry.ts           # Per-frame LOD-group selector (policy/state machine)
├── lod-selector-math.ts            # Selector math: world-box fold, box→area/diagonal projections, hysteresis pick
├── lod-blend.ts                    # Pure opacity math: coverage cross-fade + energy compensation
├── lod-fade.ts                     # Material-level fade appliers (clone-on-first-fade)
├── lod-eviction.ts                 # VRAM-budget LRU eviction policy for LOD levels
├── lod-display-gate.ts             # Never-downgrade display gate (energy-threshold release)
├── lod-freshness.ts                # Pure LOD freshness + settle helpers for the registry
├── synthetic-scene.ts              # Synthetic perf-bench scene generators (lines, points, gsplats)
└── README.md                       # This documentation
```

---

## Components

### 1. Scene Manager

The `SceneManager` is the central hub for all 3D scene operations.

**Responsibilities:**

- THREE.js scene initialization
- Camera and renderer setup
- Control system integration
- Scene graph manipulation
- Bounding box calculations
- Center point management

**Core Features:**

```typescript
class SceneManager extends THREE.EventDispatcher {
  // Scene setup (populated by init())
  scene: THREE.Scene;
  camera: LuxarCamera; // PerspectiveCamera | OrthographicCamera
  renderer: Renderer; // WebGLRenderer or WebGPURenderer
  capabilities: RendererCapabilities;
  controls: ControlsManager;
  postProcessing: PostProcessingManager;

  // Lifecycle
  init(options: {
    canvas: HTMLCanvasElement;
    debug?: boolean;
    renderer?: 'webgl' | 'webgpu';
    webgpuForceWebGL?: boolean;
    perfTimestamp?: boolean;
  }): Promise<void>;
  loadSceneData(
    src: string,
    loaderConfig?: LoaderConfig,
    options?: SceneLoadOptions
  ): Promise<void>;
  updateSize(): void;
  dispose(): void;

  // Camera control
  updateFOV(delta: number): void;
  updateClippingPlanes(near: number, far: number): void;
  autoAdjustClippingPlanes(): { near: number; far: number };
  setDynamicClipping(enabled: boolean): void;
  getDynamicClippingState(): { enabled: boolean; near: number; far: number };
  setControlType(type: 'orbit' | 'fly' | 'ortho'): void;
  getControlType(): ControlType;

  // Centering
  centerCameraOnScene(): void;
  toggleCentering(): void;
  getCurrentCenter(): THREE.Vector3;

  // WebGL context-loss
  isWebGLContextLost(): boolean;
}
```

Scene mutation goes through the standard Three.js API on the
`scene` field (e.g. `sceneManager.scene.add(group)`); SceneManager
itself does not expose a wrapping `addToScene` method.

**Usage Example:**

```typescript
const canvas = document.getElementById('app') as HTMLCanvasElement;
const sceneManager = new SceneManager();
await sceneManager.init({ canvas });

// Add objects directly through the THREE.Scene field
sceneManager.scene.add(pointCloud);

// Update camera FOV
sceneManager.updateFOV(10); // Increase FOV by 10 degrees

// Toggle between native and bbox center
sceneManager.toggleCentering();
```

### 2. Animation Controller

The `AnimationController` manages the render loop with intelligent idle detection.

**Features:**

- Automatic pause when scene is static
- Performance monitoring integration
- Event-driven animation triggers
- Configurable idle timeout
- Frame timing management

**Animation States:**

```typescript
// Start animation (triggered by events)
animationController.startAnimation();

// Animation automatically pauses after idle timeout
// Default: 2000ms of no activity

// Manual control
animationController.stopAnimation();
animationController.startAnimation();
```

**Render Loop:**

```typescript
private animate = (): void => {
  if (!this.isAnimating) return;

  // Schedule the next frame: a bare requestAnimationFrame, or — once
  // consecutive frames have been pathologically slow — a bounded cooldown
  // first, so the main thread gets a slot (#1724). See
  // `animation/animation-controller.ts`.
  this.scheduleNextFrame();

  // Update controls
  this.sceneManager.controls.update();

  // Update performance stats
  this.performanceStats?.begin();

  // Render scene
  this.postProcessing.render();

  this.performanceStats?.end();

  // Check for idle timeout
  this.checkIdleTimeout();
}
```

### 3. Scene Dimensions Manager

The `SceneDimsManager` coordinates n-dimensional navigation across all scene objects.

**Capabilities:**

- Unified dimension system for all nD objects
- Dimension metadata management
- Slice position coordination
- Step size calculations
- Display status tracking

**Dimension Structure:**

```typescript
interface SimpleDims {
  ndim: number; // Total dimensions
  displayed: number[]; // Indices of displayed dims (e.g., [0,1,2])
  currentStep: number[]; // Current position in each dimension
  metadata?: DimensionMetadata[]; // Names, units, ranges
}

interface DimensionMetadata {
  name: string; // e.g., "time", "x", "wavelength"
  unit: string; // e.g., "ms", "μm", "nm"
  scale: number; // Array-index → real-world units
  range?: [number, number]; // Optional min and max values
  step?: number; // Step size for navigation
  discrete?: boolean; // Whether dimension is discrete
  cyclic?: boolean; // Whether dimension wraps (angles, etc.)
  spatial?: boolean; // Whether points extend through this dim
  categories?: string[]; // Optional categorical labels
  display?: boolean; // Show in the 3D scene by default
  description?: string; // UI tooltip
}
```

**Usage:**

```typescript
// Initialize from scene
sceneDimsManager.initFromScene(scene);

// Get unified dimensions
const dims = sceneDimsManager.getDims();

// Navigate dimensions
sceneDimsManager.setDimensionValue(dimIndex, value);

// Listen for changes
sceneDimsManager.addListener(() => {
  updateVisualization();
});
```

### 4. Dimension Animation Manager

The `DimensionAnimationManager` provides automated playback through dimension ranges.

**Responsibilities:**

- FPS-based animation control
- Loop mode management (once, loop, bounce)
- Per-dimension animation state
- Frame throttling and measurement
- Integration with AnimationController

**Key Features:**

```typescript
class DimensionAnimationManager extends THREE.EventDispatcher {
  // Playback control
  play(dimIndex: number, options?: PlayOptions): boolean;
  pause(dimIndex: number): boolean;
  togglePlay(dimIndex: number): boolean;
  stop(dimIndex: number): void;

  // Speed control
  setTargetFPS(dimIndex: number, fps: number): void;
  increaseSpeed(dimIndex: number): void;
  decreaseSpeed(dimIndex: number): void;

  // Configuration
  setLoopMode(dimIndex: number, loopMode: LoopMode): void;

  // State queries
  isAnimating(dimIndex: number): boolean;
  getState(dimIndex: number): DimensionAnimationState | undefined;

  // Lifecycle
  dispose(): void;
}

interface PlayOptions {
  targetFPS?: number; // Default: 10
  loopMode?: 'once' | 'loop' | 'bounce'; // Default: 'loop'
  direction?: 'forward' | 'backward'; // Default: 'forward'
}
```

**Usage:**

```typescript
// Create animation manager
const animManager = new DimensionAnimationManager(sceneDimsManager, animationController);

// Start animating time dimension at 10 FPS
animManager.play(3, { targetFPS: 10, loopMode: 'loop' });

// Adjust speed
animManager.increaseSpeed(3); // Next preset: 15 FPS
animManager.setTargetFPS(3, 30); // Set exact FPS

// Change loop mode
animManager.setLoopMode(3, 'bounce'); // Reverse at boundaries

// Listen for events
animManager.addEventListener('play', (e) => {
  console.log(`Dimension ${e.dimIndex} started`);
});

animManager.addEventListener('complete', (e) => {
  console.log(`Dimension ${e.dimIndex} finished (loop: once)`);
});

// Pause/stop
animManager.pause(3);
animManager.stop(3); // Also removes state
```

**Animation Modes:**

- **Loop**: Wrap to start when reaching end (continuous playback)
- **Once**: Stop at end boundary (single pass)
- **Bounce**: Reverse direction at boundaries (ping-pong)

**FPS Presets:** 1, 2, 5, 10, 15, 30, 60 Hz

### 5. LOD Group Registry

The `LODGroupRegistry` is the per-frame selector for `lod_group`
scene-graph nodes. For each registered group it picks which child
(level of detail) renders, kicks deferred geometry loads for lazy
levels, and bounds resident VRAM with an LRU eviction pass.

**Per-frame algorithm** (one entry):

1. Fold each child's nD `positionBounds` into a cached local-space
   `BoundingBox`, mapping nD axes onto X/Y/Z via the current
   `displayDims`.
2. Lift that box to world space through the group's `matrixWorld`
   (`transformBoundingBox`). When any child publishes `lodBounds`, fold
   a second box the same way for metric sizing, falling back per child
   to `positionBounds`.
3. **Off-screen gate**: if the complete-geometry world box is entirely
   outside the camera frustum, hold the group at its coarsest _ready_
   level instead of loading a fine level the renderer would
   frustum-cull. Eviction also keeps using this complete-geometry box.
4. Otherwise, project the 8 corners of the metric world box to NDC and
   measure how much of the screen the group covers, in the units its
   `selector` attr names. A DERIVED ladder stamps `screen-area`: the
   metric is the screen-space AABB's **area** as a fraction of the viewport area
   (`projectBoxAreaFraction` — each NDC axis spans 2, so the fraction is
   the product of the per-axis half-extents after clipping to the
   viewport, viewport-size independent by construction and topping out
   at exactly `1.0` for any finite projection; sub-pixel-thin content
   ramps to its linear span instead, so an edge-on plane is not pinned
   to the coarsest level). Those thresholds are literal area fractions,
   so nothing is normalised: a whole-object ladder anchors its finest at
   `0.5` (half the screen occupied) and steps one level coarser per
   halving of occupied area, while a partition-bound one anchors at
   `1.0` (the tile alone fills the screen). The LEGACY `coverage`
   selector — pre-v3.4 stores and explicitly authored
   `coverage_fractions=[...]` lists — measures the pixel diagonal of
   that AABB instead (`projectBoxDiagonalPx`) and normalises it to a
   dimensionless coverage metric
   (`diagonalPx / (FILL_FACTOR * fittedAxisPx)`, `FILL_FACTOR = 0.5`,
   `fittedAxisPx = min(viewport.width, viewport.height)` — the extent
   `calculateCameraDistance` actually fits), so its finest anchor
   (`coverage_fraction` 1.0) is reached once the projected diagonal is
   half of the fitted screen axis. Both projections are `w`-aware: if
   any corner is at/behind the camera near plane (camera inside or
   straddling the box), they return `+Infinity` so the selector
   saturates to the finest level — instead of the collapsed/garbage
   value an unguarded perspective divide would produce on close
   approach. Under an ORTHOGRAPHIC projection nothing degenerates (`w`
   stays 1), so neither function ever saturates and each metric's plain
   value is used directly.
5. Pick the finest child whose `coverageFraction` threshold (the
   per-child value read from the zarr attr `coverage_fraction`, in
   whichever units step 4's `selector` names) is satisfied by that
   metric, with 10% asymmetric, spacing-aware hysteresis on the
   downgrade direction to suppress threshold-edge flicker
   (`pickChildWithHysteresis`).
6. Swap visibility atomically when the desired child differs; lazy
   targets that are not yet committed kick `ensureLoaded()` and swap
   on a later frame once `ready` flips true — unless the
   **hidden-layer load gate** vetoes it (below).
7. **Hidden-layer load gate**: no deferred load (initial, settled
   reload, or cross-fade partner pre-load) is _started_ while the
   group is not effectively visible — its own `visible` flag or any
   ancestor's is `false` (`isEffectivelyVisible` in
   `utils/object-visibility.ts`). A layer authored `visible=false`, or
   toggled off in the layers panel, hides the LAYER object while the
   levels underneath keep their own flags, so without the gate the
   selector kept fetching, decoding and committing multi-million-element
   levels that cannot be drawn — competing with the visible layer for
   the shared fetch gate, the worker pool and VRAM (measured: roughly
   double the scene load time). The gate only stops _starting_ work:
   nothing already resident is unloaded, and the eager `default_level`
   still loads at scene-load time so the layer paints instantly when
   shown. It is re-evaluated every frame, so the panel's eye toggle
   (`applyVisibility` → `requestRender`) resumes loading on the next
   frame. An explicit user retry
   (`retryLazyChildByNodePath`) deliberately bypasses it.
   A latched archive fault stops every automatic deferred kick through the owning
   `SceneLoader` latch. The branch that observed it retains its authored node
   path beside the anonymous placeholder, so the loading monitor can show the
   missing branch and Retry can re-kick that exact child without assigning a
   duplicate name/kind to the placeholder.
8. **Never-downgrade display gate**: a lazy level flips `ready` after
   its _first_ additive chunk commits, so an ungated swap to a
   fresh-but-still-streaming aspiration would pop displayed quality
   down to chunk-1 (on zoom in, zoom out, or after a scrub settles)
   and climb back. The registry holds the previously-displayed level
   while the streaming aspiration's committed geometry is strictly
   worse than what is shown (`shouldHoldPreviousDisplay` in
   `lod-display-gate.ts`), releasing on ladder completion (the
   commit-time `committedLadderComplete` stamp — committed, not just
   fetched), the **committed-energy threshold** (quality-stamped
   datasets: the aspiration's committed prefix carries ≥ 60% of its
   total self-energy — `committedEnergyFraction` ≥
   `ENERGY_RELEASE_THRESHOLD`, the primary and much earlier release;
   energy-ordered ladders front-load energy, so this fires chunks
   before raw counts cross), count crossover (the unstamped-dataset
   fallback), ladder failure, or the previous level losing freshness.
   Bypassed for explicit level locks and off-screen groups; a group
   with nothing better on screen still swaps at first paint.
   **Nested-group levels** (a child that is a whole subtree, e.g. the
   `overview` recipe's fine `kind=partition` branch, or arbitrary
   lod/partition nestings) participate through
   `subtreeDisplayProgress`, which folds the subtree's _visible_
   stamped leaves into one aggregate (count sum / all-complete /
   all-fresh / `reference_energy`-weighted committed energy, with
   known-empty leaves excluded and any missing stamp poisoning the
   energy aggregate to null → count fallback) — inner lod_groups' own
   level toggling shapes the
   aggregate to exactly what would render. Deferred-group activation
   also kicks the sweep refinement orchestrator
   (`SceneLoader.kickRefinementIfIdle`) so the freshly-registered part
   ladders stream to completion instead of stalling at chunk-1 until
   the next slice change. Activation is single-shot: those nested leaves
   resolve later staleness through the ordinary update sweep, while re-firing
   the group loader would attach a duplicate copy of the subtree.

**Selector modes:**

```typescript
// Auto (view-driven) — the default after register()
registry.setSelectorMode(path, 'auto');

// Lock to a specific level (0-based, coarsest→finest).
// Out-of-range lockLevel is clamped into [0, n-1] with a warning.
registry.setSelectorMode(path, { lockLevel: 2 });
```

**Capture quiescence:** `registry.isCaptureQuiescent()` answers "is
every lod_group and partition part that contributes pixels to the current
view already showing final committed quality?" — i.e. would one more
frame of waiting improve what is on screen. The offline turntable
capture drains on it (bounded) before exporting each frame, because the
rAF loop — and therefore this frustum-aware selector — runs for the
whole sweep: a tile that leaves the frustum mid-orbit is demoted, may
have its fine level released by the byte budget, and reloads
asynchronously on re-entry, which a one-rAF-per-frame capture would
otherwise film at the coarse level (#1695). Off-screen entries are
skipped (they draw nothing and are held coarse on purpose), and so are
entries that are not _effectively_ visible — a layer toggled off or
authored `visible=false` anywhere up the ancestor chain. That second
skip is load-bearing, not tidiness: a hidden group cannot start a
deferred load (`kickDeferredLoadIfVisible` refuses), while the
frustum-only selector still records a fine `desiredChildIndex` for it,
so blocking on it would never resolve and every capture frame would
burn its whole drain budget. An entry whose aspiration index holds no
child is skipped for that same reason — `children` is empty when every
level failed its `getObjectByName` attach at load — since nothing at a
missing index can ever become ready. An entry blocks while its displayed
level differs from the aspiration, while the aspiration is not ready /
not fresh / still streaming additive LODs, while any of its children is
`loading`, or while the selector's `desiredChildIndex` differs from the
aspiration **at all** — a finer level it has not got yet, but equally a
coarser one the aspiration has not moved onto. That last clause is the
subtle one: `activeChildIndex` only advances onto a READY level, so in
the frame where a lazy level's load lands (`ready` true, `loading`
cleared) the swap has not happened yet — a predicate reading only
ready/loading/displayed would call that window settled. A `desired`
level that has `failed` does not block, since it can never become ready
this frame.

Visible partition parts additionally block while a frustum rising edge is
pending, while their targeted load pass is queued or committing, while their
stamped leaves describe an older view version, or while a committed
progressive ladder is incomplete. Hidden/re-culled parts do not block, and an
archive fault skips waits that cannot make progress. A per-part loader failure
without an archive fault can leave a stale stamp, so the bounded capture timeout
and consecutive-timeout latch provide the escape instead of reporting the part
quiescent.

"Still streaming additive LODs" takes **both** available signals, and
either one alone leaves a hole. A lazy child's live `hasMoreLODs()` thunk
answers first. Then the commit-time `committedLadderComplete` stamp,
surfaced as `subtreeLadderComplete` through the registry's
`childFreshAndCount` — read off the child itself for a tracked **leaf**,
and folded over the visible stamped leaves
(`subtreeDisplayProgress().complete`) for a deferred **group** child, i.e.
a nested `kind=partition` / `kind=lod` subtree such as the `overview`
recipe's fine branch. Neither is sufficient on its own: a group child
carries no thunk, so without the fold the drain released the frame the
instant the fine branch became ready, with its part leaves at chunk-1 by
construction; and `load-lod-group-node` attaches the thunk only on the
DEFERRED path, so without the stamp the eagerly-loaded default level —
still climbing its ladder under the sweep-driven background refinement
loop — read as complete. Either way the frame is exported at the first
additive chunk and refines in over the next seconds, which is the exact
artifact class #1695 exists to remove. Anything carrying no stamp at all
(never committed, or a non-progressive loader) counts as complete and
never blocks.

Forcing the finest level instead was rejected: a capture
visits the whole scene, so peak residency would be the entire dataset.

**Retention + VRAM budget:** swaps never `release()` outgoing
geometry — retention keeps loaded levels resident so swapping back is
a sub-millisecond visibility toggle. Memory is bounded once per frame
by `enforceResidentByteBudget`, which (only when the GPU pool's live
resident byte total exceeds the shared budget) demotes evictable
levels — hidden-layer (undrawable) first, then off-screen, then
furthest-from-camera, then coldest-`lastVisibleTick`. A resident level
that has never been displayed is back-filled with the coldest sentinel
so it ages ahead of every level the user has actually seen. The visible
level of each group and eager
fallback levels (no `release` thunk) are never evicted. "Visible" here
is **effective** visibility (`isEffectivelyVisible`): under a hidden
ancestor — a layer toggled off — nothing of that group is on screen, so
all of its ready levels are ordinary cold candidates, including the
one the selector nominally displays.

**Partition frustum gating + targeted resync:** a `kind=partition`
group registers through `registerPartition`, and every frame each part
is tested against a frustum padded by `PARTITION_FRUSTUM_MARGIN` (10 %,
so a cold part preloads just before it enters). A part outside it is
hidden and stamped `userData.partitionFrustumVisible = false` — on
EVERY object the part emitted, since one part node may produce several,
and a part counts as re-entering when any of them was culled; the
scene loader's sweep (`isPartitionPathVisible`) and refinement skip its
loaders, dropping their predictive-prefetch baseline. Because a culled
part misses slice updates, its RE-ENTRY requests a resync of exactly
that part's loaders — `deps.requestReprocess(partPaths)` with the
re-entering parts' registered node paths (the wrapper path when a part
has none), coalesced per wrapper across frames and held while a view
pass is in flight or queued (`isLoadPassInProgress`; a refinement hold
does not count — the loader parks a resync that lands during one and
cancels the hold into a targeted pass). The loader runs that resync
under the **unchanged view version**: its `updateView` bumps `currentViewVersion`
only when a query determinant changes (`viewStatesEqual`). Lazy fine
levels never join the sweep and are never re-stamped by it, so a bump on
an unchanged view read every resident fine level scene-wide as stale and
dropped ALL groups to coarse on camera motion (the #2366 regression on
the 44-part h2afva scene).

**Wiring:** the SceneLoader instantiates one registry per scene and
hooks `evaluatePerFrame()` into `AnimationController` alongside the
dynamic-clipping callback. The injected `LODGroupRegistryDeps` supply
the camera, viewport size, `displayDims`, the partition resync hooks
(`requestReprocess`, `isUpdateInProgress`), and the optional resident
byte budget / measurement — omitting the budget accessors yields pure
retention (the unit-test default). `projectBoxAreaFraction`,
`projectBoxDiagonalPx` and `pickChildWithHysteresis` are exported as
pure functions for testing.

### 6. Projected-Density Guard

`projected-density.ts` measures, once per frame and per committed emissive data
mesh (points, lines, and gsplats; shaded triangle meshes are excluded), how
many visible elements land on each **drawing-buffer** pixel of the node's
projected bounding sphere (`elementsPerPixel`; buffer pixels, not CSS pixels,
so the guard never fights the adaptive-DPR controller). Frame cost on
element-dense views tracks that density, not the pixel count: the 2026-09
audit measured a 1.5 M-point example framed into ~1 600 px at 42 ms per frame
at DPR 1 and 83 ms at DPR 0.5, while the same points dollied 4× closer ran at
120 fps. Records are exposed through `__luxarDebug.getPerf().density`.

`density-guard.ts` consumes those records through the tracker's `onVisit`
hook and thins over-dense nodes on the GPU: every leaf material (points /
lines / gsplats, GLSL and TSL, visual and picking) carries a `uDensityDrop`
uniform, and the vertex stage discards an element when a hash of its
ordering-resolved storage index falls below the dropped fraction
(`luxarDensityDropped()` / `densityDroppedNode()`). The keep fraction is a
quantised ladder (1, 1/2, 1/4, … `config.densityGuard.minKeepFraction`) with
hysteresis (`enterRatio` / `leaveRatio` around `capElementsPerPixel`), and
every step change is reported to the adaptive-DPR controller as a content
change. Only the blendable modes (additive / luminous / volumetric) are
thinned, and `applyLodFade` multiplies the node's opacity by `1/keep`
(`densityCompensation`) so the composited brightness stays at the unthinned
aggregate; `max` / `normal` / `opaque` nodes are never thinned. The pick pass
mirrors the visual material's drop per node so a thinned-away element cannot
be picked. `?no-density-guard` disables both the walker and the ladder for a
session.

---

## Scene Graph Structure

### Hierarchy

```
Scene (THREE.Scene)
├── Ambient Light
├── Data Groups
│   ├── Point Cloud Groups
│   │   ├── Transform Group
│   │   │   └── Points (THREE.Points)
│   │   └── Direct Points
│   └── Lines Groups
│       └── Lines (THREE.Mesh with InstancedBufferGeometry)
├── Camera (managed separately)
└── Helper Objects (grids, axes, etc.)
```

**Note:** Lines use `THREE.Mesh` with `InstancedBufferGeometry` (not `THREE.InstancedMesh` or `THREE.LineSegments`) to enable thick lines with world-space width while staying under WebGL's 16 attribute location limit.

### Object Management

**Adding Objects:**

```typescript
// Add directly through the THREE.Scene field
sceneManager.scene.add(object);

// Batch operations
const group = new THREE.Group();
group.add(points1, points2, points3);
sceneManager.scene.add(group);
```

**Clearing Scene:**

`loadSceneData` internally clears prior loaded content (lights and
background are preserved) before adding the new dataset's root
group. Manual disposal helpers used by that path are in
`scene-manager/render-pipeline/scene-disposal.ts`
(`clearLoadedSceneContent`, `disposeSceneGraphResources`).

---

## Camera Management

### FOV Control

The scene manager provides comprehensive FOV management:

```typescript
// Adjust FOV with mouse wheel + shift
sceneManager.updateFOV(delta);

// FOV limits: 10° to 170° (must be <180°)
// Sensitivity: 0.05 per wheel unit
```

**Professional FOV Presets** (35mm equivalent, horizontal FOV):

- **28mm Wide** (75°): Ultra-wide for landscapes
- **35mm** (63°): Wide angle for environmental shots
- **50mm Normal** (47°): Natural human vision equivalent
- **85mm Portrait** (29°): Telephoto for subject isolation
- **135mm Tele** (18°): Strong telephoto for extreme focus

### Clipping Plane Control

Advanced Z-buffer management for optimal rendering precision:

```typescript
// Manual clipping plane adjustment
sceneManager.updateClippingPlanes(near, far);

// Auto-calculate optimal planes from scene bounds
const { near, far } = sceneManager.autoAdjustClippingPlanes();

// Enable dynamic clipping (auto-adjusts each frame)
sceneManager.setDynamicClipping(true); // enable sphere-based clipping

// Trigger an explicit dynamic update (normally called per-frame
// by AnimationController)
sceneManager.updateDynamicClippingPlanes();

// Get current dynamic clipping state
const state = sceneManager.getDynamicClippingState();
// { enabled: boolean, near: number, far: number }
```

**Z-Buffer Best Practices:**

- Keep near/far ratio under 10,000:1 for best precision
- Use auto-adjust after loading new datasets
- Lower near values see closer objects but reduce precision

### Dynamic Clipping Planes

The scene manager supports automatic per-frame clipping plane adjustment:

**How It Works:**

1. Each frame, computes a bounding sphere from the cached scene bounds (with safety margin)
2. Near plane = `max(nearPlaneFloor(R, far), distToCenter - R)` where `R` is the safety-expanded radius and `nearPlaneFloor(R, far) = max(minNearForRadius(R), far / MAX_NEAR_FAR_RATIO)` -- smoothly transitions to that floor as the camera enters the sphere
3. Far plane = `distToCenter + R` -- distance to farthest point on the sphere
4. The bounds cache is invalidated on scene load/clear; zero per-frame scene-graph traversal in steady state

**Benefits:**

- Always-optimal Z-buffer precision as camera moves
- **Direction-independent clipping** -- no sharp jumps at bounding box edges
- Near plane drops to the floor as camera enters the scene, but no further: `MAX_NEAR_FAR_RATIO = 1200` caps the near/far ratio so 24-bit depth stays usable (an unbounded ratio z-fights, and pops while orbiting). The cap sits below the `nearCull` depth at which all four geometry types have already faded out: Points/GSplats/Mesh are rejected outright below the shared 0.01 threshold, so it costs those three nothing, and a line is attenuated to under 1% of its authored contribution (its own discard is a separate colour test the fade never enters)
- Eliminates need for manual clipping adjustment
- Perfect for exploring large-scale scenes from any viewpoint

**Configuration:**

```typescript
// Enable sphere-based dynamic clipping
sceneManager.setDynamicClipping(true);
```

### Centering Modes

Two centering modes for camera focus:

1. **Native Center**: Scene's inherent center point
2. **Bounding Box Center**: Computed from all objects

```typescript
// Toggle with 'C' key
sceneManager.toggleCentering();

// Get current center
const center = sceneManager.getCurrentCenter();
controls.lookAt(center);
```

---

## Animation System

### Performance Optimization

The animation controller implements smart rendering:

```typescript
// Only renders when needed
- User interaction (mouse, keyboard)
- Control updates (movement, rotation)
- Data changes (dimension navigation)
- Programmatic updates

// Automatically pauses after idle
- Default timeout: 2000ms
- Saves GPU/CPU resources
- Resumes instantly on activity
```

### Event Triggers

Animation is triggered by various events:

```typescript
// Control events
controls.addEventListener('change', startAnimation);
controls.addEventListener('start', startAnimation);

// User interaction
canvas.addEventListener('mousedown', startAnimation);
canvas.addEventListener('touchstart', startAnimation);

// Data updates
sceneDimsManager.addListener(startAnimation);
```

---

## nD Visualization

### Dimension Slicing

For data with >3 dimensions, the system provides slicing:

```typescript
// 5D data example
const dims = {
  ndim: 5,
  displayed: [0, 1, 2], // Show x, y, z
  currentStep: [0, 0, 0, 50, 100], // Position in 5D space
};

// Points visible if within radius of slice
const radius = point.radius || 1.0;
const distance = calculateDistance(point.position, slicePosition);
point.visible = distance <= radius;
```

### Dimension Initialization

**Initial Slice Position** (on viewer startup):

When nD datasets load, dimensions initialize based on their type:

| Dimension Type          | Initial Position  | Example                                            |
| ----------------------- | ----------------- | -------------------------------------------------- |
| **Time/Frames**         | Start (t=0)       | Time-series starts at first frame                  |
| **Channels**            | First channel (0) | Multi-channel data shows first channel (DAPI, GFP) |
| **Discrete dimensions** | Minimum value     | Frame 0, category 0                                |
| **Continuous spatial**  | Center (0)        | 4th spatial dimension (W) centers at 0             |

**Behavior**:

```typescript
// Time-series (discrete, range 0-50ns)
Initial position: t = 0 ns ✓ (not 25ns)

// Multi-channel (discrete, 0-3)
Initial position: channel = 0 ✓ (not channel 2)

// 4D spatial (continuous, -100 to +100)
Initial position: W = 0 ✓ (center makes sense)
```

**Why It Matters**:

- Slider position matches displayed data from first load
- Time-series animations start at beginning, not middle
- Multi-channel data shows first channel first
- No confusing mismatch between UI and visualization

**Implementation Details**:

The initialization happens in two phases:

1. `sceneDimsManager.initFromScene()` sets up dimension state
2. `inputHandler.initDimensionSliders()` triggers initial data load

See `scene-dims-manager.ts` (`initFromScene`) for the initialization
logic and `input/input-handler.ts` (`initDimensionSliders`) for the
initial update trigger.

### Keyboard Navigation

Navigate through dimensions with keyboard:

```typescript
// Select dimension
'1'-'9': Select non-displayed dimension 1-9

// Navigate selected dimension
'[': Move backward in dimension
']': Move forward in dimension

// Step sizes
- Discrete dims: Use defined step
- Continuous dims: 1% of range
```

### nD Transform Inverse-Query

The viewer uses an **inverse-query** approach for `nd_transform`: instead of transforming millions of point coordinates forward (O(N)), the query (slicePosition + tolerance) is inverse-transformed from world to local space once (O(1)). This is implemented in `data/transforms/nd-transform.ts` and applied transparently during geometry slicing. No loader internals change.

### Auto-Ranging from Position Bounds

When `Dimension.range` is not explicitly set, `SceneDimsManager.initFromScene()` uses `positionBounds` from the zarr root attrs as the slider range for that dimension. These bounds are pre-computed by the Python compiler with nd_transform expansion applied, so non-displayed dimensions reflect world-space extents. Fallback order: explicit `Dimension.range` > `positionBounds` > `[0, 1]`.

---

## Performance Monitoring

### Integration with Performance Stats

```typescript
// Initialize with performance monitoring
const performanceStats = new PerformanceMonitor(renderer);
animationController.setPerformanceStats(performanceStats);

// Automatic frame timing
- Begin() before render
- End() after render
- Updates FPS display
- Tracks frame times
```

### Optimization Techniques

1. **Frustum Culling**: Automatic via THREE.js
2. **Idle Detection**: Pause rendering when static
3. **LOD Support**: Ready for level-of-detail
4. **Batch Operations**: Group scene updates
5. **Disposal Management**: Proper cleanup

---

---

## WebGL Context Loss Recovery

### What is Context Loss?

WebGL contexts can be lost due to:

- GPU driver crashes or resets
- System sleep/hibernate
- Too many WebGL contexts (browser limit ~16)
- Out of GPU memory
- GPU overheating

When this happens, all WebGL resources are lost and rendering stops.

### How Luxar Handles It

The SceneManager automatically handles context loss and restoration:

```typescript
// Automatic recovery - no user action needed
canvas.addEventListener('webglcontextlost', (event) => {
  event.preventDefault(); // Allow restoration
  // Show user message: "Graphics context lost - attempting to restore..."
});

canvas.addEventListener('webglcontextrestored', async () => {
  // Recreate WebGL resources
  renderer.resetState();
  // Trigger re-render
  // Show success message: "Graphics context restored"
});
```

### User Experience

**During Context Loss**:

1. Rendering stops
2. User sees message: "Graphics context lost - attempting to restore..."
3. Loading indicator appears

**During Restoration**:

1. WebGL resources automatically recreated
2. Rendering resumes
3. User sees: "Graphics context successfully restored"

**If Restoration Fails**:

- Error message: "Failed to restore graphics. Please refresh the page."
- User can continue using UI, but rendering disabled

### For Developers

Check context status:

```typescript
// Check if context is lost
if (sceneManager.isWebGLContextLost()) {
  // Skip rendering operations
  return;
}

// Safe to render (one frame through the post-processing pipeline)
sceneManager.postProcessing.render();
```

**Best Practice**: Don't create WebGL-dependent operations during context loss. Wait for restoration.

## Usage Examples

### Complete Setup

```typescript
import { SceneManager } from './scene/scene-manager';
import { AnimationController } from './scene/animation/animation-controller';
import { sceneDimsManager } from './scene/scene-dims-manager';

// Initialize scene
const canvas = document.getElementById('app') as HTMLCanvasElement;
const sceneManager = new SceneManager();
await sceneManager.init({ canvas });
const animationController = new AnimationController(
  sceneManager.controls,
  sceneManager.postProcessing
);

// Setup dimensions for nD data
sceneDimsManager.initFromScene(sceneManager.scene);

// Start render loop
animationController.startAnimation();
```

### Loading a Zarr Scene

```typescript
// One-shot: clears prior content, loads zarr scene tree,
// applies any viewer_config camera overrides, auto-frames the
// camera, and auto-adjusts clipping planes.
await sceneManager.loadSceneData(zarrUrl);

// Manual recentering on bbox (also bound to 'F' key in app)
sceneManager.centerCameraOnScene();
```

### Handling Window Resize

```typescript
window.addEventListener('resize', () => {
  // updateSize() schedules an rAF-coalesced resize through the
  // ResizeOrchestrator, which propagates to renderer, camera
  // aspect, post-processing, and material uniforms.
  sceneManager.updateSize();
});
```

---

## Configuration

### Scene Settings

```typescript
const sceneConfig = {
  backgroundColor: 0x000000, // Pitch black — zero radiance under the HDR exposure chain
  ambientLight: {
    color: 0xffffff,
    intensity: 1.0,
  },
  camera: {
    fov: 60,
    near: 0.1,
    far: 1000,
    position: { x: 0, y: 0, z: 8 },
  },
};
```

### Animation Settings

```typescript
const animationConfig = {
  idleTimeoutMs: 2000, // Pause after 2 seconds
  targetFPS: 60, // Target frame rate
  adaptiveQuality: true, // Reduce quality if FPS drops
  pacing: {
    enabled: true, // Frame pacing on (false = the back-to-back rAF loop)
    slowFrameMs: 250, // A frame past this is "pathologically slow" (4 fps)
    maxCooldownMs: 250, // Ceiling on the inserted gap
  },
};
```

---

## Best Practices

### Scene Organization

1. **Use Groups**: Organize related objects
2. **Name Objects**: Set meaningful names for debugging
3. **Clean Disposal**: Always dispose geometries/materials
4. **Batch Updates**: Group scene modifications
5. **Cache Bounds**: Update bbox only when needed

### Performance Tips

1. **Minimize Draw Calls**: Use instanced rendering
2. **Optimize Geometries**: Merge when possible
3. **Texture Atlas**: Combine textures
4. **Frustum Culling**: Let THREE.js handle it
5. **Smart Animation**: Use idle detection

### Memory Management

```typescript
// Proper disposal
function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.geometry?.dispose();
      child.material?.dispose();
    }
  });

  // Remove from parent
  object.parent?.remove(object);
}
```

---

## Troubleshooting

### Common Issues

**Problem: Scene appears black**

- Check lighting setup
- Verify camera position
- Ensure objects are in view frustum

**Problem: Poor performance**

- Enable idle detection
- Reduce point count
- Check for memory leaks
- Profile with Chrome DevTools

**Problem: Objects not visible**

- Check bounding box
- Verify camera near/far planes
- Ensure materials are configured

**Problem: Dimension navigation not working**

- Initialize sceneDimsManager
- Check dimension metadata
- Verify keyboard focus

---

## API Reference

### SceneManager

| Method                                           | Description                                                                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `init(options)`                                  | Build renderer, scene, camera, controls, post-processing                                                               |
| `loadSceneData(src, loaderConfig?, options?)`    | Load zarr scene; `applyViewerConfigFov` gates pre-frame scene FOV unless an authored position carries its resolved FOV |
| `centerCameraOnScene()`                          | Frame camera on scene bounding box                                                                                     |
| `toggleCentering()`                              | Switch center mode (origin ↔ bbox)                                                                                     |
| `getCurrentCenter()`                             | Get active center point                                                                                                |
| `setFov(degrees)`                                | Set an absolute validated perspective FOV                                                                              |
| `updateFOV(delta)`                               | Adjust field of view                                                                                                   |
| `updateClippingPlanes(near, far)`                | Set camera clipping planes                                                                                             |
| `autoAdjustClippingPlanes()`                     | Calculate optimal clipping from scene                                                                                  |
| `setDynamicClipping(enabled)`                    | Enable/disable per-frame clipping update                                                                               |
| `getDynamicClippingState()`                      | Get current dynamic clipping state                                                                                     |
| `updateDynamicClippingPlanes()`                  | Manually trigger dynamic clipping update                                                                               |
| `setControlType(type)` / `getControlType()`      | Switch / read current control type                                                                                     |
| `setAdaptivePixelRatio(dpr)`                     | Set DPR override (AdaptiveDPRManager)                                                                                  |
| `updateExposure/GlobalOffset/GlobalGamma(value)` | Update post-processing tone-mapping uniforms                                                                           |
| `isWebGLContextLost()`                           | Query WebGL context-loss state                                                                                         |
| `updateSize()`                                   | Handle resize (rAF-debounced)                                                                                          |
| `dispose()`                                      | Clean up resources                                                                                                     |

### AnimationController

| Method                           | Description                                               |
| -------------------------------- | --------------------------------------------------------- |
| `startAnimation()`               | Begin render loop                                         |
| `stopAnimation()`                | Stop render loop                                          |
| `addPerFrameCallback(id, fn)`    | Add named per-frame callback (e.g., for dynamic clipping) |
| `removePerFrameCallback(id)`     | Remove per-frame callback by ID                           |
| `hasPerFrameCallback(id)`        | Check if callback exists                                  |
| `setAdaptiveDPRManager(manager)` | Set adaptive DPR manager for dynamic resolution           |
| `dispose()`                      | Clean up resources                                        |

### SceneDimsManager

| Method                        | Description           |
| ----------------------------- | --------------------- |
| `initFromScene(scene)`        | Initialize dimensions |
| `getDims()`                   | Get dimension info    |
| `setDimensionValue(idx, val)` | Update position       |
| `addListener(callback)`       | Subscribe to changes  |
| `reset()`                     | Clear dimensions      |

---

## License

Part of the Luxar project. See root LICENSE file for details.

---

_For implementation details, see the source files in this directory._

---

## Top-Level Files

- `scene-manager.ts` — `SceneManager` orchestrator: renderer, camera,
  controls, post-processing, resize, disposal.
- `scene-dims-manager.ts` — Singleton dimension state across all nD
  objects in the scene (exported as both class `SceneDimsManager`
  and lazy-Proxy singleton `sceneDimsManager`).
- `dimension-loading.ts` — Applies the current scene-dimension selection,
  propagates animation frame budgets, warms the projected next slice during
  playback, and releases shadow-loader resources after playback stops.
- `lod-group-registry.ts` — `LODGroupRegistry`: per-frame `lod_group`
  child selector (pick on the group's screen-area or legacy diagonal
  metric + frustum off-screen gate
  - asymmetric hysteresis), lazy-load gating, and the display/fade/
    eviction orchestration. Re-exports `projectBoxAreaFraction`,
    `projectBoxDiagonalPx` and `pickChildWithHysteresis` from
    `lod-selector-math.ts`.
- `lod-selector-math.ts` — The selector's camera-geometry math:
  `computeEntryWorldBox` (nD raw or robust bounds → world box via
  displayDims), `projectBoxAreaFraction` (world box → fraction of the
  viewport area) and `projectBoxDiagonalPx` (→ screen-space pixel
  diagonal) — both with near-plane saturation — and
  `pickChildWithHysteresis`.
- `lod-fade.ts` — Material-level appliers for the two LOD anti-popping
  mechanisms: `applyLodFade` (write coverage-weight × `1/e(k)` opacity
  per fadeable leaf, clone-on-first-fade) and `isBlendableSubtree`
  (uniformly additive/luminous/volumetric check — `BLENDABLE_MODES`).
  The material-touching counterpart of `lod-blend.ts`'s pure math.
- `lod-eviction.ts` — `enforceResidentByteBudget`: the VRAM-pressure
  policy — while the GPU pool reports over-budget, demote hidden LOD
  levels hidden-layer-first / off-screen-first / furthest-first /
  coldest-first, clearing `visible` on each released level. "Hidden" is
  ancestor-aware (`utils/object-visibility.ts`), so a level under a
  toggled-off layer is reclaimable — and reclaimed before anything a
  visible layer could still draw.
- `lod-display-gate.ts` — The never-downgrade display gate for the
  registry: `shouldHoldPreviousDisplay` holds the previously-displayed
  level while a streaming upgrade is strictly worse than what is shown,
  releasing on committed energy e(k) ≥ `ENERGY_RELEASE_THRESHOLD`
  (0.6), ladder completion, count crossover, or freshness loss;
  `subtreeDisplayProgress` aggregates commit stamps over nested-group
  levels. Pure policy over commit-time stamps — no THREE import.
- `lod-freshness.ts` — Pure, dependency-free freshness + settle
  primitives for the registry: `coarsestFreshIndex` (which level's
  committed geometry reflects the current view version) and
  `SettleTracker` ("has the version been stable for N ticks?" — the
  debounce behind deferred fine-level reloads).
- `synthetic-scene.ts` — Mulberry32-seeded synthetic scene generators
  used by the perf bench and the `__luxarDebug.injectSyntheticScene`
  debug API. `generateSyntheticLines` (random-walk segments — output
  pinned byte-for-byte by the 10 M-segment bench contract), plus
  `generateSyntheticPoints` / `generateSyntheticGSplats` built on a
  shared gaussian-blob cluster sampler (`sampleClusteredPositions`).
  `syntheticLinesBoundsDiagonal` reports the generation volume's
  diagonal so the injector can size its line primitive by rendered
  width the way a compiled node does (`types/line-primitive.ts`).
  GSplats get valid lower-triangular Cholesky factors with varied
  scale, anisotropy, and orientation so depth-sorted 'normal'
  blending is order-dependent; the injector emits the production
  commit signals (`committedData` stamp + `noteDepthSortCommit`) so
  the sort subsystem engages on injected nodes.

## Subpackages

- [animation](./animation/README.md) — `AnimationController` render
  loop and `DimensionAnimationManager` per-dimension playback.
- [dims](./dims/README.md) — Pure step-size, wrap, and non-displayed-dimension
  selection helpers shared by scene and input code.
- [scene-manager](./scene-manager/README.md) — Extracted SceneManager
  helpers (camera setup/framing/materials/mode, clipping policy and
  bounds cache, render-pipeline setup, viewport DPR and resize).
