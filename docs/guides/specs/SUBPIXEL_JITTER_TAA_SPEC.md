# Sub-Pixel Jitter for Temporal Anti-Aliasing (TAA)

> **Status:** Not yet implemented. This document records the complete design and implementation from the now-deleted `feature/subpixel-jitter` branch for future reference.

## Overview

Sub-pixel camera jittering applies a small, sub-pixel offset to the camera position each frame. Over successive frames, this provides better spatial sampling coverage across the pixel grid, reducing aliasing artifacts on edges and small features (e.g., thin point cloud structures, sharp geometry boundaries).

The technique is a prerequisite for accumulation-based temporal anti-aliasing (TAA), where jittered frames are blended together to produce a smoother final image.

## Architecture

### Components

1. **`JitterGenerator`** class (singleton) — generates per-frame sub-pixel offsets
2. **`AnimationController`** integration — applies/restores jitter around each render call
3. **Config** — three new `RenderingSettings` fields
4. **UI** — lil-gui controls nested under the existing Anti-Aliasing folder

### Data Flow (per frame)

```
AnimationController.animate()
  │
  ├─ controls.update()          // Process user input first
  │
  ├─ Restore previous jitter    // camera.position.sub(previousOffset)
  ├─ Apply new jitter           // jitterGenerator.applyToCamera(camera, viewport)
  │                              //   → converts pixel offset to world-space
  │                              //   → shifts camera along its local right/up axes
  │                              //   → returns the world-space offset vector
  ├─ postProcessing.render()    // Render the jittered frame
  │
  └─ Restore camera position    // camera.position.sub(currentOffset)
                                 // camera.updateMatrixWorld()
                                 // (so controls/picking work on unjittered position)
```

**Key invariant:** The camera position is always restored after rendering, so orbit controls, raycasting, and other camera-dependent logic operate on the true (unjittered) camera position.

## Jitter Patterns

### 1. Halton Sequence (default, recommended)

Low-discrepancy quasi-random sequence providing optimal spatial coverage with minimal clustering.

**Algorithm:**
```
halton(index, base):
    f = 1, r = 0, i = index
    while i > 0:
        f = f / base
        r = r + f * (i % base)
        i = floor(i / base)
    return r    // result in [0, 1)
```

- Uses **base 2** for X and **base 3** for Y (standard choice for 2D Halton)
- Index starts at 1 (index 0 returns 0 for all bases)
- Output is centered: `(halton - 0.5) * amount * 2` to distribute offsets symmetrically around zero

**Offset calculation:**
```
x = (halton(frameIndex + 1, 2) - 0.5) * amount * 2
y = (halton(frameIndex + 1, 3) - 0.5) * amount * 2
```

**Properties:**
- Deterministic (same sequence every time)
- Progressive (each additional sample improves coverage)
- No clustering unlike pure random
- First 8 samples in base 2: 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625

### 2. Random (uniform disk)

Uniform random sampling within a circular region.

**Algorithm:**
```
angle  = random() * 2 * PI
radius = sqrt(random()) * amount    // sqrt for uniform area distribution
x = cos(angle) * radius
y = sin(angle) * radius
```

The `sqrt(random())` is critical: without it, points cluster toward the center of the disk. The square root produces a uniform distribution over the disk area by compensating for the fact that area grows quadratically with radius.

### 3. Grid (4xMSAA positions)

Cycles through 4 fixed sub-pixel positions matching standard 4xMSAA sample locations:

| Sample | X | Y |
|--------|------|------|
| 0 | -0.25 | -0.25 |
| 1 | +0.25 | -0.25 |
| 2 | -0.25 | +0.25 |
| 3 | +0.25 | +0.25 |

**Offset:** `position[frameIndex % 4] * amount * 2`

Repeats every 4 frames. Simplest pattern, adequate for static scenes.

## Pixel-to-World Conversion

The jitter is specified in pixel units but must be applied in world space. The conversion pipeline:

```
pixel offset
    → NDC offset:   ndcX = (pixelX / viewportWidth) * 2.0
                     ndcY = (pixelY / viewportHeight) * 2.0
    → world scale:  distance = camera.position.length()   // approx distance to scene center
                     fovRad = camera.fov * PI / 180
                     worldHeight = 2 * tan(fovRad / 2) * distance
                     worldWidth = worldHeight * camera.aspect
    → world offset: worldX = ndcX * worldWidth * 0.5
                     worldY = ndcY * worldHeight * 0.5
    → camera-space: extract right vector = matrixWorld column 0
                     extract up vector    = matrixWorld column 1
                     finalOffset = right * worldX + up * worldY
    → apply:        camera.position += finalOffset
                     camera.updateMatrixWorld()
```

**Note on distance:** Uses `camera.position.length()` as an approximation for the distance to the scene center (assumes scene is centered at origin). For scenes with a known focus point, using the actual camera-to-target distance would be more accurate.

## Configuration

Three fields added to `RenderingSettings`:

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `jitterEnabled` | `boolean` | `false` | — | Master enable/disable |
| `jitterAmount` | `number` | `0.5` | 0–2.0 px | Jitter radius in pixels |
| `jitterPattern` | `string` | `'halton'` | `'random'` \| `'halton'` \| `'grid'` | Sampling pattern |

Settings are persisted via the existing `saveSettings()`/`loadSettings()` mechanism.

## UI Integration

Controls are placed under **Anti-Aliasing > Temporal Jitter (TAA)** in lil-gui:

```
Anti-Aliasing/
  ...existing SSAA, FXAA, MSAA controls...
  Temporal Jitter (TAA)/          ← subfolder, hidden when disabled
    Jitter Enabled     [checkbox]
    Jitter Amount (px) [slider 0–2.0, step 0.01]
    Pattern            [dropdown: random/halton/grid]
    Frame Index        [read-only, updates every 100ms]
```

- The subfolder auto-shows/hides based on the enabled checkbox
- Frame Index is a read-only debug display updated via `setInterval(fn, 100)`
- Tooltip on the enable checkbox explains the feature and its tradeoffs

## AnimationController Integration

The `AnimationController` stores:
- `private jitterOffset: THREE.Vector3` — the world-space offset applied in the current frame
- References to `renderer` and `camera` (changed from unused `_renderer`/`_camera` to active `private` fields)

**Lifecycle per frame:**
1. `controls.update()` — must happen on unjittered camera
2. Subtract previous `jitterOffset` from camera (restore true position)
3. Call `jitterGenerator.applyToCamera(camera, viewport)` — returns new offset
4. `postProcessing.render()` — renders with jittered camera
5. Subtract current `jitterOffset` from camera (restore for next frame's controls)
6. `camera.updateMatrixWorld()` — ensure matrix reflects restored position

## Performance Characteristics

- **Zero heap allocation in hot path:** `getNextOffset()` reuses a cached `THREE.Vector2`
- **`applyToCamera()` allocates:** `new THREE.Vector2()`, `new THREE.Vector3()` (x3) per frame — these could be hoisted to instance fields for zero-alloc rendering if needed
- **Halton computation:** O(log n) per dimension per frame (negligible)
- **No GPU cost:** Jitter is purely a CPU-side camera position shift

## Limitations and Known Issues

1. **No frame accumulation/blending:** The branch only implements the jitter generation and camera application. A full TAA solution would also need a temporal resolve pass that blends the current jittered frame with a history buffer (reprojected using motion vectors). Without this, the jitter produces a subtle per-frame shimmer rather than smooth anti-aliasing.

2. **Distance approximation:** `camera.position.length()` assumes the scene is at the origin. Scenes offset from the origin will get incorrect jitter magnitudes.

3. **Allocations in `applyToCamera`:** Creates 4 new Three.js vector objects per frame. Should be converted to reusable instance fields for production use.

4. **Singleton pattern:** `jitterGenerator` is a module-level singleton. This works for single-viewer apps but would need refactoring for multi-viewer scenarios.

5. **`setInterval` for frame counter UI:** The 100ms polling interval for the debug frame counter is crude; an event-driven update from the animation loop would be cleaner.

6. **No motion vector generation:** Without motion vectors, a future temporal resolve pass cannot accurately reproject the history buffer, which may cause ghosting on moving objects.

## Future Work (to complete TAA)

1. **History buffer:** Render to an off-screen texture, keep previous frame's result
2. **Temporal resolve shader:** Blend current frame with reprojected history using exponential moving average (typical alpha: 0.1–0.2 for current frame weight)
3. **Motion vectors:** Generate per-pixel velocity from camera delta (and optionally object motion) for accurate reprojection
4. **Neighborhood clamping:** Clamp history color to the min/max of the current frame's 3x3 neighborhood to reject stale/ghosted samples
5. **Sharpening pass:** Compensate for the slight softening that accumulation introduces (e.g., CAS or simple unsharp mask)
