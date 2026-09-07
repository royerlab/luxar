# Luxar Rendering Package

> Dual-stack WebGL2/WebGPU rendering pipeline (GLSL `ShaderMaterial` default, TSL `NodeMaterial` opt-in via `?renderer=webgpu`) with a custom mega-shader for high-quality nD scientific visualization

## Overview

The Luxar Rendering package provides a high-performance rendering pipeline built on Three.js r185. Each of the 9 production shaders (3 geometry visual + 3 geometry picking + mega-shader + FXAA + bloom-threshold) ships as a `ShaderSource` pair: a `WebGLRenderer`-targeted GLSL3 string and a `WebGPURenderer`-targeted TSL factory. `MaterialManager` dispatches on `RendererCapabilities.apiSurface` so the same scene graph renders identically through either backend. Post-processing runs through a hand-written **mega-shader** that fuses all per-pixel effects into a single fullscreen fragment pass — bloom is a separate pre-pass (needs neighbor reads) and FXAA is a separate post-pass (edge detection on the LDR output).

The default backend is `THREE.WebGLRenderer` (GLSL `ShaderMaterial`). `WebGPURenderer` (TSL `NodeMaterial`) is selectable via `?renderer=webgpu` or `VITE_LUXAR_USE_WEBGPU=1`; it falls back to its internal WebGL2 backend when no WebGPU adapter is available. Every TSL shader is validated against its GLSL counterpart through `tsl-shader-parity.spec.ts`. For diagnostics, `?renderer=webgpu&webgpu-force-webgl` constructs `WebGPURenderer({ forceWebGL: true })`: Luxar still uses TSL `NodeMaterial` shaders and the WebGPURenderer API surface, but Three.js routes rendering through its internal WebGL2 backend instead of native WebGPU.

**The TSL half is lazily loaded.** Because WebGL is the default, the entire `three/webgpu` cone — the 9 TSL material classes and every TSL graph factory — sits behind a single `await import()` in `rendering/tsl/load.ts` and is fetched only when `selectBackend()` actually chooses WebGPU. That keeps ~182 kB gzipped off the initial payload for the default session (issue #1679). Consequences worth knowing before you edit a material: the `MaterialManager` dispatch tables hold **thunks**, not classes (`VISUAL_FACTORIES[kind][backend]()`); the `ShaderSource.webgpu` closures obtain their factory from `requireTslMaterials()` rather than importing it; `material-sync-helpers.ts` uses structural probes instead of `instanceof` on TSL classes; and a value import of `three/webgpu` from production code outside the lazy cone — `rendering/tsl/registry.ts` and the `*-tsl` / `*.tsl` modules it owns — is an ESLint error (`src/tests/**` is exempt: it ships nothing, and the parity harness drives the WebGPU path directly). See `tsl/README.md`.

### Key Features

- **HDR Rendering Pipeline**: 16-bit float (HalfFloat) buffers for true HDR support
- **Mega-shader post-processing**: One fused fragment pass for bloom mix, detector noise, EOG, tone mapping, vignette, chromatic lens distortion, and sRGB encoding
- **Custom Shader System**: Optimized shaders for Points, Lines, GSplats, and Mesh
- **Line Rendering**: Instanced quad geometry for thick lines with seamless joints
- **Material Management**: Per-node materials with lifecycle tracking for Points, Lines, GSplats, and Mesh
- **World-Space Point Sizing**: Physically accurate scaling
- **Anti-Aliasing Options**: FXAA, MSAA, and SSAA support

### Package Architecture

```
rendering/
├── material-manager.ts                 # Per-node material creation + global camera updates
├── node-factory.ts                     # Scene-node factories for Points / Lines / GSplats
├── gpu-buffer-pool.ts                  # Geometry reuse with count and byte-budget eviction
├── gpu-byte-budget.ts                  # Single adaptive VRAM budget (pool + LOD registry share it)
├── adaptive-dpr-manager.ts             # Adaptive resolution
├── pixel-ratio-cap.ts                  # The max DPR the viewer may render at (high DPR is opt-in)
├── colormap-textures.ts                # Built-in/custom DataTexture creation and cache disposal
├── colormap-data.ts                    # Built-in colormap lookup tables (auto-generated)
├── renderer-capabilities.ts            # WebGL2 vs WebGPU capability detection
├── blending-state.ts                   # THREE blending state for every Luxar mode
├── material-colormap-helpers.ts        # Shared scalar-colormap guards and uniform helpers
├── material-sync-helpers.ts            # Geometry-commit material sync helpers
├── webgl-blend-warmup.ts               # WebGL-only pre-link of every reachable blend-mode program variant (`?no-blend-warmup`)
├── tsl/                                # The lazy three/webgpu boundary — registry.ts (sole entry to the cone) + load.ts (sole `await import()`) + slot.ts (zero-runtime-import accessor). See tsl/README.md.
├── display-range.ts                    # Pure display-window ↔ shader intensity/offset math + resolveColormapWindow (shared by all 3 node factories)

├── line-geometry.ts                    # Line quad base + 6-texel layout/texel writer + mesh create/update
├── gsplat-geometry.ts                  # Instanced GSplat mesh creation/update helpers
├── element-texture-layout.ts           # RGBA32F element-texture layout authority (gsplat + point + line bindings)
├── element-storage.ts                  # Shared texture-backed element storage + aSortedIndex writers
├── depth-sort-coordinator.ts           # Main-thread side of the depth-sort worker + camera re-sort scheduler (Phases 2-3)
├── render-layers.ts                    # Object3D layer bits — two, used transiently by the refraction split
├── point-geometry.ts                   # Point quad base + 3-texel layout/texel writer
├── mesh-geometry.ts                    # Plain indexed BufferGeometry for Mesh (NOT instanced/texture-backed): WebGPU-safe colour dtypes, capacity-sized vertex buffers + index + drawRange
├── widen-to-float32.ts                 # Dtype widening for the texel writers
│
├── materials/                          # Per-geometry material and shader stacks
│   ├── point/   { material-glsl, material-tsl, shader-glsl, shader-tsl }
│   ├── line/    { material-glsl, material-tsl, shader-glsl, shader-tsl }
│   ├── gsplat/  { material-glsl, material-tsl, shader-glsl, shader-tsl, math }
│   └── _shared/ { camera-aware-material, colormap-aware-material, camera-uniforms,
│                  uniform-helpers, material-builder, tsl-helpers, glsl-lib, shader-source }
│
├── material-manager/                   # MaterialManager helper modules
│   ├── factories.ts                    # VISUAL/PICKING/MEGA_SHADER_FACTORIES + backend dispatch
│   ├── lifecycle.ts                    # subscribeToDispose + removeFromRegistries + SOFT_DISPOSE_FLAG
│   └── stats.ts                        # getCacheStats snapshot
│
├── node-factory/                       # NodeFactory helper modules
│   ├── validation.ts                   # validateLoadedPointsData / ColorMode / TransformFormat
│   ├── transforms.ts                   # applyTransform
│   ├── create-points-node.ts           # createPointsGeometry + createPointsMaterial
│   ├── create-lines-node.ts            # createLinesNode + createEmptyLinesNode
│   └── create-gsplats-node.ts          # createGSplatsNode + createEmptyGSplatsNode
│
├── post-processing/                    # HDR post-processing pipeline
│   ├── post-processing-manager.ts      # Public API — HDR pipeline orchestrator
│   ├── bloom/                          # chain / shaders / bloom.tsl
│   ├── fxaa/                           # pass / shaders / fxaa.tsl
│   ├── fullscreen/                     # pass / geometry
│   ├── mega/                           # material / material-tsl / shader.glsl / shader.tsl
│   ├── hdr/                            # pixel-utils / capture
│   ├── render-target-sizing.ts
│   └── post-processing-manager/        # PostProcessingManager helper modules
│       ├── resource-lifecycle.ts       # buildTransientResources / disposeTransientResources / sizing
│       ├── settings.ts                 # bloom / msaa / vignette / chromatic-lens setters
│       ├── pipeline.ts                 # runPipeline (scene → HDR → bloom → mega → FXAA)
│       └── capture.ts                  # captureHDRPixels / captureHDRAsEXR / renderToImageData
│
├── picking/                            # GPU picking materials + orchestration
│   ├── picking-system.ts               # Orchestrator
│   ├── PICKING_DESIGN.md               # Backend readback strategy + 1-frame-latency rationale
│   ├── point/    { material, material-tsl, shaders (GLSL), pick.tsl (TSL) }
│   ├── line/     { material, material-tsl, shaders (GLSL), pick.tsl (TSL) }
│   ├── gsplat/   { material, material-tsl, shaders (GLSL), pick.tsl (TSL) }
│   └── picking-system/                 # PickingSystem helper modules
│       ├── registration.ts             # disposePickMaterial / unregisterAllPickMaterials / PickNodeEntry
│       ├── ray-aabb.ts                 # rayHitsAnyNode / getOrComputeWorldBox / invalidateBoxCache
│       ├── pick-render.ts              # voteWinner (brightness-weighted majority over 5×5)
│       ├── lens-distortion.ts          # applyLensDistortion (sync screen-space picking with mega-shader)
│       ├── settle-loop.ts              # HOVER_SETTLE_MS + evaluateSettle decision logic
│       └── settle-scheduler.ts         # Debounced settle-pass scheduling
│
├── gpu-buffer-pool/                    # Per-type adapters + eviction
│   ├── {points,lines,gsplats}-adapter.ts
│   ├── eviction-policy.ts / pool-stats.ts
│   ├── capacity.ts                     # chooseCapacity (1.5× growth capped at 262,144 elements of headroom, min-instance floor)
│   ├── geometry-bytes.ts               # estimateGeometryBytes + cached size invalidation
│   └── byte-budget-evictor.ts          # Cross-type byte-budget enforcement
│
├── shaders/                            # Barrel only — re-exports GLSL constants from materials/<kind>/shader-glsl.ts
│   └── index.ts                        # Stable re-export spelling; no importer today (knip-ignored)
│
├── index.ts                            # Public-API barrel
└── README.md                           # This documentation
```

---

## Getting Started

### Step 1: Initialize Post-Processing

```typescript
import { PostProcessingManager } from './rendering/post-processing/post-processing-manager';
import { createRendererCapabilities } from './rendering/renderer-capabilities';

const capabilities = createRendererCapabilities(renderer);
const postProcessing = new PostProcessingManager(renderer, capabilities, scene, camera, {
  width: window.innerWidth,
  height: window.innerHeight,
});

// In your render loop
function animate() {
  requestAnimationFrame(animate);
  postProcessing.render(); // Renders scene with all effects
}
```

### Step 2: Enable Bloom (HDR Glow)

```typescript
postProcessing.updateBloomSettings(
  0.3, // strength: how intense the glow (0-1)
  0.85, // radius: how far it spreads (0-1)
  0.01 // threshold: HDR values above this glow
);
```

### Step 3: Choose Tone Mapping

```typescript
// ACES Filmic: cinematic, the recommended default
postProcessing.setToneMapping(THREE.ACESFilmicToneMapping);

// Or try others:
// THREE.AgXToneMapping       - Neutral, film-like
// THREE.ReinhardToneMapping  - Classic, simple
// THREE.LinearToneMapping    - No tone mapping
```

### Step 4: Add Anti-Aliasing

```typescript
// FXAA: Single-pass post-process AA (recommended for real-time)
postProcessing.setFXAAEnabled(true);

// OR MSAA: Hardware-accelerated multisample (sharp, but additive blending caveat)
postProcessing.setMSAAEnabled(true);
postProcessing.setMSAASamples(4);
```

**You're done!** Your scene now has professional HDR rendering with bloom, tone mapping, and anti-aliasing.

---

## Components

### 1. PostProcessing Manager (mega-shader)

The `PostProcessingManager` runs the mega-shader pipeline: a custom fragment shader that fuses tone mapping, EOG, vignette, detector noise, chromatic lens distortion, and sRGB encoding into a single fullscreen pass. Bloom is a separate pre-pass (neighbor reads). FXAA is a separate post-pass (edge detection on the LDR output). See `post-processing/README.md` for the full pipeline.

**Key Advantages:**

- One fused fullscreen pass for per-pixel effects: fewer rasterizations, fewer texture binds, no ping-pong target pair
- No third-party post-processing dependency
- A single GLSL/TSL shader pair (`mega/shader.glsl.ts` + `mega/shader.tsl.ts`) backs both the WebGL2 and WebGPU backends

**Core API:**

```typescript
// Initialize with HDR support
const postProcessing = new PostProcessingManager(renderer, capabilities, scene, camera, {
  width,
  height,
});

// Configure bloom
postProcessing.updateBloomSettings(/* strength */ 0.3, /* radius */ 0.85, /* threshold */ 0.01);

// Set tone mapping
postProcessing.setToneMapping(THREE.ACESFilmicToneMapping);

// Enable effects
postProcessing.setVignetteEnabled(true, 0.5, 0.5);
postProcessing.setChromaticLensDistortionEnabled(true, -0.05, -0.05);
```

### 2. Point Material

Advanced shader material for points rendering with custom vertex and fragment shaders.

**Vertex Shader Features:**

- World-space sizing from view-space depth (matches lines/gsplats — no
  off-axis shrink)
- 1.5px sprite floor (anti-rasterization-gap, matches lines); no
  sharpness size compensation — the shifted-truncated super-Gaussian
  truncates exactly at the sprite edge
- FOV-independent sizing
- Automatic viewport adaptation
- Unified `perspectiveNearFade` near handling (shared with lines/gsplats)

**Fragment Shader Features:**

- Shifted-truncated super-Gaussian falloff `max(exp(-K·ρ^β) - C, 0)/(1 - C)`
  with `β = 2^(6·sharpness - 2)` (C⁰-continuous at the sprite edge)
- Sub-pixel energy compensation: alpha × `sizeScale²` for sprites below
  the 1.5px floor (energy ∝ area)
- Per-node GOG (Gain-Offset-Gamma) color adjustment: `color * intensity + offset; clip; pow(color, 1/gamma)`
- Optimized with pre-computed uniforms

### 3. Line Material

Specialized shader material for thick lines using instanced quad geometry.

**Key Features:**

- **Super-Gaussian Kernel**: shifted-truncated super-Gaussian perpendicular falloff `max(exp(−K·p^β) − C, 0)/(1 − C)` (K=ln(100)≈4.605, C=0.01); sharpness is a normalised `[0, 1]` knob mapping to `β = 2^(6s − 2)` (s=0.5 → β=2 Gaussian, the default)
- **Seamless Joints**: Cap factor calculation ensures correct additive blending at joints
- **World-Space Width**: Lines have consistent thickness regardless of distance
- **nD Clipping**: Clipped endpoints use full intensity for correct visual appearance
- **Per-Endpoint Values**: Color, width, and sharpness interpolate along segments
- **Aspect-Ratio Correct**: Perpendicular direction computed in pixel space for correct line width
- **Anti-Aliasing for Thin Lines**: Minimum pixel width (1.5px) prevents sub-pixel rendering gaps; intensity scaling preserves visual weight of thin lines; smooth edge falloff using smoothstep

**Architecture Note:** Lines use `THREE.Mesh` with `InstancedBufferGeometry` (not `THREE.InstancedMesh`). Since the lines texture-storage migration their per-segment data does NOT live in vertex attributes: it lives in an RGBA32F **line texture** (6 texels/segment — startPos+startWidth, endPos+endWidth, startColor+startSharpness, endColor+endSharpness, segLen+cap-suppression scalars, scalars+alphas; layout documented in `line-geometry.ts`) fetched in the vertex stage via `texelFetch`, indexed by the double-buffered ordering pair `aSortedIndex`/`aSortedIndexB` (Uint32). This retired both the interleaved-buffer packing AND the colormap attribute-set toggle (the fixed layout always carries the scalar slots — no `GL_MAX_VERTEX_ATTRIBS` pressure), and lets the SortWorker permute draw order for `normal`-mode depth sorting (segment-midpoint keys) without rewriting segment data.

### 4. GSplat Material

Specialized shader material for volumetric Gaussian splatting with nD slicing support.

**Key Features:**

- **Oriented Anisotropic Gaussian**: Full 3D covariance via Cholesky factors
- **Perspective Projection**: Projects 3D covariance to 2D screen space using Jacobian
- **Shifted-Truncated Gaussian Falloff**: `max(exp(-½·r²) - C, 0)/(1 - C)` (C⁰-continuous at the truncation radius)
- **Sum/Max Projection Modes**: Ray integration for additive, peak value for max blending
- **Unified Near Fade**: shared `perspectiveNearFade` (smoothstep across `[uNearCull, 2·uNearCull]`, behind-camera → 0, ortho → 1) + an independent screen-coverage fade
- **GPU Optimizations**: Flat interpolation, reciprocal precomputation, early discard at 3σ

**Culling Strategy:**

- **Unified near handling** (`perspectiveNearFade`, shared across all
  four geometry types, visual + pick — the stage each one evaluates it in
  is tabulated in `materials/_shared/README.md`): perspective = 0 behind
  the camera, smoothstep fade across `[uNearCull, 2·uNearCull]`;
  ortho = fade 1 with NDC clipping as the sole cull authority.
  Prevents white-screen artifacts when navigating inside datasets.
  What each type does with the value differs, and the table is the
  authority — gsplats and points reject the VERTEX below 0.01 and
  multiply the survivor into amplitude; a mesh rejects per FRAGMENT at the
  same 0.01 (because `opaque` always, and `normal` at opacity ≥ 0.99,
  write depth) and either ramps its shaded RGB or folds the fade into
  coverage depending on the mode; lines have no 0.01 reject at all — the
  fade multiplies into the intensity chain and a fragment leaves through
  a colour-only `max(rgb) < 1e-4` discard in non-opaque modes or an
  alpha-weighted RGB-contribution discard in `opaque`
- **Screen-coverage fade**: unconditional amplitude fade toward the
  extent clamp (no hard-edged clamped rectangles, any sigma scale)

**Architecture Note:** GSplats use `THREE.Mesh` with `InstancedBufferGeometry` for instanced quad rendering, similar to the line material approach — but since the depth-sorting Phase 1 migration their per-splat data does NOT live in vertex attributes: it lives in an RGBA32F **splat texture** (4 texels/splat; layout authority in `element-texture-layout.ts`) fetched in the vertex stage via `texelFetch`, indexed by the double-buffered ordering pair `aSortedIndex`/`aSortedIndexB` (Uint32; both buffers are allocated at attach so the vertex layout is invariant — see the depth-sorting paragraph below). Points storage migrated the same way (3 texels/point — center+radius, color+sharpness, scalar+alpha; layout documented in `point-geometry.ts`, ≈56 B/point), and lines followed (6 texels/segment; layout in `line-geometry.ts`), so all three geometry types share the `element-storage.ts` texture+ordering machinery. This decouples draw order from storage order so the SortWorker can permute draw order without rewriting element data. The texture shares its geometry's lifetime (`element-storage.ts::attachElementStorage` registers a geometry-`dispose` listener) and costs ≈72 B/splat — 64 B of texels + 8 B for the ordering pair (≈ +38% vs the 52 B interleaved era; see `gpu-byte-budget.ts`; ≈56 B/point and ≈104 B/segment on the same rule). ALL visual materials are therefore **per node** (each binds its node's `uSplatTex` / `uPointTex` / `uLineTex`) — the historical material-manager LRU died with the lines migration.

**Depth sorting (Phase 2):** `depth-sort-coordinator.ts` is the main-thread side of the persistent depth-sort worker (`workers/sort-worker.ts`), serving **all four geometry types**. The description below is the instanced (`aSortedIndex`) apply used by gsplats, points and lines; mesh shares everything up to the apply and diverges there — see the mesh paragraph after it. Every non-noop commit of a sortable node calls `noteDepthSortCommit`, which bumps the node's **generation** counter; when the node's live blending mode is order-dependent (`needsDepthSort(mode)` — the three emissive types sort in both `normal` and `volumetric`; mesh has no `volumetric`, so its sorted set is just `normal`), the projected 3D centers are TRANSFERRED to the SortWorker and one back-to-front sort is requested from the current camera pose (a freshly derived `inverse(camera.matrixWorld) × mesh.matrixWorld` — both matrices are refreshed at sort time because the renderer-maintained caches can be stale when a commit fires before the next frame). Gsplats pass their fresh `centers3D` array directly; points pass a LAZY provider that fresh-copies `data.positions` only on the sorted path (the committed/lineage array's buffer must never itself be transferred); lines pass a LAZY provider computing fresh segment midpoints (`(start+end)/2` — the standard approximation; artifacts only when long segments interleave). Resolved orderings are applied to `aSortedIndex` via `writeSortedIndexOrdering` only when the generation still matches AND the node's `committedData` stamp survives (drops results raced by newer commits or LOD demotion); at most one sort is in flight per node, with one queued re-sort. Every ordering is **double-buffered** (depth-sorting spec §2.1 tier 3): each geometry carries an `aSortedIndex`/`aSortedIndexB` pair and a `uSortedIndexSlot` uniform selects which one the shaders read. A resolve only STAGES; the per-frame scheduler streams one `SORTED_INDEX_CHUNK_ELEMENTS` slice (1M indices = 4 MB) per rendered frame into the buffer that is NOT being drawn, and flips the slot once that buffer holds the whole permutation. So the per-frame memcpy+upload stays bounded at 4 MB instead of a 40 MB hitch (measured p99 119–563 ms at 10M splats) AND every rendered frame samples a complete ordering. BOTH buffers are allocated at attach (+4 B/element on every node, sorting or not). An earlier revision aliased them onto one attribute and split on a node's first sort so commutative-mode nodes paid nothing; that is unsafe on the native WebGPU backend and silently so. Three keys a pipeline's vertex-buffer layout by BufferAttribute IDENTITY, but `getGeometryCacheKey` hashes only attribute NAMES/itemSize/normalized and `RenderObjects.get` answers `needsGeometryUpdate` with a bare `setGeometry()` that leaves the cached pipeline alone — so after the split the draw bound three vertex buffers into a two-buffer layout, every later attribute shifted a slot, the quad-corner attribute read the ordering buffer's u32s as `vec2<f32>`, and the scene rendered BLACK with no validation error. WebGL binds by program location and was unaffected; so was `tsl-shader-parity`, which runs `WebGPURenderer({ forceWebGL: true })` and never builds a WebGPU vertex layout. The invariant — _the ordering attribute objects are fixed for the geometry's lifetime_ — is guarded by a unit test in `splat-texture-storage.test.ts`. Streaming into the live attribute is what the original L8 shipped, accepting a bounded old/new index mix — measured 2026-07-29 at 27–33% of frames on a 1.9M-splat volumetric node and 80% on an 8M-point node, because under a continuous orbit a new sort arrives about as fast as a stream drains, so the mix was the steady state rather than a transient. There is **no dispatch gate**: sorting and applying run concurrently, since the displayed ordering stays whole throughout. A newer ordering arriving mid-stream is **held** (at most one, latest wins) and started after the current one flips — restarting from slice 0 under a continuous orbit never converged. Any identity write (commit path), a cleared `committedData` stamp, node release, or geometry dispose cancels the remaining slices and the held ordering, which is safe because it abandons an un-drawn buffer. The WebGPU backends ignore attribute update ranges (full re-upload per flush), so SLICING is gated off there (`configureSortedIndexChunkedApply`, wired in renderer-setup) — they write the ordering in one slice and flip on the next pump; double-buffering itself is unconditional. Runtime blending-mode switches route through `noteDepthSortBlendingModeSwitch` (TO an effective sorted mode: clear the noop stamp + reprocess so the next commit registers; AWAY: release). The camera + render/reprocess callbacks are injected once at app init (`configureDepthSort` in `core/app/init/pipeline.ts`); disposal runs in the app dispose pipeline and per-mesh in `scene-disposal.ts`. The worker is spawned + initialized at APP INIT (`warmUpDepthSortWorker`), not by the first order-dependent commit — that commit lands exactly when the main thread is saturated decoding the scene, which is where startup used to lose the deadline race. Worker init runs under the data pool's shared `worker-pool/lifecycle/init-with-guard.ts`, raced against the `config.depthSort.workerInitTimeoutMs` deadline (default 30 s) plus the worker's own `error`/`messageerror` events, so the cached `initPromise` always SETTLES (otherwise every commit's continuation, closing over its centers provider, piles up on a forever-pending promise), and the failure is CLASSIFIED (issue #1694): a dead script / rejected `initialize` / constructor throw is permanent, while a missed deadline — main-thread starvation during a multi-million-element load, not a broken worker — is retried up to 3 attempts total (≈96 s worst case) with a growing backoff, driven from the per-frame scheduler and only when a visible, committed, order-dependent node actually wants sorting and no offline capture is draining (that path is time-bounded and cannot await a 30 s init); since the on-demand loop idle-pauses after `config.animation.idleTimeoutMs` and would otherwise never run the scheduler again, arming a retry also arms ONE `setTimeout` that calls `requestRender` just after the backoff expires (re-arming itself, rather than being spent, if there is no `requestRender` to call at that instant — an offline capture nulls it deliberately). Between attempts the cached rejection still absorbs every commit, which is what bounds the closure pile-up. A late success then has to RE-REGISTER, because the fresh worker holds no registrations and the coordinator retains no centers by design (they were transferred, or their thunks were never paid): it invalidates both freshness stamps on every unregistered still-committed sorted node and requests one reprocess, exactly as the switch-to-sorted mode change does, accepting one transient cost — a substitutive-LOD group drops to its coarsest ready level until the settle-gated reload climbs back. The stamp-less node keeps its exact cross-node `renderOrder`; only its within-mesh permutation stays stale until the re-commit. Every module write past an `await` in the init closure is epoch-guarded, because the 30 s deadline timer lives in that closure and a dispose cannot cancel it: without the epoch an orphaned timer would classify a miss against the NEXT app's healthy worker and spawn a third worker over it. `getDepthSortWorkerStatus()` exposes the resulting state (`idle`/`ready`/`starved`/`failed` + the deadline-miss count), also on `__luxarDebug`.

**The first frame after a commit (#2290):** the worker round trip means a commit's
ordering does not exist yet when the commit returns, so the frame that renders next
draws on whatever the commit path wrote. That used to be storage order
(`writeSortedIndexIdentity`) for every commit whose element count differed from the
previous one — and an nD re-slice changes the resident count at almost every step, so
a timelapse hit it at EVERY timepoint. On the `cloud` demo (4D Points, `volumetric`)
that was ~14 unsorted frames a second, and it reads as a flash rather than as
staleness. Two changes close it, and they compose. First, the commit paths now pass
`repairFromCount` when the count moved but the buffers did not, and the adapters call
`repairSortedIndexForCount` instead: the existing permutation is compacted (shrink) or
extended (grow) into a valid permutation of the new `[0, count)` rather than discarded.
Second, `noteDepthSortCommit` computes the ordering SYNCHRONOUSLY, on the main thread,
for eligible instanced nodes within `config.depthSort.syncSortMaxElements` (default
250,000), and publishes it live via `writeSortedIndexOrderingLive` — no staging, because
a permutation computed in one shot has no partial state to hide. The ceiling is a
shared element budget between frame evaluations, not a per-node allowance. The kernel
is the TypeScript reference `sort_splats_by_depth` (exact-parity with the Rust one, and
the WASM module lives in the worker), a counting sort measured at 0.8 ms for 34k
elements, 2.5 ms for 250k and 16.2 ms for 1M — enough to bound the whole commit batch.
The async pipeline is deliberately untouched: the node still registers and still
dispatches, because keeping the ordering current as the camera moves is still its job;
the worker's answer is the same permutation landing as a no-op overwrite. Measured on
the live demo at a frozen camera
pose, as the fraction of sampled element pairs composited in correct back-to-front
order across a timepoint step: storage order 0.617, repaired 0.858, sorted 1.000. (When
measuring anything like this, freeze the camera pose first — demos open in cinematic
mode with the camera orbiting at ~13.85 deg/s, and an ordering scored against a pose it
was not sorted for reads 0.161, which is stale in CAMERA, not in data.)

**Depth sorting a mesh (the indexed apply):** a mesh is one indexed `drawElements`, not `elementCount` instanced quads, so it has no `aSortedIndex` indirection to permute — the draw order of its triangles IS the order of the index buffer. Everything upstream is shared and unchanged: `commit-mesh-geometry.ts` calls the same `noteDepthSortCommit` with a LAZY provider computing face centroids (the mean of a triangle's three projected vertices), and the same worker and kernel return the same back-to-front permutation. The apply lives in `depth-sort-coordinator/triangle-ordering.ts` and differs in two ways that are forced rather than chosen. First, it is **atomic**: `geometry.index` is bound state, so no uniform can select between two index buffers and the double-buffer + chunked-stream trick does not transfer — and a half-written index buffer is not a permutation (some triangles drawn twice, others not at all), so slicing it would tear rather than merely lag. The whole visible prefix goes in one `set` + one update range, which is exactly the upload `applyMeshIndices` already performs on every slice move. Second, it permutes the **canonical** triples the commit produced (`ProjectedMeshData.indices`, retained on the node's sort state only while it is being sorted), never the live buffer — permuting the buffer would compose successive permutations, which looks perfect on the first sort and scrambles the surface on the second. Two things fall out for free: `gl_VertexID` for an indexed draw is the fetched index, so **picking is invariant under the permutation** (no slot syncing needed), and `mesh.geometry.index` staying the same attribute object means no `RenderObject` invalidation. Mesh also needs the `committedData` stamp for the coordinator's demotion gate, so `commitMeshGeometry` writes it even though mesh has no memoized-noop path of its own.

**Camera-triggered re-sorts (Phase 3):** `evaluateDepthSortPerFrame` runs as the `'depth-sort-scheduler'` per-frame callback (registered beside `'lod-group-selector'`, same allocation-free invariant). Each dispatch records the pose it sorted from — specifically the model-view matrix's z-row (axis direction + normalized offset), which fully determines the kernel's permutation — and a re-sort fires when the node-relative view axis rotates past `config.depthSort.angleThresholdDeg` (default 3°) or the camera translates along it past `config.depthSort.translationFraction` (default 0.05) × the node's bounding-sphere radius (translation orthogonal to the view axis provably cannot change a view-z ordering and is ignored). Hysteresis is dispatch-updates-reference: a triggered node goes quiet until the camera moves past the threshold again. The scheduler skips nodes while a view update is in flight (the pending commit sorts anyway), while a sort is already outstanding, and for invisible (own flag OR any ancestor — an LOD level can be a hidden group)/LOD-demoted/mode-switched-away meshes. The camera reaches the coordinator through a live `getCamera` getter (never a captured reference — the ortho-mode toggle replaces the camera object). `?depthSort=0` (embedders: `LuxarAppOptions.depthSort = false`; or `config.depthSort.enabled = false`) pins identity storage ordering and prevents the SortWorker from spawning, while the cross-node pass still applies authored `layer_order` bands and the physical-glass draw-first bias (deterministic E2E/visual runs). Each sort is recorded as a 'Depth Sort' pass in the update profiler spanning the whole dispatch→applied lifecycle: the pass stays open across the chunked apply and the slot flip, then closes only from the render mesh's `onAfterRender` acknowledgement after THREE has consumed the pending attribute ranges and drawn the selected buffer (byte tag upgraded `sched` → `up`, plus an `applyMs` resolve→applied split). A hidden/off-screen selection is not falsely reported as uploaded; if it is superseded or torn down before a completed draw, the pass closes still labeled `sched`. The data-loading monitor's Performance tab shows it as a third timing section.

**Cross-mesh (inter-node) ordering:** depth sorting orders elements WITHIN a mesh, but a `kind=partition` gsplat is multiple sibling meshes, and THREE orders transparent _objects_ by their `matrixWorld` origin — which is the shared world origin for every part (element centers are baked into the geometry), so THREE's per-object key is identical and the parts would draw in fixed creation order, not back-to-front. So `evaluateDepthSortPerFrame` ALSO assigns every visible sorted-mode mesh the coordinator tracks (all four geometry types) a `renderOrder` each frame — THREE sorts transparent objects by `renderOrder` before `z`, ascending, so the lowest (farthest) draws first. Cleared to 0 when a mesh leaves the sorted modes.

Because `renderOrder` is compared **globally** across all transparent meshes, the assignment is a collect-then-assign pass (`assignGlobalRenderOrder`, in the `depth-sort-coordinator/render-order.ts` submodule) that puts every sorted-mode mesh on ONE sequential integer scale (1..M, farthest first): meshes group by partition wrapper (single leaves are groups of one), groups order by the mean view-z of their members' content centroids (a documented approximation — wrappers/leaves are normally spatially disjoint datasets), except that a group whose bounding sphere strictly contains another group's is forced to draw FIRST (a priority topological pass — an embedded node, e.g. a tiny reference marker inside a huge cloud, would otherwise be erased by the container's transmittance for ~half of all camera orientations in an order-dependent mode; container-first lets embedded content composite on top, and the strictly-larger→smaller edges cannot cycle), and within a group the order comes from one of two paths:

- **BSP tree.** Native `partition=` adders and gsplat spatial-partition producers (`tiles`/`adaptive`, content/uniform tiling, and batch merge) store their recursive split planes as a `bsp_tree` attr (see `docs/guides/user/LUXAR_ZARR_FORMAT.md` and `docs/specs/GSPLATS_ZARR_FORMAT.md`). `load-partition-group-node.ts` validates the tree structure and, when the parts have `position_bounds`, checks its split geometry; malformed or unsound trees drop to the centroid fallback, while well-formed trees without verifiable bounds retain the stored ordering. Accepted trees are stashed on the wrapper `THREE.Group`'s `userData.bspTree`, and each part object is stamped with its `userData.partIndex`. The coordinator transforms the camera into the wrapper's local space and traverses the tree back-to-front (Fuchs–Kedem–Naylor: at each split recurse the far side of `split` first), numbering the parts into ranks (0 = farthest). Each split `axis` is a **stored center-column** index, so `bspAxisToComponent` maps it to the on-screen component through the current `displayDims` before traversal — the two coincide only for `displayDims == [0, 1, 2]`, and a scene displaying e.g. `[1, 2, 3]` would otherwise order along the wrong axis. The result is exact for point/gsplat BSP cells and approximate for centroid-split lines/mesh or overlapping uniform tiles, where geometry can cross a cut. Memoized per wrapper per frame.
- **Centroid (fallback).** A single-leaf scene (one mesh — a harmless no-op), a legacy/streamed partition with no `bspTree`, or a partition **any of whose split axes are not currently displayed** (its plane then carries no on-screen depth information, so `bspAxisToComponent` returns `null` and BSP ordering would be along a hidden axis) falls back to the part's content-centroid view-space z (bounding-sphere center through the model-view matrix). Per-_object_ ordering: approximate, and it degenerates when the camera is inside the volume — which is why the BSP path exists.

Both run independent of the within-mesh re-sort hysteresis (cheap). The global scale orders all sorted-mode (effective `needsDepthSort`) meshes against each other — all four geometry types, mixed freely (multiple partitions, partition + leaf, multiple leaves). Live ranks use 1..M, reserving THREE's default `renderOrder` 0 for empty parts that have not committed and for geometry outside the sorted set (commutative modes); those objects draw before the farthest sorted mesh, and depth interleaving with them stays out of scope. Even with a correct part order, elements whose footprints spill across a tile boundary can still interleave — only per-element / single-mesh sorting is exact there. Additive/luminous/max are commutative so their order is irrelevant.

### 5. Mesh Material

The odd one out, and deliberately so: **mesh is the only geometry type that shades.**
The other three are soft, emissive, per-element sprites whose fragment stage computes a
falloff and emits colour, with no notion of a surface orientation. A triangle has one.
See `materials/mesh/README.md` and `docs/specs/MESH_NODE_SPEC.md` §6.2.

**Structurally different from the three above:**

- **A plain indexed `BufferGeometry`**, not an instanced quad — so there is no element
  texture, no `texelFetch` prologue, no `aSortedIndex` indirection and no buffer pool.
  Per-vertex data arrives in ordinary vertex attributes.
- **Camera-aware for half the contract.** A mesh has no screen-space footprint to
  size, so `fov` / `resolution` are ignored — but `uIsOrtho` / `uNearCull` are bound
  and broadcast, because the shared `perspectiveNearFade` applies to a surface as
  much as to a sprite. Mesh evaluates it PER FRAGMENT (a triangle spans depth) with a
  per-fragment reject below 0.01; see the stage table in
  `materials/_shared/README.md`.
- **`opaque` by default**, unlike the siblings' `additive`: the only mode
  unconditionally correct without per-triangle depth sorting (§9 defers that), and what
  a surface should look like. The default is per geometry type
  (`defaultBlendingMode` in `types/geometry-capabilities`) and applied by each consumer
  when the composed `blending_mode` is `undefined` — composition itself preserves the
  unset state, so "nothing in the ancestry set a mode" survives to the consumer.

**Key features:**

- **View-anchored offset key light**, no scene light and no scene-graph change. Wrapped diffuse uses a fixed above-left `L`; additive Blinn–Phong uses the constant half-vector between `L` and the fixed view axis `V = (0, 0, 1)`.
- **Two normal sources, chosen at COMPILE time** (`LUXAR_MESH_FLAT_NORMAL`): the stored
  `normal` attribute when `shading == "smooth"` AND `normal_dims` equals the displayed
  axes, else screen-space derivatives of the view position. A compile-time variant
  rather than a runtime branch because a declared-but-unbound attribute reads
  `(0, 0, 0, 1)` — there is no runtime value meaning "no normals".
- **`opaque` is a hard alpha CUTOUT**, so node `opacity` sweeps a threshold rather than
  dimming; a smooth fade means selecting `normal`.
- **Five per-node appearance knobs** — `ambient`, `shade_exponent`, `specular`, `shininess`, `alpha_cutoff` —
  exposed as the mesh-only Layers-panel sliders.

**Two hazards worth knowing before touching these shaders**, both of which fail on
exactly one backend and neither of which the parity harness can see (it compiles TSL
_to_ GLSL):

- `cross(dFdx(P), dFdy(P))` carries the sign of the fragment-space y axis, and GLSL's
  `dFdy` is bottom-up where WGSL's `dpdy` is top-down. The derivative normal is
  therefore **forced** viewer-facing (`z >= 0`), which makes it convention-independent.
  Measured, so this is not overstated: on Chrome + Apple Silicon a real-WebGPU A/B with
  the flip REMOVED renders a face-on flat quad identically to WebGL, so the conventions
  coincide there and the flip is currently inert on that platform. Kept because it costs
  one instruction, is correct under either convention, and neither spec promises they
  agree — insurance, not a fix for an observed bug.
- Three's `transformNormalToView` normalizes internally, and the writer accepts
  zero-length normals with a warning (degenerate triangles legitimately produce them) —
  `normalize(vec3(0))` is NaN, which then interpolates across every triangle touching
  that vertex. The transform is spelled out as
  `viewMatrix · (modelNormalMatrix · n)` instead.

### 6. Material Manager

Singleton manager for material creation across Points, Lines, GSplats and Mesh (ALL per node — each instanced material binds its node's element texture; the historical Lines LRU died with the lines texture-storage migration). Runtime blending changes use `blending-state.ts` so UI updates apply the same complete THREE.js state as material creation.

```typescript
// Get the per-node point material
const pointMaterial = materialManager.getPointMaterial({
  blendingMode: 'additive',
  opacity: 1.0,
  gamma: 1.0,
});

// Get the per-node line material
const lineMaterial = materialManager.getLineMaterial({
  blendingMode: 'additive',
  opacity: 1.0,
  intensity: 1.0,
  offset: 0.0,
});

// Update global parameters (updates both point and line materials)
materialManager.updateCameraParams(fov, resolution);
```

#### Material Lifecycle and Memory Management

**Automatic Disposal**: Materials are automatically registered with the MaterialManager when created and unregistered when disposed. This prevents memory leaks.

```typescript
// Materials are cached and reused automatically
const material1 = materialManager.getPointMaterial({ opacity: 1.0 });
const material2 = materialManager.getPointMaterial({ opacity: 1.0 }); // Distinct instance (per-node)

// When disposing geometry/points, material is automatically handled
points.geometry.dispose(); // Frees GPU buffers
// Material manager keeps material alive if other objects use it
```

**Global Updates**: When camera settings change, MaterialManager automatically updates ALL registered materials - no manual scene traversal needed. Global exposure/offset/gamma are handled inside the mega-shader post-processing pass, not per-material.

```typescript
// Updates all materials in the scene automatically
materialManager.updateCameraParams(newFov, newResolution);
```

**Memory Leak Prevention**: Always dispose geometries and points when done. The material system handles cleanup automatically.

**Key Points**:

- Line materials are cached by properties (opacity, gamma, intensity, offset, blending mode); point and gsplat materials are per node (each carries its node's element texture)
- Global uniform updates affect all materials simultaneously
- Disposal is automatic - no manual material cleanup needed
- Thread-safe caching prevents duplicate material creation

### 7. GPU Buffer Pool

The `GPUBufferPool` manages geometry reuse for Points, Lines, and GSplats, eliminating per-frame GPU allocations.

**Key Features:**

- Size-based bucketing: reuses geometries when size and type match (0ms GPU allocation)
- In-place data updates (fused texel writes for all three geometries — lines joined in the PR-C texture-storage migration)
- Count-based and byte-budget eviction (`gpuPoolMaxBytes`, `gpuPoolEvictBatchSize`)
- Multi-type support: Points, Lines, and GSplats (all three carry a fixed scalar/alpha texel slot; lines stamp presence via `userData.hasScalars`)

### 8. Adaptive DPR Manager

The `AdaptiveDPRManager` dynamically adjusts device pixel ratio based on real-time FPS, trading resolution for frame rate when needed. It is a facade over pure, timestamp-driven modules in `adaptive-dpr/` (FPS tracker, stall detector, refresh-rate estimator, hysteresis tracker, probe controller, bounds ledger — see that folder's README).

**Control loop:**

- Samples FPS using a 1-second sliding window, evaluated every 500 ms; a frame gap counts as dead time (window reset, pending probe voided) only when it is BOTH > `gapResetMs` and strictly more than 4× the median of the four PRECEDING inter-frame intervals (the interval under test is never part of the median it is judged against — with a short post-boundary memory that made a startup stall read as "the frame rate"), so an isolated stall or idle-resume is still discarded while a genuinely slow cadence is kept and the loop keeps adapting below ~1000/`gapResetMs` fps (a software rasterizer at 0.5 fps used to sit at native DPR forever); the sliding window likewise retains a two-sample minimum so the estimate stays defined slower than the window itself. A dead-time reset no longer touches the evaluation clock at all; what keeps dead time from counting as progress toward the next evaluation is that a tick with no frame rate to judge is not an evaluation and does not consume the interval budget. Pinned to the frame’s own timestamp (the pre-fix line), a hitch recurring more often than 500 ms re-pinned the clock forever and the loop never evaluated at all (measured: zero evaluations in 43 s on a 16.7/16.7/400 ms cadence). Clamping that pin to one interval before the frame was tried and measured INERT — identical evaluations, DPR, floor and applied-ratio counts on six cadences — so the return-value rule is the whole fix. Up to two dead intervals in a row stay outliers and are discarded; by the third, half the cadence memory is dead time and it is absorbed as the frame rate, so the window it lands in is treated as unrepresentative for a couple of intervals (scale-downs apply, nothing is learned). The protection is bounded and counted in INTERVALS — a burst of up to four teaches nothing — and that is the only form the guarantee takes: what it buys in wall clock depends on the interval length and on how much of its own window a probe can still gather, so it has to be measured per cadence rather than stated as a number (measured for consecutive 400 ms hitches inside a 60 fps session: eight, 3.2 s, leave the floor untouched; nine, 3.6 s, pin one). Residual limitation: a dead period alternating one-for-one with a SINGLE fast frame lands the median between the phases, so the dead time is kept as "the frame rate" and the FPS window mixes it with render cost — the manager still adapts, but the rate it reports is not the rate the user perceives
- Thresholds are RELATIVE to the display's estimated achievable rAF rate: scale down below `scaleDownFpsRatio × cap`, count toward scale-up above `scaleUpFpsRatio × cap` (works unchanged on 30/60/120/144 Hz; the estimator holds a high-water mark, lower-bounded by `refreshRateFallback` until genuine rAF throttling is detected)
- Scale-up fires after `hysteresisSeconds` of sustained high FPS, with a small mid-band grace so isolated dropped-frame samples don't restart the wait
- The DPR walks multiplicatively below the session CEILING — the LIVE `window.devicePixelRatio` (re-read on every evaluation and public read; a monitor/zoom change rebases all learned state and clamps an engaged override — no supersampling on a lower-DPI display), capped by the allow-high-DPR setting, which is OFF by default and pins the ceiling at 1.0 even on a HiDPI display (see `pixel-ratio-cap.ts`) — and stops strictly above `max(minDPR, learned floor)`

**U-shape probe and learned bounds:**

- Every scale-down is probe-verified: if FPS did not improve ≥ `probeImprovement` on a clean sample (`probeMinSamples`, half-window span, not load-suppressed), it reverts and the probed DPR becomes a floor; unclean probes void as inconclusive (nothing learned)
- Repeated identical rejections back off exponentially (`floorTtlMs` × `backoffMultiplier`^n, capped) — no eternal probe/blur cycle on scenes DPR reduction can't help
- On HiDPI displays, the operating ceiling demotes from native to exactly 1.0 for the session (TTL-decayed, backed off on re-demotion) on either kind of evidence: repeated "punished ascents" (a scale-up above 1.0 followed by an FPS collapse), or sustained sub-throttle DISTRESS — FPS below what any real display mode can produce (< ~22 fps) for a sustained period, which the estimator reports instead of ever latching its throttle verdict down there (pre-fix that latch collapsed the cap onto the loaded FPS and parked the DPR at native on exactly the scenes that needed help)
- `notifyContentChanged()` (dataset/layer/LOD changes) pulls learned-bound expiries forward and resets backoff streaks; `setLoadActivityPredicate()` suppresses all learning while data loads
- A probe whose measurement window spans a content change is CONFOUNDED — its baseline was measured on the old content — so it teaches nothing at all: the reduction is kept, the DPR is never reverted upward, no floor is pinned, and the rejection-backoff streak is untouched. It does NOT stop the walk: under sustained content churn no clean experiment exists, so the pixel ratio keeps descending unratified toward `minDPR`, which is the distress response (fewer pixels never hurt a stuttering loop) and lifts again through the normal scale-up hysteresis once content settles. What discarding the verdict buys is the absence of thrash — measured over 20 minutes of per-frame churn at 0.33 fps, acting on those verdicts produced 100 reductions and 98 reverts back up (202 renderer pixel-ratio applications), while discarding them produced 13 monotone reductions and no revert. An earlier revision held the walk on a churn clock instead; that mechanism was measured to invert its own goal (its window was the same 5 s constant the notifications coalesce on, so churn arriving just slower than that window was fully acted upon and completely invisible to the hold, which then walked further down — to `minDPR` — than not having it at all, while per-frame churn pinned a 0.33 fps scene at a ratio of 1.62 indefinitely) and was removed

**Idle/pause integration:**

- `notifyPaused()` (from `stopAnimation`) clears session state only — learned bounds survive
- At idle-pause the resting frame is restored to full native sharpness (`prepareIdleFrame()` + one direct render); `notifyResumed()` snaps back to the remembered operating DPR in one step
- Frames are not recorded while the rendering context is lost. Known limitation: under `?renderer=webgpu`, device loss is handled via a pipeline latch (unrecoverable this release), and WebGPU async pipeline-compile jank inside probe windows is not specially detected — the probe backoff bounds the damage
- `?dpr=<value>` pins a fixed pixel ratio and locks adaptation off for the session (deterministic E2E/visual runs)

### 9. Colormap Data

`colormap-data.ts` contains the auto-generated lookup tables (LUTs) for all built-in colormaps (e.g., viridis, magma, turbo). Each LUT is a flat Uint8 array of RGBA values.

### 10. Colormap Textures

`colormap-textures.ts` manages creation and caching of `THREE.DataTexture` instances from built-in and custom colormap LUTs. Built-in textures live for the app lifetime; custom LUT textures are bounded and can be disposed on dataset unload.

### 11. Global EOG (Exposure-Offset-Gamma)

Applied inside the mega-shader fragment before the tone-mapping operator, in a single fullscreen pass:

**EOG Uniforms:**

- `uExposure`: Log2 stops (`color * 2^exposure`)
- `uGlobalOffset`: Additive shift (`color + offset`)
- `uGlobalGamma`: Power curve (`pow(color, 1/gamma)`)

---

## Effects Library

### Core Effects

#### Bloom

HDR bloom via the separate `BloomChain` pre-pass:

- Rec.709 luma threshold with soft knee
- Configurable intensity, radius, and mip-count (1-12)
- Output texture sampled and additively mixed by the mega-shader

#### Tone Mapping

Multiple tone mapping operators (all run inside the mega-shader via THREE's `<tonemapping_pars_fragment>` chunk):

- ACES Filmic (default) - Industry standard cinematic look
- Neutral - Gentle rolloff; well above 1.0 it desaturates hard (hue is kept, chroma is not) — inside [0, 1] use None instead, which is exact
- AgX - Modern alternative
- Reinhard / Cineon - Classic operators
- Linear - Clamp/saturate to [0, 1]

### Anti-Aliasing

#### FXAA

Fast Approximate Anti-Aliasing:

- Very fast performance
- Good quality for most cases
- Single-pass implementation
- **Recommended for general use**
- Works perfectly with additive blending

#### MSAA

Multisample Anti-Aliasing:

- Hardware-accelerated
- Sample counts: 2x, 4x, 8x
- **⚠️ WARNING**: Incompatible with additive blending
- Causes brightness multiplication artifacts with points
- Only use with normal blending mode

#### SSAA

Super-Sample Anti-Aliasing:

- Renders at higher resolution (1.5x, 2x, 3x, 4x)
- Best possible quality
- **Heavy performance cost**
- Recommended only for screenshots or high-end GPUs

### Cinematic Effects

#### Vignette

Screen edge darkening (multiplicative stage in the mega-shader):

- Adjustable darkness
- Configurable offset
- No alpha-overflow artifacts (mega-shader forces alpha = 1.0 at the final write)

#### Chromatic Lens Distortion

Physically accurate lens distortion with wavelength-dependent chromatic aberration (per-channel sampling stage in the mega-shader):

- **Wavelength-dependent distortion**: Blue refracts more than red (optical dispersion)
- **Realistic chromatic fringing**: Follows lens geometry (stronger at edges)
- Full camera model: Distortion, principal point, focal length, skew
- Barrel/pincushion distortion for wide angle/telephoto simulation

#### Detector Noise (Physics-Based)

Realistic camera/detector noise simulation for scientific imaging:

- **Shot Noise (Poisson)**: Signal-dependent noise from photon statistics
- **Readout Noise (Gaussian, temporal)**: Signal-independent electronic noise, varies per frame
- **Fixed Pattern Noise (Gaussian, static)**: Per-pixel offset from detector non-uniformities
- Configurable photon gain for low-light simulation
- Uses efficient GPU approximations (Bob Jenkins hash, clamped logistic, Anscombe transform)
- **Recommended for scientific visualization aesthetics**

**Physics model:**

```
I_observed = Poisson(I_true / gain) × gain + Gaussian_temporal(0, σ_read²) + FPN(pixel)
```

```typescript
// Enable physics-based detector noise
postProcessing.setDetectorNoiseEnabled(
  true, // enabled
  0.01, // readoutSigma: Temporal readout noise (0-0.1)
  0.01, // photonGain: Shot noise visibility (0.0001-0.1)
  0.005 // fpnSigma: Fixed pattern noise (0-0.05)
);

// Update parameters dynamically
postProcessing.updateDetectorNoiseSettings({
  photonGain: 0.05, // Simulate low-light conditions
  fpnSigma: 0.01, // Add more fixed pattern noise
});
```

---

## Anti-Aliasing Recommendations

### For Best Results with Point Clouds

1. **General Use**: Enable **FXAA** - fast and effective
2. **Maximum Quality**: Enable **SSAA** at 2x (heavy performance cost)
3. **Avoid MSAA with additive blending** (used by Points / GSplats)

### Troubleshooting

#### SSAA Issues

- If viewport appears cropped, restart the viewer
- Performance impact scales quadratically with multiplier

#### MSAA Not Working

- Check console for GPU support warnings
- Under WebGL2, MSAA requires float-buffer extensions on the active context; check `RendererCapabilities.maxMSAASamples > 0`
- Under WebGPU, MSAA is native (no extension required); the same `maxMSAASamples` query reports the adapter's supported sample counts
- Will not show effect with additive blending
- Try switching to normal blending mode to verify

#### Performance Tips

- Start with FXAA for best performance
- SSAA should only be used for final renders
- Monitor FPS when enabling AA effects

---

## Pipeline Architecture

### Rendering Flow

```
Scene Geometry
    ↓
Custom Point / Line / GSplat Shaders (HDR colors)
    ↓
HDR Render Target (HalfFloatType, optional MSAA + SSAA)
    ↓
Bloom pre-pass (if enabled): threshold + mip pyramid → bloom texture
    ↓
Mega-shader fullscreen pass: fuses
   ChromaticLensDistortion → Bloom mix (additive) → DetectorNoise →
   EOG → ToneMapping → Vignette → sRGB encode
    ↓
Optional FXAA post-pass on the tone-mapped LDR output
    ↓
Canvas backbuffer
```

### Performance Optimizations

1. **Single fused pass** for per-pixel effects (one rasterization, one set of binds)
2. **Cached programs**: Toggle defines trigger lazy recompile, then the program is cached by define-set
3. **Material caching**: Reuse materials with same properties
4. **Selective AA**: Choose AA method based on performance

---

## Configuration

### HDR Configuration

```typescript
// Renderer setup for HDR (the PostProcessingManager pins these)
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;

// HDR render target (managed by PostProcessingManager)
new THREE.WebGLRenderTarget(width, height, {
  type: THREE.HalfFloatType,
});
```

---

## Usage Examples

### Basic Setup

```typescript
import { PostProcessingManager } from './rendering/post-processing/post-processing-manager';
import { materialManager } from './rendering/material-manager';

// Initialize post-processing
const postProcessing = new PostProcessingManager(renderer, capabilities, scene, camera, {
  width: canvas.width,
  height: canvas.height,
});

// Create point material
const material = materialManager.getPointMaterial({
  blendingMode: 'additive',
  opacity: 1.0,
  gamma: 1.0,
});
```

### Configuring Effects

```typescript
// Enable multiple effects
postProcessing.updateBloomSettings(0.3, 1.0, 0.01);
postProcessing.setToneMapping(THREE.ACESFilmicToneMapping);
postProcessing.setFXAAEnabled(true);

// Add cinematic effects
postProcessing.setVignetteEnabled(true, 0.5, 0.5);
postProcessing.setChromaticLensDistortionEnabled(true, -0.05, -0.05, 0.03);
```

### Render Loop

```typescript
function animate() {
  requestAnimationFrame(animate);

  // Update controls
  controls.update();

  // Render with post-processing
  postProcessing.render();
}
```

---

## Performance Guidelines

### Optimization Strategies

1. **Start Simple**: Begin with bloom and tone mapping only
2. **Add Selectively**: Enable effects based on performance budget
3. **Profile First**: Measure impact before adding effects
4. **Use Quality Presets**: Match quality to hardware capability
5. **Monitor FPS**: Disable effects if FPS drops below target

### Performance Impact (1M points, 1080p, indicative)

| Effect                                        | Performance Cost |
| --------------------------------------------- | ---------------- |
| Base Rendering                                | ~5ms             |
| Bloom (BloomChain)                            | ~2ms             |
| Mega-shader fused pass (everything per-pixel) | ~0.5-1ms         |
| FXAA                                          | ~0.5ms           |

---

## Troubleshooting

### Common Issues

**Problem: Black screen after enabling effects**

- Check browser console for WebGL/WebGPU errors (the message prefix tells you which backend is active)
- Verify HDR buffer support via the backend-agnostic `RendererCapabilities.hdr.floatTextures` (don't probe `renderer.capabilities.isWebGL2` — that's WebGL-only and silently undefined under WebGPU)
- Try disabling effects one by one to isolate the issue
- Verify tone mapping mode is set (required for HDR pipeline)
- See `post-processing/README.md` Troubleshooting for known causes (e.g. missing `toneMapped: false` on a custom material)

**Problem: Poor performance with all effects**

- Disable bloom first (~2 ms at 1080p, plus mip allocations)
- Reduce bloom mipmap levels: `setBloomLevels(3)` instead of default 8
- Disable detector noise if not needed
- Lower SSAA multiplier or disable: `setSSAAEnabled(false)`

**Problem: Colors look wrong**

- Verify tone mapping operator: try 'AgX' or 'ACES Filmic' instead of 'Reinhard'
- Check exposure value in HDR controls
- Ensure proper color space: `renderer.outputColorSpace = THREE.SRGBColorSpace`
- Verify bloom threshold isn't too low (washing out colors)
- Check gamma correction in materials (should be 1.0 for linear workflow)

**Problem: Effects not visible**

- Check effect enabled state in debug console
- Verify threshold values (bloom threshold too high will gate everything)
- Confirm `LUXAR_TONE_MAPPING_MODE` matches the THREE constant you set
- Confirm the mega-shader is the bound material (capture-mode defines bypass downstream stages)

**Problem: Thin lines have aliasing/gaps**

- Line material automatically handles this with 1.5px minimum width
- Ensure anti-aliasing is enabled (FXAA or SSAA)
- Check that line widths are properly set (not zero or NaN)
- For very thin lines, increase width slightly or use higher SSAA

**Problem: Bright artifacts in dark areas (additive blending)**

- The mega-shader forces fragColor.a = 1.0 at the final write, so this is unlikely now. If you see it, check that no upstream custom material is propagating NaN/Inf into the HDR target.

**Problem: nD slicing shows no points**

- Points with zero effective radius are filtered in fragment shader
- Navigate to a different slice where points intersect the hyperplane
- Check dimension ranges and current slider positions
- Verify the dataset has points in the current nD region

**Problem: Material cache thrashing (many materials created)**

- Materials use integer bucketing to group similar values
- Small variations (e.g., opacity 0.999 vs 1.0) create separate materials
- Use consistent values: prefer 1.0, 0.5, 0.25 instead of arbitrary floats
- Check cache statistics: `materialManager.getCacheStats()`

**Problem: Out of memory with large datasets**

- Enable chunked loading in data loader
- Reduce bloom mipmap levels
- Use lower SSAA multiplier
- Consider using lower encoding mode (AGGRESSIVE)

---

## API Reference

### PostProcessingManager

| Method                                                       | Description                                                                                                                                                                                                                       |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `render()`                                                   | Execute rendering pipeline                                                                                                                                                                                                        |
| `setBloomEnabled(enabled, strength?, radius?, threshold?)`   | Enable/disable bloom (and update settings)                                                                                                                                                                                        |
| `updateBloomSettings(strength?, radius?, threshold?)`        | Update bloom settings                                                                                                                                                                                                             |
| `setBloomLevels(levels)`                                     | Set bloom mip pyramid depth (1-12)                                                                                                                                                                                                |
| `setToneMapping(mode)`                                       | Set tone mapping operator (THREE.ToneMapping)                                                                                                                                                                                     |
| `updateExposure(value)` / `getExposure()`                    | EOG exposure (log2 stops)                                                                                                                                                                                                         |
| `updateGlobalOffset(value)` / `updateGlobalGamma(value)`     | EOG offset and gamma                                                                                                                                                                                                              |
| `setFXAAEnabled(enabled)`                                    | Toggle FXAA post-pass                                                                                                                                                                                                             |
| `setMSAAEnabled(enabled)` / `setMSAASamples(n)`              | Toggle MSAA on the HDR target / set sample count                                                                                                                                                                                  |
| `setSSAAEnabled(enabled)` / `setSSAAMultiplier(value)`       | Toggle SSAA / set supersampling factor                                                                                                                                                                                            |
| `setDetectorNoiseEnabled(enabled, sigma?, gain?, fpnSigma?)` | Configure physics-based detector noise                                                                                                                                                                                            |
| `updateDetectorNoiseSettings(params)`                        | Update detector noise parameters                                                                                                                                                                                                  |
| `setVignetteEnabled(enabled, darkness?, offset?)`            | Configure vignette                                                                                                                                                                                                                |
| `setChromaticLensDistortionEnabled(enabled, ...params)`      | Configure chromatic lens distortion                                                                                                                                                                                               |
| `updateChromaticLensDistortion(params)`                      | Update chromatic lens distortion params                                                                                                                                                                                           |
| `getLensDistortionParams()`                                  | Read distortion uniforms (cloned, for picking)                                                                                                                                                                                    |
| `captureHDRPixels(mode?)` / `captureHDRAsEXR(opts?)`         | Read HDR/LDR pixels for EXR export                                                                                                                                                                                                |
| `renderToImageData()`                                        | Render once and read back as ImageData (sRGB)                                                                                                                                                                                     |
| `rebuildAfterContextRestore()`                               | Rebuild GPU resources after a WebGL2 `webglcontextrestored` event. WebGPU device loss uses a different model (`device.lost` promise) and is currently treated as unrecoverable — see `scene-manager.ts::setupContextLossHandling` |
| `setDPRScale(value)`                                         | Apply an adaptive DPR scale                                                                                                                                                                                                       |
| `startDeferRebuild()` / `endDeferRebuild()`                  | Defer rebuilds during bulk changes (no-op in mega-shader pipeline)                                                                                                                                                                |
| `dispose()`                                                  | Clean up resources                                                                                                                                                                                                                |

---

## Future Enhancements

- Temporal Anti-Aliasing (TAA)
- Screen Space Reflections (SSR)
- Motion Blur
- Lens Flare
- God Rays
- Color Grading with LUTs
- Custom Effect API

---

## Contributing

When extending the rendering system:

1. **Add per-pixel effects to the mega-shader** rather than as a new full-screen pass — see `post-processing/mega/shader.glsl.ts` and `post-processing/mega/material.ts`
2. **Bracket new effects with `#ifdef USE_*` defines** so disabled effects compile out entirely
3. **Test performance** across different hardware
4. **Document settings** and performance impact
5. **Maintain HDR pipeline** integrity (custom materials must set `toneMapped: false`)

---

## License

Part of the Luxar project. See root LICENSE file for details.

---

_For implementation details, see the source files in this directory._
