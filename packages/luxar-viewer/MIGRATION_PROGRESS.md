# WebGPU migration progress tracker

Single source of truth for where the migration stands. Edit the
checkboxes in-place as port PRs land.

For *prep work* status, see git history on branch
`chore/threejs-r184-migration` — all prep is committed; see
`BROWSER_SUPPORT_POLICY.md`, `WEBGPU_FALLBACK_REPORT.md`,
`THREE_VERSION_NOTES.md`, `BLENDING_PORT_NOTES.md`,
`MATERIAL_WRAPPER_DESIGN.md`, and
`src/rendering/picking/PICKING_DESIGN.md` for the decision artefacts.

## Blockers

These must be true *before* the renderer-port PRs begin in earnest.
Each blocker is an external dependency or design call.

- [x] **Raw-GL seam closed.** All `renderer.getContext()` calls hide
  behind `RendererCapabilities` (`renderer-capabilities.ts`); pre-
  renderer probes (canvas-context creation, WebGPU-availability
  detection) are enumerated in `scene-manager.ts`'s allow-list
  comment.
- [x] **Shader-source registry in place.** Every GLSL string lives
  in a `*-shaders.ts` / `*.glsl.ts` module exporting a `ShaderSource`
  value; materials consume `webgl.vertex/fragment` from the source.
- [x] **Capture paths Promise-typed.** `captureHDRPixels`,
  `renderToImageData`, `readBackbufferPixels` all return Promises.
- [x] **`Renderer` type alias exists** (`renderer-capabilities.ts`),
  ready to widen from single-arm to
  `WebGLRenderer | WebGPURenderer` at port time.
- [x] **Browser-support policy decided** (`BROWSER_SUPPORT_POLICY.md`).
- [x] **Picking strategy decided** (`PICKING_DESIGN.md` — async
  readback with stale-tooltip suppression).
- [x] **Three.js pinned to `~0.184.x`** (`THREE_VERSION_NOTES.md`).
- [x] **Material wrapper strategy decided** (`MATERIAL_WRAPPER_DESIGN.md`
  — in-place rewrite with `buildMaterial` helper).
- [x] **`onBeforeCompile` zero-use confirmed** (see
  `BROWSER_SUPPORT_POLICY.md` "Codebase-side hard guarantees").
- [x] **External `.uniforms.X.value =` writes eliminated**
  (`material-colormap-helpers.ts` delegates to `ColormapAwareMaterial`
  setters; see same section in `BROWSER_SUPPORT_POLICY.md`).
- [x] **Blending-mode port reference drafted** (`BLENDING_PORT_NOTES.md`).
- [x] **WebGPU fallback E2E experiment run** — initial attempt
  in M3 surfaced a playwright-config blocker (existing dev
  server reused, env var didn't propagate). Documented as
  deferred-to-M5 in `WEBGPU_FALLBACK_REPORT.md` — first
  meaningful per-shader data lands when FXAA's TSL factory is
  testable.

## Non-shader migration tasks

Tracked separately because they unblock the per-shader rows below.

- [ ] **TSL strict-typing playbook for scene materials** — TSL's
  type system requires explicit attribute generic types (e.g.
  `attribute('radius', 'float')` returns `AttributeNode<'float'>`
  which then dispatches to `NumExtensions<'float'>` and gains
  `.mul()`, `.add()` etc.). The dispatch keys off literal string
  generics — `attribute('radius')` with no `nodeType` yields
  `AttributeNode<unknown>` with no operators. M11 attempted a
  speculative port of `point` and surfaced this gap.

  Discovered separately (May 2026) while building the
  `tsl-shader-parity` harness (`src/tests/e2e/tsl-shader-parity.spec.ts`):
  - **`PointsNodeMaterial` is sprite-based in r184**, not
    `gl_PointSize`-driven. The TSL counterpart of our existing
    `THREE.Points + ShaderMaterial` setup is `THREE.Points +
    PointsNodeMaterial` with `material.sizeNode` (pixels) and
    `material.colorNode`. `gl_PointCoord` becomes the sprite's
    `uv` attribute. M11-M16 ports need to accept this
    architectural shift, not translate `gl_PointSize` literally.
  - **`screenUV` is Y-flipped under `WebGPURenderer({ forceWebGL:
    true })`** — it follows the WebGPU convention
    (`vec2(x, size.y - y) / size`). Fullscreen passes that want to
    match a GLSL `vUv = position.xy * 0.5 + 0.5` convention must
    use `uv()` (the geometry-attribute reader) instead. Fxaa + the
    three bloom factories were updated to use `uv()` after the
    harness caught the flip.
  - **`uniform(float(value))` creates a const, not a live
    uniform.** The TSL `float(...)` wrap builds a `VarNode<float,
    ConstNode>` which `uniform()` then bundles as an inlined
    constant — writes to `uniforms.X.value` no longer reach the
    shader. Pass the raw JS value (or `IUniform.value`) straight
    to `uniform()`. Bloom factories were updated.

  Patterns are now validated for fullscreen-pass shaders. The
  Points-sprite path remains the next prerequisite before M11
  resumes — needs a hello-world that exercises typed attribute
  reads against a real `THREE.Points` mesh.
- [ ] **`tsconfig.json moduleResolution` bump from `Node` to
  `Bundler`** — required to resolve `three/webgpu` types
  (currently shipped under `@types/three/build/three.webgpu.d.ts`
  which the legacy Node resolver can't find via the `exports`
  field). Discovered in M1 (Phase 0 pre-flight) attempting the
  type-only smoke import: `tsc` complained "Cannot find module
  'three/webgpu'". Flipping to `Bundler` resolves the WebGPU
  types but surfaces a zarrita typing mismatch in
  `lines-spatial-index-loader.ts:865`,
  `points-spatial-index-loader.ts:{1085, 1212}` — the
  `loadColorRanges` signature accepts
  `zarr.Array<zarr.DataType, zarr.FetchStore>` but callers pass
  `zarr.Array<zarr.DataType, zarr.Readable>`. Under Node
  resolution these widen to compatible types; under Bundler
  they're precise and incompatible. Resolution requires either
  widening the helper signature to accept `zarr.Readable` or
  narrowing the caller sites' array types. Estimated half-day.
  Must complete before M2 lands.
- [ ] **`MegaShaderMaterial` consumer delegation** — the mega TSL
  factory (`mega.tsl.ts`) lands in M9 with parity-tested coverage
  for default, USE_BLOOM, and USE_VIGNETTE configurations, but the
  consumer-side material (`mega-shader-material.ts`) still
  `extends THREE.ShaderMaterial`. Switching it to compose-via-
  `buildMaterial` is invasive because the existing setters
  (`toggleVignette`, `setToneMapping`, etc.) flip GLSL `defines`,
  which TSL doesn't have a runtime analog for — the TSL path has
  to re-call the factory on each toggle. Defer until the M18
  default-flip needs it; production stays on the WebGL2 path
  until then.
- [ ] **Mega detector-noise TSL port (M9-bis)** — the GLSL3
  `applyDetectorNoise` uses a Bob Jenkins hash chain + Anscombe
  variance-stabilising transform to approximate Poisson shot
  noise. TSL has the primitives (`bitcast`, `floatBitsToUint`,
  `shiftLeft`, `bitXor`) but the port is non-mechanical because
  the hash is a long sequence of `uint`-only ops. The mega TSL
  factory throws if `config.useDetectorNoise` is true so callers
  see a loud failure rather than a silent missing effect.
- [ ] **`setupRenderer` becomes async** — `scene-manager.ts`. Branches
  on `import.meta.env.VITE_LUXAR_USE_WEBGPU_RENDERER` to construct
  either `WebGLRenderer` or `WebGPURenderer({forceWebGL: true})`.
- [ ] **`Renderer` union widened** from `THREE.WebGLRenderer` to
  `THREE.WebGLRenderer | THREE.WebGPURenderer`
  (`renderer-capabilities.ts`).
- [ ] **`buildMaterial(source, uniforms, config)` helper** —
  per `MATERIAL_WRAPPER_DESIGN.md`. Lands alongside the first TSL
  port.
- [ ] **`readBackbufferPixels` body** swaps WebGL2 `gl.readPixels`
  for the WebGPU `buffer.mapAsync` equivalent
  (`renderer-capabilities.ts`). M17 deferred to M17-bis: needs
  `renderToImageData` (post-processing-manager) to render the
  full pipeline into an offscreen `WebGLRenderTarget` first,
  then `readRenderTargetPixelsAsync(target, …)`. WebGPURenderer
  doesn't expose the canvas backbuffer for direct readback. M17
  today documents the error path under WebGPU; the wider
  refactor lands when M18 flips the default (since today the
  WebGL2 path covers this case fine).
- [ ] **`readbackAndVote` becomes async** — per `PICKING_DESIGN.md`.
  `performPick` already gates on `_dirty`; the only edit is adding
  `async` to both methods and `await` at the readback call.
- [ ] **Stale-tooltip suppression** wired through the picking
  callback at `app.ts:782`.

## Shader / material port matrix

Twelve shader-pairs total. Each row tracks the per-shader migration.
A row is "ready to delete GLSL3" when every preceding box is checked.

Legend:
- **G**: GLSL3 strings present in `ShaderSource.webgl`.
- **T**: TSL factory implemented in `ShaderSource.webgpu`.
- **F**: Passes `shader-material-compile.spec.ts` under
  `WebGPURenderer({forceWebGL: true})`.
- **R**: Passes real-WebGPU smoke test (Chrome/Edge stable).
- **D**: GLSL3 strings deletable (after both F and R green).

### Scene materials

| Shader            | G | T | F | R | D |
|-------------------|---|---|---|---|---|
| `point`           | x |   |   |   |   |
| `line`            | x |   |   |   |   |
| `gsplat`          | x |   |   |   |   |

### Post-processing

| Shader              | G | T | F | R | D |
|---------------------|---|---|---|---|---|
| `mega`              | x | x | x |   |   |
| `bloom-threshold`   | x | x |   |   |   |
| `bloom-downsample`  | x | x |   |   |   |
| `bloom-upsample`    | x | x |   |   |   |
| `fxaa`              | x | x |   |   |   |

### Picking

| Shader        | G | T | F | R | D |
|---------------|---|---|---|---|---|
| `point-pick`  | x |   |   |   |   |
| `line-pick`   | x |   |   |   |   |
| `gsplat-pick` | x |   |   |   |   |

## Recommended port order

Smallest to largest, to surface design issues early and bound risk:

1. **`fxaa`** — single fullscreen pass, 2 uniforms, no
   instancing or custom blending. Lowest-risk first TSL port.
   This is where `buildMaterial` helper lands.
2. **`bloom-{threshold,downsample,upsample}`** — three small
   fullscreen passes sharing one vertex shader.
3. **`mega`** — single fragment, but the largest one (~380 lines)
   with many `#define`-gated branches. Tests every TSL conditional
   construct.
4. **`point`** — first real geometry material. Validates uniform
   parity (pointSizeFactor, etc.) and `LUXAR_MAX_RGB_CONTRIBUTION`
   conditional path.
5. **`point-pick`** — picking parity test against the visual point.
6. **`line` + `line-pick`** — instanced quad expansion; validates
   instanced-attribute access in TSL.
7. **`gsplat` + `gsplat-pick`** — most complex shaders (covariance
   projection, per-channel alpha blending). Last because they're
   the hardest and benefit from every preceding lesson.

## Sign-off

A row's `D` box flips only when *both* `F` (fallback path) and
`R` (real WebGPU) are green. Once `D` is checked, a follow-up PR
removes the GLSL3 strings from `*-shaders.ts` / `*.glsl.ts` and
the `webgl` field becomes optional on the corresponding
`ShaderSource`. When every row has `D` checked, the `webgl` field
can be removed from `ShaderSource` entirely.
