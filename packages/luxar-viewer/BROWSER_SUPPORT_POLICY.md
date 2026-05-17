# Browser-support policy for the WebGPU migration

## Codebase-side hard guarantees

These are verified facts about Luxar's current state that the
migration depends on. Each is a grep-resistant invariant — a CI
check enforcing them would be appropriate later.

- **`onBeforeCompile` is not used anywhere in `src/`.** Three.js's
  WebGPURenderer manual states modifications via `onBeforeCompile()`
  are not supported. Verified by `grep -rn 'onBeforeCompile' src/`
  → zero matches. Keep this clean through the port — a future
  consumer should reach for a `ShaderSource.webgpu` factory
  instead, never for a build-time shader patch.
- **All `material.uniforms.X.value =` writes happen inside material-
  owner files.** `material-colormap-helpers.ts` is the historical
  exception and was refactored to delegate to material setters
  (`ColormapAwareMaterial.setColormapTexture/setScalarRange`).
  Confirmed by the spot-check grep in the migration plan.

## Hard constraint up front: `ShaderMaterial` blocks `WebGPURenderer`

Three.js's `WebGPURenderer` does **not** support `THREE.ShaderMaterial` or
`THREE.RawShaderMaterial`. From the Three.js manual:

> Custom materials based on `ShaderMaterial`, `RawShaderMaterial` and
> modifications of built-in materials via `onBeforeCompile()` are not
> supported in `WebGPURenderer`. This part of your application must
> be ported to node materials and TSL.
>
> — *Three.js manual, "Using WebGPURenderer"*

Status (post wrapper-layer wiring): every scene-material,
picking-material, and post-processing-shader pair has a TSL /
NodeMaterial counterpart living in `*.tsl.ts` files alongside the
GLSL3 originals in `*-shaders.ts` / `*.glsl.ts`. Pixel parity is
verified by `tsl-shader-parity.spec.ts`. Production wrapper classes
(`PointTSLMaterial`, `LineTSLMaterial`, `GSplatTSLMaterial`, and
the three picking equivalents) extend `NodeMaterial` and ship
one-for-one with their GLSL `ShaderMaterial` counterparts;
`MaterialManager.getXxxMaterial` dispatches on `caps.apiSurface ===
'webgpu'`. The GLSL wrappers are kept live behind
`VITE_LUXAR_USE_LEGACY_WEBGL=1` as the parity reference (per
project policy — GLSL is never deleted), not as a fallback for
production users.

Historical note: `{ forceWebGL: true }` does not rescue a
`ShaderMaterial` pipeline. `WebGPURenderer`'s WebGL2 backend
dispatches `NodeMaterial`, not `ShaderMaterial`. A `ShaderMaterial`
handed to `WebGPURenderer` (with or without `forceWebGL`) will not
render. The migration was therefore the TSL ports *and* the
dispatcher rewire, not a renderer swap.

## Userbase

WebGPU shipped in Chrome 113 (May 2023), Edge 113 (May 2023), and
Safari 18 (Sep 2024). Firefox has it behind `dom.webgpu.enabled` on
non-Linux platforms; Firefox 141+ may flip the default on Windows /
macOS but Linux remains gated. Today's userbase therefore breaks
roughly into:

- **WebGPU-native**: Chrome stable, Edge stable, Safari ≥ 18.
- **WebGL2-only**: Firefox stable (all platforms), Safari < 18,
  legacy Chromium / Edge.
- **Neither**: very old browsers we won't try to support.

This document records the policy decision for which of these
populations Luxar Viewer targets *after* the migration completes,
and what we run *today*.

## Historical — before TSL ports landed

The original framing of this section:

- We ran `THREE.WebGLRenderer` exclusively. There was no alternative
  because every Luxar material was a `THREE.ShaderMaterial({
  glslVersion: GLSL3 })`.
- Equivalent to "Option A" in spirit, but framed by force rather
  than choice.
- The runtime-detection helper at `src/utils/webgpu-availability.ts`
  could identify a WebGPU-capable browser, but knowing this
  changed nothing about what renderer we handed to a user.

This is no longer the only path — see "Today" below.

## Today — WebGL default, WebGPU as the opt-in second path

- Both renderer constructions exist:
  - **Default: `THREE.WebGLRenderer` (GLSL `ShaderMaterial`).** Constructed
    when no URL flag / env var opts in to WebGPU. This is the production
    default because per-scene performance measurements were below the
    WebGL baseline on the WebGPU path; WebGL stays the safe choice until
    those gaps close.
  - Opt-in WebGPU path via `?renderer=webgpu` or `VITE_LUXAR_USE_WEBGPU=1`:
    constructs `WebGPURenderer`, which acquires a real WebGPU adapter
    when the browser provides one and transparently falls back to its
    internal WebGL2 backend otherwise — the same TSL graphs target both
    backends from one source. TSL `NodeMaterial` wrappers are dispatched
    by `MaterialManager` when `caps.apiSurface === 'webgpu'`. The TSL ↔ GLSL
    parity harness keeps both stacks in sync; per-shader GLSL3 sources
    are never deleted, per project policy.

### Backend selection — precedence

`SceneManager.setupRenderer` resolves the active backend through
three levels, highest precedence first:

1. **`?renderer=webgl` / `?renderer=webgpu` URL parameter.**
   Per-load override threaded from `UrlParams.renderer` →
   `LuxarAppOptions.renderer` → `SceneManager.init({ renderer })`.
   Useful for A/B diagnostics: reload the same page with
   `?renderer=webgpu` to compare against the default WebGL dispatch
   without restarting the dev server or rebuilding the bundle.
   Case-insensitive; `webgl2` is accepted as an alias for `webgl`;
   any other value falls through. Add `?webgpu-force-webgl` alongside
   `?renderer=webgpu` to construct `WebGPURenderer({ forceWebGL: true })`:
   Luxar still uses the WebGPURenderer API surface and TSL `NodeMaterial`
   shaders, but Three.js routes draw calls through its internal WebGL2
   backend instead of requesting a native WebGPU adapter.
2. **`VITE_LUXAR_USE_WEBGPU=1` env var.** Build / dev-server-time
   opt-in to the WebGPU path. Useful for running the whole E2E suite
   against WebGPU. Anything other than `'1'` (or unset) → next level.
   The transitional `VITE_LUXAR_USE_WEBGPU_RENDERER` alias is honoured
   for the same purpose.
3. **Default: `webgl`.**

The legacy `VITE_LUXAR_USE_LEGACY_WEBGL=1` env var is a no-op alias
now that WebGL is the default — kept harmless so existing CI scripts
that set it keep working.

#### Common URL invocations

```
http://localhost:5173/                    # default (webgl)
http://localhost:5173/?renderer=webgl    # pin the WebGL path explicitly
http://localhost:5173/?renderer=webgpu   # opt into WebGPURenderer + TSL path
http://localhost:5173/?renderer=webgpu&webgpu-force-webgl
                                          # WebGPURenderer + TSL, backed by WebGL2
http://localhost:5173/?src=...&renderer=webgpu&debug
                                          # combines with other URL flags
```
- TSL factories for all 12 shaders ship with TSL `NodeMaterial`
  wrappers (`{Point,Line,GSplat}TSLMaterial`,
  `{Point,Line,GSplat}PickingTSLMaterial`, `MegaShaderTSLMaterial`)
  one-for-one with the GLSL wrappers. `MaterialManager.setCaps`
  is wired from both `SceneManager.setupRenderer` arms so dispatch
  is consistent across paths.
- `RendererCapabilities.apiSurface` reports which **renderer surface** is in
  use, not the physical backend: `'webgl2'` when Luxar instantiated
  `THREE.WebGLRenderer` (the production default); `'webgpu'` whenever
  Luxar instantiated `WebGPURenderer` — **including** runs where
  WebGPURenderer falls back to its internal WebGL2 backend, because
  the callable API surface (readback signatures, render-target
  wiring, row-padding rules) still follows the WebGPURenderer
  contract. Callers branching on `caps.apiSurface` are picking which
  signature contract to follow, not probing the GPU backend.

## Target end-state (post-TSL-ports)

Three options are conceivable once `ShaderMaterial` is no longer
in the picture (i.e. every material has a `NodeMaterial` /
TSL counterpart in its `ShaderSource.webgpu` slot).

### Option A — Require WebGPU

Drop the WebGL renderer entirely. Smallest codebase, simplest QA
matrix, but loses the entire Firefox-stable population and any
Safari users still on 17 or below.

### Option B — Dual-stack

Maintain both `WebGLRenderer` and `WebGPURenderer` in the bundle.
Runtime detect, branch. Every shader needs both GLSL3 and TSL
implementations *running*, not just present in the source. Visual
parity must hold across the two stacks for every effect. Highest
maintenance burden — every PR that touches a material gets reviewed
against both pipelines.

### Option C — Single renderer, Three.js internal fallback

Use `WebGPURenderer` exclusively. Three.js's `three.webgpu.js`
build includes an internal WebGL2 fallback inside the
`WebGPURenderer` itself: when no adapter is available (or
`forceWebGL: true` is set), the same `WebGPURenderer` instance
dispatches through WebGL2. Our code only knows about *one* renderer;
Three.js handles the dispatch. Crucially, this option requires that
the material pipeline is `NodeMaterial`-based — `WebGPURenderer`'s
internal WebGL2 fallback dispatches `NodeMaterial`, not
`ShaderMaterial`. So Option C is a strict subset of post-TSL-ports
state.

## Target decision

**Option C — single renderer with Three.js internal fallback**,
contingent on:

1. The TSL ports landing for every material (the migration itself).
2. The `WEBGPU_FALLBACK_REPORT.md` experiment confirming that the
   ported `NodeMaterial` pipeline renders correctly under
   `WebGPURenderer({ forceWebGL: true })`. The experiment will be
   meaningful **only after** Item 1 (the first TSL port) lands —
   running it today, with the existing `ShaderMaterial`s, would
   simply confirm what this section already states.

If the post-TSL fallback experiment finds an artefact we can't fix,
fall back to Option A (the dual-stack maintenance cost in Option B
is not justified for a Firefox-stable-only audience, which is what
B costs us).

Rationale (for Option C, assuming the experiment passes):

1. **One renderer to maintain.** The mega-shader refactor already
   collapsed our post-processing surface to a single fullscreen
   pass; keeping two parallel implementations of that pass would
   undo the architectural win.
2. **Three.js's fallback path is upstream-maintained.** We get
   WebGL2 coverage without owning the dispatch layer.
3. **TSL writes one shader that compiles to both targets.** The
   GLSL3 strings can be removed; the TSL factory slot in
   `ShaderSource.webgpu` becomes the only source; Three.js routes
   it through the active backend. We never write two parallel
   shader sources.
4. **Audience coverage is acceptable.** WebGPU-native users get
   the fast path; WebGL2-only users still get a working viewer
   (potentially slower, but not broken). Only true "neither" users
   are excluded — and they're already excluded today (we require
   WebGL2).

## Runtime detection

`src/utils/webgpu-availability.ts` exposes:

```ts
getRendererAPI(): Promise<'webgpu' | 'webgl2' | 'unsupported'>
getRendererAPISync(): 'webgpu' | 'webgl2' | 'unsupported'
```

These are **diagnostic** signals — for a "fast path enabled" badge,
for telemetry on which backend users land on, for the `?backend=`
URL parameter override during testing. They do **not** drive the
renderer choice (Three.js's internal fallback handles that
transparently, once we're on Option C).

`RendererCapabilities.apiSurface` reports the API the renderer actually
runs on, which may differ from the page-load probe if the user has
flipped a flag mid-session (rare) or if `WebGPURenderer` decided to
fall back at init (common on WebGPU-unsupported hardware).

## What "unsupported" means

If `getRendererAPI()` returns `'unsupported'`, the page renders an
informational error message ("Luxar Viewer requires a browser with
WebGPU or WebGL2 support. Try Chrome, Edge, Firefox, or Safari 18+.")
and refuses to construct the renderer. This case is already covered
by the existing pre-renderer error path in `scene-manager.ts`.

## Out of scope

- **Mobile browsers.** Touch input, layout, viewport sizing are
  separate concerns. The graphics-API decision applies to mobile
  too, but mobile-specific bugs are a separate triage stream.
- **Headless / SSR environments.** Luxar Viewer is a client-side
  WebGL/WebGPU app; SSR rendering of the canvas isn't a goal. The
  `getRendererAPI()` helper guards against missing `document` so
  test environments don't crash.
- **WebGPU-only feature flags.** If a future Luxar feature is
  *only* possible on WebGPU (compute shaders for advanced picking,
  for example), gate it behind `caps.apiSurface === 'webgpu'`. The flag
  belongs to the feature, not to the renderer.

## Revisit if

- Three.js drops the `ShaderMaterial` restriction on
  `WebGPURenderer`. Then Option C becomes available *before* the
  TSL ports complete, which would change the sequencing argument.
- Firefox's WebGPU default flips to "on" on Linux. At that point
  WebGL2 fallback covers a much smaller population, and Option A
  becomes more attractive.
- Three.js drops the WebGL2 fallback path inside `WebGPURenderer`.
  At that point Option C becomes Option A by force.
- We hit a shader where TSL can't express what the GLSL version
  does. We'd need either a per-backend hand-written variant
  (Option B in microcosm) or to redesign the shader.

## References

- [Three.js manual, "Using WebGPURenderer"](https://threejs.org/manual/en/webgpurenderer.html) —
  authoritative statement of the `ShaderMaterial` non-support.
- [GitHub three.js #26719 — custom shader support for `WebGPURenderer`](https://github.com/mrdoob/three.js/issues/26719) —
  upstream tracking issue.
