# Browser-support policy for the WebGPU migration

## Hard constraint up front: `ShaderMaterial` blocks `WebGPURenderer`

Three.js's `WebGPURenderer` does **not** support `THREE.ShaderMaterial` or
`THREE.RawShaderMaterial`. From the Three.js manual:

> Custom materials based on `ShaderMaterial`, `RawShaderMaterial` and
> modifications of built-in materials via `onBeforeCompile()` are not
> supported in `WebGPURenderer`. This part of your application must
> be ported to node materials and TSL.
>
> — *Three.js manual, "Using WebGPURenderer"*

Every one of Luxar's materials today is `THREE.ShaderMaterial({
glslVersion: GLSL3 })`. There are roughly 34 such instances across
the scene materials, post-processing materials, and picking materials.

`{ forceWebGL: true }` does not rescue this: it instructs
`WebGPURenderer` to dispatch through its WebGL2 backend, but the
materials it accepts are still `NodeMaterial`-based. A
`ShaderMaterial` handed to `WebGPURenderer` (with or without
`forceWebGL`) will not render.

Consequence: until the TSL ports land — which **are** the migration —
`WebGPURenderer` is not usable. The migration is the TSL ports, not
a renderer swap.

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

## Today

Until TSL ports land:

- We run `THREE.WebGLRenderer`. There is no alternative.
- Equivalent to "Option A" in spirit, but framed by force rather
  than choice.
- The runtime-detection helper at `src/utils/webgpu-availability.ts`
  exists and is honest about what it reports — it can identify a
  WebGPU-capable browser, but knowing this changes nothing about
  what renderer we hand to a user today.

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

`RendererCapabilities.api` reports the API the renderer actually
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
  for example), gate it behind `caps.api === 'webgpu'`. The flag
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
