# Browser-support policy for the WebGPU migration

## Context

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
populations Luxar Viewer targets post-WebGPU-port.

## Options

### Option A — Require WebGPU

Drop the WebGL renderer entirely. Smallest codebase, simplest QA
matrix, but loses the entire Firefox-stable population and any
Safari users still on 17 or below.

### Option B — Dual-stack

Maintain both `WebGLRenderer` and `WebGPURenderer` in the bundle.
Runtime detect, branch. Every shader gets two implementations
(GLSL3 + TSL). Visual parity must hold across the two stacks for
every effect. Highest maintenance burden — every PR that touches a
material has to be reviewed against both pipelines.

### Option C — Single renderer, Three.js internal fallback

Use `WebGPURenderer` exclusively in our code. Three.js's
`three.webgpu.js` build includes an internal WebGL2 fallback path
inside `WebGPURenderer` itself: when no adapter is available, the
same `WebGPURenderer` instance dispatches through WebGL2. Our code
only knows about *one* renderer; Three.js handles the dispatch.

Material pipeline: one source-of-truth per shader (the
`ShaderSource` modules already in place from Item 2 of the prep
plan), each providing a GLSL3 string today and a TSL factory after
the port. Three.js routes the active one through the active backend.

## Decision

**Option C — single renderer with Three.js internal fallback.**

Subject to Item 5 of the migration plan (running the existing E2E
suite against `WebGPURenderer` in `{ forceWebGL: true }` mode)
showing acceptable visual parity. If the fallback path produces
unacceptable artefacts on our shaders, fall back to Option A (the
dual-stack maintenance cost is not justified for a Firefox-only
audience).

Rationale:

1. **One renderer to maintain.** The mega-shader refactor already
   collapsed our post-processing surface to a single fullscreen pass;
   keeping two parallel implementations of that pass would undo the
   architectural win.
2. **Three.js's fallback path is already tested by the upstream
   project.** We get WebGL2 coverage without owning the dispatch
   layer.
3. **TSL writes one shader that compiles to both targets.** The
   GLSL3 strings stay around as the WebGL2 path; the TSL factory
   slot in `ShaderSource` becomes the WebGPU path; Three.js picks
   the one that matches the active backend. We never write two
   parallel shader sources.
4. **Audience coverage is acceptable.** WebGPU-native users get
   the fast path; WebGL2-only users still get a working viewer
   (potentially slower, but not broken). Only true "neither"
   users are excluded — and they're already excluded today (we
   require WebGL2).

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
transparently).

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

- Firefox's WebGPU default flips to "on" on Linux. At that point
  WebGL2 fallback covers a much smaller population, and Option A
  becomes more attractive.
- Three.js drops the WebGL2 fallback path inside `WebGPURenderer`.
  At that point Option C becomes Option A by force.
- We hit a shader where TSL can't express what the GLSL version
  does. We'd need either a per-backend hand-written variant
  (Option B in microcosm) or to redesign the shader.
