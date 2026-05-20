# Test Mocks

Mock infrastructure that lets Vitest unit tests run under Node.js
without a real browser, GPU, or storage backend. Globally installed
once from `src/tests/setup.ts` via `installAllMocks()`; individual
modules can also be imported on demand.

The parent [`src/tests/README.md`](../README.md) covers the bigger
picture — mock vs. E2E trade-offs, the test category map, and when
to add a new mock. This README focuses on what each file provides
and how the pieces fit together.

---

## File Structure

```
mocks/
├── index.ts                 # Re-exports + installAllMocks() aggregator
├── webgl.mock.ts            # MockWebGLRenderingContext + installWebGLMock
├── browser-apis.mock.ts     # matchMedia, ResizeObserver, IntersectionObserver,
│                            #   requestAnimationFrame/cancelAnimationFrame, performance.now
├── opfs.mock.ts             # navigator.storage stub that rejects getDirectory()
├── orbit-controls.mock.ts   # OrbitControls class stub for three/examples
└── three.mock.ts            # ~1.5k-line drop-in mock of the `three` module
```

---

## How mocks are wired

1. `src/tests/setup.ts` imports `installAllMocks` from this folder and
   calls it before any test runs. That single call installs the WebGL,
   browser-API, and OPFS mocks onto `globalThis`.
2. THREE.js is mocked at the module boundary, not via globals — tests
   that need it use:

   ```ts
   vi.mock('three', () => import('./mocks/three.mock'));
   ```

   The path is relative to the test file. Because the mock re-exports
   every symbol the production code reaches for (math classes,
   geometries, materials, the `WebGLRenderer`, constants), most tests
   need no per-test plumbing beyond the `vi.mock` line.
3. `orbit-controls.mock.ts` is re-exported by name from `index.ts`
   (`export { OrbitControls } from './orbit-controls.mock'`) so it can
   be supplied alongside `three.mock.ts` when stubbing
   `three/examples/jsm/controls/OrbitControls`.

---

## What each mock provides

### `index.ts`

Central re-export point. The aggregator `installAllMocks()` calls
`installWebGLMock()`, `installAllBrowserMocks()`, and
`installOPFSMock()` in that order. Use this from `setup.ts`; in a
single-file test, importing the specific install function is fine.

### `webgl.mock.ts`

`MockWebGLRenderingContext` is a class that exposes the WebGL2
surface THREE.js touches during construction and a single draw —
shader/program/buffer/texture/framebuffer/uniform/attribute calls,
plus `getExtension`, `getParameter`, and the handful of GL constants
read at runtime. `getExtension` returns truthy stubs for the HDR
extensions (`EXT_color_buffer_float`,
`EXT_color_buffer_half_float`, `WEBGL_color_buffer_float`) and for
`WEBGL_debug_renderer_info` so HDR-detection code (`hdr-detection.ts`)
sees a supported environment.

`installWebGLMock()` overrides `HTMLCanvasElement.prototype.getContext`
so any `canvas.getContext('webgl' | 'webgl2')` returns a fresh
`MockWebGLRenderingContext`.

### `browser-apis.mock.ts`

Five small installers, one aggregator:

| Installer | Globals patched |
|---|---|
| `installMatchMediaMock` | `window.matchMedia` (returns `matches: false`, suitable for HDR/P3 negative paths) |
| `installResizeObserverMock` | `ResizeObserver` |
| `installIntersectionObserverMock` | `IntersectionObserver` |
| `installAnimationFrameMock` | `requestAnimationFrame`, `cancelAnimationFrame`, plus a `__clearAllAnimationFrames` helper |
| `installPerformanceMock` | `performance.now` (delegates to `Date.now`) |

`installAllBrowserMocks()` runs them all.

The `requestAnimationFrame` mock backs each frame with a `setTimeout`
and tracks pending IDs in a map so `cancelAnimationFrame` actually
cancels and so `globalThis.__clearAllAnimationFrames()` can be called
in `afterEach` to drain leftover frames before teardown.

### `opfs.mock.ts`

`installOPFSMock()` replaces `navigator.storage` with:

- `getDirectory()` — `vi.fn().mockRejectedValue(new Error('OPFS not available in test environment'))`
- `estimate()` — resolves to `{ quota: 0, usage: 0 }`

The cache layer is expected to handle this rejection by falling back
to in-memory only; cache tests assert that fallback.

### `orbit-controls.mock.ts`

A hand-written `OrbitControls` class that mirrors the public surface
of `three/examples/jsm/controls/OrbitControls`: all the tunables
(`enabled`, `target`, `minDistance`/`maxDistance`,
`enable*`, `dampingFactor`, `mouseButtons`, `touches`, …), the
lifecycle methods (`update`, `dispose`, `saveState`, `reset`,
`listenToKeyEvents`, …), and the event-emitter surface
(`addEventListener`/`removeEventListener`/`dispatchEvent`). `update`
returns `false` (THREE.js convention for "no change"), all other
methods are `vi.fn()` so calls can be asserted.

### `three.mock.ts`

The largest mock by far. It re-implements just enough of `three` for
unit tests to construct a scene, mutate it, and run a render pass
without GPU access. Highlights:

- **Math classes** — `Vector2`/`Vector3`/`Vector4`, `Quaternion`,
  `Euler`, `Matrix4`, `Box3`, `Color`. Methods are chainable (`set`,
  `copy`, `clone`, `add`, `sub`, `multiplyScalar`, `normalize`,
  `equals`) and behave correctly enough that tests can do real math
  on them.
- **Scene graph** — `Object3D` (real class with `add`/`remove`/
  `traverse`), `Group`, `Scene`, `Camera`, `PerspectiveCamera`,
  `OrthographicCamera`.
- **Geometry/material/mesh** — `BufferGeometry`, `BufferAttribute`,
  `BoxGeometry`, `Material`, `ShaderMaterial`, `PointsMaterial`,
  `Mesh`, `Points`, `Texture`, `WebGLRenderTarget`.
- **Renderer** — `WebGLRenderer` with `render`, `setSize`,
  `setPixelRatio`, `getContext`, `dispose`, and the tone-mapping /
  output-color-space setters that the production renderer touches.
- **Utilities** — `Frustum`, `Raycaster`, `Timer`.
- **Constants** — every enum value the codebase imports: blending
  modes, depth functions, sides, encodings, color spaces, tone
  mapping, shadow maps, draw-usage hints, `GLSL3`.

Because the mock matches the import surface of `three` rather than
mocking individual call sites, tests get realistic behaviour by
default and only need bespoke mocking for truly external dependencies
(network, file system).

---

## Adding or extending a mock

1. Check whether the symbol already exists — `three.mock.ts` covers a
   surprising amount.
2. For a new browser API, add an `installXMock()` to
   `browser-apis.mock.ts` and call it from `installAllBrowserMocks()`.
3. For a new THREE.js class or constant, add it to `three.mock.ts`
   beside its peers (math classes together, geometries together,
   constants together).
4. Keep mocks minimal but realistic: `vi.fn()` for methods that just
   need to be called; small real implementations for methods whose
   return value affects test logic (math, traversal, dispose
   tracking).
5. If you add a new file, re-export it from `index.ts` so tests have
   one canonical import path.

---

## See Also

- [`src/tests/README.md`](../README.md) — full test-suite overview, mock vs. E2E policy, and category-by-category test map
- [`src/tests/setup.ts`](../setup.ts) — installs these mocks before any test runs
- [`src/tests/builders/`](../builders/) — fluent data builders that pair with these mocks
