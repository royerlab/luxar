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
└── opfs.mock.ts             # navigator.storage stub (rejecting default + in-memory luxar/ root)
```

---

## How mocks are wired

1. `src/tests/setup.ts` imports `installAllMocks` from this folder and
   calls it before any test runs. That single call installs the WebGL,
   browser-API, and OPFS mocks onto `globalThis`.
2. Tests that need THREE.js mock it at the module boundary with an
   inline `vi.mock('three', …)` factory in the test file itself.

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

| Installer                         | Globals patched                                                                            |
| --------------------------------- | ------------------------------------------------------------------------------------------ |
| `installMatchMediaMock`           | `window.matchMedia` (returns `matches: false`, suitable for HDR/P3 negative paths)         |
| `installResizeObserverMock`       | `ResizeObserver`                                                                           |
| `installIntersectionObserverMock` | `IntersectionObserver`                                                                     |
| `installAnimationFrameMock`       | `requestAnimationFrame`, `cancelAnimationFrame`, plus a `__clearAllAnimationFrames` helper |
| `installPerformanceMock`          | `performance.now` (delegates to `Date.now`)                                                |

`installAllBrowserMocks()` runs them all.

The `requestAnimationFrame` mock backs each frame with a `setTimeout`
and tracks pending IDs in a map so `cancelAnimationFrame` actually
cancels and so `globalThis.__clearAllAnimationFrames()` can be called
in `afterEach` to drain leftover frames before teardown.

### `opfs.mock.ts`

Two flavours. `installOPFSMock()` (the global default) replaces `navigator.storage` with:

- `getDirectory()` — `vi.fn().mockRejectedValue(new Error('OPFS not available in test environment'))`
- `estimate()` — resolves to `{ quota: 0, usage: 0 }`

The cache layer is expected to handle this rejection by falling back
to in-memory only; cache tests assert that fallback.

`createFakeOpfsRoot(options?)` builds an in-memory OPFS root for the cache
tests that need a WORKING L2. It models exactly the levels production walks:
the origin root accepts ONLY the viewer's `luxar/` namespace dir
(`OPFS_NAMESPACE_DIR`, throwing for any other name and `NotFoundError` on a
cold `create: false` lookup), the `luxar/` dir hands out one flat dataset dir
for every `zarr-cache-*` id (recording them in `datasetIds`, with `vi.fn`
wrapped `getDirectoryHandle` / `removeEntry` for call assertions), and the
dataset dir keeps `files` / `metaFiles` by name (bucket dirs collapse onto
it). `install({ datasetDir?, estimate? })` stubs `navigator.storage`; a test
with its own instrumented dataset handle passes it as `datasetDir` plus an
`onRemoveDataset` wipe of its maps.

---

## Adding or extending a mock

1. For a new browser API, add an `installXMock()` to
   `browser-apis.mock.ts` and call it from `installAllBrowserMocks()`.
2. Keep mocks minimal but realistic: `vi.fn()` for methods that just
   need to be called; small real implementations for methods whose
   return value affects test logic (math, traversal, dispose
   tracking).
3. If you add a new file, re-export it from `index.ts` so tests have
   one canonical import path.

---

## See Also

- [`src/tests/README.md`](../README.md) — full test-suite overview, mock vs. E2E policy, and category-by-category test map
- [`src/tests/setup.ts`](../setup.ts) — installs these mocks before any test runs
- [`src/tests/builders/`](../builders/) — fluent data builders that pair with these mocks
