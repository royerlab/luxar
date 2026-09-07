# luxar-viewer source tree

> Navigational hub for the Luxar Viewer TypeScript source. Each
> subfolder below is its own subpackage with a dedicated `README.md`;
> this file is the index that ties them together and documents the
> two top-level entry modules.

The viewer is a GPU-accelerated WebGL2 / WebGPU renderer for
arbitrarily large nD scientific datasets streamed from Zarr archives.
The source is organized as a strict layered architecture — see
[Layered architecture](#layered-architecture) below — with a single
side-effect-free public entry (`index.ts`) and a single orchestration
class (`LuxarApp` in `core/`) that wires every subpackage together.

For the package-level documentation (features, controls, embedding,
URL parameters, troubleshooting) see the parent
[`../README.md`](../README.md). For cross-cutting conventions every
subpackage follows (naming, BEM, logging, `Result<T,E>`, disposal,
layer order) see [`../CONVENTIONS.md`](../CONVENTIONS.md).

## Top-level entry points

Only two source files live directly in `src/` — everything else is
inside a subpackage.

| File                                           | Purpose                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`index.ts`](./index.ts)                       | Public ES-module barrel. Side-effect-free: re-exports `LuxarApp`, `bootstrapStandalone`, URL helpers, `StorageKeys`, `LoaderConfig`, `ZarrViewerConfig`, `buildInfo`, `buildInfoLine`, `BuildInfo`, and a small set of stable rendering helpers (`getCompleteBlendingState`, `applyColormapTextureToMaterial`, …). This is what `import { LuxarApp } from '@luxar/viewer'` resolves to. |
| [`lib-styles-entry.ts`](./lib-styles-entry.ts) | CSS-only entry for the library build. Imports `styles/index.css` so Vite emits `dist/lib/luxar-viewer.css`, which consumers reach via `import '@luxar/viewer/styles.css'`. Exists solely to keep `index.ts` free of side effects.                                                                                                                                                       |

The actual application class (`LuxarApp`) and the standalone
bootstrap live in [`core/`](./core/README.md); the public barrel just
re-exports them.

## Subpackages

```
src/
├── index.ts                # public ES-module barrel (side-effect-free)
├── lib-styles-entry.ts     # CSS-only entry for the library build
├── cache/                  # three-level cache (L0 in-mem → L1 segmented LRU → L2 OPFS)
├── config/                 # unified, section-based configuration with URL-param overrides
├── controls/               # camera navigation: orbit / fly / ortho modes
├── core/                   # LuxarApp orchestrator + standalone bootstrap
├── data/                   # zarr loading, nD slicing, spatial indexing, accumulators
├── input/                  # context-aware keyboard / mouse routing
├── profiling/              # hierarchical timing instrumentation with EMA smoothing
├── rendering/              # WebGL2 / WebGPU pipeline, materials, post-processing
├── scene/                  # 3D scene, camera, clipping, animation, render pipeline
├── styles/                 # CSS architecture (embed-safe library + standalone-app entries)
├── tests/                  # unit (vitest) and E2E (playwright) test suites
├── themes/                 # runtime theming via CSS custom properties (dark / light / glass)
├── types/                  # shared TypeScript types for points, lines, gsplats, mesh, dims, zarr
├── ui/                     # panels, overlays, custom GUI framework, layers panel, monitors
├── utils/                  # cross-cutting helpers: log, Result, EventGroup, HDR, clamp, …
├── wasm/                   # Rust→WASM hot path + pure-TS fallback (spatial, decode, gsplats)
└── workers/                # worker pool + data-worker for off-main-thread compute
```

Per-subpackage one-liners (lifted from each subfolder's own README):

- [`cache/`](./cache/README.md) — Three-level caching system with
  intelligent prefetching for zarr chunks enabling offline viewing,
  instant reloads, and reduced bandwidth.
- [`config/`](./config/README.md) — Unified configuration system that
  centralizes all configuration values to ensure consistency and make
  the application easy to customize.
- [`controls/`](./controls/README.md) — Camera navigation and
  interaction system (orbit / fly / ortho modes) for the Luxar Viewer.
- [`core/`](./core/README.md) — Application initialization and
  lifecycle management; contains `LuxarApp` and the standalone
  bootstrap that orchestrate every other subpackage.
- [`data/`](./data/README.md) — High-performance zarr data loading and
  nD slicing for scientific visualization.
- [`input/`](./input/README.md) — Advanced input handling system with
  context-aware keyboard and mouse management.
- [`profiling/`](./profiling/README.md) — Hierarchical timing
  instrumentation for the scene update pipeline with EMA smoothing and
  session-based nesting.
- [`rendering/`](./rendering/README.md) — Dual-stack WebGL2 / WebGPU
  rendering pipeline (GLSL `ShaderMaterial` default, TSL `NodeMaterial`
  opt-in) with a custom mega-shader for high-quality nD visualization.
- [`scene/`](./scene/README.md) — Core 3D scene management and
  orchestration for nD scientific visualization.
- [`styles/`](./styles/README.md) — CSS architecture split between an
  embed-safe library entry (`index.css`) and a standalone-app entry
  (`standalone.css`).
- [`tests/`](./tests/README.md) — Comprehensive Vitest unit tests and
  Playwright E2E tests for the Luxar Viewer.
- [`themes/`](./themes/README.md) — Runtime theming system with CSS
  custom property injection, persistent user preferences, and advanced
  glass-effect themes.
- [`types/`](./types/README.md) — Shared TypeScript types for nD
  points, lines, gsplats, mesh, dimension metadata, zarr-store schemas,
  animation state, and data-loading monitor contracts.
- [`ui/`](./ui/README.md) — Responsive UI components for nD
  visualization and control (panels, overlays, custom GUI framework).
- [`utils/`](./utils/README.md) — Cross-cutting utility functions:
  HDR color, geometry-buffer accounting, typed event bus,
  `EventGroup`, `Result<T,E>`, `clamp`, `escapeHtml`, log, ….
- [`wasm/`](./wasm/README.md) — Rust-compiled WebAssembly for spatial
  queries, nD visibility computation, and array decoding, with a pure
  TypeScript fallback.
- [`workers/`](./workers/README.md) — Multi-threaded worker pool for
  offloading spatial queries, nD visibility, projection, clipping,
  and array decoding from the main thread.

## Layered architecture

The dependency direction is enforced at **severity `error`** by
`dependency-cruiser` (see `.dependency-cruiser.cjs` at the package
root; run `pnpm check:layers` to verify):

```
types → config → cache → rendering → data → scene → input → ui → core
```

Each layer may import from layers to its **left**. Cross-cutting
helpers — `utils/`, `themes/`, `wasm/`, `workers/`, `profiling/`,
`controls/` — may be imported anywhere. Type-only imports
(`import type`) are exempt because they are erased at compile time.

When a lower layer needs behavior implemented by a higher layer (DOM,
WebGL, THREE state), the dependency is inverted via a **port
interface + factory** rather than relaxed. See `CONVENTIONS.md` §12
for the established pattern and examples
(`SceneLoaderMonitorPort`, `DimensionSlidersFactory`,
`LabelTooltipFactory`).

## Side-effect contract

`import '@luxar/viewer'` resolves to [`./index.ts`](./index.ts), which
must remain **side-effect-free**:

- No `console` patching.
- No CSS injection into the host document.
- No DOM mutation at import time.
- No singleton instantiation.

CSS is opt-in via a separate import path
(`import '@luxar/viewer/styles.css'`, served from
[`./lib-styles-entry.ts`](./lib-styles-entry.ts) → `styles/index.css`).
Standalone-app conveniences (URL parsing, theme from `?theme`,
console interceptor, codec warming) live in `core/bootstrap.ts` and
are only activated when an embedder explicitly opts in by calling
`bootstrapStandalone()` instead of constructing `LuxarApp` directly.

## See also

- [`../README.md`](../README.md) — package overview, features,
  controls, embedding guide, URL parameters, troubleshooting.
- [`../CONVENTIONS.md`](../CONVENTIONS.md) — naming, BEM, logging,
  error-handling, `Result<T,E>`, worker safety, layer order,
  dependency-inversion patterns, disposal rules.
- [`../ARCHITECTURE-DIAGRAMS.md`](../ARCHITECTURE-DIAGRAMS.md) —
  diagrams of the data-flow and component hierarchy.
- `core/README.md` — the orchestrator that wires every subpackage
  here into a running viewer.
