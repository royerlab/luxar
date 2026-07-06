# sections

Per-slice configuration storage for the unified config package. Each immediate subfolder is one section of the `AppConfig` object — the parent [../README.md](../README.md) composes their literals in `../index.ts`, re-exports their public types through `../types.ts`, and dispatches per-section validation from `../validation.ts`. This folder owns no top-level source files of its own; it exists purely to group the section trios.

Every section conforms to the section-trio pattern: `data.ts` exports the literal (e.g. `cameraConfig: CameraConfig`), `types.ts` defines the interface, and — when the section has cross-field invariants — `validate.ts` exports a section validator (`validateCamera`, `validateBloomConsistency`, `validateAdaptiveDPR`, …) that the central dispatcher invokes. Sections without invariants (`animation/`, `dimension-animation/`, `ui/`) deliberately omit `validate.ts`.

## Layout

```text
sections/
├── adaptive-dpr/          # FPS-driven DPR scaling: thresholds, factors, hysteresis
├── animation/             # Render-loop idle timeout (power-saving knob)
├── cache/                 # OPFS three-level cache budgets (L0/L1/L2) + TTL
├── camera/                # Initial position, FOV limits, lens-distortion presets
├── controls/              # Per-mode fly/orbit knobs + scene-scale multipliers
├── data-loading/          # Composite section — see Subpackages below
│   ├── spatial/           # nD slicing tolerance + max splat radius
│   ├── network/           # Fetch lifetimes and retry policy for zarr stores
│   ├── memory/            # Heap-usage targets and check cadence
│   ├── monitor/           # Event ring-buffer limits and alert thresholds
│   └── performance/       # Accumulators, workers, WASM, GPU buffer pool
├── dimension-animation/   # FPS-based playback through dimension ranges
├── input/                 # Sensitivity + keyboard shortcuts + fly/dim keys
├── rendering-controls/    # User-adjustable rendering settings (single source of truth)
├── scene/                 # Background color, fit ratio, ShaderConfig placeholder
├── ui/                    # z-index, timings, spinner, debug console, components
└── webgl/                 # Context attrs, renderer options, render-target config
```

## Subpackages

- **[adaptive-dpr/](adaptive-dpr/README.md)** — defaults + validation for the FPS-driven DPR control loop: refresh-rate-relative scaling ratios, multiplicative factors, the DPR floor, U-shape probe knobs, learned floor/ceiling TTLs with exponential backoff, and session-hygiene timings (gap reset, content-change recheck, hysteresis).
- **[animation/](animation/README.md)** — the idle-timeout used by the render loop to pause continuous rendering when no input or scene change has occurred. Not the dimension-animation playback engine.
- **[cache/](cache/README.md)** — three-level cache hierarchy budgets (L0 decompressed, L1 in-memory LRU, L2 OPFS) plus the OPFS operation timeout and external-dataset TTL.
- **[camera/](camera/README.md)** — initial position, FOV zoom limits and sensitivity, and the photography-style FOV / lens-distortion preset tables. Note: `fov`/`near`/`far` live in `rendering-controls/`, not here.
- **[controls/](controls/README.md)** — defaults for the `fly` and `orbit` control modes plus the scene-scale multipliers used to adapt control parameters to the bounding-box diagonal.
- **[data-loading/](data-loading/README.md)** — composite section that bundles five sub-slices (`spatial`, `network`, `memory`, `monitor`, `performance`) into a single `DataLoadingConfig`; dispatches per-sub-section validation.
- **[dimension-animation/](dimension-animation/README.md)** — defaults and presets for FPS-based playback through dimension ranges, including loop mode (`once` / `loop` / `bounce`), direction, frame-time floors, and target-vs-actual FPS feedback.
- **[input/](input/README.md)** — default adjustment sensitivity, the global keyboard shortcut map, the fly-mode movement key set, the dimension-navigation keys, and mouse timings such as `doubleClickDelay`.
- **[rendering-controls/](rendering-controls/README.md)** — the single source of truth for user-adjustable rendering settings: camera FOV/clipping, bloom, global EOG, anti-aliasing, tone mapping, vignette, detector noise, lens distortion, navigation mode, and adaptive-DPR toggles.
- **[scene/](scene/README.md)** — canvas background color, default fit-to-bounds framing ratio, and a forward-compatible `ShaderConfig` placeholder reserved for future per-geometry shader knobs.
- **[ui/](ui/README.md)** — z-index layering, timing constants for transient UI, loading-spinner geometry, debug-console panel/interceptor/style settings, the scale-bar overlay, and per-component border-radius/padding tokens.
- **[webgl/](webgl/README.md)** — WebGL2 context attributes, `THREE.WebGLRenderer` constructor options, and the post-processing render-target settings (MSAA is off — incompatible with the additive blending used for points/gsplats).

## See Also

- [../README.md](../README.md) — config-package overview, composition flow, and end-to-end usage examples.
- `../index.ts` — composes the section literals into the `config: AppConfig` export.
- `../types.ts` — re-exports every section type plus the `AppConfig` interface.
- `../validation.ts` — central dispatcher that invokes each section's `validate*` function.
