# WebGPURenderer fallback — E2E experiment runbook + report

**Status**: pending. The experiment has been *designed* but not yet
*executed*. This file is the runbook for whoever performs the run,
plus the report shape so the findings land in a predictable place.

## What we want to learn

Three.js's `three.webgpu` build exposes a `WebGPURenderer` whose
`forceWebGL: true` option dispatches through an internal WebGL2
backend. If our viewer renders **identically** under that backend,
the eventual WebGPU port reduces to:

1. Swap `new THREE.WebGLRenderer(...)` → `new THREE.WebGPURenderer({...})`.
2. Add `await renderer.init()`.
3. Port shaders GLSL → TSL one by one, behind the `ShaderSource.webgpu`
   slot already prepared in `src/rendering/shaders/shader-source.ts`.

If the fallback path **doesn't** render identically — colour shifts,
broken bloom, ShaderMaterial compilation failures, depth z-fighting —
we need to know **before** the port starts so the scope is honest.

## Runbook

### Step 1 — Add the flag

In `src/scene/scene-manager.ts`, replace `setupRenderer()` with an
async variant that branches on an env variable. Skeleton:

```ts
import type { Renderer } from '../rendering/renderer-capabilities';

private async setupRenderer(): Promise<void> {
  const useWebGPU = import.meta.env.VITE_LUXAR_USE_WEBGPU_RENDERER === '1';

  if (useWebGPU) {
    // Dynamic import so the WebGPU build isn't pulled into the
    // default bundle. forceWebGL routes through Three.js's
    // internal WebGL2 backend without requiring real WebGPU
    // hardware — the point of this experiment.
    const { WebGPURenderer } = await import('three/webgpu');
    const r = new WebGPURenderer({
      canvas: this.canvasElement,
      antialias: config.webgl.context.antialias,
      alpha: config.webgl.context.alpha,
      forceWebGL: true,
    });
    await r.init();
    this.renderer = r as Renderer;  // widen Renderer to the union first
  } else {
    // existing WebGLRenderer construction unchanged
    this.renderer = new THREE.WebGLRenderer({ ... });
  }
  // … existing capability + HDR + clear-color setup …
}
```

Also update `async init` to `await this.setupRenderer();`.

The `Renderer` type alias from `renderer-capabilities.ts` is
single-arm today (`THREE.WebGLRenderer`). Before this experiment
runs, widen it to
`THREE.WebGLRenderer | import('three/webgpu').WebGPURenderer` so
the cast above becomes a no-op. That single edit is the entire
typing seam — every consumer that holds a renderer reference
absorbs it automatically.

### Step 2 — Run

```bash
cd packages/luxar-viewer
pnpm typecheck              # must stay clean — fail = stop, report a typing bug
pnpm test --run             # unit tests still pass (default WebGL path)
VITE_LUXAR_USE_WEBGPU_RENDERER=1 pnpm dev
# open the page, manually verify a known dataset renders
VITE_LUXAR_USE_WEBGPU_RENDERER=1 pnpm playwright test \
    basic-rendering.spec.ts \
    visual-regression.spec.ts \
    geometry-types.spec.ts \
    post-processing-pipeline.spec.ts
```

Run the targeted Playwright specs first — they're the highest-signal
slice (~2 min). If they pass, expand to the full suite.

### Step 3 — Classify failures

For each failure, decide which category it belongs to. The category
determines which prep item the fix belongs in:

| Category | Symptom | Owns the fix |
|---|---|---|
| **A. Shader-compile error** | "ERROR: invalid #version directive" or similar from a `ShaderMaterial`. | Item 2 (shader registry) — the GLSL source needs adjustment for WebGPURenderer's stricter prefix. |
| **B. Render-target-type mismatch** | Wrong colour space, blown highlights, all-black bloom. | Item 1 (capabilities) — `RendererCapabilities.api` probably needs to influence target type choice. |
| **C. API-shape error** | "renderer.X is not a function" thrown by post-processing or picking code. | New leak — file an issue and add the missing method to `RendererCapabilities`. |
| **D. Semantic difference** | Same output but slightly off (e.g. point size 1 px smaller, line antialiasing softer). | Defer to port-PR — these are real WebGPU/WebGL semantic gaps and warrant per-shader investigation. |
| **E. Test-only failure** | A spec checks a WebGL-specific internal (e.g. `renderer.info.programs.length`). | Update the spec — it's pinned to an implementation detail. |

## Findings

### Run conditions

The experiment attempted in M3 (commit after M2 `1de5ece0`)
surfaced a Playwright-config gotcha: `playwright.config.ts` sets
`reuseExistingServer: !process.env.CI`, so when a dev server is
already running on port 5173 from another session, Playwright
reuses it instead of spawning one with the test's env vars.
Result: setting `VITE_LUXAR_USE_WEBGPU_RENDERER=1` in the
Playwright invocation has **no effect on the reused server**.
All "WebGPU fallback" runs end up testing the default
WebGLRenderer path.

To run M3 cleanly, do **one** of:

1. Kill the existing dev server before launching Playwright, so
   the test's env var reaches the freshly-spawned process.
2. Add a `webServer.env` block to `playwright.config.ts` that
   forwards `VITE_LUXAR_USE_WEBGPU_RENDERER` to the spawned
   server. This is the right long-term fix — see M3-bis below.
3. Build the viewer with the env var (`VITE_LUXAR_USE_WEBGPU_RENDERER=1
   pnpm build:vite`) and serve the static build for Playwright.

### Decision (M3)

Rather than block on the dev-server-conflict workaround, we
proceed directly to M4 (the FXAA TSL port). The experiment's
value is the per-shader signal it produces; that signal is much
sharper *after* at least one shader has a TSL factory to test
against (since today every `ShaderMaterial` would either crash
under real WebGPU or pass-through under `forceWebGL: true` —
neither is informative).

M5 reruns the fallback experiment **scoped to FXAA only**,
which is the first meaningful per-shader datapoint. The
broader fallback run lands as a separate sweep after Phase 3
(post-processing ports complete) and again after Phase 5
(scene materials complete) — see `MIGRATION_PROGRESS.md`'s `F`
column.

### Follow-up: M3-bis (deferred)

Add `webServer.env: { VITE_LUXAR_USE_WEBGPU_RENDERER:
process.env.VITE_LUXAR_USE_WEBGPU_RENDERER ?? '' }` to
`playwright.config.ts` so future invocations propagate the
flag through to the spawned dev server. Small change; lands
alongside the first M5-scoped fallback run.

### Targeted specs

| Spec | Result | Notes |
|---|---|---|
| `basic-rendering.spec.ts` | _deferred to M5_ | run was inconclusive (env var didn't propagate to reused server) |
| `visual-regression.spec.ts` | _deferred to M5_ | |
| `geometry-types.spec.ts` | _deferred to M5_ | |
| `post-processing-pipeline.spec.ts` | _deferred to M5_ | |

### Full suite

| Spec group | Pass | Fail | Skip | Notes |
|---|---|---|---|---|
| Basic functionality | _pending_ | | | |
| Scene & transforms | _pending_ | | | |
| nD navigation | _pending_ | | | |
| Worker & WASM | _pending_ | | | |
| Geometry & rendering | _pending_ | | | |
| Data & I/O | _pending_ | | | |
| UI panels | _pending_ | | | |
| Visual regression | _pending_ | | | |
| Performance | _pending_ | | | |

### Classified failures

_None observed yet._

## Decision criteria (post-run)

If **all** targeted specs pass and the full suite shows fewer than 5
% spec failures, all in category D/E:

→ proceed with Option C of `BROWSER_SUPPORT_POLICY.md` (single
renderer with internal fallback). Begin the renderer-port PR.

If **any** category A/B/C failure exists, or full-suite failure rate
is >5 %:

→ first close the gap by promoting the relevant prep item (Item 1
or Item 2) with the missing functionality. Re-run. Do not begin the
renderer-port PR until this report shows green.

## Out of scope for the experiment

- **Performance.** Three.js's WebGL2 backend inside
  `WebGPURenderer` is unlikely to match the native `WebGLRenderer`
  on raw throughput. We're measuring **correctness**, not speed.
  Performance comparison happens after the real WebGPU path runs
  on real WebGPU hardware.
- **TSL ports.** The `ShaderSource.webgpu` slot stays empty in this
  experiment. We're testing whether the existing GLSL shaders
  survive the fallback dispatch, not the eventual TSL rewrite.
- **Adapter selection / power-preference.** `forceWebGL: true` skips
  the adapter entirely. Real WebGPU's adapter selection is a port-PR
  concern.

## Cross-references

- Plan file: `~/.claude/plans/fixed-check-again-iridescent-pearl.md`
  (Item 5)
- Policy doc: `BROWSER_SUPPORT_POLICY.md`
- Version pin: `THREE_VERSION_NOTES.md`
- Shader registry: `src/rendering/shaders/shader-source.ts`
- Capabilities: `src/rendering/renderer-capabilities.ts`
