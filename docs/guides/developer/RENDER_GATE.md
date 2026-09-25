# Render gate: exactness and performance, build against build

The render gate measures a **candidate** viewer build against a **baseline** build
on the machine it runs on, and fails when the candidate renders differently than
its commit says it should, or renders slower. It is an opt-in tool for changes to
the render path (shaders, materials, the frame loop, view-dependent CPU logic),
not a CI gate: CI has no GPU, and the answer it gives is only meaningful on real
hardware.

```bash
make render-gate BASE=origin/main CAND=HEAD                   # exactness, IDENTICAL class
make render-gate BASE=HEAD~1 CAND=HEAD CLASS=ULP              # a math-moving commit
make render-gate BASE=HEAD~1 CAND=HEAD INTENDED=env_splats    # declared intended change
make render-gate BASE=origin/main CAND=HEAD SUITE=perf        # performance only
make render-gate BASE=HEAD CAND=WORKTREE ONLY=mixed           # uncommitted work, one case
```

The script behind it is `packages/luxar-viewer/scripts/render-gate/run-gate.mjs`
(pass extra flags with `GATE_ARGS=`, e.g. `GATE_ARGS="--backends webgl --dsf 1"`).
Reports land in `delme/gate/<label>/`: `report.md`, `report.json` and, for every
view that differed, a heatmap PNG (black: identical; blue: drift; red: flip).

## How a run works

1. **Builds.** Each ref is extracted with `git archive` into
   `delme/gate-builds/<sha>/`, installed from the lockfile and built with
   `vite build`, then cached by commit SHA. When a ref's Rust sources match the
   current checkout, its WASM module is copied rather than recompiled. `WORKTREE`
   builds the working tree as it is.
2. **Serving.** Each build's `dist/` is served by its own small static server
   with the checkout's `datasets/` mounted at `/datasets/`: same origin, no CORS.
3. **Driving.** System Chrome (`--channel chrome`), headless. On Linux the
   harness adds `--use-angle=vulkan --enable-features=Vulkan,WebGPU`, without
   which headless Chrome renders on SwiftShader. A page reporting a software
   renderer is refused rather than measured. The page is driven only through
   debug surfaces present on main (`scripts/render-gate/page-ops.mjs`), so the
   baseline may predate the gate.
4. **Scenes** (`scripts/render-gate/gate-scenes.json`). Seeded synthetic points,
   lines (quad and capsule) and splats in every blending mode, plus stores written
   by `scripts/render-gate/generate_gate_scenes.py` into `datasets/gate/`:
   `mixed` (all four geometry types, pickable), `env_splats` (splats reflected in a
   scene-captured environment), `partition_normal`, `glass` (`refract_data`),
   `lod_ladder` (one copy under a rotated parent) and `tiny_units_ortho`. Every
   exact case runs per backend (WebGL, WebGPU), per device scale factor (1 and 2;
   an odd 1277×719 viewport at 1.5 for two cases), per projection and per pose.
   The device scale factor comes from Playwright, not `?dpr=`, which is clamped to
   the native ratio.

## Exactness

For every view the harness captures the raw scene HDR target
(`captureHDRPixels('raw-scene-hdr')`), the visible LDR image before sRGB
encoding (`'visible-ldr'`) and, for pickable stores, the pick-ID buffer. Post
effects that vary with time or add nothing (detector noise, vignette, lens
distortion, bloom) are switched off first.

**Units.** The HDR and LDR targets are HalfFloat, so differences are measured in
half-float steps of each pixel's own value (**ULP16**). A differing pixel is
either **drift** (at most 2 ULP16: the arithmetic producing it moved) or a
**flip** (more: a discrete decision such as a discard at a Gaussian cutoff or a
coverage edge went the other way). Pick IDs admit no drift: any change is a
mismatch.

**Verdict classes.** Each commit declares one:

| class | allowed |
|---|---|
| `IDENTICAL` | nothing: 0 ULP16 in every buffer, 0 pick mismatches, equal drawn element counts |
| `ULP` | HDR: flips ≤ 3e-3 of pixels, p99.99 ≤ 32 ULP16. LDR: flips ≤ 1e-2, p99.99 ≤ 64. Pick mismatches ≤ 3e-3. Equal drawn element counts. |
| `INTENDED` (per case, `INTENDED=a,b`) | any change in the named cases; reported, never failed. Each must be backed by its own targeted test. Every other case keeps the commit's class. |

**How the `ULP` limits were set.** They are calibrated, not guessed. The gate
was run against builds whose point size factor, or splat focal length, was
scaled by (1 + ε). It covered points and splats in every blending mode, on both
backends at DPR 1 and 2 (Apple M4 Max, 2026-09-25). Per view:

| ε | HDR p99.99 (ULP16) | HDR flips | LDR flips |
|---|---|---|---|
| 2⁻²³ (1 float32 ULP) | ≤ 4 | ≤ 2.2e-4 | ≤ 4.0e-4 |
| 2⁻²⁰ (8 float32 ULPs) | ≤ 18 | ≤ 1.2e-3 | ≤ 4.8e-3 |
| 1e-4 | ≥ 13 | ≥ 6.7e-3 | ≥ 1.3e-3 |
| 1e-3 | ≥ 415 | ≥ 0.17 | ≥ 0.13 |

The HDR flip fraction is the metric that separates a rounding-level change from
a real 1e-4 error in every view. For splats, p99.99 and the tone-mapped LDR
buffer overlap between those two cases, so they are looser guards against gross
errors. `ULP` therefore accepts up to about 8 float32 ULPs of change in a
derived scale and rejects a 1e-4 relative error. A pixel-sized change is far
more visible than its ULP16 count suggests: a point or splat edge moving by a
fraction of a pixel flips the pixels it crosses. Re-run the sweep after
changing the scenes or the scorer.

A drawn-element-count difference between the builds fails a case outright (it
means LOD selection or residency diverged, which would otherwise surface as a
baffling pixel difference). The report lists the largest flip relative to the
frame's peak for review; there is no generic cutoff for it, so look at the
heatmap.

**Control arm.** The baseline is captured twice, from two page loads, and each
capture is taken twice within its page. A view whose baseline does not agree with
itself is **excluded**, and any exclusion makes the run **INCOMPLETE** (exit 3),
never a pass: a gate that could not see a view has not certified it.

**First-load effect (ANGLE/Metal).** On WebGL the first page load of a scene in a
fresh browser renders a few ULP16 differently from every later load (measured:
loads 2..N agree bit-for-bit, load 1 does not). Warming up with an unrelated
scene, or `--disable-gpu-program-cache`, does not remove it; WebGPU does not show
it. The harness therefore runs one discarded warm-up pass per build through the
same views before the measured arms.

## Performance

Per case and backend the harness runs R rounds (7 by default); each round loads
the baseline, the candidate and the baseline again, in an order that rotates
between rounds. Each arm records:

- **`gpuMs`**, GPU cost at a settled pose. Five repetitions of 20 back-to-back
  full-pipeline renders, each followed by a GPU sync (a 1-pixel `readPixels` on
  WebGL, `queue.onSubmittedWorkDone()` on WebGPU). The fastest repetition is the
  least-interfered estimate and is the value kept. It depends on neither vsync
  nor timer queries, which ANGLE/Metal reports unreliably.
- **`frameMs`, `frameP95Ms`, `cpuMs`, `rendersPerFrame`**: a deterministic orbit
  of 180 frames, one step per animation frame, with
  `--disable-gpu-vsync --disable-frame-rate-limit`. Motion matters: LOD
  selection, depth-sort scheduling, the density guard and dynamic clipping only
  work while the view changes, so a static camera measures almost none of the
  frame loop. The four figures are:
  - `frameMs`: wall time per presented frame;
  - `frameP95Ms`: p95 frame interval;
  - `cpuMs`: script time per frame, from CDP `Performance.getMetrics`
    `ScriptDuration`;
  - `rendersPerFrame`: `postProcessing.render` calls per frame. A frame that
    renders twice costs twice.

A metric fails when its point ratio `median(candidate) / median(baseline)`
exceeds `1 + floor`, and is reported as a win below `1 − floor`. `floor` is the
band a no-change comparison spans: the bootstrap 95% CI of the two baseline arms'
ratio, never below 1%. The candidate's own CI is reported but does not decide,
because testing one CI edge against another CI's half-width counts the sampling
noise twice and fails unchanged builds at small round counts.

Run the perf suite on the machine whose numbers you care about, with nothing else
heavy on its GPU. In headless Chrome with vsync off, rAF does not wait for the
GPU, so `frameMs` is effectively CPU-bound frame cost and `gpuMs` carries the GPU
cost. Keep the two apart when reading a result.

## When a run is not a pass

- **FAIL** (exit 1): a case broke its class, element counts diverged, a page
  errored or did not settle, or a perf metric regressed beyond its floor.
- **INCOMPLETE** (exit 3): no failure, but some views were excluded as
  nondeterministic. Fix the nondeterminism or remove the case; do not ignore it.
- exit 2: the harness itself crashed (build failure, server port in use).
