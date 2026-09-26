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
| `ULP` | frame energy change within ±1e-5; p99.9 of the 4×4 box-filtered tile change ≤ 1.3e-4 of the brightest tile; pick NODE mismatches ≤ 3e-3; equal drawn element counts |
| `INTENDED` (per case, `INTENDED=a,b`) | any change in the named cases; reported, never failed. Each must be backed by its own targeted test. Every other case keeps the commit's class. |

**Why `ULP` is judged on light, not on pixels.** A per-pixel count cannot tell a
real error from rounding jitter. Take a line shader that reads
`projectionMatrix[1][1]` where it used to read a CPU uniform of the same value.
On ANGLE/Metal that changes how the surrounding matrix products compile, and a
dense, far-away line scene moves about one float32 ulp per vertex. That flips
2.7e-3 of the frame's pixels, yet the frame's energy moves 1e-7, exactly like a
1-ulp nudge of the uniform. A width that is really 1e-4 too large moves the
energy by 1e-4. So `ULP` compares what a viewer could see:

- the signed relative change of the frame's total energy;
- the change of each 4×4 box-filtered tile relative to the brightest tile. The
  filter cancels light moved between neighbouring pixels and keeps light added
  or removed.

Per-pixel drift, flips and p99.99 stay in the report for reading.

Pick buffers are judged on NODE identity. Where one node's elements overlap, the
winning element is a near-tie that any rounding flips. A 1-ulp nudge of the line
width changed the element at up to 35% of a dense line scene's pick pixels, and
changed the node at none.

**How the `ULP` limits were set.** They are calibrated, not guessed. Calibration
arms scale one derived quantity by (1 + ε): the point size factor, the line
width scale or the splat focal length. The real commits of the
projection-in-shader work were measured next to them. Every case covered
points, lines and splats in every blending mode, on both backends (Apple M4 Max,
2026-09-25). Worst value per view:

| arm | abs(energy change) | tile p99.9 |
|---|---|---|
| rounding level: 1 or 8 float32 ulps, the real commits | ≤ 2.2e-6 | ≤ 9.4e-5 |
| real error: ε = 1e-4 | ≥ 4.4e-6 (see below) | ≥ 1.7e-4 |

Max- and normal-blended splats at a close pose barely change energy under a
1e-4 focal error, because those modes do not add light. There the tile metric
catches the error, at ≥ 2.2e-3. The tile maximum overlaps between the two rows
of the table and is only reported. The margins are about 1.4× on the tile metric
and about 5× on energy wherever energy separates. Re-run the calibration arms
(`--cand-dist` accepts a hand-patched build) after changing the scenes or the
scorer.

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

A browser that dies mid-run (a GPU-process crash, or the OOM killer on a busy
host) is relaunched, and the case it interrupted is retried once; the report
header then says how many relaunches happened. A case that kills the browser a
second time is reported as an error. On a laptop, run the gate under
`caffeinate -dims` (macOS): a machine that sleeps stalls the run without
failing it.
