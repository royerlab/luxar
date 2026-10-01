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
make render-gate BASE=HEAD~1 CAND=HEAD SUITE=counters GATE_ARGS="--expect exp.json"
make render-gate BASE=origin/main CAND=HEAD SUITE=all GATE_ARGS="--heavy"
```

`SUITE` is `exact`, `perf`, any suite declared in the manifest's `suites` block
(`counters`, `playback`, `hosted`, `trees`, `cache`; see
[Workload suites](#workload-suites)), or `all` for every one of them.

The script behind it is `packages/luxar-viewer/scripts/render-gate/run-gate.mjs`
(pass extra flags with `GATE_ARGS=`, e.g. `GATE_ARGS="--backends webgl --dsf 1"`).
Reports land in `delme/gate/<label>/`: `report.md`, `report.json` and, for every
view that differed, a heatmap PNG (black: identical; blue: drift; red: flip).
An excluded control view also records the A/A HDR and LDR scores (each with the
`bbox` and count of its differing pixels, so lens-confined vs scene-wide reads off
`report.json`), heatmaps, pick mismatches (or `lengthMismatch` when the two pick
buffers differ in size), element counts, camera matrices, and within-page
stability in `report.json`; the report links the control heatmaps and summarizes
those fields.

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
   the native ratio. Exact cases load with `lodFinest` (deterministic finest
   content) unless the case sets `"liveLod": true`, as `lod_ladder` does, so LOD
   selection itself is compared; a case's own `urlParams` are appended to the
   defaults, not substituted for them.

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
itself (HDR, LDR, or pick ids or pick buffer size) is **excluded**, and any exclusion makes the run **INCOMPLETE** (exit 3),
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
  frame loop. Every orbit frame ends with the same GPU sync as `gpuMs`, and the
  frame is timed after it. The four figures are:
  - `frameMs`: wall time per frame, CPU work plus GPU work. The sync
    serializes the two (no CPU/GPU overlap), so this is an honest upper bound
    on a frame's cost rather than a throughput figure;
  - `frameP95Ms`: p95 of the same per-frame times;
  - `cpuMs`: script time per frame, from CDP `Performance.getMetrics`
    `ScriptDuration`. CPU cost only: `ScriptDuration` does not count the time
    spent blocked in the sync (measured on ANGLE/Vulkan with
    `perf-splats-5m-volumetric`: 2.3 ms of script in a 55 ms synced frame),
    and the WebGPU wait is asynchronous;
  - `rendersPerFrame`: `postProcessing.render` calls per frame. A frame that
    renders twice costs twice.
- **`wakeRenders`, `wakeBlockMs`**: waking a stopped loop the way an input
  handler does (`startAnimation()` after two idle frames), repeated nine times.
  `wakeRenders` counts renders from the wake up to and including the first
  animation frame after it (one is the minimum; a wake that also renders inside
  the handler costs two). `wakeBlockMs` is how long the call blocked its caller:
  input latency added to the handler. Chrome coarsens `performance.now()` to
  100 µs here, so `wakeBlockMs` differences within 0.2 ms are never judged (a
  0/0 or x/0 ratio would otherwise decide the verdict).

A metric fails when its point ratio `median(candidate) / median(baseline)`
exceeds `1 + floor`, and is reported as a win below `1 − floor`. `floor` is the
band a no-change comparison spans: the bootstrap 95% CI of the two baseline arms'
ratio, never below 1%. The candidate's own CI is reported but does not decide,
because testing one CI edge against another CI's half-width counts the sampling
noise twice and fails unchanged builds at small round counts.

Run the perf suite on the machine whose numbers you care about, with nothing else
heavy on its GPU. In headless Chrome with vsync off, rAF does not wait for the
GPU: without the per-frame sync an orbit frame times only the CPU submitting it
(0.1-0.3 ms against a 20-110 ms `gpuMs`), a GPU-bound change is invisible, and a
CPU stall on a resource the GPU still holds (a blocking `bufferSubData`) moves the
number by two orders of magnitude at unchanged GPU cost. So read the figures as:
`frameMs`/`frameP95Ms` are the only frame metrics that include the GPU (the
report says so under its Performance heading); `gpuMs` is GPU cost alone at a
settled pose; `cpuMs` is CPU cost alone. The sync is part of the harness, so both
arms carry it and the comparison stays like for like; a report measured before
it existed is not comparable with one measured after.

## Workload suites

A perf programme gates each commit on two things: exactness (the exact suite)
and a MEASURED improvement of a metric the commit declares. The workload suites
provide the second. Each is a key of the manifest's `suites` block and is run by
`scripts/render-gate/suites.mjs`:

| suite | what its starter cases do |
|---|---|
| `counters` | idle after a small drag (renders, bytes uploaded, stale frame); WebGPU uploads while idle |
| `playback` | play a hidden time axis (achieved fps, renders, last timepoint shown, duplicate decodes); a slider drag over it |
| `hosted` | cold loads over the simulated hosted link (first frame, settle, requests and bytes to first frame, concurrency) |
| `trees` | single timeline steps on a time-partitioned tree (step latency, lookups, console calls); an orbit over a LOD ladder |
| `cache` | several playback loops (duplicate decodes, pinned bytes, server requests) |

### Declaring a case

```json
"suites": {
  "counters": {
    "defaults": { "rounds": 5, "backends": ["webgl", "webgpu"],
                  "viewport": { "width": 960, "height": 540 }, "dsf": 1,
                  "urlParams": "noBlendWarmup", "serverProfile": "local" },
    "cases": [
      { "id": "idle_after_drag_mixed", "store": "gate/mixed.luxar.zarr",
        "pose": { "position": [14, 9, 16], "target": [0, 0, 0], "up": [0, 1, 0] },
        "workload": { "kind": "drag", "px": 4, "tailMs": 3000 },
        "staleFrameCheck": true,
        "metrics": [
          { "name": "render.count", "better": "lower", "kind": "counter" },
          { "name": "ext.gpuUploadBytes", "better": "lower", "kind": "counter" } ] } ] } }
```

A case takes `store` (relative to `datasets/`) or `synthetic` (as in the exact
suite), optional `urlParams` (appended to the defaults), `backends`, `pose`,
`projection`, `heavy` and `staleFrameCheck`. A missing store makes the case an
`error` row, not a crash. `heavy: true` cases run only with `--heavy`.

Per arm the runner opens the case in a fresh context, applies the pose, settles
(skipped for a cold load), zeroes the counters, runs the workload and reads the
metric object. A metric `name` may be:

- a field the workload returns (below);
- any viewer perf counter by name (`render.count`, `gpu.uploadBytes`,
  `decode.duplicates`, ...: every counter in `__luxarDebug.getPerf().counters`
  is in the object, so a newly added counter can be declared without touching
  the harness);
- an ext counter, prefixed `ext.` (below);
- a server-side figure: `serverRequests`, `serverBytes` (dataset requests of
  the arm), `maxInflight` (the server's peak concurrent requests), and after a
  cold load `requestsToFirstFrame`, `bytesToFirstFrame`, `serialDepth`.

`kind: "counter"` is judged by `judgeCounter`: when both baseline arms report
one single value, the medians are compared EXACTLY (`|delta| <= tol` passes; a
move in the `better` direction is a win, the other way a fail). A counter that
varies between baseline arms falls back to the A/A floor. `kind: "timing"` is
judged by `judgePerf` against the A/A floor, in the declared `better`
direction. A metric an arm does not report (an older baseline has no viewer
counters) reads `n/a`: never a pass for an expected metric.

### Workloads

The in-page halves are self-contained functions in `page-ops.mjs`. They use only
surfaces older builds also have (`__luxarDebug.{app, camera, renderOnce,
getPerf, getSceneLoader, inputHandler, sceneDimsManager}`, `app.getCameraPose /
setCameraPose / getDimensions / setDimensionValue / awaitDimensionUpdate`,
`postProcessing.render` wrapped to count renders), and the counter APIs only
behind `?.`.

| kind | params | returns |
|---|---|---|
| `idle` | `ms` | `renders`, `frames` |
| `drag` | `px`, `tailMs` | `renders` (drag + tail), `tailRenders`, `tailFrames` |
| `playback` | `dim?`, `fps`, `durationMs` or `loops` | `ticks`, `achievedFps`, `renders`, `rendersPerTick`, `lastTimepointShown`, `wraps`, `tickLatencyP50Ms`/`P90Ms` (records only) |
| `scrub` | `dim?`, `mode: step\|drag`, `steps`, `intervalMs` | step: `stepMs` (p50), `stepP90Ms`; drag: `commitsDuringDrag`, `dragFrames` |
| `orbit` | `radius?`, `steps`, `revolutions` | `levelFlips`, `renders` |
| `zoom` | `from`, `to`, `steps` | `levelFlips`, `renders` |
| `coldLoad` | (case `pose`) | `firstFrameMs`, `ttfpMs`, `settledMs` |

Notes on method:

- `drag` uses real input (Playwright's mouse, dispatched through CDP): an
  in-page synthetic `PointerEvent` has no active pointer, and the controls'
  `setPointerCapture` throws on it.
- `playback` drives the viewer's own playback
  (`inputHandler.getAnimationManager().play(dim, { targetFPS, loopMode: 'loop' })`)
  after moving the axis to its minimum; `dim` defaults to the first hidden
  dimension. Ticks come from the `playback.tick` records when the build has
  them, else from the axis value observed each animation frame.
  `lastTimepointShown` is 1 when the range maximum was ever applied.
- `scrub` in drag mode counts the frames in which committed geometry changed,
  from a build-independent signature of every geometry node (effective
  visibility, geometry id, attribute and index `version`s, drawn count, draw
  range, and the `version` of every texture uniform: points keep positions in a
  data texture).
- `orbit`/`zoom` `levelFlips` polls each LOD group's `displayedChildIndex`
  (`getSceneLoader().getDefaultLoader().lodGroupRegistry.list()`), the same
  way on every build; the viewer's own `lod.levelSwaps` counter is separate.
- `coldLoad` runs right after navigation and forces nothing. `firstFrameMs` is
  the earliest `timeline.firstCommit`, in ms since navigation start (so it
  includes fetching the viewer); `settledMs` is when `isSettled` has held for 3
  frames. A case `pose` is applied at once and again at `sceneLoaded`, because
  loading the scene frames the camera and no URL parameter sets a pose.
- `staleFrameCheck: true` screenshots the page without forcing a render, then
  renders once and screenshots again. The PNGs must be byte-identical; a
  candidate arm that differs fails the row with reason `stale-frame` (a stale
  baseline is only reported).

### Ext counters

`ext-counters.mjs` is installed with `context.addInitScript`, before any page
script, and wraps browser APIs rather than viewer code, so its numbers exist on
ANY baseline build. It counts bytes and calls of WebGL `bufferData`,
`bufferSubData`, `texImage2D/3D`, `texSubImage2D/3D` and WebGPU
`GPUQueue.writeBuffer/writeTexture` (`ext.gpuUploadBytes`, `...Calls`, split
`...Buffer`/`...Texture`), main-thread `fetch` calls, resource-timing
`transferSize`/`encodedBodySize` sums and peak resources in flight,
`console.{log,info,warn,error,debug}` calls (`ext.consoleCalls`) and
`Worker.postMessage` calls. Fetches and resource timing are per realm: requests
a worker makes are not seen (the server-side figures do see them).

### Expectations: `--expect`

A commit that claims an improvement declares it:

```json
{ "idle_after_drag_mixed": { "render.count": "zero", "ext.gpuUploadBytes": "win" } }
```

After judging, an unmet expectation fails the row with reason `expectation`:
`win` needs a win verdict; `zero` a candidate median of exactly 0; `same` a pass
with equal medians (for counters); `pass` no regression (pass or win). `n/a`
never meets an expectation. A case or metric the manifest does not declare is a
harness error (exit 2). An expectation on a `heavy` case that was skipped fails
too; pass `--heavy`.

### Server profiles

The workload suites serve both builds from servers configured by a profile in
the manifest's `serverProfiles` (the suite's `defaults.serverProfile`, or
`--server-profile` for the whole run):

- `local`: the plain server plus single-range `Range` support (206 with
  `Content-Range`, 416 when unsatisfiable, suffix `bytes=-n` included), which a
  `.zarr.zip` store needs;
- `hosted`: 100 ms latency before every response's headers, ONE shared token
  bucket at 25 Mbit/s, HTTP/2 over TLS (a self-signed certificate generated
  once into `delme/gate-certs/`; the browser gets
  `--ignore-certificate-errors`), a strong `ETag` with `If-None-Match` → 304,
  and `Cache-Control: max-age=300`.

The exact and perf suites keep the unconfigured server, whose responses are
unchanged. Every server logs its requests (`GET /__gate/requests`, cleared by
`POST /__gate/reset`). `serialDepth` approximates the longest dependent request
chain before the first frame as the number of start WAVES: dataset requests
sorted by start, where a start at least `max(latencyMs, 20)` ms after the
previous one opens a new wave (a dependent request cannot start before its
parent's response, which takes at least the latency). It is an approximation;
give it a `tol`.

### Report

Each suite adds a table (case, backend, metric, base median, candidate median,
ratio, verdict, and any expectation) to `report.md`; `report.json` carries the
rows under `suites.<name>`.

## When a run is not a pass

- **FAIL** (exit 1): a case broke its class, element counts diverged, a page
  errored or did not settle, a perf metric regressed beyond its floor, a
  workload suite row regressed, showed a stale frame or missed an expectation,
  or a declared store is missing.
- **INCOMPLETE** (exit 3): no failure, but some views were excluded as
  nondeterministic. Fix the nondeterminism or remove the case; do not ignore it.
- exit 2: the harness itself crashed (build failure, server port in use), or
  the invocation was bad (unknown suite, profile, or expectation).

A browser that dies mid-run (a GPU-process crash, or the OOM killer on a busy
host) is relaunched, and the case it interrupted is retried once; the report
header then says how many relaunches happened. A case that kills the browser a
second time is reported as an error. On a laptop, run the gate under
`caffeinate -dims` (macOS): a machine that sleeps stalls the run without
failing it.

`report.json` is written before `report.md`. To rewrite the markdown from a
saved run without measuring again, pass
`--from-json <dir>/report.json`.
