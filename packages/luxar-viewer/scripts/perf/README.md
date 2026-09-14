# Timelapse-nav perf bench

Before/after harness for GPU-pool / upload-path changes. Scrubs the
time dimension of a 4D gsplat scene and records per-timepoint wall
time, rAF frame-delta percentiles (captures upload hitches CPU timers
can't see), GPU-pool stats, and `renderer.info.memory` (WebGPU) across
three phases: initial ladder load, N full scrubs, and a dataset-switch
churn loop.

```bash
# CORS-enabled data access relies on chromium --disable-web-security;
# serve the repo root plainly:
(cd <repo-root> && python3 -m http.server 9009 &)

# one run per checkout, distinct ports:
node scripts/perf/timelapse-nav-bench.mjs --viewer-dir <checkout>/packages/luxar-viewer \
  --port 5199 --label after --out /tmp/perfbench/after-webgl.json --scrubs 2
node scripts/perf/timelapse-nav-bench.mjs ... --renderer webgpu --tps 40  # capped scrub

python3 scripts/perf/timelapse-nav-compare.py /tmp/perfbench   # markdown tables
```

Caveats learned the hard way:

- Never point two runs at the same vite port; verify no stale listener
  (`lsof -iTCP:<port> -sTCP:LISTEN`) — a server from another checkout
  silently poisons results.
- Wall-clock medians are machine-load-noisy across single runs; treat
  <2x deltas as suggestive, use frame percentiles + memory counters as
  the hard signals.
- Dataset switches DISPOSE the scene loader (and its pool), so pool
  counters reset across the churn phase — read pool deltas within a
  phase only.
- The headless `?renderer=webgpu` path occasionally crashes the page
  mid-scrub (device-loss recovery reload); rerun or cap with --tps.

## Native WebGPU (real Dawn/Metal adapter)

Playwright's BUNDLED chromium has `navigator.gpu` but **no Metal
adapter** (`requestAdapter()` → null), so every `?renderer=webgpu` run
on it silently exercises the WebGL2 fallback. Real Chrome
(`channel: 'chrome'`, works headless) has a full Metal-3 adapter — pass
`--channel chrome` to the bench / `--chrome` to the probes. Two traps:

- `navigator.gpu` only exists in SECURE contexts — probing it on
  `about:blank` reads as "no WebGPU" (goto a localhost page first).
- `--use-gl=egl` (our E2E GPU-acceleration flag) forces ANGLE-GL and
  DISABLES the Dawn/Metal adapter — the tools omit it in channel mode.

## grow-leak-probe.mjs

Forces the GPU pool's GROW path (same nodeId, doubling counts, rendered
each step), then releases + LRU-evicts, reporting
`renderer.info.memory.attributes` per step. On native WebGPU this
demonstrates the in-place-rebuild strand directly: pre-fix (main), each
grow permanently pins the replaced buffer generation (+5 views/gen,
never reclaimed — +23.4 MB after 5 doublings of a 64K node); post-fix
(grow = release + reacquire), everything returns to the clean floor
after eviction.

## opfs-deep-pass-bench.mjs

Reproduces a single pinned-depth timelapse transition while sweeping the
page-wide OPFS read cap. Generate the 51-frame H2AFVA scene with
`hatch run python -m luxar.demos.demo_gsplats_4d_h2afva_timelapse --no-serve`,
serve the resulting store on port 9011, and run the viewer dev server on port 5198. Use a persistent Chrome profile and run the harness twice: the first pass
fills missing L2 entries; only compare the second pass when every arm reports
`misses=0`. As with the timelapse navigation bench, verify that neither port has
a stale listener from another checkout before recording results. Run each server
command below in its own terminal, then run the harness from the viewer directory.

```bash
(cd <repo-root> && \
  hatch run python -m luxar.demos.demo_gsplats_4d_h2afva_timelapse --no-serve)
(cd <repo-root> && \
  hatch run luxar serve datasets/demos/gsplats_4d_h2afva_timelapse.luxar.zarr --port 9011)
pnpm dev --port 5198

node scripts/perf/opfs-deep-pass-bench.mjs \
  --url 'http://127.0.0.1:5198/?src=http://127.0.0.1:9011' \
  --profile-dir /tmp/luxar-opfs-profile \
  --out /tmp/opfs-warmup.json --concurrency 64

node scripts/perf/opfs-deep-pass-bench.mjs \
  --url 'http://127.0.0.1:5198/?src=http://127.0.0.1:9011' \
  --profile-dir /tmp/luxar-opfs-profile \
  --out /tmp/opfs-sweep.json --concurrency 8,64,512,4096
```

The harness disables predictive prefetch by default; pass `--prefetch` or
`--prefetch true` to reproduce production scheduling. In `once` mode playback
does not apply the tick that reaches or crosses `range[1]`, so the harness
mirrors that boundary and discrete grid snapping to derive the coordinate that
leaves exactly one applied advance. An explicit `--start-frame` must equal that
coordinate so the measurement remains exactly one transition at integer
`--ladder-depth` (default 6). Pass `--clear-first` to
clear every cache level before the first arm. It records transition and settle timings separately, update and
LOD-refinement profiler trees, L2 hit/miss deltas, read-gate occupancy, write
queue `depth`/`inFlight`, and the live viewer renderer/backend. Chrome runs
headed by default so Linux does not silently benchmark SwiftShader; pass
`--headless` or `--headless true` when the host's accelerated headless path is known.

The exploratory trace behind #2731 reported a mixed-cache wave with 339 L2
misses: cap 512 showed 9.884 s wall / 6.011 s aggregate `Load Arrays`, while
cap 64 showed 2.941 s / 0.802 s. Those absolute numbers are provenance only:
the old harness included settle/refinement work in wall time and stale profiler
rows in the aggregate. The cross-arm result remained useful: once warm
(`misses=0`), the pre-fix harness reported five wall-time passes at 4.99–5.79 s
for cap 64 and 3.72–5.24 s for the unbounded-ish arm, with no 6–30 s tail or
cap-dependent trend. Those ranges include the pre-fix settle/quiet-window tax.
`ValidationQueue` runs only during store initialization, not per read. The
evidence attributes the reported stall to mixed L2-miss fan-out/browser
contention; the cap added in #2732 is sufficient, with no additional scheduling
change justified. New artifacts expose `animationMs`, `settleMs`,
transition/settle updates, and refinement roots separately.

The corrected September 13, 2026 baseline used the setup above with
`Time range=[0,50]`, `step=1`, and 51 timepoints. Because `once` stops one
coordinate short of `range[1]`, each of the four warm sweeps measured 48→49:
cap 64 reported `animationMs` 0.121–2.923 s and `settleMs` 0.187–1.744 s; cap
4096 reported `animationMs` 0.103–0.131 s and `settleMs` 0.064–1.555 s. Three
cap-4096 arms stamped `missedUpdates=1` (cap 64 stamped 0–1), so the phase split
remains diagnostic for the fastest arms, but the baseline still shows no 6–30 s
warm tail or cap-dependent trend.

This untyped harness depends on the debug surface names
`getSceneLoader`, `getDefaultLoader`, `getProfiler`, `inputHandler`,
`sceneDimsManager`, and `renderer`; cache stats `l2.activeReads`,
`l2.queuedReads`, `l2.misses`, `l2WriteQueue.depth`, and
`l2WriteQueue.inFlight`; and profiler methods `getTimings` and
`getRefinementTimings`. Keep this list in sync when those typed APIs change.
