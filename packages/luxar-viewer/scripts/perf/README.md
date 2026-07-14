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
