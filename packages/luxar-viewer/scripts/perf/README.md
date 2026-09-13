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
page-wide OPFS read cap. Point it at an already-served 4D scene, use a
persistent Chrome profile, and run it twice: the first pass fills missing
L2 entries; only compare the second pass when every arm reports
`misses=0`.

```bash
node scripts/perf/opfs-deep-pass-bench.mjs \
  --url 'http://127.0.0.1:5198/?src=http://127.0.0.1:9011' \
  --profile-dir /tmp/luxar-opfs-profile \
  --out /tmp/opfs-warmup.json --concurrency 64

node scripts/perf/opfs-deep-pass-bench.mjs \
  --url 'http://127.0.0.1:5198/?src=http://127.0.0.1:9011' \
  --profile-dir /tmp/luxar-opfs-profile \
  --out /tmp/opfs-sweep.json --concurrency 8,64,512,4096
```

The harness disables predictive prefetch, moves to `--start-frame`
(default 49), then plays exactly one transition at `--ladder-depth`
(default 6). It records the update-profiler tree, L2 hit/miss deltas,
read-gate occupancy, and the actual GPU renderer. Chrome runs headed by
default so Linux does not silently benchmark SwiftShader; pass
`--headless true` when the host's accelerated headless path is known.
