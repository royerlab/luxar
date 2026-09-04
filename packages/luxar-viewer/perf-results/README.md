# perf-bench output

This directory holds per-commit JSON results from the line perf bench.

Layout:

    perf-results/
      <commit-sha>/
        results.json

Compare two:

    pnpm perf:diff perf-results/<base>/results.json perf-results/<new>/results.json

## Viewer audit bench (`viewer-audit-perf-bench.spec.ts`)

End-to-end load + frame numbers over six real scenes, cold (fresh browser context
per repetition) plus one warm re-load, medians of `LUXAR_PERF_AUDIT_REPEATS`
(default 3) with `spread` = (max−min)/median per metric. Rows land in the same
`results.json` under `audit`, keyed `audit-<scene>[-hosted]/<backend>`; the
`## Viewer audit (load + frames)` section of `pnpm perf:diff` compares them.

    pnpm build   # the audit bench measures the PRODUCTION bundle
    LUXAR_PERF_PREVIEW=1 LUXAR_PERF_HEADLESS=1 LUXAR_PERF_CHROME_ARGS=--use-angle=metal \
      pnpm test:perf:e2e -g 'viewer audit'

- Datasets are the checkout's `datasets/examples` + `datasets/demos` stores
  (`make run-examples`; demos via `luxar demo run <key>`), served by the config's
  data server; a missing store skips its row with the path in the reason.
- Metrics come from `__luxarDebug.getPerf()` (load-timeline milestones,
  `isSettled`), a long-task observer, request counters, and the rAF cadence under
  forced continuous rendering at DPR 1 / 0.5 and dollied 4x closer. Never WebGL
  timer queries: on ANGLE/Metal they report 40–56 ms for scenes that run at 120 fps.
- `LUXAR_PERF_AUDIT_NET=hosted` throttles to 25 Mbps / 30 ms via CDP;
  `LUXAR_PERF_AUDIT_SCENES=dense-points,cmu1-2d` restricts scenes (the line bench
  owns `LUXAR_PERF_SCENARIO_FILTER` and rejects unknown ids).
- A separate `audit-dense-points-adaptive` row runs WITHOUT the `dpr=1` pin and
  records where the adaptive-DPR controller settles after 30 s.
- A `spread` above ~0.15 on a headline metric means the host was busy; re-run
  before reading deltas.

## Running the bench reliably

- **Headless on a Linux/NVIDIA box**: `LUXAR_PERF_HEADLESS=1` alone silently lands
  WebGL on SwiftShader (frames of seconds, timeouts). Pass
  `LUXAR_PERF_CHROME_ARGS=--use-angle=vulkan` as the config header documents; verify
  the captured `apiSurface` names the real adapter.
- **Headless system Chrome can misreport frame cadence on macOS**: Chrome 152 headless was
  observed to pick a 30 Hz BeginFrame cadence on some launches (every forced-continuous-render
  frame metric reads 33.3 ms with a 96 %-idle main thread) and occasionally to stop firing rAF
  altogether, hanging a cadence measurement until the test timeout. When the frame rows look
  quantised or a row runs into its timeout, re-run with `LUXAR_PERF_BROWSER=chromium` (the
  bundled Chromium ran the same pages at the display's 120 Hz throughout) and compare only
  runs taken with the same browser.
- **Never a headed run on a remote box without a live desktop**: a login-screen X
  session keeps Chrome occluded, which throttles rendering to ~1 fps and produces
  plausible-looking ~1000 ms medians. Garbage with no error anywhere.
- **Never override `launchOptions` at the Playwright project level**: it REPLACES the
  top-level args wholesale (dropping the WebGPU/Vulkan flags), silently moving WebGPU
  to a fallback adapter. Use the built-in envs instead.
- **Control-arm canary**: before believing any A/B delta, check an arm that ran
  identical code across the runs (e.g. the `default` arm, or a backend fallback arm).
  If the control moved more than a few percent, the box was loaded — discard the run.
  Shared machines can wake background work mid-run.
- **Absolute WASM floors**: `pnpm test:perf` reports misses without failing because
  wall-clock throughput follows host load. On a controlled quiet host, run
  `pnpm test:perf:strict` (equivalent to `LUXAR_PERF_QUIET_HOST=1 pnpm test:perf`) to
  enforce them.
- **Subset runs**: `LUXAR_PERF_SCENARIO_FILTER=<id,id>` restricts scenarios (ids are
  validated); `LUXAR_PERF_LINE_PRIMITIVES=default,<primitive>` crosses the primitive
  axis. One scenario × one primitive ≈ minutes, the full matrix ≈ tens of minutes.
- **Count sweeps**: `LUXAR_PERF_SYNTHETIC_COUNTS=250k,1M,4M` re-parameterizes each
  active synthetic scenario at every listed segment count (rows keyed `<id>-n<count>`,
  never colliding with authored-count rows). Note the `default` primitive arm builds
  whatever production would for that scene's size — under the auto policy that flips
  to the quad once count × the rendered-width factor reaches the threshold, so a wide
  scenario flips well below the swept count — which makes cross-era `default`-arm
  comparisons invalid; sweep with explicit `-capsule`/`-screen-space` arms.
- `&dpr=1` is pinned in the bench URLs — AdaptiveDPR must never run during sampling.
