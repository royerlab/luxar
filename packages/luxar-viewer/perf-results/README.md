# perf-bench output

This directory holds per-commit JSON results from the line perf bench.

Layout:

    perf-results/
      <commit-sha>/
        results.json

Compare two:

    pnpm perf:diff perf-results/<base>/results.json perf-results/<new>/results.json

## Running the bench reliably

- **Headless on a Linux/NVIDIA box**: `LUXAR_PERF_HEADLESS=1` alone silently lands
  WebGL on SwiftShader (frames of seconds, timeouts). Pass
  `LUXAR_PERF_CHROME_ARGS=--use-angle=vulkan` as the config header documents; verify
  the captured `apiSurface` names the real adapter.
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
- **Subset runs**: `LUXAR_PERF_SCENARIO_FILTER=<id,id>` restricts scenarios (ids are
  validated); `LUXAR_PERF_LINE_PRIMITIVES=default,<primitive>` crosses the primitive
  axis. One scenario × one primitive ≈ minutes, the full matrix ≈ tens of minutes.
- **Count sweeps**: `LUXAR_PERF_SYNTHETIC_COUNTS=250k,1M,4M` re-parameterizes each
  active synthetic scenario at every listed segment count (rows keyed `<id>-n<count>`,
  never colliding with authored-count rows). Note the `default` primitive arm builds
  whatever production would for that count — under the auto policy that flips to the
  quad at/above the threshold — so cross-era `default`-arm comparisons are invalid;
  sweep with explicit `-capsule`/`-screen-space` arms.
- `&dpr=1` is pinned in the bench URLs — AdaptiveDPR must never run during sampling.
