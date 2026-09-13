# perf-bench output

This directory holds per-commit JSON results from the line perf bench.

Layout:

    perf-results/
      <commit-sha>/
        results.json

Compare two:

    pnpm perf:diff perf-results/<base>/results.json perf-results/<new>/results.json

## Viewer audit bench (`viewer-audit-perf-bench.spec.ts`)

End-to-end load + frame numbers over thirteen real scenes, cold (fresh browser context
per repetition) plus one warm re-load, medians of `LUXAR_PERF_AUDIT_REPEATS`
(default 3) with `spread` = (max−min)/median per metric. Rows land in the same
`results.json` under `audit`, keyed `audit-<scene>[-lod-bias-N][-hosted]/<backend>`; the
`## Viewer audit (load + frames)` section of `pnpm perf:diff` compares them.

    pnpm build   # the audit bench measures the PRODUCTION bundle
    LUXAR_PERF_PREVIEW=1 LUXAR_PERF_HEADLESS=1 LUXAR_PERF_CHROME_ARGS=--use-angle=metal \
      pnpm test:perf:e2e -g 'viewer audit'

- Datasets are the checkout's `datasets/examples` + `datasets/demos` stores.
  `make run-examples` builds `dense-points`, `bench-100-nodes`, and the two
  checked-in LOD examples. Build each remaining row with its corresponding
  `luxar demo run <key> -- --no-serve` command. ZebraHub specifically needs
  `luxar demo run zebrahub_velocity_streamlines -- --preset hifi --streamline-lod --no-serve`,
  which writes the `_hifi_lod` store. A missing store skips its row with the path
  in the reason; set `LUXAR_PERF_AUDIT_REQUIRE_SCENES=1` to make missing coverage
  fail the run.
- Metrics come from `__luxarDebug.getPerf()` (load-timeline milestones,
  `isSettled`), a long-task observer, request counters, and the rAF cadence under
  forced continuous rendering at DPR 1 / 0.5 and dollied 4x closer. Never WebGL
  timer queries: on ANGLE/Metal they report 40–56 ms for scenes that run at 120 fps.
- `LUXAR_PERF_AUDIT_NET=hosted` throttles to 25 Mbps / 30 ms via CDP;
  `LUXAR_PERF_AUDIT_SCENES=dense-points,cmu1-2d` restricts scenes (the line bench
  owns `LUXAR_PERF_SCENARIO_FILTER` and rejects unknown ids).
- `LUXAR_PERF_AUDIT_LOD_BIASES=1,2,4` crosses only scenes carrying a
  substitutive ladder; non-ladder scenes remain single-arm frame/load rows. Bias
  1 is the neutral existing key, while non-neutral rows are keyed `-lod-bias-N`.
  The bench pins `no-lod-fade` so committed counts and active levels describe one
  selected level rather than a cross-fade pair. Rows include committed
  visible-element totals under substitutive groups (or the whole scene when
  none exists) plus the active level of every substitutive group at
  the opening pose and after the 4x dolly. The audit scene catalog includes the
  Hilbert/ocean/ZebraHub dense-line cases, neuromast/zebrafish timelapses, the
  Tribolium recipes contract check, and small checked-in Lines/GSplat LOD examples;
  missing generated stores skip cleanly. Both Lines LOD rows use GSplat beads
  for their coarse levels until #2679; ZebraHub is the dense, real-scale Lines
  arm, not a genuine-Lines-levels arm. Active-level strings record the selector
  units and whether footprint stamps are present; stamp presence does not imply
  the footprint selector applies to the current display dimensions.
- A separate `audit-dense-points-adaptive` row runs WITHOUT the `dpr=1` pin and
  records where the adaptive-DPR controller settles after 30 s.
- A `spread` above ~0.15 on a headline metric means the host was busy; re-run
  before reading deltas.

### LOD-bias decision (#2685)

The final three-repeat sweep ran on September 13, 2026 at `666172e44` with
bundled Chromium, WebGL over ANGLE/Vulkan, and an NVIDIA RTX PRO 6000 Blackwell.
The filtered run included Lines LOD example, ZebraHub Lines, Zebrafish 4D,
GSplat LOD example, and Tribolium recipes: five of the catalog's six
substitutive-ladder scenes. CMU-1 2D, the other stamped ladder and only other
non-toy stamped arm, was unavailable because its pinned archive endpoint returned
HTTP 504. The remaining seven catalog scenes have no substitutive ladder, so the
bias axis does not cross them. `LUXAR_PERF_AUDIT_REQUIRE_SCENES=1` required this
filtered set, not all thirteen catalog scenes.

The examples were rebuilt at or after #2718, so the GSplat example carried its
footprint stamps. The code checkout predates the #2658 corpus rebuild required by
the general capture-discipline rule below; that does not affect these rows because
#2658 changes different scenes and gives them additive rather than substitutive
ladders. Every headline frame spread was at most `0.006`, but median rAF cadence
was exactly `16.7 ms` at both viewpoints in every arm. The rig was vsync-bound at
60 Hz, so this run could not resolve frame-cost differences or headroom above
60 fps; the decision therefore rests on selected levels, residency, requests,
and bytes.

| Scene              | Selector provenance                  | Bias 1 opening / 4x elements | Bias 2 opening / 4x elements | Bias 4 opening / 4x elements |                      Requests / bytes (bias 1; 2; 4) |
| ------------------ | ------------------------------------ | ---------------------------: | ---------------------------: | ---------------------------: | ---------------------------------------------------: |
| Lines LOD example  | occupancy; coarse GSplat beads       |               1,735 / 27,835 |                6,948 / 8,000 |               27,835 / 8,000 |             77 / 1.20 MB; 86 / 1.29 MB; 99 / 1.49 MB |
| ZebraHub Lines     | occupancy; Lines have no stamps      |        1,078,802 / 4,324,178 |        4,324,178 / 4,324,178 |        4,324,178 / 4,324,178 | 1,357 / 22.33 MB; 3,620 / 34.57 MB; 3,620 / 34.57 MB |
| Zebrafish 4D       | occupancy; store predates its stamps |                  341 / 1,369 |                1,369 / 1,369 |                1,369 / 1,369 |          121 / 2.04 MB; 121 / 1.99 MB; 121 / 1.99 MB |
| GSplat LOD example | footprint; toy pinned at finest      |                      45 / 45 |                      45 / 45 |                      45 / 45 |             59 / 1.05 MB; 59 / 1.05 MB; 59 / 1.05 MB |
| Tribolium recipes  | mixed; `levels` uses footprint       |            111,029 / 152,791 |            111,029 / 152,791 |            111,029 / 375,263 |       930 / 16.73 MB; 930 / 16.73 MB; 930 / 16.73 MB |

The Lines example changes from 27,835 GSplat beads at level 2 to the original
8,000-segment Lines node at level 3 after the dolly, so its falling element count
is a geometry switch, not coarsening. Zebrafish's equal 121-request arms select a
different level sequence across the opening pose and dolly; that different chunk
mix is consistent with the finer bias-2 arm being 43 KB smaller, so transfer is
not monotonic with selected element count.
The GSplat example is a 45-splat toy already pinned at its finest level in every
arm and does not discriminate the footprint policy. Tribolium `levels` is the
only stamped ladder that moves; its `footprint_dims` matched the displayed
dimensions, confirming that footprint selection, not occupancy fallback, made
the level 1 → 2 switch at bias 4.

Decision:

- Keep `WHOLE_OBJECT_FINEST_ANCHOR = 0.5`. Bias 2 promotes ZebraHub's opening
  pose from 1.08M to 4.32M committed elements, adds 2,263 requests, and transfers
  55% more bytes. That is exactly the eager-finest dense-Line regime the anchor
  exists to prevent; the high-end benchmark GPU absorbing it at 60 fps is not a
  reason to make every client pay the residency and network cost. Bias 2 is the
  decision-relevant transfer arm because ZebraHub is already finest there; bias
  4 selects the same levels and has identical request and byte totals.
- Keep `lod-bias` available to both occupancy and footprint selection. It is a
  no-op for bias ≥ 1 when a stamped ladder is already saturated at the finest
  level, but the stamped Tribolium `levels` ladder changes from level 1 to level
  2 after the 4x dolly at bias 4, raising committed scene elements 2.46x without
  changing its coarse opening contract. Bias below 1 can instead coarsen a
  footprint-selected ladder. The knob therefore remains a useful explicit
  quality override rather than a fallback-store compatibility switch.
- Keep the current viewer policy of a `1.5 px` median-footprint limit. The moving
  stamped evidence is the single real Tribolium `levels` ladder; CMU-1 was not
  available and the checked-in GSplat toy was already finest. Revisit this limit
  with #2734: its population at the 0.5-voxel initialization scale can bias the
  stored median footprint low and make the selector accept a coarser level.

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
- **Capture discipline**: run the sweep from `dev` at or after the corpus rebuilds
  in #2658, #2715, #2717, #2721, and #2710, then verify the SHA recorded in the
  result row before comparing it. Reusing an older generated store can otherwise
  change selector provenance without a code change.
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
