# Viewer Developer Tools

Standalone developer tools for driving a real Chromium, generating benchmark
fixtures, and serving range-readable data. Sibling to `scripts/` (which holds
build / quality / perf-diff utilities); these are local development helpers,
not part of any build pipeline.

The two browser drivers share the same recipe: launch headless Chromium
via `@playwright/test` with GPU acceleration flags, navigate to a Luxar
URL with `?debug` enabled, wait for the scene to initialize, poke at
`window.__luxarDebug`, and write a PNG to disk. The E2E server-identity
and worker-sizing helpers are test-harness infrastructure rather than
browser drivers.

## Contents

```
tools/
├── agent-driver.ts          # Browser debugging driver
├── capture-hires.ts         # High-resolution figure capture for papers
├── e2e-server-identity.ts   # Checkout identity + Playwright preflight helpers
├── e2e-workers.ts           # Local Playwright parallelism sized to the machine
├── make-zip-bench-fixtures.py # Build directory/STORED/DEFLATE benchmark fixtures
└── range-http-server.py     # Range server + hermetic slow-body test endpoint
```

## `e2e-server-identity.ts`

Prevents local Playwright runs from silently reusing Vite or dataset
servers rooted in another clone/worktree. It derives a deterministic
identity from the checkout's canonical repository path, writes a small
gitignored marker under `.luxar-e2e-identities/`, exposes the marker
through Vite middleware, and validates the exact response during global
setup. The repository-root Python data servers serve the same marker directly.

The standard, performance, screenshot, and video Playwright configs use
the checkout-specific Vite marker as their `webServer.url` readiness
probe. A sibling checkout therefore returns 404, while a same-checkout
Vite server remains reusable. The standard and performance configs also
probe the repository-root data marker; screenshot/video generation uses
`luxar serve`, which has no identity endpoint, so those configs disable
data-server reuse instead. The helper also performs the HTTP availability
checks for required E2E datasets so filesystem presence cannot mask a
mis-rooted server.

## `e2e-workers.ts`

Decides how many workers a local Playwright run may use, so a busy machine
gets a slower suite instead of a red one. `playwright.config.ts` takes the
count; the E2E global setup prints one line with the run's ceiling and the
inputs the count was sized from (there, not at config load, because Playwright
applies `--workers=N` afterwards):

```text
[🧵] [E2E] parallelism: max 1 worker — 16 cpus, load1 22.2 (capacity)
```

`max` because the stamped number is `config.workers`, a ceiling: Playwright
narrows it to `min(workers, maxConcurrentTestGroups)` after global setup, so a
single-file run can be stamped `max 3` and then report "using 1 worker". A
ceiling that differs from what was sized prints as
`(capacity sized 3, run with 1)` — both numbers, no cause claimed, since
`--workers=N`, `--ui`, a watch session and `playwright.perf.config.ts`'s own
`workers: 1` all get there.

Four workers is still the ceiling — the binding resource is the Python dataset
server on port 9000, not the GPU — and the count is that ceiling scaled by the
box's free fraction, `clamp(round(4 * (cpus - load1) / cpus), 1, 4)`. The bands
are fractions of the box: 4 while at least 7/8 of it is free, 3 down to 5/8, 2
down to 3/8, 1 below that — on 16 cores, 4 up to load 2, 3 up to load 6, 2 up to
load 10, then 1. An **idle** box of any size keeps the ceiling, so this only ever
backs off under load. Measured on a 16-core box at a 1-minute load of 12–24,
`dimension-animation.spec.ts` failed 15 of 21 tests at four workers and passed
21 of 21 at one, every failure a bare action timeout with the element already
visible/enabled/stable. Only those two counts were measured — 16 cores at load
12 → 1 is the green configuration — and the 2 and 3 the formula can also pick
are interpolation.

The signal is imperfect on purpose: the 1-minute load average lags and is read
once at startup (a run launched just after a burst crawls on a box that is
already idle), and on Linux it counts uninterruptible-sleep tasks, so a
concurrent `git lfs pull` throttles the suite for I/O it does not compete for.
Sampling `os.cpus()[].times` twice would be lag-free but needs ~150 ms, and
Playwright re-evaluates the config module in every worker process as well as the
parent — so that wait would be paid N+1 times per run. Nothing prevents an
`await` there (the package is ESM and the config is loaded with
`await import()`); the cost is simply not worth a lag-free reading.

The arithmetic lives in the pure `chooseLocalWorkers({ cpus, load1, override })`
(covered by `src/tests/unit/config/e2e-workers.test.ts`); `resolveE2EWorkers()`
is the thin wrapper that reads `os.availableParallelism()`, `os.loadavg()`, `CI`,
and `LUXAR_E2E_WORKERS`. Precedence: Playwright's own `--workers=N` overrides the
config outright (and `--debug` forces one worker), then
`LUXAR_E2E_WORKERS=N` — an integer, clamped to `[1, cpus]` purely as a typo guard
(`=40` on a 16-core box), so a pin may exceed the ceiling of four but not the
machine — then the heuristic; `CI` is unconditionally serial. The heuristic
itself is deliberately NOT capped by the core count: the binding resource is that
single-threaded dataset server rather than the cores, and capping would lower the
historical default on a 1- or 2-core box, which nothing has measured as needed.
Windows has no load average, and is reported as no signal rather than as an idle
box, which means the ceiling.

## `agent-driver.ts`

Lets Claude Code (or any AI agent) "see" the viewer without a monitor.
Mirrors all browser console output to the terminal — including
`pageerror`, `requestfailed`, and any HTTP response ≥400 — then dumps a
JSON snapshot of `__luxarDebug.scene` / `.camera` / `.renderer` / the
`getState()` performance block, and screenshots the current view.

Wired into the viewer package via two pnpm scripts (see
`package.json`):

```bash
pnpm agent:debug              # headless
pnpm agent:debug:visible      # visible browser (--headless=false)
```

Direct invocation accepts `--url=`, `--headless=true|false`, and
`--wait=<ms>` flags (the wait controls how long to give Three.js to
finish first-frame initialization). The default URL is
`http://localhost:5173/?debug` (or `APP_URL` from the environment); the
driver appends `?debug` if you forget to. Output paths:

- `test-results/debug/debug-view.png` — success screenshot.
- `test-results/debug/error-state.png` — written on any thrown error
  before the process exits 1.

The driver requires the viewer to be running locally and the scene to
have been opened with `?debug` so `window.__luxarDebug` is populated —
otherwise the state dump returns `{ error: 'Debug mode not enabled' }`.

See the parent README's "AI-Assisted Debugging" section for the full
debug-loop workflow.

## `capture-hires.ts`

High-resolution figure capture for paper and slide figures. Same
launch pattern as `agent-driver.ts`, but driven entirely by environment
variables so it slots cleanly into figure-generation scripts:

| Env var            | Purpose                                                | Default                        |
| ------------------ | ------------------------------------------------------ | ------------------------------ |
| `APP_URL`          | Viewer URL (required; must include `?debug`)           | —                              |
| `OUT`              | Output PNG path                                        | `test-results/debug/hires.png` |
| `WAIT`             | Total ms budget for scene init + post-camera streaming | `15000`                        |
| `WIDTH` / `HEIGHT` | Viewport in CSS pixels                                 | `2400` / `1600`                |
| `DSF`              | `deviceScaleFactor` (for retina-quality output)        | `1`                            |
| `CAMERA_ZOOM`      | Scale distance from orbit target                       | `1`                            |
| `CAMERA_POS`       | Absolute camera position `x,y,z` (world units)         | —                              |
| `CAMERA_TARGET`    | Orbit target `x,y,z`                                   | —                              |
| `CAMERA_FOV`       | Vertical FOV override (degrees)                        | —                              |
| `SLICE`            | nD slice override, e.g. `"time:240,channel:0"`         | —                              |

Camera and slice overrides are applied after the first 15s of the
`WAIT` budget is consumed (so auto-fit and initial chunks land first);
the remainder of the budget then gives the new frustum a chance to
stream in fresh chunks before the screenshot fires.

```bash
APP_URL="http://localhost:5173/?src=/data/demo.zarr&debug" \
OUT=figure-1.png WAIT=20000 WIDTH=3200 HEIGHT=2400 DSF=2 \
  pnpm exec tsx tools/capture-hires.ts

# nD navigation + camera framing for a multi-channel time series
APP_URL="http://localhost:5173/?src=/data/4d.zarr&debug" \
SLICE="time:240,channel:0" CAMERA_ZOOM=1.5 \
  pnpm exec tsx tools/capture-hires.ts
```

The script logs a readiness summary of the `getState()` snapshot before
screenshotting. If the scene is not ready it warns
`Warning: scene not ready — <reason>` followed by the four counts in
parentheses; the wording stays vague about WHY on purpose, because the
reason may be that there was no debug state, an unexpected `getState()`
shape or a probe that threw, in which case those counts are placeholders
rather than measurements. A ready verdict that still carries a caveat —
a total that was counted as 0 because it was `Infinity`/`NaN`, or
because a partial snapshot never carried it — prints as `Note: <reason>`,
naming the fields, instead of being swallowed.

All four geometry types are reported — `totalPoints`, `totalGSplats`,
`totalLines`, `totalTriangles`, plus `totalElements`, which is the
**maximum** of the snapshot's own `totalElements` field and the sum of
those four (a version-skew hedge: the tool captures against whatever
viewer build is served at `APP_URL`, and a stale or partial snapshot must
never under-claim against the per-type totals it is carrying — the
current viewer sets the field to exactly that sum, so the max is inert on
a live snapshot). The per-node counts `pointCloudCount`, `gsplatCount`,
`lineCount` and `meshNodeCount` come along too, so a mesh-only or
lines-only scene is recognised as loaded rather than reading as empty. A
nonzero `totalDroppedElements` rejects the capture as renderer-truncated;
split or partition the oversized node before capturing. Other warnings
almost always mean the slice position is wrong or the dataset URL is
stale. The converse does not hold: the verdict measures the scene
graph (hidden nodes and all LOD levels included), so an all-hidden scene
passes and can still capture blank.

The verdict itself is computed in Node by
`summarizeCaptureReadiness()` (`../src/core/app/debug/capture-readiness.ts`);
the browser closure only hands back `getState()` verbatim. That split is
deliberate — deciding readiness inside `page.evaluate` is untestable, and
the previous inline version read the totals from a `state.performance`
sub-object that `getState()` has never returned, silently reporting every
scene as empty (#1579). The probe is only a diagnostic: if `getState()`
throws, the failure is logged as the not-ready reason and the screenshot
is still written.

## Requirements

- The viewer dev server must be running (`pnpm dev`) — both tools talk
  to a live browser instance.
- `@playwright/test` Chromium must be installed (`pnpm exec playwright
install chromium` if first-time setup).
- The target URL must include `?debug` so `window.__luxarDebug` is
  exposed; `agent-driver.ts` auto-appends it, `capture-hires.ts` does
  not.

## See Also

- `../scripts/README.md` — build / lint / perf-diff scripts.
- Parent `../README.md` — "AI-Assisted Debugging" section.
- `../src/core/app/debug/debug-interface.ts` — assembles the
  `window.__luxarDebug` object (`scene`, `camera`, `renderer`,
  `controls`, `sceneDimsManager`, `getState()`, `renderOnce()`).
- `../src/core/app/debug/debug-state.ts` — the pure scene-walking
  computer behind `__luxarDebug.getState()` (point / gsplat / line /
  triangle counts + camera + dim reporting).
- `../src/core/app/debug/capture-readiness.ts` — the pure readiness
  verdict `capture-hires.ts` runs over that snapshot.
