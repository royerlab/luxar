# E2E Tests (Playwright)

Full-browser end-to-end specs that exercise the viewer through real
WebGL, OPFS, and network paths. The folder also ships the **shared
test fixture, helper toolbox, global setup, and TSL/GLSL parity
harness** that every spec is built on — those are described first
because most spec-authoring questions are really questions about how
the supporting infrastructure works.

For the wider testing strategy (unit vs E2E, mocks, builders, fixture
data) see the parent [tests README](../README.md).

## Quick Start

```bash
# Run the whole suite
pnpm test:e2e

# Interactive UI
pnpm test:e2e:ui

# Single spec
pnpm exec playwright test src/tests/e2e/basic-rendering.spec.ts

# Open the HTML report after a run
pnpm test:e2e:report
```

New specs import from `./fixtures`, not from `@playwright/test`
directly — see [Shared Fixture](#shared-fixture-fixturests) below.

### Mobile / touch suite

```bash
make test-e2e-mobile          # from the repository root; refreshes fixtures
pnpm test:e2e:mobile          # direct playwright.mobile.config.ts invocation
```

`src/tests/e2e/mobile/` runs under real device emulation (iPhone 14 portrait +
landscape, iPad Pro 11 and Pixel 7), all on **Chromium**: both the GPU promotion
runner and the hosted PR job use Chromium, and the gestures are synthesised through CDP
`Input.dispatchTouchEvent` (`mobile/touch-helpers.ts`: pinch, twist, one-finger
drag, 2→1 release, long-press, double-tap), which WebKit does not expose. The
main config ignores this folder; the mobile config only matches it. The suite
runs in PR CI for TypeScript changes.

What it covers: the media queries actually match under emulation, pinch dollies
the camera while `visualViewport.scale` stays 1, twist rolls, a finger lifting
out of a pinch continues as a rotate, double-tap re-frames, tap picks + shows
the tooltip, long-press opens the element / rail menus, fly mode looks and flies
by touch, the rail / help / monitor / layers geometry stays inside a phone
viewport, and the DPR cap and GPU budget resolve to the mobile values. Real iOS
Safari behaviour (no `contextmenu` on long-press, no Fullscreen on iPhone,
dynamic toolbar) is the manual device checklist's job, not this suite's.

This suite is the integration check for the touch series planned in #2582, not
a standalone test of this branch. `gestures.spec.ts` and `fly.spec.ts` require
parts A and C (gesture ownership and touch controls), and the double-tap case in
`gestures.spec.ts` also requires B. `pick.spec.ts` requires B,
`layout.spec.ts` requires D1, and `runtime.spec.ts` requires E. The complete
suite is validated with #2595 present so held pointer gestures keep the render
loop awake. Run it against the complete series; expected failures on an earlier
stack are not harness flakiness.

Two helper rules keep the gesture specs honest under load (a shared Mac at a
1-minute load of 26 ran a 16-step CDP drag in 5 s):

- **Read the camera after two animation frames** (`cameraPose`). Controls apply
  input inside their per-frame `update()`, so a pose read straight after the
  last touch event can predate the frame that applies it — under software GL
  with two workers a frame can take a second or more, and every gesture then
  "fails" with the camera exactly at its home pose.
- **Queue a double-tap's four touch events without awaiting** (`doubleTap`).
  Awaited CDP round trips put 300–700 ms between the two lifts, past the
  viewer's 300 ms double-tap window, so the gesture read as two single taps.
  Queued on one session they land milliseconds apart whatever the box is doing.

The worker count is the desktop plan's, capped at two — the phone viewports
are cheap but the gesture timing is not, and the load-sizing in
`tools/e2e-workers.ts` is what stops a busy box from inventing failures here.

Under load the first touch move can arrive seconds after the press; before
#2595 the loop had idle-paused by then and the whole drag moved nothing — on
`dev` with a mouse too.

### Parallelism is sized to the machine

The local worker count is not a constant. `playwright.config.ts` asks
`tools/e2e-workers.ts` for it, and the global setup prints the run's ceiling as
one line before the pre-flight checks:

```text
[🧵] [E2E] parallelism: max 1 worker — 16 cpus, load1 22.2 (capacity)
```

That number is `config.workers`, the ceiling for the run — Playwright narrows it
to `min(workers, maxConcurrentTestGroups)` after global setup, so a single-file
run of a spec pinned to one worker is stamped `max 3` and then reports
"using 1 worker". When the ceiling is not what was sized the line carries both,
without guessing at the cause (`--workers=N`, `--ui`, a watch session and
`playwright.perf.config.ts`'s own `workers: 1` all produce it):

```text
[🧵] [E2E] parallelism: max 1 worker — 16 cpus, load1 1.9 (capacity sized 3, run with 1)
```

Four workers remains the ceiling — the binding resource is the Python dataset
server on port 9000, not the GPU — and the count is that ceiling scaled by the
box's free fraction, `clamp(round(4 * (cpus - load1) / cpus), 1, 4)`. The bands
are fractions of the box, so they hold at any size: 4 while at least 7/8 of it is
free, 3 down to 5/8, 2 down to 3/8, 1 below that. On 16 cores that is 4 up to load
2, 3 up to load 6, 2 up to load 10, then 1. An **idle** box of any size keeps the
ceiling, so the sizing only ever backs off under load. On a workstation shared
with CI runner slots that matters a lot: at a 1-minute load of 12–24 on 16 cores,
`dimension-animation.spec.ts` failed 15 of 21 tests at four workers and passed
21 of 21 at one, every failure a wall-clock action timeout
(`page.click: Timeout 10000ms exceeded`, element already visible/enabled/stable)
with no product cause. A loaded box now gets a slower run instead of a red one,
so treat a burst of timeouts across unrelated specs as a capacity report before
filing a viewer bug. Only those two counts were measured, though — 16 cores at
load 12 → 1 is the configuration measured green, and the 2 and 3 the formula can
pick are interpolation — and the protection is probabilistic either way: the
1-minute load average lags, is read once at startup, and on Linux includes tasks
blocked on I/O elsewhere on the machine.

Because concurrency is now load-dependent, the two wall-clock guards in
`nd-navigation.spec.ts` and `worker-wasm-integration.spec.ts` have more headroom
on a loaded box **than the same box would give them at four workers**. That is
not a claim about idle machines versus busy ones: at one worker on the loaded box
a real DOM click still took 869–1216 ms, so its absolute margin against a fixed
wall clock is not obviously better than an idle box's at three.

Pin the count with `LUXAR_E2E_WORKERS=N` (an integer, clamped to `[1, cpus]`;
anything else is ignored), or with Playwright's own `--workers=N`, which
overrides the config outright — as does `--debug`, which forces one worker. CI is
unconditionally serial.

### Which script runs which specs

| Script                       | Selection                                                                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `pnpm test:e2e`              | Everything under `src/tests/e2e/`, minus `*perf-bench.spec.ts` and `src/tests/e2e/mobile/` (`testIgnore`)                                        |
| `pnpm test:e2e:ci`           | The same, minus tests tagged `@visual` — a **title grep**, not a file list                                                                       |
| `pnpm test:e2e:visual`       | Local run of tests tagged `@visual`; snapshot assertions are active on Linux                                                                     |
| `pnpm test:e2e:smoke`        | An explicit five-file allowlist: `viewer-initialization`, `url-parameters`, `dataset-switching`, `controls-interaction`, `keyboard-input-system` |
| `pnpm test:e2e:smoke:strict` | The smoke allowlist with strict browser-console handling                                                                                         |
| `pnpm test:e2e:browsers`     | Three fixture-backed specs under Chromium, Firefox and WebKit                                                                                    |
| `pnpm test:e2e:mobile`       | Only `src/tests/e2e/mobile/`, under `playwright.mobile.config.ts`                                                                                |
| `pnpm test:perf:e2e`         | Only `*perf-bench.spec.ts`, under `playwright.perf.config.ts` (which shares this global setup)                                                   |

The smoke subset is deliberately narrow and pulls no Git LFS. **Do not add a spec that
reads `tests/fixtures/` to it** — that would make the job depend on the Python
fixture generator, and would newly expose `test:e2e:smoke:strict` (which drops
the 4xx/5xx allow-list) to fixture-server 404s.

`mesh-rendering.spec.ts` is therefore in `test:e2e` / `test:e2e:ci` but **not**
in smoke. The GitHub `e2e-tests` job runs only the mobile suite; software WebGL
on standard runners remains too slow for the full rendering-heavy corpus, which
runs locally (`make test-e2e`) or on the GPU promotion runner.

### Generated zarr fixtures are a hard dependency

20 specs read `packages/luxar-viewer/tests/fixtures/*.zarr`, generated by
`tests/fixtures/generate_test_data.py`. `make test-e2e` and `pnpm test` (vitest)
regenerate them automatically; direct `pnpm test:e2e` does not hide the 1–2 minute
generator run inside global setup. Instead that setup fails fast when fixtures are
missing or stale, naming `pnpm test:generate-fixtures`. Both harnesses read the manifest
through
`tools/fixture-manifest.ts` so they cannot disagree about which fixtures exist.

**Except for the two suites that read no fixtures**, which set
`LUXAR_E2E_NO_FIXTURES=1` to skip the check:

- **smoke** — its five specs are chosen precisely so none of them touches
  `tests/fixtures/`;
- **perf** (`pnpm test:perf:e2e`) — a different config, but the same global
  setup, and none of its `*perf-bench.spec.ts` files reads `tests/fixtures/`.

Requiring fixtures in either would break a suite built not to need them. The
flag is set on the same `package.json` line as the file list (or the
`--config`), so the two move together rather than drifting apart. The same
variable is the escape hatch for a one-off local run of a spec you know does
not read fixtures.

## Folder Layout

```
e2e/
├── fixtures.ts          # Re-extended `test` fixture; auto console-error guard
├── helpers.ts           # ~40 Playwright helper utilities (wait/get/assert)
├── global-setup.ts      # Parallelism stamp; pre-flight: servers, datasets, fixtures; makes dirs
├── render-ticks.ts      # Confirmed render-tick flushing for detector specs
├── harnesses/
│   └── tsl-harness.ts   # TSL ↔ GLSL parity harness (loaded by tsl-harness.html)
├── mobile/              # touch-helpers.ts + five device-emulated specs
├── *.spec.ts            # Playwright specs (one per feature area)
└── *.spec.ts-snapshots/ # Visual-regression baselines (auto-managed)
```

A spec's filename hints at its scope: `basic-rendering`,
`dimension-animation`, `cache-system`, `tsl-shader-parity`, etc. See
the parent README's "Running E2E tests in chunks" section for a
suggested grouping.

`render-ticks.ts` is for the detector specs that need their draws to have
actually happened: `flushRenderTicks(page, n)` kicks the renderer and confirms
each tick against the renderer's own frame counter (bounded per tick _and_ in
aggregate), and `reportFlushVerdict(page, report)` decides — from a follow-up
probe rather than from the tick count — whether a flush that confirmed nothing
is worth failing the run over (#1651). It lives outside `helpers.ts` so it can
be unit-tested against a fake page; see
`src/tests/unit/tests/e2e-render-ticks.test.ts`.

## Shared Fixture (`fixtures.ts`)

`fixtures.ts` exports a `test` that re-extends `@playwright/test`'s
`test` so **a spec that imports it dismisses the timed control-rail hint before
navigation and auto-asserts no console errors after each test**. Specs that
import `test` from `@playwright/test` directly get neither normalization nor
fixture teardown. New specs must use:

```ts
import { test, expect } from './fixtures';
```

There is **one** gate, and it is the Playwright-side one: `page.on('console')`
where `msg.type() === 'error'`, plus `page.on('pageerror')`. That is the wider
of the two sources available — it catches errors fired **before** the viewer's
in-app debug interceptor installs (a mistyped asset, a pre-init
`ReferenceError`), uncaught exceptions and browser-generated errors the app
never routed through `console`, it accumulates into an unbounded array rather
than the interceptor's ring buffer (which evicts at `DEFAULT_MAX_BUFFER_SIZE`),
and it survives navigation, which resets that buffer. Nothing in the
interceptor's `errors` bucket is missing from it: every write into the buffer
goes through the private `captureMessage`, reachable only from the five
`patch()` closures, each of which re-emits through the original `console`
method.

The fixture therefore does **not** read the in-app buffer. It used to, via
`assertNoConsoleErrors` — a second, narrower opinion on the verdict just
rendered, bought with a `page.evaluate` round trip a saturated main thread can
withhold for minutes (#1651/#1746/#1747/#1760). A spec that wants that buffer
specifically — its `warnings` / `logs` buckets have no Playwright-side gate, and
a bare `assertNoConsoleErrors(page)` runs with no allow-list at all — still
calls the helper itself, and `hover-tooltip`, `hover-overlay`,
`mouse-interactions` and `recording-panel` deliberately do so from their own
`test.afterEach`.

The captured entries are filtered against `DEFAULT_ALLOWED_CONSOLE_ERRORS` by
the exported pure function `unexpectedConsoleErrors(captured, allowed)` — split
out of the teardown closure so the decision is unit-testable without a browser
(`src/tests/unit/tests/e2e-fixture-console-gate.test.ts`). The allow-list is
intentionally narrow:

| Pattern                                                                                               | Why allowed                                                                                                                                                                                                                                              |
| ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/WebGL context lost/`                                                                                | Headless Chromium occasionally drops the context under GPU pressure; the viewer recovers. Real context-loss bugs surface as black-canvas or frame-count divergences caught by the spec body.                                                             |
| `Failed to load resource: …status of 4xx/501` (default; off when `LUXAR_E2E_STRICT_CONSOLE=1` is set) | The zarr loader probes optional resources during normal scene loading (`.zattrs` / `.zgroup` / `zarr.json` detection chain, optional overlays, fallback PROPFIND for directory listing). Each miss is a benign 404/501 the loader's `try/catch` handles. |

Specs that **intentionally** trigger console errors (e.g.
`error-recovery.spec.ts`, `webgl-errors.spec.ts`) opt out per test:

```ts
import { test, ALLOW_CONSOLE_ERRORS } from './fixtures';

test('handles a bad URL gracefully', async ({ page }) => {
  test.info().annotations.push({
    type: ALLOW_CONSOLE_ERRORS,
    description: 'Bad-URL recovery surfaces a console.error by design.',
  });
  // ... test body ...
});
```

The annotation type is checked verbatim against `ALLOW_CONSOLE_ERRORS`
— typos turn into hard failures rather than silent opt-outs. The
fixture also skips the assertion when the test already failed or
timed out, so the original error stays prominent. Both opt-out paths, and the
gate's own throw, run inside a `try` whose `finally` detaches the two page
listeners: they must come off or they leak across tests sharing a browser
context, and they must come off _after_ the verdict so an error emitted once
the body has ended is still counted.

## Helpers (`helpers.ts`)

`helpers.ts` is the toolbox specs use to wait for asynchronous viewer
state, query the debug interface, and assert on rendering / console /
WebGL outcomes. It is the largest single file in the folder; the
exports group into the categories below.

### Lifecycle and readiness

| Helper                                 | Use when                                                                                                       |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `waitForLuxarReady`                    | Wait for `window.__luxarDebug.getState().initialized` (45 s default). The first thing nearly every spec calls. |
| `waitForDebugInterfaceReady`           | Lighter check that just asserts `window.__luxarDebug` is present.                                              |
| `waitForConsoleInterceptor`            | Wait for the viewer's debug-console interceptor to install before reading captured logs.                       |
| `waitForDataLoaded`                    | Wait for at least one geometry to have committed data.                                                         |
| `waitForPointsLoaded(page, minPoints)` | Wait until `getState().totalPoints >= minPoints`.                                                              |

### Render / animation pacing

| Helper                            | Use when                                                                                                                                                                                                                                      |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `renderOnce`                      | Trigger a single deterministic frame and wait one paint cycle — used before screenshots.                                                                                                                                                      |
| `waitForNextRender(page, frames)` | Best-effort yield for N animation frames. If the frame counter is unreadable or stalls, it falls back to a state wait plus a short buffer, warns once, and returns `false` — so a starved page may advance one or two frames rather than N.   |
| `waitForRenderStable`             | Best-effort wait for the frame counter to advance by `minFrames` (default 3) — the screenshot-pacing helper. Same fallback and one-line warning as `waitForNextRender`; it returns `void`, so read the warning to know a capture was unpaced. |
| `waitForAnimationStep`            | Wait for the animation manager to advance one logical step.                                                                                                                                                                                   |

### nD / dimension navigation

| Helper                                                           | Use when                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `waitForDimensionSystemReady`                                    | Wait for the scene dimensions manager to expose its API.                                                                                                                                                                                                                                                                                                                                   |
| `waitForDimensionSelected(page, index)`                          | Wait until a specific dimension is the active one.                                                                                                                                                                                                                                                                                                                                         |
| `waitForDimensionNavigation`                                     | Wait for a dimension-change to commit (debounce-aware).                                                                                                                                                                                                                                                                                                                                    |
| `waitForNavigationComplete` / `waitForNavigationCompleteOrThrow` | Wait for queued navigation events to drain; the `OrThrow` variant fails the test on timeout. The plain variant tolerates a condition that never settles — it returns after one warning, naming how many probes went out, how many yielded no usable state, and whether the `isLoading` flag was readable — but THROWS if every state probe FAILED, since it confirmed nothing (see #1726). |
| `waitForSpatialQuery` / `waitForSpatialQueryOrThrow`             | Wait for the spatial-index query that drives nD slicing. The `OrThrow` variant is a bare `page.waitForFunction` and fails the test on timeout, with no probe of its own to warn about. The plain variant has the same two-way give-up as `waitForNavigationComplete`.                                                                                                                      |

### Console and error assertions

| Helper                                                                 | Use when                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `captureConsoleMessages(page)`                                         | Attach a synchronous capture object that accumulates `errors` / `warnings` / `logs`.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `raceEvaluate(evaluation, timeout, onTimeout)`                         | Bound one already-started `page.evaluate` — it carries no timeout of its own, so otherwise only the whole test budget stops it (#1640, #1651). Pass a sentinel the in-page function can never return.                                                                                                                                                                                                                                                                                                                                      |
| `getConsoleMessages(page, timeout?)`                                   | Read the viewer's debug interceptor (formatted, captured in-app). Deadline-bounded (45 s): throws when the page never answers, rather than fabricating empty buckets. Called by specs directly and by the four assertions below — **not** by the shared fixture (#1760), so its cost is paid only where the in-app buffer was asked for.                                                                                                                                                                                                   |
| `assertConsoleContains(page, pattern)` / `assertConsoleDoesNotContain` | Positive / negative assertion on the captured stream.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `assertNoConsoleErrors(page, allow)`                                   | Strict no-error gate on the in-app buffer against an allow-list. NOT called by the [shared fixture](#shared-fixture-fixturests) (#1760) — specs opt in, four of them from their own `test.afterEach` with no allow-list at all.                                                                                                                                                                                                                                                                                                            |
| `assertNoShaderErrors(page)`                                           | Read the debug renderer for shader-compile / link failures specifically.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `getWebGLErrors(page)`                                                 | Drain accumulated WebGL errors from the renderer.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `waitForWebGLError(page, predicate)`                                   | Block until accumulated WebGL errors satisfy a predicate. Each read is bounded by the remaining budget, after the first one none is dispatched into a remainder shorter than one poll interval (a discarded read has already drained the GL queue), and a read that spent the whole remainder is not followed by one more poll past that deadline. It does not throw on give-up (it returns data) but warns once, saying whether any read answered and how many errors the union held; a rejecting read or a closed page still propagates. |

### Cache, UI, and input

| Helper                                                            | Use when                                                                       |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `waitForCacheStable`                                              | Wait for in-flight prefetch / write-through traffic to drain.                  |
| `waitForUIState(page, predicate)`                                 | Generic wait that polls a UI-state predicate.                                  |
| `dismissDatasetBrowser(page)`                                     | Close the first-time-UX picker so the spec can drive a known dataset.          |
| `focusCanvas(page)`                                               | Move keyboard focus onto the canvas (required before `1-9` / `[]` navigation). |
| `openLayersPanel(page)`                                           | Open the Layers UI from a known closed state.                                  |
| `ctrlScroll` / `shiftScroll`                                      | Synthetic modifier-scroll events (used by zoom / depth tests).                 |
| `getInputHandler` / `getAnimationManager` / `getSceneDimsManager` | Pull singleton managers off the debug interface for direct inspection.         |

### Scene introspection and pixel sampling

| Helper                                                      | Use when                                                                                                                                                                                                                                              |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `getLuxarState(page, timeout?)`                             | Read `__luxarDebug.getState()` with a clear error if the interface isn't ready. Deadline-bounded (45 s): throws a diagnostic naming #1651 / #1724 when the page never answers the probe, rather than returning a fallback the caller would assert on. |
| `getSceneObjectNames(page)`                                 | Flat list of every named object in the scene graph.                                                                                                                                                                                                   |
| `getLayerMaterialState(page, layer)`                        | Inspect uniforms / blending / depth state of a specific layer's material.                                                                                                                                                                             |
| `getPostProcessingState(page)`                              | Read the post-processing pipeline state (tone mapping mode, bloom, exposure).                                                                                                                                                                         |
| `probeWebGPUBackend(page)`                                  | Which backend physically runs behind `?renderer=webgpu` — skip gate for specs that must not run on the WebGL2 fallback.                                                                                                                               |
| `validateSceneAttributes(page)`                             | Audit every geometry's attribute buffers against the format spec.                                                                                                                                                                                     |
| `SampledPixel`, `ElementPixelStats`                         | Types returned by the pixel-sampling helpers.                                                                                                                                                                                                         |
| `captureElementScreenshot(page, selector, route, timeout?)` | Capture with explicit `framebuffer` or `composited` intent. Framebuffer pixels follow drawing-buffer DPR, so pin `?dpr=1` for stable counts and sample locations; `timeout` covers readiness only.                                                    |
| `samplePixelAt(...)` / `samplePixelsAt(..., route)`         | Decode one explicit-route capture and read one or many pixels from the same frame.                                                                                                                                                                    |
| `captureCanvasRGBA(page, selector, route)`                  | Decode one explicit-route capture into a full-frame RGBA buffer for whole-image / multi-region analysis on a single identical frame.                                                                                                                  |
| `getElementPixelStats(page, ..., route)`                    | Pixel-statistics rollup over one explicit-route capture.                                                                                                                                                                                              |

For a multi-sample scalar sweep, call `postProcessing.renderToImageData()`
in-page and reduce each `ImageData` there instead of serializing every frame.

### Camera placement

Writing `camera.position` from a spec does **not** move the camera: the active
controls own target / orientation / distance, and `runUpdateStep` step 8
re-applies them to the camera every frame. A placement only sticks if the
controls `reinitialize()` from it first — that one trap made three specs
silently inert (#1930), so use these instead of hand-rolling the sequence, and
**assert the returned placement** rather than discarding it.

| Helper                                                                  | Use when                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `placeCameraAt(page, position, { target?, up? })`                       | Move the camera and make it stick (pivot defaults to the active controls' target). Returns the POST-clamp position/distance plus `viaOrbitControls`, so a spec can PROVE the camera moved rather than assuming it did. `null` means the debug camera was missing.                                                        |
| `getCameraPivot(page)`                                                  | Read the active controls' pivot, to express a pose relative to the content centre before placing. REJECTS when there is no pivot (no orbit target, no `getFocusTarget()`) instead of answering with the world origin — that silent fallback is half of #1930.                                                            |
| `withOrbitDistanceLimits(page, limits, body)`                           | Rarely. Runs `body` with the orbit distance clamp widened, restoring the previous limits in a `finally`, and THROWS if the widening could not be applied. The window must span the placement AND the sampling — restoring early lets the next frame pull the camera back.                                                |
| `UNCLAMPED_ORBIT_DISTANCE_LIMITS`                                       | "Wherever I put it, leave it" limits for the above.                                                                                                                                                                                                                                                                      |
| `InPageCameraApi`, `Vec3Like`, `CameraPlacement`, `OrbitDistanceLimits` | Types. `InPageCameraApi` types `window.__luxarE2ECamera`, the in-page object the helpers install — a spec that places the camera many times inside ONE `page.evaluate` (the lod-group sweep) drives that object directly, so `placeCameraAt`, that sweep and `ortho-mode.spec.ts` all share one definition of the idiom. |

The distance clamp is **not** a second trap, despite how often it gets blamed
for one. After auto-framing it is `[D / ZOOM_IN_FACTOR, D * ZOOM_OUT_FACTOR]` =
`[D / 1000, D × 10000]` around the framing distance `D` (`camera-framing.ts`;
before framing, `deriveScaleLimits` applies the same 1/1000 … 10000 factors to
the scene diagonal). Measured on `test_lines`: `D = 13.799`, so
`minDistance = 0.0138` and `maxDistance = 137990`. Ordinary placements —
"right up against the geometry", "two decades back" — are nowhere near that, so
reach for `withOrbitDistanceLimits` only when the arithmetic says you are
outside it, and check that arithmetic before adding it.

A camera **sweep** has a real second trap on top of the `reinitialize()` one: it
can outrun data loading. An LOD registry requests a level's chunks only when that
level is first needed and keeps the currently resident level on screen until the
replacement has arrived, so a sweep that crosses a boundary faster than the fetch
completes skips that band without any error. Warm the residency up first — coarse
passes over the same range, repeated until every level has been observed visible
at least once — rather than one guessed sleep up front. The waiting still happens
(a ≤500 ms backoff between passes, to let in-flight fetches land), but it sits
INSIDE a loop whose exit is the condition, so a fast page pays one pass and a slow
one keeps going. Bound the warm-up on **wall clock alone**, so that an incomplete
warm-up has by construction spent its whole budget
(`lod-group.spec.ts`'s volumetric cross-fade sweep is the worked example). A pass
count is a tempting second bound and is the wrong one: residency is gated on
CHUNK FETCHES, so the only useful response to "not resident yet" is to keep
waiting, and stopping early with budget left gives up on the one resource that
helps. It also makes the exit condition a function of RENDER SPEED — that sweep
once failed hard on "out of passes with budget to spare", which worked out to
"a pass ran faster than ~1 s", i.e. a fast machine was more likely to be called
broken than a slow one.

Bounded warm-ups and in-page deadlines mean a run can end up with less evidence
than the assertion wants, so let the strength of the claim follow the evidence —
but **narrow the contract rather than dropping it**. That sweep makes its strict
every-boundary claim when the warm-up reached residency AND the sweep was not
truncated; when the warm-up came up short it still requires a genuine cross-fade
for every adjacent pair among the levels that DID become resident (a level that
never arrived cannot fade, but every pair not touching it still has to — on a
three-level ladder a missing middle level leaves no pair at all, and the run
says so), and only a
truncated sweep — which never visited part of the range, so no subset of pairs is
implied — falls all the way back to "at least one genuine cross-fade". Every
narrowing is stated in the failure text and in a `degraded` annotation, naming
the levels that were never displayed, so a weakened run is never a quiet one.
Size an in-page deadline from the budget you actually have, too
(that sweep's is 90 s inside a 150 s test), or the "exceptional" degraded path
becomes the normal CI outcome and the strict branch is never exercised.

Give any test with in-page deadlines its own `test.setTimeout` that
covers their sum with headroom — being killed by the suite-wide budget
mid-`page.evaluate` throws away the attachment and the annotation, which is the
least diagnosable way for a diagnostic test to fail.

`pnpm check:e2e-timeout-budgets` enforces that rule for deadlines above half the
default project's test timeout (30 s with today's 60 s budget). It reads
explicit `timeout` options, numeric arguments passed to wait helpers, local
wait-helper defaults, and module-level timeout/deadline constants used by each
test — except where such a constant IS the budget, since the argument to
`test.setTimeout`, `test.slow` or `describe.configure` is what the deadline has
to beat rather than a deadline of its own.
Imported helper defaults, deadlines inside local helper bodies, and
`beforeAll`/`afterAll` hooks remain out of scope; #2897 tracks that gap. So is
the sum — deadlines are modelled as the largest single wait, not their total,
so three sequential 40 s waits under a 45 s budget pass the check even though
the rule above asks for headroom over their sum. Summing across branches and
loops is not something a static pass can honestly claim to do, so the budget
still has to be sized by hand.
`test.slow()`, `test.setTimeout()`, or an enclosing
`test.describe.configure({ timeout })` supplies the budget. `test.slow()` is
modelled as Playwright implements it, which differs by scope. In a describe
body or at file level nothing executes: it is a static annotation, so the
declared timeout always wins and the tripling happens afterwards — order is
irrelevant and it reaches down through nested suites. In a **test body** it is
a live call against the resolved slot, so a preceding `setTimeout` is what gets
tripled and a following one replaces the result. Either way it lands **once per
test**, since a suite-level `slow` makes an in-test one a no-op, and a
_conditional_ `test.slow(cond)` declares nothing at all — the condition is a
run-time value, so the check falls back to demanding a real budget. If the
static
heuristic cannot model a case, add its file, source line, test title, and a specific reason
to `scripts/e2e-timeout-budget-exceptions.json`; stale exceptions fail the
check and must be removed when the test gains a budget or stops using the long
deadline.

### Pattern for a new helper

Helpers follow a few conventions worth matching:

- All async helpers take `page: Page` as the first argument.
- Wait helpers expose an explicit `timeout` parameter (default 45 s
  for top-level readiness and for the two bounded probes, `getLuxarState` and
  `getConsoleMessages`, 5–15 s otherwise) and throw with a
  message that names the expected condition.
- Helpers that read the debug interface go through `getLuxarState`
  rather than poking `window.__luxarDebug` directly — the wrapper
  surfaces a clean "Debug interface not ready" error when the page
  hasn't initialized.
- "Wait then throw" variants are spelled `XxxOrThrow` and exist
  alongside the polling versions so specs can choose between a hard
  gate on the condition and a wait that tolerates it never settling —
  note that the plain variants are not silent either: they throw when
  no probe ever ANSWERED, since they then confirmed nothing (#1726).

## Global Setup (`global-setup.ts`)

Runs once before any spec (wired in `playwright.config.ts`, and shared
by `playwright.perf.config.ts`). It first stamps the run's parallelism
ceiling on stderr — see
[Parallelism is sized to the machine](#parallelism-is-sized-to-the-machine) —
then runs five preflight checks:

1. **Checkout identity** — both the Vite server and the repository
   dataset server must return the deterministic marker for the current
   checkout. Playwright's readiness URLs use the same marker, so a
   server on ports 5173 or 9000 from a sibling clone/worktree is not
   silently reused. An unrelated catch-all server that returns 200 is
   rejected by checking the marker body here.
2. **Examples directory** — `datasets/examples/` in the identified
   checkout is verified to exist and, when present, its producer stamp is
   checked through `scripts/run_examples.py --check`. A stale stamp, an
   unavailable checker, or a missing directory warns and continues because
   31 of the 71 specs do not read example datasets. The other 40 still run
   their normal dataset-specific assertions against the existing stores. If
   one fails after a stale verdict, the shared page fixture repeats the
   `make run-examples` guidance beside the failure. That reminder covers 37
   of the 40 example-reading specs; three performance specs bypass the shared
   fixture, as do eight specs in the full corpus, and receive only the global
   warning. Example-independent specs (`basic-rendering`,
   `viewer-initialization`, `test-fixtures-rendering`, `geometry-types`) remain
   runnable while one example producer is stale or unavailable.
   This warning path is for package-level Playwright commands run directly,
   including `pnpm test:e2e`. The repository `make test-e2e`,
   `make test-e2e-browsers`, `make test-e2e-mobile`, `make test-e2e-smoke`,
   `make test-e2e-smoke-strict`, and `make test-perf-e2e` targets regenerate
   examples first. The CI mobile job also runs `make run-examples`, so CI
   coverage is not weakened by the warning behavior.

3. **Required datasets** — checks for the eight required `*.zarr`
   directories and then issues an HTTP `HEAD` request for each one
   found locally. A fixture that exists on disk but is not reachable
   from the server is therefore a preflight error rather than a later
   viewer timeout.
4. **Generated zarr fixtures** — every name in the manifest must exist
   under `tests/fixtures/` and be complete (`.zmetadata` present, which
   the compiler writes last — otherwise an interrupted generator's stump
   directory passes for a fixture), and the first of them must be
   reachable over HTTP (one probe: they all share a serving root, so
   they answer the same question). Unlike the examples check this **throws**, and
   it is skipped when `LUXAR_E2E_NO_FIXTURES=1` — see
   [Generated zarr fixtures are a hard dependency](#generated-zarr-fixtures-are-a-hard-dependency).
5. **Output directories** — creates `test-results/` and
   `test-results/debug/` if they don't exist.

Identity markers live in the gitignored
`packages/luxar-viewer/.luxar-e2e-identities/` directory. Their names
are hashes of the checkout's canonical project-root path; the marker
contains only that hash. Servers from the same checkout remain
reusable locally, while a foreign occupied port causes a prompt,
explicit startup failure. The file uses ESM (`import.meta.url`) to
reconstruct `__dirname` because the package is `"type": "module"`.

## TSL ↔ GLSL Parity Harness (`harnesses/tsl-harness.ts`)

Loaded by `packages/luxar-viewer/tsl-harness.html` (Vite serves it
at `/tsl-harness.html`). The harness renders every registered shader
through **both** the GLSL3 path (`THREE.WebGLRenderer` +
`THREE.ShaderMaterial`) and the TSL path
(`WebGPURenderer({ forceWebGL: true })` + `NodeMaterial` /
`MeshBasicNodeMaterial`), then exposes both readbacks plus the
generated GLSL strings to the Playwright spec for pixel-diffing.

Why a dedicated page rather than reusing the main viewer:

- Construction order is explicit and minimal — no app/state machine
  to wait on, no scene graph to mock around.
- Both backends live side by side; the test toggles between them per
  call rather than per page load.
- The TSL path drives `GLSLNodeBuilder` so the generated GLSL
  strings are recoverable for snapshot diff.

Not in scope: real WebGPU dispatch. That requires Chrome stable +
`?renderer=webgpu` and runs in a separate spec
(`webgpu-native-smoke.spec.ts`). This harness validates the
WebGL2-via-WebGPU-backend fallback parity, which is what
`forceWebGL: true` covers.

### Window API exposed by the harness

```ts
window.__tslHarness = {
  ready: Promise<void>,
  renderGLSL: (shaderName: string) => Uint8Array,
  renderTSL: (shaderName: string) => Promise<{
    pixels: Uint8Array;
    vertexShader: string;
    fragmentShader: string;
  }>,
  renderBloomChainGLSL: (fixture?: 'radial' | 'ramp-y') => Promise<{
    pixels: Uint8Array;
    mipCount: number;
  }>,
  renderBloomChainTSL: (fixture?: 'radial' | 'ramp-y') => Promise<{
    pixels: Uint8Array;
    mipCount: number;
  }>,
  listShaders: () => string[],
};
```

The bloom methods render the production multi-pass pyramid rather than a
shader-registry entry, including additive upsample accumulation onto existing mips.
The `fixture` argument picks the input pattern: `radial` (default) is a centred
Gaussian, and `ramp-y` a monotone vertical ramp. Both cases run — the radial one
is mirror-invariant and so cannot see a Y-orientation fault, which is how #2584's
per-tap flip stayed hidden. Either way the fixture is staged into a render target
first, because that is the only input for which the two backends agree on Y
(three's `TextureNode` normalises render-target sampling but not a raw
`DataTexture`), so the two readbacks compare directly with no row flipping.

`renderTSL` patches the renderer's internal
`NodeManager._createNodeBuilderState` once per call to capture the
generated vertex/fragment GLSL — there is no public Three.js API
for "give me the source", so the harness reaches into the backend's
pipeline cache during the build phase and restores the original
method immediately after.

### Registered shaders

Each entry in the (private) `SHADER_REGISTRY` provides:

- `source` — a `ShaderSource` whose `glsl3`, `webgpu`, and
  uniform-binding metadata is the same one the production renderer
  consumes.
- `buildUniforms()` — default uniforms for the parity test.
- `buildDefines?()` — optional GLSL3 `#define`s (used by shaders
  like `mega` that gate feature toggles).
- `buildTSLMaterial?(uniforms)` — TSL-side override when a
  shader-specific factory config is required (e.g.
  `megaWebGPUFactory(uniforms, { toneMappingMode: 1 })`).
- `buildMesh?(material)` — override for non-fullscreen-quad cases
  (`point`, `line`, `gsplat` need instanced quad geometry).
- `vertexColors?` — set on the GLSL3 material so Three emits the
  `in vec3 color` attribute declaration; the TSL side reads the
  same attribute via `attribute('color', 'vec3')` and doesn't need
  a parallel flag.

The registry currently covers FXAA, bloom threshold, the mega
post-processing pass, point / line / gsplat / mesh materials, the
three picking shaders, and the shared-math `erf` entry whose GLSL
and TSL twins have no production shader consumer of their own.
Adding a new shader to `SHADER_REGISTRY` is the only step needed
to make it diff-testable from the spec.

Helpers shared by point/line/gsplat entries:
`buildPointInstancedMesh`, `buildLineInstancedMesh`,
`buildGSplatInstancedMesh` — each builds a small instanced quad
mesh seeded with deterministic per-instance attributes so both
backends render the same scene.

The 8×8 `buildTestTexture()` produces a deterministic input
texture: an `x`/`y` gradient with one high-contrast pixel near the
centre so FXAA's edge-detection path actually triggers.

### Specs that use the harness

- `tsl-shader-parity.spec.ts` — pixel diff (GLSL vs TSL) for every
  entry in the registry.
- `tsl-codegen-snapshot.spec.ts` — snapshot the generated GLSL
  strings into `src/tests/__codegen__/` so unintended codegen
  regressions surface as text diffs.

## Visual-Regression Snapshots

Folders named `<spec>.spec.ts-snapshots/` hold per-spec PNG
baselines used by `expect(...).toHaveScreenshot(...)`. They are a
**local developer aid, not a CI contract**: GitHub CI runs only non-visual
Playwright subsets, and `test:e2e:ci` deliberately excludes every `@visual`
test. A green pull request therefore says nothing about whether these pixels
still match.

The checked-in corpus is Linux Chromium only. This is the one platform
the project can reproduce consistently; do not add Darwin or Windows
copies that no maintained runner refreshes. On Linux, run the visual
subset with `pnpm test:e2e:visual`. Update its baselines deliberately
with `pnpm test:e2e:visual:update`, then inspect every PNG diff before
committing it. The unit suite rejects non-Linux baseline filenames so
an unsupported platform corpus cannot silently return.

The 44 baselines added with this policy were recorded on Ubuntu 24.04.4
LTS with Playwright 1.62.1. Font configuration is not pinned, so a
font-driven local diff is expected on another distro; re-record it for
local inspection rather than blessing it into Git. The visual fixtures
come from `datasets/examples/`: when their producer stamp moves and the
examples are rebuilt, refresh and inspect the affected baselines too.

Keeping one reproducible Linux corpus leaves room for a future CI job
covering the DOM/CSS-only `@visual` specs without the WebGL rasterizer
variability that keeps the full desktop E2E corpus on the GPU promotion runner.

## Conventions for New Specs

1. **Import from `./fixtures`**, never `@playwright/test` directly.
2. Use the `?src=<dataset>&debug` URL form (not `?data=`) so
   `window.__luxarDebug` is exposed.
3. Wait for `waitForLuxarReady(page)` before reading state or dispatching keyboard/mouse input.
4. Prefer 3D datasets for general specs — 4D/nD slicing may show 0
   points at arbitrary slice positions. Specs that test nD behavior
   should drive to a slice known to contain geometry.
5. **Use the canonical `?src=` spelling without a trailing slash** — both
   forms are accepted, but one spelling keeps fixtures and logs consistent.
6. If a spec intentionally produces console errors, annotate it
   with `ALLOW_CONSOLE_ERRORS`; do not broaden
   `DEFAULT_ALLOWED_CONSOLE_ERRORS` unless the noise is genuinely
   environmental.
7. Reach for an existing helper before writing a new wait loop —
   the `*OrThrow` variants exist precisely so specs don't reinvent
   them.

## See Also

- Parent: [Luxar Viewer Test Suite](../README.md)
- Sibling unit tests: `../unit/`
- Sibling mocks: `../mocks/`
- Playwright config: `../../../playwright.config.ts`
- Playwright reference guide:
  [`docs/guides/developer/PLAYWRIGHT_GUIDE.md`](../../../../../docs/guides/developer/PLAYWRIGHT_GUIDE.md)
- E2E quick reference:
  [`docs/guides/user/E2E_TESTING_GUIDE.md`](../../../../../docs/guides/user/E2E_TESTING_GUIDE.md)
