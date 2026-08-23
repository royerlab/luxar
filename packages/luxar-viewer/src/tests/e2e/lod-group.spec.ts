/**
 * E2E tests for the lod_group scene-graph node.
 *
 * Covers:
 *   - Loading a multi-level lod_group fixture into the viewer (no console
 *     errors, no WebGL errors).
 *   - The layers panel renders an "Active level" dropdown for the
 *     lod_group, populated with ``auto`` + one ``lock to level <n>``
 *     option per child (label is 1-based; the option value stays 0-based
 *     to match the registry's lockLevel API).
 *   - Manual override: locking to a specific level via the dropdown
 *     swaps which child mesh is visible. ``auto`` mode resumes
 *     view-driven selection.
 *
 * Uses the ``test_lod_group.luxar.zarr`` fixture (3 levels, splat counts
 * 8 / 32 / 128, thresholds 0 / 50 / 200 px) — small enough to render
 * instantly, large enough that the registry's selector picks a
 * meaningful level on a default-zoom view.
 */

import { test, expect } from './fixtures';
import { EXPECTED_BLEND_STATE } from './blending-expected-state';
import {
  waitForLuxarReady,
  waitForNextRender,
  openLayersPanel,
  assertNoConsoleErrors,
  getWebGLErrors,
  focusCanvas,
  withOrbitDistanceLimits,
  UNCLAMPED_ORBIT_DISTANCE_LIMITS,
  type InPageCameraApi,
} from './helpers';

const FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_lod_group.luxar.zarr';

test.describe('lod_group node', () => {
  test.beforeEach(async ({ page }) => {
    // `&no-opfs` on every load: this spec never asserts the L2 OPFS tier, and
    // automated Chromium's OPFS stalls systemically (10s per op — issue #1645),
    // starving scene readiness past the test budget. The circuit breaker only
    // helps un-flagged real sessions (it still pays ~3 timeouts per fresh page).
    await page.goto(`/?src=${FIXTURE}&debug&no-opfs`);
    await waitForLuxarReady(page);
    // Three meshes (one per LOD child) should attach to the scene
    // graph; wait for them to settle.
    await page.waitForFunction(
      () => {
        const debug = (
          window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } }
        ).__luxarDebug;
        if (!debug?.scene) return false;
        let lodChildren = 0;
        (debug.scene as { traverse: (cb: (o: { name?: string }) => void) => void }).traverse(
          (o) => {
            if (o.name && o.name.startsWith('/multires/child_')) lodChildren++;
          }
        );
        return lodChildren >= 3;
      },
      { timeout: 15000 }
    );
  });

  test('loads without errors and exposes three lod children', async ({ page }) => {
    await assertNoConsoleErrors(page);
    expect(await getWebGLErrors(page)).toEqual([]);
  });

  test('auto mode shows the active level, plus at most its cross-fade partner', async ({
    page,
  }) => {
    // HISTORY: this test used to assert "exactly one visible child" and
    // FLAKED under parallel runs — a different assertion failing each run.
    // That premise predates the coverage CROSS-FADE (lod-blend.ts): in the
    // blend band the registry deliberately shows TWO ADJACENT levels with
    // blended opacities, and this fixture's default view sits in the
    // level-0/1 band, settling on {child_0, child_1} a few frames after
    // load. The old test only passed by sampling before the fade engaged;
    // one-shot reads under CPU load landed after it. The real invariant:
    // the visible set is {active} or {active, active+1} — never a
    // non-adjacent pair, never all three, never empty.
    await page.waitForFunction(
      () => {
        const debug = (
          window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } }
        ).__luxarDebug;
        if (!debug?.scene) return false;
        const vis: Record<string, boolean> = {};
        (
          debug.scene as {
            traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void;
          }
        ).traverse((o) => {
          const m = o.name?.match(/\/multires\/child_(\d+)$/);
          if (m) vis[m[1]] = !!o.visible;
        });
        const idxs = Object.entries(vis)
          .filter(([, v]) => v)
          .map(([k]) => Number(k))
          .sort((a, b) => a - b);
        // Settled: ≥1 visible, ≤2 visible, and if 2 they are adjacent.
        return (
          Object.keys(vis).length >= 3 &&
          idxs.length >= 1 &&
          idxs.length <= 2 &&
          (idxs.length === 1 || idxs[1] === idxs[0] + 1)
        );
      },
      { timeout: 10000 }
    );
    // Pin the settled shape: at most two adjacent levels are visible (a single
    // level, or a cross-fading pair). Which level that is depends on the framing
    // and the FILL_FACTOR anchor, so the assertions below stay level-agnostic.
    const visibleIdxs = await page.evaluate(() => {
      const debug = (window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } })
        .__luxarDebug;
      const vis: Record<string, boolean> = {};
      (
        debug!.scene as {
          traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void;
        }
      ).traverse((o) => {
        const m = o.name?.match(/\/multires\/child_(\d+)$/);
        if (m) vis[m[1]] = !!o.visible;
      });
      return Object.entries(vis)
        .filter(([, v]) => v)
        .map(([k]) => Number(k))
        .sort((a, b) => a - b);
    });
    expect(visibleIdxs.length).toBeGreaterThanOrEqual(1);
    expect(visibleIdxs.length).toBeLessThanOrEqual(2);
    if (visibleIdxs.length === 2) {
      expect(visibleIdxs[1]).toBe(visibleIdxs[0] + 1); // cross-fade partners are adjacent
    }
  });

  test('debug snapshot reports the lod_group with its active level', async ({ page }) => {
    // The monitor's LOD-awareness is fed from the same scene markers the
    // debug snapshot reads. Assert getState().lodGroups surfaces the
    // multires group, that exactly one level is active, and that the
    // reported activeLevel matches the visible child index.
    type LodGroupInfo = { name: string; levelCount: number; activeLevel: number };
    // HISTORY: this test used to demand a single visible child equal to
    // activeLevel and FLAKED under parallel runs — the coverage cross-fade
    // (lod-blend.ts) deliberately shows the active level's ADJACENT partner
    // in the blend band, so the real contract is: activeLevel is AMONG the
    // visible children and every visible child is activeLevel or its +1
    // partner. Poll until the snapshot and scene agree on that settled
    // shape before pinning details (one-shot reads catch mid-selection
    // states under CPU load).
    await page.waitForFunction(
      () => {
        const debug = (
          window as unknown as {
            __luxarDebug?: {
              getState?: () => {
                lodGroups: { name: string; levelCount: number; activeLevel: number }[];
              };
              scene?: { traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void };
            };
          }
        ).__luxarDebug;
        const g = debug?.getState?.()?.lodGroups?.find((x) => x.name === '/multires');
        if (!g) return false;
        const vis: Record<string, boolean> = {};
        debug?.scene?.traverse((o) => {
          const m = o.name?.match(/\/multires\/child_(\d+)$/);
          if (m) vis[m[1]] = !!o.visible;
        });
        const idxs = Object.entries(vis)
          .filter(([, v]) => v)
          .map(([k]) => Number(k))
          .sort((a, b) => a - b);
        return (
          idxs.length >= 1 &&
          idxs.length <= 2 &&
          idxs[0] === g.activeLevel &&
          (idxs.length === 1 || idxs[1] === g.activeLevel + 1)
        );
      },
      { timeout: 10000 }
    );
    const { lodGroups, visibleIdxs } = (await page.evaluate(() => {
      const debug = (
        window as unknown as {
          __luxarDebug?: {
            getState?: () => {
              lodGroups: { name: string; levelCount: number; activeLevel: number }[];
            };
            scene?: { traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void };
          };
        }
      ).__luxarDebug;
      const state = debug?.getState?.();
      const vis: Record<string, boolean> = {};
      debug?.scene?.traverse((o) => {
        const m = o.name?.match(/\/multires\/child_(\d+)$/);
        if (m) vis[m[1]] = !!o.visible;
      });
      return {
        lodGroups: state?.lodGroups ?? [],
        visibleIdxs: Object.entries(vis)
          .filter(([, v]) => v)
          .map(([k]) => Number(k))
          .sort((a, b) => a - b),
      };
    })) as { lodGroups: LodGroupInfo[]; visibleIdxs: number[] };

    const multires = lodGroups.find((g) => g.name === '/multires');
    expect(multires).toBeDefined();
    expect(multires!.levelCount).toBe(3);
    // The active level is the PRIMARY visible child; any second visible
    // child is its cross-fade partner (active + 1).
    expect(visibleIdxs[0]).toBe(multires!.activeLevel);
    expect(visibleIdxs.length).toBeLessThanOrEqual(2);
    if (visibleIdxs.length === 2) {
      expect(visibleIdxs[1]).toBe(multires!.activeLevel + 1);
    }
    expect(multires!.activeLevel).toBeGreaterThanOrEqual(0);
  });

  test('layers panel shows an "Active level" dropdown with auto + 3 lock options', async ({
    page,
  }) => {
    await openLayersPanel(page);

    // Find the lod_group row in the layer list — the layer name is the
    // last path segment, so look for "multires".
    const lodRow = page.locator('.luxar-layer-row__name', { hasText: 'multires' }).first();
    await expect(lodRow).toBeVisible({ timeout: 5000 });
    await lodRow.click();
    await waitForNextRender(page);

    // The Active-level dropdown lives in the shared controls section
    // and is shown only when the primary selected layer is an lod_group.
    // The dropdown's own label says "Active level".
    const lodSelect = page
      .locator('.luxar-layers-panel__control-group', { hasText: 'Active level' })
      .locator('select')
      .first();
    await expect(lodSelect).toBeVisible();

    const optionValues = await lodSelect
      .locator('option')
      .evaluateAll((opts) => opts.map((o) => (o as HTMLOptionElement).value));
    expect(optionValues).toEqual(['auto', '0', '1', '2']);
  });

  // Selects by option VALUE '2' (0-based child index 2 = finest), whose
  // 1-based label reads "lock to level 3". Asserting the value, not the
  // label text, keeps this robust to label wording.
  test('locking the finest level (value 2) makes child_2 the visible mesh', async ({ page }) => {
    await openLayersPanel(page);
    const lodRow = page.locator('.luxar-layer-row__name', { hasText: 'multires' }).first();
    await lodRow.click();
    await waitForNextRender(page);

    const lodSelect = page
      .locator('.luxar-layers-panel__control-group', { hasText: 'Active level' })
      .locator('select')
      .first();
    await lodSelect.selectOption('2');
    // child_2's geometry is loaded lazily on first activation, so the
    // swap takes a frame to fire ``ensureLoaded`` plus the async fetch.
    // Poll until child_2 is the sole visible mesh (auto-retries absorb
    // the deferred load).
    await page.waitForFunction(
      () => {
        const debug = (
          window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } }
        ).__luxarDebug;
        if (!debug?.scene) return false;
        let child2Visible = false;
        let otherVisible = false;
        (
          debug.scene as {
            traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void;
          }
        ).traverse((o) => {
          if (!o.name || !o.name.startsWith('/multires/child_')) return;
          if (o.name === '/multires/child_2') child2Visible = !!o.visible;
          else if (o.visible) otherVisible = true;
        });
        return child2Visible && !otherVisible;
      },
      { timeout: 5000 }
    );

    const visibility = await page.evaluate(() => {
      const debug = (window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } })
        .__luxarDebug;
      const out: Record<string, boolean> = {};
      if (!debug?.scene) return out;
      (
        debug.scene as { traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void }
      ).traverse((o) => {
        if (o.name && o.name.startsWith('/multires/child_')) {
          out[o.name] = !!o.visible;
        }
      });
      return out;
    });
    expect(visibility['/multires/child_2']).toBe(true);
    expect(visibility['/multires/child_0']).toBe(false);
    expect(visibility['/multires/child_1']).toBe(false);
  });

  test('monitor visible-splat count tracks the active LOD level (not the coarsest)', async ({
    page,
  }) => {
    // Regression for the "always shows the coarsest count" bug: a
    // substitutive-LOD swap happens per-frame (camera/lock) with no data
    // reload, so the monitor's visible tally must be refreshed on the
    // switch — otherwise it stays pinned to the default/coarsest level.
    // Reads the value the MONITOR actually displays (not a scene recompute)
    // so it guards the full wiring. Fixture levels: 8 / 32 / 128.
    const visibleSplats = page.locator('[data-field="visible-splats"]').first();

    // Expand the monitor (hidden → mini → expanded) so the Overview metric
    // cards render and the polling loop patches them.
    await focusCanvas(page);
    await page.keyboard.press('m');
    await page.keyboard.press('m');
    await expect(visibleSplats).toBeVisible({ timeout: 5000 });

    // Lock to the finest level (128) via the layers dropdown.
    await openLayersPanel(page);
    const lodRow = page.locator('.luxar-layer-row__name', { hasText: 'multires' }).first();
    await lodRow.click();
    await waitForNextRender(page);
    const lodSelect = page
      .locator('.luxar-layers-panel__control-group', { hasText: 'Active level' })
      .locator('select')
      .first();

    await lodSelect.selectOption('2');
    // toHaveText auto-retries, absorbing the per-frame swap + ~100ms poll.
    await expect(visibleSplats).toHaveText('128', { timeout: 5000 });

    // Lock to the coarsest level (8) — the monitor count must drop, proving
    // it tracks the active level rather than staying high or summing levels.
    await lodSelect.selectOption('0');
    await expect(visibleSplats).toHaveText('8', { timeout: 5000 });
  });
});

// ---------------------------------------------------------------------------
// Volumetric blendable lod_group — the same 3-level shape with
// blending_mode="volumetric" + absorption authored on the group node.
// Volumetric is in BLENDABLE_MODES (scene/lod-fade.ts): the coverage
// cross-fade and streaming energy compensation apply to it as they do to
// additive/luminous, because opacity linearly scales optical depth τ — the
// fade interpolates monotonically between the two levels' absorptions
// (exact at the endpoints; NOT invariant mid-fade in general — see
// VOLUMETRIC_BLENDING_SPEC.md §6 and the volumetric-math unit suite).
// ---------------------------------------------------------------------------

const VOLUMETRIC_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_lod_group_volumetric.luxar.zarr';

/**
 * Steps in the cross-fade distance sweep (inclusive, so 61 samples over the
 * 0.15× → ~15× span). Each step multiplies the distance by 10^(1/30) ≈ 1.08, so
 * the coverage metric moves ~8% per step — several samples inside a
 * `CROSSFADE_BAND_FRACTION = 0.4` band, which is ±40% of the local threshold gap.
 */
const SWEEP_STEPS = 60;

/**
 * Warm-up pass stride, in {@link SWEEP_STEPS} indices — 16 poses over the same
 * span, enough for every level's band to be entered at least once.
 */
const WARMUP_STRIDE = 4;

/**
 * Bounds on the residency warm-up: at most this many coarse passes, and at most
 * {@link WARMUP_BUDGET_MS} of wall clock, whichever comes first. Both are
 * fail-open — if residency never settles the measuring sweep runs anyway and the
 * per-boundary assertion reports what was actually observed, because a silently
 * skipped band is exactly what #1930 is about.
 */
const WARMUP_MAX_PASSES = 6;

/**
 * Wall-clock budget for the whole warm-up. Sized against the 60 s test timeout:
 * worst case is this plus the 20 s sweep deadline, and the observed cost is
 * ~3–5 s (the fixture's levels land within ~4 s of first being requested).
 */
const WARMUP_BUDGET_MS = 12000;

test.describe('lod_group node — volumetric blendable', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${VOLUMETRIC_FIXTURE}&debug&no-opfs`);
    await waitForLuxarReady(page);
    await page.waitForFunction(
      () => {
        const debug = (
          window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } }
        ).__luxarDebug;
        if (!debug?.scene) return false;
        let lodChildren = 0;
        (debug.scene as { traverse: (cb: (o: { name?: string }) => void) => void }).traverse(
          (o) => {
            if (o.name && o.name.startsWith('/multires/child_')) lodChildren++;
          }
        );
        return lodChildren >= 3;
      },
      { timeout: 15000 }
    );
  });

  test('camera driven into a boundary band cross-fades two adjacent volumetric levels, in the volumetric blend state', async ({
    page,
  }) => {
    // This test must FAIL against the pre-change hard swap, so it is not
    // enough to accept "one or two visible": a single visible child is
    // exactly what the old behavior produced. Sweep the camera distance to
    // cross the fixture's coverage boundaries and require an actual blend —
    // two ADJACENT levels, both with weights strictly inside (0, 1),
    // complementary to 1 (the registry's coverage-band contract).
    //
    // HISTORY (#1930): this sweep was INERT and reported "one level at opacity
    // 1" for a whole run. It read the pivot from `debug.controls.target`, which
    // does not exist on `ControlsManager` (→ silent fallback to the world
    // origin), and wrote `camera.position` without `reinitialize()`, so
    // `runUpdateStep` step 8 restored the settled framing on the next line.
    // All 61 iterations therefore sampled the SAME opening pose — outside both
    // bands of the fixture's 0 / 0.5 / 1.0 ladder, which is why a single level
    // at opacity 1 was the correct answer for the pose the test was in.
    // `withOrbitDistanceLimits` + `__luxarE2ECamera.place()` (helpers.ts) fix
    // both, and the distance-spread assertion below makes going inert again a
    // failure rather than a false pass.
    //
    // HISTORY, PART 2 (#1930): with the camera moving, the trace was still
    // DISCONTINUOUS — the 0↔1 band cross-faded but the 1↔2 one, which the
    // fixture's ladder puts squarely inside the swept range, never appeared.
    // Not a fade bug either: the registry requests a level's chunks only when
    // that level is first NEEDED, and holds the currently-resident level at
    // opacity 1 until the replacement has arrived. A sweep that crosses a
    // boundary faster than the fetch completes therefore skips the band
    // silently. Hence the residency warm-up below, and hence the assertion
    // requiring EVERY adjacent pair to have been seen cross-fading: "some band
    // exists somewhere" passes happily while a boundary is being skipped.
    const { sweep, states } = await withOrbitDistanceLimits(
      page,
      UNCLAMPED_ORBIT_DISTANCE_LIMITS,
      async () => {
        const sweep = await page.evaluate(
          async (cfg) => {
            const { steps, warmupStride, warmupMaxPasses, warmupBudgetMs } = cfg;
            type Mat = { uniforms?: { uOpacity?: { value: number } } };
            type Obj = { name?: string; visible?: boolean; material?: Mat };
            type Level = { idx: number; opacity: number };
            const w = window as Window &
              typeof globalThis & {
                __luxarE2ECamera?: InPageCameraApi;
                __luxarDebug?: {
                  scene?: { traverse: (cb: (o: Obj) => void) => void };
                  camera?: { position: { x: number; y: number; z: number } };
                  renderer?: { info?: { frame?: number; render?: { frame?: number } } };
                };
              };
            const debug = w.__luxarDebug;
            const camApi = w.__luxarE2ECamera;
            if (!debug?.scene || !debug.camera || !camApi) return null;
            const cam = debug.camera;
            const tgt = camApi.pivot();
            const d0 = {
              x: cam.position.x - tgt.x,
              y: cam.position.y - tgt.y,
              z: cam.position.z - tgt.z,
            };
            const sample = (): Level[] => {
              const out: Level[] = [];
              // The ambient __luxarDebug declaration types traverse's callback as
              // THREE.Object3D, so narrow to the mesh shape this test reads.
              debug.scene!.traverse((node) => {
                const o = node as unknown as Obj;
                const m = o.name?.match(/\/multires\/child_(\d+)$/);
                if (m && o.visible && o.material?.uniforms?.uOpacity) {
                  out.push({ idx: Number(m[1]), opacity: o.material.uniforms.uOpacity.value });
                }
              });
              return out.sort((a, b) => a.idx - b.idx);
            };
            // Every level the lod_group published, visible or not — the warm-up
            // and the per-boundary assertion are both derived from this rather
            // than from a hard-coded 3, so the test still reads correctly if the
            // fixture gains a level.
            const levelIndices = (): number[] => {
              const found: number[] = [];
              debug.scene!.traverse((node) => {
                const m = (node as unknown as Obj).name?.match(/\/multires\/child_(\d+)$/);
                if (m) {
                  const idx = Number(m[1]);
                  if (!found.includes(idx)) found.push(idx);
                }
              });
              return found.sort((a, b) => a - b);
            };
            // The LOD registry re-selects and re-writes the fade opacity during the
            // frame's update, so sample only after the renderer's frame counter has
            // actually advanced — a fixed rAF count can land ahead of a paced frame
            // (renderOnce() arms a cooldown of up to 250 ms). Budgeted, so a stalled
            // loop costs one budget instead of hanging the sweep.
            const frameNo = (): number | null => {
              const info = debug.renderer?.info;
              const f = info?.render?.frame ?? info?.frame;
              return typeof f === 'number' ? f : null;
            };
            const awaitFrames = async (n: number, budgetMs: number): Promise<void> => {
              const t0 = performance.now();
              const f0 = frameNo();
              let ticks = 0;
              // do/while: always yield at least one animation frame, even with a
              // spent budget — sampling in the same task as `place()` would read
              // the pre-update opacities every time.
              do {
                await new Promise((r) => requestAnimationFrame(() => r(null)));
                ticks++;
                if (f0 == null) {
                  if (ticks >= n) return; // no counter: fall back to rAF ticks
                } else {
                  const f = frameNo();
                  if (f != null && f >= f0 + n) return;
                }
              } while (performance.now() - t0 < budgetMs);
            };
            // Keep the render loop turning for `ms` without asserting anything —
            // the between-pass backoff that lets in-flight chunk fetches land.
            const pumpFor = async (ms: number): Promise<void> => {
              const until = performance.now() + ms;
              while (performance.now() < until) {
                await new Promise((r) => requestAnimationFrame(() => r(null)));
              }
            };
            // Geometric sweep across ~2 decades of distance: the coverage metric is
            // inversely proportional to distance, so this crosses every boundary of
            // the fixture's 0.0 / 0.5 / 1.0 ladder.
            const kAt = (i: number): number => 0.15 * Math.pow(10 ** (1 / 30), i); // 0.15 → ~15
            const placeAtK = (k: number) =>
              camApi.place({ x: tgt.x + d0.x * k, y: tgt.y + d0.y * k, z: tgt.z + d0.z * k }, tgt);

            // --- Residency warm-up ------------------------------------------
            // The registry fetches a level's chunks on FIRST NEED and keeps the
            // resident level at opacity 1 until the replacement has arrived, so
            // the measuring sweep would race the loader and skip whichever band
            // it reached first. Drive a coarse pass over the same range to make
            // every level be requested, and repeat until each one has actually
            // been seen VISIBLE — the in-page proxy for "this level is resident".
            // Bounded twice (pass count AND wall clock) and fail-open: if it
            // never settles, the measuring pass still runs and the trace plus the
            // per-boundary assertion say exactly what was missing.
            const allLevels = levelIndices();
            const residentSeen: number[] = [];
            const noteVisible = (levels: Level[]): void => {
              for (const l of levels) if (!residentSeen.includes(l.idx)) residentSeen.push(l.idx);
            };
            const allResident = (): boolean => allLevels.every((i) => residentSeen.includes(i));
            const warmupDeadline = performance.now() + warmupBudgetMs;
            const warmupStart = performance.now();
            let warmupPasses = 0;
            while (
              warmupPasses < warmupMaxPasses &&
              performance.now() < warmupDeadline &&
              !allResident()
            ) {
              warmupPasses++;
              for (let i = 0; i <= steps; i += warmupStride) {
                placeAtK(kAt(i));
                await awaitFrames(
                  2,
                  Math.min(250, Math.max(0, warmupDeadline - performance.now()))
                );
                noteVisible(sample());
                if (performance.now() >= warmupDeadline) break;
              }
              if (!allResident()) {
                await pumpFor(Math.min(500, Math.max(0, warmupDeadline - performance.now())));
              }
            }
            const warmup = {
              passes: warmupPasses,
              elapsedMs: performance.now() - warmupStart,
              resident: residentSeen.slice().sort((a, b) => a - b),
              complete: allResident(),
            };

            // --- Measuring sweep ---------------------------------------------
            const trace: { k: number; distance: number; levels: Level[] }[] = [];
            const blends: { distanceScale: number; levels: Level[]; sum: number }[] = [];
            // A healthy run spends ~2 frames (~33 ms) per step; the global deadline
            // only binds when the loop is stalling, and keeps a fully starved page
            // inside the 60 s test budget instead of 61 × 500 ms of waiting.
            const sweepDeadline = performance.now() + 20000;
            for (let i = 0; i <= steps; i++) {
              const k = kAt(i);
              const placed = placeAtK(k);
              await awaitFrames(2, Math.min(500, Math.max(0, sweepDeadline - performance.now())));
              const levels = sample();
              trace.push({ k, distance: placed ? placed.distance : NaN, levels });
              if (
                levels.length === 2 &&
                levels[1].idx === levels[0].idx + 1 &&
                levels.every((v) => v.opacity > 0.02 && v.opacity < 0.98)
              ) {
                blends.push({
                  distanceScale: k,
                  levels,
                  sum: levels[0].opacity + levels[1].opacity,
                });
              }
            }
            // Park at a pose where a blend was seen, so the blend-state read below
            // samples a cross-fading PAIR rather than whatever single level the far
            // end of the sweep happens to leave on screen.
            if (blends.length) {
              const kb = blends[0].distanceScale;
              placeAtK(kb);
              await awaitFrames(2, 500);
            }
            // The whole sweep always runs (the trace is the diagnostic); `blends`
            // holds EVERY crossing that met the cross-fade contract, so the caller
            // can require each adjacent pair rather than just one band.
            return { blends, trace, levels: allLevels, warmup };
          },
          {
            steps: SWEEP_STEPS,
            warmupStride: WARMUP_STRIDE,
            warmupMaxPasses: WARMUP_MAX_PASSES,
            warmupBudgetMs: WARMUP_BUDGET_MS,
          }
        );

        // Every VISIBLE child renders in the pinned volumetric blend state
        // (One/OneMinusSrcAlpha premultiplied emission–absorption, depthWrite
        // unconditionally false) with its authored κ — the fade drives opacity
        // only, never the mode. Read INSIDE the widened-limits window: restoring
        // the scene-derived clamp first would let the next frame pull the camera
        // off the blend pose (`runUpdateStep` step 6 re-clamps every frame).
        const states = await page.evaluate(() => {
          const debug = (
            window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } }
          ).__luxarDebug;
          const out: {
            idx: number;
            blending: number;
            blendEquation: number;
            blendSrc: number;
            blendDst: number;
            depthTest: boolean;
            depthWrite: boolean;
            transparent: boolean;
            mode: string | undefined;
          }[] = [];
          (
            debug!.scene as {
              traverse: (
                cb: (o: { name?: string; visible?: boolean; material?: unknown }) => void
              ) => void;
            }
          ).traverse((o) => {
            const m = o.name?.match(/\/multires\/child_(\d+)$/);
            if (!m || !o.visible || !o.material) return;
            const mat = o.material as {
              blending: number;
              blendEquation: number;
              blendSrc: number;
              blendDst: number;
              depthTest: boolean;
              depthWrite: boolean;
              transparent: boolean;
              userData?: { blendingMode?: string };
            };
            out.push({
              idx: Number(m[1]),
              blending: mat.blending,
              blendEquation: mat.blendEquation,
              blendSrc: mat.blendSrc,
              blendDst: mat.blendDst,
              depthTest: mat.depthTest,
              depthWrite: mat.depthWrite,
              transparent: mat.transparent,
              mode: mat.userData?.blendingMode,
            });
          });
          return out;
        });

        return { sweep, states };
      }
    );

    expect(
      sweep,
      'the in-page sweep could not run: no debug scene/camera, or the camera-placement helper was not installed'
    ).not.toBeNull();
    const { blends, trace, levels, warmup } = sweep!;

    // The full per-step record goes to the report as an attachment rather than
    // to stdout, so a passing run stays quiet but a future failure can tell
    // "the band exists but is narrow" from "the opacities are only ever 0/1".
    await test.info().attach('lod-volumetric-sweep.json', {
      body: JSON.stringify({ levels, warmup, trace }, null, 2),
      contentType: 'application/json',
    });
    const warmupText =
      `residency warm-up: ${warmup.passes} pass(es), ${warmup.elapsedMs.toFixed(0)} ms, ` +
      `levels seen resident [${warmup.resident.join(', ')}] of [${levels.join(', ')}]` +
      (warmup.complete ? '' : ' — INCOMPLETE, so a band may have been skipped by the loader');
    const traceText = trace
      .map(
        (s) =>
          `k=${s.k.toFixed(3)} d=${s.distance.toFixed(4)} → ` +
          (s.levels.length
            ? s.levels.map((l) => `${l.idx}:${l.opacity.toFixed(3)}`).join(' ')
            : '(nothing visible)')
      )
      .join('\n');
    // Every failure message below embeds this: the per-step trace is the
    // diagnostic, and the warm-up line says whether a missing band is a fade
    // bug or the loader never having made that level resident.
    const report = `${warmupText}\n${traceText}`;

    // THE inertness guard (#1930): the sweep is only evidence about cross-fading
    // if the camera actually moved. Distances are measured AFTER the controls
    // re-applied their state, so this also catches a distance clamp silently
    // pinning every step to the same pose.
    expect(
      trace.every((s) => Number.isFinite(s.distance) && s.distance > 0),
      `every sweep step must report a finite positive camera distance:\n${report}`
    ).toBe(true);
    const distances = trace.map((s) => s.distance);
    const spread = Math.max(...distances) / Math.min(...distances);
    expect(
      spread,
      'the sweep did not move the camera — requested a ~100x distance span but ' +
        `observed ${spread.toFixed(2)}x, so nothing was actually sampled across the ` +
        `LOD boundaries:\n${report}`
    ).toBeGreaterThan(10);

    // Every observed cross-fade obeys the coverage-band contract: an ADJACENT
    // pair, both weights strictly partial, complementary to 1.
    for (const b of blends) {
      expect(b.levels).toHaveLength(2);
      expect(b.levels[1].idx).toBe(b.levels[0].idx + 1);
      for (const lvl of b.levels) {
        expect(lvl.opacity).toBeGreaterThan(0.02);
        expect(lvl.opacity).toBeLessThan(0.98);
      }
      // Complementary weights — the coverage-band invariant (w + (1−w) = 1).
      expect(b.sum).toBeCloseTo(1, 5);
    }

    // EVERY boundary must have cross-faded, not just one of them. "At least one
    // band exists" is the assertion that let the loader-race above hide: the
    // 0↔1 band passed the test while the 1↔2 band was being skipped outright.
    // Pairs are derived from the levels the scene actually published, so adding
    // a level to the fixture tightens this automatically.
    expect(
      levels.length,
      `the fixture must publish at least two LOD levels:\n${report}`
    ).toBeGreaterThanOrEqual(2);
    const pairs = (idxs: number[]): string =>
      idxs.length ? idxs.map((i) => `${i}↔${i + 1}`).join(', ') : 'none';
    const faded = Array.from(new Set(blends.map((b) => b.levels[0].idx))).sort((a, b) => a - b);
    const missing = levels.slice(0, -1).filter((i) => !faded.includes(i));
    expect(
      missing,
      `no coverage-band cross-fade was observed for level pair(s) ${pairs(missing)} ` +
        `(observed pairs: ${pairs(faded)}). Each boundary must show two adjacent ` +
        `levels with opacities strictly inside (0.02, 0.98) somewhere in the sweep:\n${report}`
    ).toEqual([]);

    // The blend-state read above ran at the parked blend pose, so this is the
    // cross-fading pair's material state.
    expect(states.length).toBeGreaterThanOrEqual(1);
    const expected = EXPECTED_BLEND_STATE.volumetric;
    for (const s of states) {
      expect(s.mode).toBe('volumetric');
      expect(s.blending).toBe(expected.blending);
      expect(s.blendEquation).toBe(expected.blendEquation);
      expect(s.blendSrc).toBe(expected.blendSrc);
      expect(s.blendDst).toBe(expected.blendDst);
      expect(s.depthTest).toBe(expected.depthTest);
      expect(s.depthWrite).toBe(expected.depthWrite);
      expect(s.transparent).toBe(expected.transparent);
    }
    await assertNoConsoleErrors(page);
    expect(await getWebGLErrors(page)).toEqual([]);
  });
});
