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
    // `&noOpfs` on every load: this spec never asserts the L2 OPFS tier, and
    // automated Chromium's OPFS stalls systemically (10s per op — issue #1645),
    // starving scene readiness past the test budget. The circuit breaker only
    // helps un-flagged real sessions (it still pays ~3 timeouts per fresh page).
    await page.goto(`/?src=${FIXTURE}&debug&noOpfs`);
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
      null,
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
      null,
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
 * The lod_group THREE node's name — the node path (`load-lod-group-node.ts`
 * sets `group.name = node.path`), whose children are the `/multires/child_<i>`
 * meshes this spec probes. Its `children.length` counts levels the name probe
 * cannot see, which is what makes the contiguity check able to catch a missing
 * TAIL level as well as a gap.
 */
const LOD_GROUP_NAME = '/multires';

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
 * Wall-clock budget for the residency warm-up, and its ONLY bound.
 *
 * The warm-up keeps placing, pumping and re-sampling until either every
 * published level has been seen visible or this budget is gone, so an
 * INCOMPLETE warm-up has always spent its clock. That invariant is what makes
 * incompleteness interpretable: residency is gated on CHUNK FETCHES, and the
 * only thing a test can do about a fetch that has not landed yet is keep the
 * loop turning and wait, so stopping early with budget left would be giving up
 * on the one resource that actually helps.
 *
 * HISTORY: an earlier revision also capped the warm-up at a pass count and
 * treated "out of passes with budget to spare" as a HARD product failure. The
 * arithmetic made that a coin-flip on render speed rather than a signal — six
 * passes cost `6 × (one pass + 500 ms backoff)`, so the hard-failure branch
 * fired whenever a pass ran faster than ~1 s, i.e. under ~62 ms per pose. The
 * measured healthy reference run sat 5% from that boundary (16 poses ≈ 1049 ms)
 * and a 60 fps machine (~33 ms/pose) landed squarely inside it. Worse, the
 * reasoning was inverted: a FASTER renderer leaves more budget unspent and was
 * therefore MORE likely to be called a product bug, while the real bottleneck —
 * the Python dataset server shared by up to 4 workers — is not measured by the
 * render loop at all.
 *
 * Worst case is this plus {@link SWEEP_DEADLINE_MS}, which is why the test
 * raises its own timeout (`test.setTimeout`) rather than relying on the
 * suite-wide 60 s one. Observed cost is ~3–5 s (the fixture's levels land
 * within ~4 s of first being requested).
 */
const WARMUP_BUDGET_MS = 12000;

/**
 * Hard cap on the measuring sweep, in wall clock.
 *
 * Blowing it sets `truncated`, which downgrades the headline assertion — so it
 * must be genuinely exceptional rather than a routine CI outcome, and it is
 * sized from the budget that is actually available instead of from a round
 * number. `test.setTimeout` is 150 s; the `beforeEach` readiness wait can take
 * 15 s and the warm-up {@link WARMUP_BUDGET_MS} 12 s, and the park, the
 * blend-state read, the attachment and the assertions cost a few more seconds —
 * leaving ~115 s. 90 s uses most of that and still keeps ~25 s of margin, and a
 * healthy run spends ~2 s here, so the cap costs nothing when nothing is wrong.
 * (The previous 20 s was BELOW the plausible cost of 61 steps × 2 paced frames
 * on a single-worker software-rendering runner whose Python dataset server is
 * saturated — i.e. the strict branch risked never being exercised in CI while
 * the test stayed green.)
 */
const SWEEP_DEADLINE_MS = 90000;

test.describe('lod_group node — volumetric blendable', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${VOLUMETRIC_FIXTURE}&debug&noOpfs`);
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
      null,
      { timeout: 15000 }
    );
  });

  test('camera swept across every boundary cross-fades each adjacent pair of volumetric levels, in the volumetric blend state', async ({
    page,
  }) => {
    // The suite-wide 60 s budget (playwright.config.ts) does not cover this
    // test's WORST case, and being killed by it is the least diagnosable
    // outcome there is: the test dies mid-`page.evaluate` with a bare
    // "Test timeout … exceeded", losing the trace attachment, the warm-up line
    // and the annotation — in the one test whose entire purpose is
    // diagnosability. The worst case is the 15 s readiness wait in `beforeEach`
    // + the 12 s warm-up budget + the 90 s sweep deadline
    // ({@link SWEEP_DEADLINE_MS}) + the park and two more evaluates; 150 s
    // covers that with ~25 s of margin, and a healthy run still finishes in a
    // few seconds, so this costs nothing when nothing is wrong.
    test.setTimeout(150_000);

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
    // at opacity 1 was the correct answer for the pose the test was in. The
    // missing `reinitialize()` was the whole cause, and
    // `__luxarE2ECamera.place()` (helpers.ts) is the fix; the distance-spread
    // assertion below makes going inert again a failure rather than a false
    // pass. `withOrbitDistanceLimits` here is DEFENSIVE only: the clamp is
    // `[D/1000, D×10000]` around the framing distance D and this sweep spans
    // 0.15·D → 15·D, two-plus decades inside it, so it cannot fire as written —
    // it is kept because this is the one site whose range is a knob, and
    // widening `kAt` past D/1000 would otherwise pin the near end silently.
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
            const { steps, warmupStride, warmupBudgetMs, sweepDeadlineMs } = cfg;
            const lodGroupName = cfg.lodGroupName;
            type Mat = { uniforms?: { uOpacity?: { value: number } } };
            type Obj = {
              name?: string;
              visible?: boolean;
              material?: Mat;
              children?: unknown[];
            };
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
            // How many children the lod_group THREE node actually holds —
            // including any attached as an anonymous placeholder `THREE.Group`
            // (a nested kind=lod / kind=partition child, `load-lod-group-node.ts`),
            // which `levelIndices()` above cannot see because it matches on the
            // `/multires/child_<i>` name. `null` when the group node itself was
            // not found under that name.
            const publishedChildren = (): number | null => {
              let n: number | null = null;
              debug.scene!.traverse((node) => {
                const o = node as unknown as Obj;
                if (o.name === lodGroupName) n = o.children?.length ?? 0;
              });
              return n;
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
            const awaitFrames = async (n: number, budgetMs: number): Promise<boolean> => {
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
                  // Still yield the requested rAF ticks, but do not call the
                  // resulting sample fresh: without a renderer counter there is
                  // no evidence that the LOD registry's update actually ran.
                  if (ticks >= n) return false;
                } else {
                  const f = frameNo();
                  if (f != null && f >= f0 + n) return true;
                }
              } while (performance.now() - t0 < budgetMs);
              return false;
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
            // Bounded by WALL CLOCK ALONE: residency is gated on chunk fetches,
            // so the only useful response to "not resident yet" is to keep the
            // loop turning and re-sample. There is no pass cap, and therefore no
            // way to return with budget unspent — an incomplete warm-up has by
            // construction spent its whole clock, which is what lets the caller
            // read incompleteness as "the data did not arrive in 12 s" and
            // nothing else. (A pass cannot run away either: every pose awaits at
            // least one animation frame and the inner loop re-checks the
            // deadline, so the overrun past it is one pose.)
            const allLevels = levelIndices();
            const residentSeen: number[] = [];
            const noteVisible = (levels: Level[]): void => {
              for (const l of levels) if (!residentSeen.includes(l.idx)) residentSeen.push(l.idx);
            };
            const allResident = (): boolean => allLevels.every((i) => residentSeen.includes(i));
            const warmupDeadline = performance.now() + warmupBudgetMs;
            const warmupStart = performance.now();
            let warmupPasses = 0;
            while (performance.now() < warmupDeadline && !allResident()) {
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
            // `passes` and `budgetMs` are diagnostics, not a contract: the only
            // exit that matters is `complete`, and an incomplete warm-up always
            // spent `budgetMs`. Pass count still rides along because "how many
            // times did we cross every band before the level showed up" is the
            // first thing you want to know from a failure.
            const warmup = {
              passes: warmupPasses,
              elapsedMs: performance.now() - warmupStart,
              budgetMs: warmupBudgetMs,
              resident: residentSeen.slice().sort((a, b) => a - b),
              complete: allResident(),
            };

            // --- Measuring sweep ---------------------------------------------
            const trace: { k: number; distance: number; levels: Level[]; fresh: boolean }[] = [];
            const blends: {
              distanceScale: number;
              distance: number;
              levels: Level[];
              sum: number;
              fresh: boolean;
            }[] = [];
            // Every placement must be re-derived by ORBIT controls, or the
            // reported distances are just the request echoed back and prove
            // nothing about the camera having moved (`CameraPlacement`'s own
            // JSDoc: assert this whenever that is the point).
            let viaOrbitControls = true;
            // A healthy run spends ~2 frames (~33 ms) per step; the global deadline
            // only binds when the loop is stalling, and keeps a fully starved page
            // inside the test budget instead of 61 × 500 ms of waiting. It is a
            // hard CAP, not just a per-step budget: once it is spent the loop
            // STOPS (`truncated`) rather than paying one more rAF per remaining
            // step for samples nobody will trust anyway — a spent budget used to
            // still cost ~61 forced frames, which is how a slow page turned a
            // diagnostic into a bare "Test timeout … exceeded" with no attachment.
            const sweepDeadline = performance.now() + sweepDeadlineMs;
            let truncatedAt: number | null = null;
            for (let i = 0; i <= steps; i++) {
              if (performance.now() >= sweepDeadline) {
                truncatedAt = i;
                break;
              }
              const k = kAt(i);
              const placed = placeAtK(k);
              if (!placed || !placed.viaOrbitControls) viaOrbitControls = false;
              const fresh = await awaitFrames(
                2,
                Math.min(500, Math.max(0, sweepDeadline - performance.now()))
              );
              const levels = sample();
              const distance = placed ? placed.distance : NaN;
              trace.push({ k, distance, levels, fresh });
              if (
                levels.length === 2 &&
                levels[1].idx === levels[0].idx + 1 &&
                levels.every((v) => v.opacity > 0.02 && v.opacity < 0.98)
              ) {
                blends.push({
                  distanceScale: k,
                  distance,
                  levels,
                  sum: levels[0].opacity + levels[1].opacity,
                  fresh,
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
            // The trace is the diagnostic; `blends` holds EVERY crossing that met
            // the cross-fade contract, so the caller can require each adjacent
            // pair rather than just one band. `truncated` and `warmup` say how
            // much of that the run earned: the full-range/all-resident case
            // claims every boundary, an incomplete warm-up claims every pair
            // whose BOTH levels appear in `warmup.resident`, and a truncated
            // sweep claims one band. Nothing here is silently absorbed — the
            // caller reports each narrowing.
            return {
              blends,
              trace,
              levels: allLevels,
              publishedChildren: publishedChildren(),
              viaOrbitControls,
              warmup,
              truncated: truncatedAt !== null,
              stepsRun: truncatedAt ?? steps + 1,
              stepsPlanned: steps + 1,
            };
          },
          {
            steps: SWEEP_STEPS,
            warmupStride: WARMUP_STRIDE,
            warmupBudgetMs: WARMUP_BUDGET_MS,
            sweepDeadlineMs: SWEEP_DEADLINE_MS,
            lodGroupName: LOD_GROUP_NAME,
          }
        );

        // Every VISIBLE child renders in the pinned volumetric blend state
        // (One/OneMinusSrcAlpha premultiplied emission–absorption, depthWrite
        // unconditionally false) with its authored κ — the fade drives opacity
        // only, never the mode. Read INSIDE the widened-limits window for the
        // same reason the wrapper is here at all — defensively, and so the whole
        // sweep-then-sample sequence runs under ONE clamp regime. As written the
        // real clamp cannot fire (it is `[D/1000, D×10000]` and the park pose is
        // decades inside it), so restoring it first would not move the camera;
        // but the sweep's range is a knob, and a widened `kAt` would make the
        // placement and the sampling disagree if only one of them were covered.
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
    const {
      blends,
      trace,
      levels,
      publishedChildren,
      viaOrbitControls,
      warmup,
      truncated,
      stepsRun,
      stepsPlanned,
    } = sweep!;

    // The full per-step record goes to the report as an attachment rather than
    // to stdout, so a passing run stays quiet but a future failure can tell
    // "the band exists but is narrow" from "the opacities are only ever 0/1".
    await test.info().attach('lod-volumetric-sweep.json', {
      body: JSON.stringify(
        {
          levels,
          publishedChildren,
          viaOrbitControls,
          warmup,
          truncated,
          stepsRun,
          stepsPlanned,
          trace,
        },
        null,
        2
      ),
      contentType: 'application/json',
    });
    // Levels the warm-up never once saw on screen. Named everywhere the run is
    // degraded — in the warm-up line, hence in the annotation and in every
    // failure message — because they are exactly the levels whose boundaries the
    // contract below stops claiming anything about.
    const neverDisplayed = levels.filter((l) => !warmup.resident.includes(l));
    const warmupText =
      `residency warm-up: ${warmup.passes} pass(es), ` +
      `${warmup.elapsedMs.toFixed(0)} ms of ${warmup.budgetMs} ms, ` +
      `levels seen resident [${warmup.resident.join(', ')}] of [${levels.join(', ')}]` +
      (warmup.complete
        ? ''
        : ` — INCOMPLETE (whole ${warmup.budgetMs} ms budget spent): level(s) ` +
          `[${neverDisplayed.join(', ')}] NEVER DISPLAYED at any pose, so every boundary ` +
          'involving them was unobservable in this run');
    const sweepText =
      `sweep: ${stepsRun}/${stepsPlanned} steps` +
      (truncated
        ? ` — TRUNCATED on the ${(SWEEP_DEADLINE_MS / 1000).toFixed(0)} s in-page deadline, ` +
          'range not fully visited'
        : '');
    const traceText = trace
      .map(
        (s) =>
          `k=${s.k.toFixed(3)} d=${s.distance.toFixed(4)} ${s.fresh ? 'fresh' : 'stale'} → ` +
          (s.levels.length
            ? s.levels.map((l) => `${l.idx}:${l.opacity.toFixed(3)}`).join(' ')
            : '(nothing visible)')
      )
      .join('\n');
    // Every failure message below embeds this: the per-step trace is the
    // diagnostic, and the warm-up / sweep lines say whether a missing band is a
    // fade bug, a level the loader never made resident, or a range the sweep
    // never got to.
    const report = `${warmupText}\n${sweepText}\n${traceText}`;

    // Whether this run is allowed to make the STRICT per-boundary claim: every
    // published level resident AND the whole distance range visited. Both
    // degradations are wall-clock stories on a loaded runner rather than product
    // bugs — the warm-up spends its entire budget before reporting
    // `complete: false` (see WARMUP_BUDGET_MS), and truncation means part of the
    // range was never sampled — so neither is failed on directly. What replaces
    // the strict claim is NOT "anything goes": see the per-boundary block below,
    // which narrows the contract to the levels this run actually observed
    // instead of dropping it.
    const conclusive = warmup.complete && !truncated;
    if (!conclusive) {
      test
        .info()
        .annotations.push({ type: 'degraded', description: `${warmupText}; ${sweepText}` });
    }

    // THE inertness guard (#1930): the sweep is only evidence about cross-fading
    // if the camera actually moved. Distances are measured AFTER the controls
    // re-applied their state, so this also catches a distance clamp silently
    // pinning every step to the same pose. `viaOrbitControls` is the other half:
    // without an orbit controller re-deriving each write, those distances are
    // just the request echoed back and prove nothing (`CameraPlacement`'s JSDoc).
    expect(
      viaOrbitControls,
      'every placement in the sweep must have been re-derived by orbit controls; without that ' +
        "the reported distances are the requested ones echoed back, not the controls' own " +
        `post-clamp answer:\n${report}`
    ).toBe(true);
    expect(
      trace.every((s) => Number.isFinite(s.distance) && s.distance > 0),
      `every sweep step must report a finite positive camera distance:\n${report}`
    ).toBe(true);
    const distances = trace.map((s) => s.distance);
    const spread = Math.max(...distances) / Math.min(...distances);
    // Measured against the span the sweep actually REQUESTED, so a truncated run
    // is held to the range it visited rather than to the full 100x (on a
    // complete run this is exactly the historical `> 10`, i.e. √100).
    const requestedSpan = trace.length > 1 ? trace[trace.length - 1].k / trace[0].k : 1;
    expect(
      spread,
      `the sweep did not move the camera — requested a ${requestedSpan.toFixed(1)}x distance ` +
        `span but observed ${spread.toFixed(2)}x, so nothing was actually sampled across the ` +
        `LOD boundaries:\n${report}`
    ).toBeGreaterThan(Math.sqrt(requestedSpan));

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

    // ...and each band is a DISSOLVE, in the right direction. Everything above
    // is satisfied by a static 50/50 split and by an INVERTED mapping (the
    // coarse level wearing the fine level's weight), which are the two mutants
    // that survive it: neither changes adjacency, strict partiality or the sum.
    // The registry's contract is that the finer level fades OUT as the camera
    // retreats, so within a band the higher-index child's opacity must be
    // non-increasing in distance (lod children are authored coarse → fine, which
    // this fixture's 0 / 0.5 / 1.0 ladder follows: child_2 is the finest) and
    // must actually MOVE rather than sit on a constant.
    const FADE_MONOTONIC_EPS = 0.02; // float noise on a weight in [0, 1]
    const FADE_MIN_SPREAD = 0.1; // a real dissolve crosses far more than this
    const FADE_MIN_SAMPLES = 3; // fewer confirmed-fresh samples are not evidence
    // Consecutive blend samples sharing the same adjacent pair = one traversal
    // of one band (the sweep visits distances in increasing order, so these are
    // already ordered by distance).
    type Blend = (typeof blends)[number];
    const bands: Blend[][] = [];
    for (const b of blends) {
      const current = bands[bands.length - 1];
      const previous = current?.[current.length - 1];
      if (current && previous && previous.levels[0].idx === b.levels[0].idx) current.push(b);
      else bands.push([b]);
    }
    for (const band of bands) {
      const [lo, hi] = [band[0].levels[0].idx, band[0].levels[1].idx];
      const bandText = band
        .map(
          (s) =>
            `  d=${s.distance.toFixed(4)} ${s.fresh ? 'fresh' : 'stale'} ` +
            s.levels.map((l) => `${l.idx}:${l.opacity.toFixed(3)}`).join(' ')
        )
        .join('\n');
      const fine = band.map((s) => s.levels[1].opacity);
      for (let i = 1; i < fine.length; i++) {
        expect(
          fine[i],
          `in the ${lo}↔${hi} band the finer level (${hi}) must fade OUT as the camera retreats, ` +
            `but its opacity rose from ${fine[i - 1].toFixed(3)} to ${fine[i].toFixed(3)} between ` +
            `d=${band[i - 1].distance.toFixed(4)} and d=${band[i].distance.toFixed(4)} — an ` +
            `inverted dissolve hands each level its partner's weight:\n${bandText}\n\n${report}`
        ).toBeLessThanOrEqual(fine[i - 1] + FADE_MONOTONIC_EPS);
      }
      const freshBand = band.filter((sample) => sample.fresh);
      if (freshBand.length >= FADE_MIN_SAMPLES) {
        const freshFine = freshBand.map((sample) => sample.levels[1].opacity);
        const fineSpread = Math.max(...freshFine) - Math.min(...freshFine);
        expect(
          fineSpread,
          `the ${lo}↔${hi} band had ${freshBand.length} confirmed-fresh samples but level ${hi}'s ` +
            'opacity barely ' +
            `moved (spread ${fineSpread.toFixed(3)}); the blend weight must VARY with distance — ` +
            'a constant split is a pop at each band edge, which is what the cross-fade ' +
            `exists to prevent:\n${bandText}\n\n${report}`
        ).toBeGreaterThan(FADE_MIN_SPREAD);
      } else {
        test.info().annotations.push({
          type: 'degraded',
          description:
            `${lo}↔${hi} spread check skipped: ${freshBand.length}/${band.length} samples were ` +
            `confirmed fresh; need ${FADE_MIN_SAMPLES}`,
        });
      }
    }

    expect(
      levels.length,
      `the fixture must publish at least two LOD levels:\n${report}`
    ).toBeGreaterThanOrEqual(2);
    // The published indices must be CONTIGUOUS from 0. `levelIndices()` only
    // sees nodes named `/multires/child_<i>` (and `sample()` additionally needs
    // a `uOpacity` uniform), so a level whose child is a nested `kind=lod` /
    // `kind=partition` group — attached as an anonymous `THREE.Group` until it
    // is first activated (`load-lod-group-node.ts`) — is invisible to both. That
    // must fail loudly rather than quietly shrinking the pair set to the levels
    // that happen to be plain meshes.
    expect(
      levels,
      'the published lod children must be contiguous from 0; a gap means a level was ' +
        'not visible to this probe (a nested lod/partition child is an anonymous Group ' +
        `until first activation), which would silently weaken the per-boundary check:\n${report}`
    ).toEqual(levels.map((_, i) => i));
    // Contiguity alone cannot catch a missing TAIL — `[0, 1]` is contiguous, and
    // the finest level is the natural place for a nested lod/partition child, so
    // exactly the case the message above describes would slip through. Compare
    // against the lod group's own `children.length`, which counts the anonymous
    // placeholder Groups the name probe cannot see. `null` = the group node was
    // not found by name, which the level names already contradict, so fail too.
    expect(
      publishedChildren,
      `the probe saw ${levels.length} named lod level(s) but ${LOD_GROUP_NAME} holds ` +
        `${publishedChildren ?? 'an unknown number of'} children: a level invisible to the ` +
        `\`${LOD_GROUP_NAME}/child_<i>\` name match (a nested lod/partition child is an ` +
        'anonymous Group until first activation) would silently drop the last boundary pair ' +
        `from the per-boundary check:\n${report}`
    ).toBe(levels.length);
    // Pairs come from CONSECUTIVE OBSERVED entries rather than from `i, i+1`, so
    // the contract follows what the scene actually published.
    const boundaries: [number, number][] = levels
      .slice(0, -1)
      .map((lo, j) => [lo, levels[j + 1]] as [number, number]);
    const pairs = (ps: [number, number][]): string =>
      ps.length ? ps.map(([lo, hi]) => `${lo}↔${hi}`).join(', ') : 'none';
    const faded = Array.from(new Set(blends.map((b) => b.levels[0].idx))).sort((a, b) => a - b);
    const fadedPairs = faded.map((lo) => [lo, lo + 1] as [number, number]);
    const missing = boundaries.filter(([lo]) => !faded.includes(lo));
    // Boundaries BOTH of whose levels the warm-up actually saw on screen — the
    // pairs a run with an incomplete warm-up still had the data to observe. A
    // pair is dropped from the contract only because one of its levels never
    // became resident, and that level is named in `warmupText`.
    const observablePairs = boundaries.filter(
      ([lo, hi]) => warmup.resident.includes(lo) && warmup.resident.includes(hi)
    );
    const missingObservable = observablePairs.filter(([lo]) => !faded.includes(lo));

    if (conclusive) {
      // THE strict claim, and only on the evidence that supports it: with every
      // level resident and the whole range visited, EVERY boundary must have
      // cross-faded. "At least one band exists" is the assertion that let the
      // loader race hide — the 0↔1 band passed while 1↔2 was skipped outright —
      // so this stays as strong as the run allows.
      expect(
        missing,
        `no coverage-band cross-fade was observed for level pair(s) ${pairs(missing)} ` +
          `(observed pairs: ${pairs(fadedPairs)}). Each boundary must show two adjacent ` +
          `levels with opacities strictly inside (0.02, 0.98) somewhere in the sweep:\n${report}`
      ).toEqual([]);
    } else if (truncated) {
      // The sweep stopped before the far end of the distance range, so a
      // boundary can be missing simply because its band lives in the part of the
      // range that was never visited — nothing about residency narrows that down
      // to a subset of pairs. This is the one case that falls all the way back to
      // "at least one genuine cross-fade", with every per-blend invariant above
      // still enforced, and it is sized to be exceptional: SEE SWEEP_DEADLINE_MS,
      // which is 90 s of a 150 s test against a ~2 s healthy cost.
      expect(
        blends.length,
        'no coverage-band cross-fade was observed at all, and the sweep was TRUNCATED, so part ' +
          'of the distance range was never visited and a missing band cannot be attributed to ' +
          `the fade: ${warmupText}; ${sweepText}. Only "at least one genuine cross-fade" is ` +
          `required here. Boundaries without a band: ${pairs(missing)} of ${pairs(boundaries)}:` +
          `\n${report}`
      ).toBeGreaterThanOrEqual(1);
    } else if (observablePairs.length) {
      // Warm-up incomplete but the whole range WAS swept. Scale the contract to
      // the evidence instead of abandoning it: require a genuine cross-fade for
      // every adjacent pair among the levels that DID become resident. This is
      // nearly as strong as the strict claim and cannot go red on a healthy
      // product, because a level that never arrived can never fade.
      //
      // Why it is well founded rather than a convenient weakening: a level that
      // never displays takes its own boundaries out of `observablePairs`, but
      // every OTHER adjacent pair stays in the contract and still has to fade.
      // A never-displayed END level costs one pair; a MIDDLE one costs two, so
      // on this fixture's three-level ladder a missing level 1 empties
      // `observablePairs` altogether and the run drops to the floor branch
      // below — a real contract collapse, not a narrowing. From four levels up
      // the untouched pairs keep carrying the claim. Either way the run has
      // spent its full warm-up clock, and every narrowing is reported
      // (annotation + `warmupText`, naming the levels never seen), never silent.
      expect(
        missingObservable,
        `no coverage-band cross-fade was observed for level pair(s) ${pairs(missingObservable)} ` +
          `(observed pairs: ${pairs(fadedPairs)}). The residency warm-up was INCOMPLETE, so ` +
          'this run only claims the boundaries whose BOTH levels were seen on screen ' +
          `(${pairs(observablePairs)} of ${pairs(boundaries)}) — but the whole distance range ` +
          'WAS swept, so each of those must show two adjacent levels with opacities strictly ' +
          `inside (0.02, 0.98) somewhere in it:\n${report}`
      ).toEqual([]);
    } else {
      // Fewer than two ADJACENT levels ever reached the screen in the full
      // warm-up budget, so there is no pair left to make a claim about: a
      // cross-fade needs two resident neighbours by definition, and requiring one
      // here would be a guaranteed red on a starved runner rather than a signal.
      // The run is annotated `degraded` and `warmupText` names the levels that
      // never displayed. The one thing still worth asserting is that the lod
      // group rendered at all.
      expect(
        warmup.resident.length,
        'not a single lod level was ever displayed during the whole residency warm-up budget, ' +
          `so nothing rendered at any pose in the sweep: ${warmupText}; ${sweepText}:\n${report}`
      ).toBeGreaterThanOrEqual(1);
    }

    // The blend-state read above ran at the parked blend pose. When a blend was
    // observed, that pose IS a cross-fade, so exactly TWO children must be
    // visible there — `>= 1` would pass on a single level and let the claim
    // "this samples the cross-fading pair" be silently false.
    if (blends.length) {
      expect(
        states.length,
        `parked at the pose of the first observed blend (k=${blends[0].distanceScale.toFixed(3)}, ` +
          `levels ${blends[0].levels.map((l) => l.idx).join('+')}), so exactly two children must ` +
          `be visible; saw ${states.map((s) => s.idx).join(', ') || 'none'}:\n${report}`
      ).toBe(2);
    } else {
      expect(states.length, `no visible lod child at the final pose:\n${report}`).toBeGreaterThan(
        0
      );
    }
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
