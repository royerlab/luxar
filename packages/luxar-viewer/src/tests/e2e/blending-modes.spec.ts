/**
 * Blending Modes E2E Tests
 *
 * Validates that per-layer blending modes are correctly applied to Three.js materials:
 * - Initial blending modes match zarr metadata
 * - Changing blending mode updates material blending, depthTest, depthWrite
 * - Different modes produce visually different results
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  waitForNextRender,
  getWebGLErrors,
  assertNoConsoleErrors,
  captureCanvasRGBA,
  samplePixelsAt,
  openLayersPanel,
} from './helpers';
import { EXPECTED_BLEND_STATE as EXPECTED_STATE } from './blending-expected-state';

const DATASET = 'http://localhost:9000/datasets/examples/rendering_modes_example.luxar.zarr';
const MULTI_DATASET = 'http://localhost:9000/datasets/examples/multiple_objects_example.luxar.zarr';
const POINTS_BLENDING_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_points_blending_modes.luxar.zarr';
const BLENDING_INHERITED_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_blending_inherited.luxar.zarr';
const GSPLAT_OVERLAP_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_normal_overlap.luxar.zarr';
const GSPLAT_OVERLAP_REVERSED_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_normal_overlap_reversed.luxar.zarr';
const POINTS_OVERLAP_REVERSED_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_points_normal_overlap_reversed.luxar.zarr';
const GSPLAT_VOLUMETRIC_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_volumetric.luxar.zarr';
const GSPLAT_VOLUMETRIC_REVERSED_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_volumetric_reversed.luxar.zarr';
const POINTS_VOLUMETRIC_REVERSED_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_points_volumetric_reversed.luxar.zarr';
const LINES_VOLUMETRIC_REVERSED_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_lines_volumetric_reversed.luxar.zarr';
const GSPLAT_RGBA_OCCLUSION_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_rgba_occlusion.luxar.zarr';

/** Read {name, mode, state, opacity} for every points mesh in the scene. */
function readPointsMaterialStates(page: import('@playwright/test').Page) {
  return page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    const states: Array<{
      name: string;
      blendingMode: string | undefined;
      blending: number;
      blendEquation: number;
      blendSrc: number;
      blendDst: number;
      depthTest: boolean;
      depthWrite: boolean;
      transparent: boolean;
      opacity: number;
    }> = [];
    debug.scene.traverse((obj: any) => {
      if (obj.userData?.nodeType === 'points' && obj.material) {
        const m = obj.material;
        states.push({
          name: obj.name ?? '',
          blendingMode: m.userData?.blendingMode,
          blending: m.blending,
          blendEquation: m.blendEquation,
          blendSrc: m.blendSrc,
          blendDst: m.blendDst,
          depthTest: m.depthTest,
          depthWrite: m.depthWrite,
          transparent: m.transparent,
          opacity: m.uniforms?.opacity?.value ?? m.uniforms?.uOpacity?.value ?? 1.0,
        });
      }
    });
    return states;
  });
}

test.describe('Blending Modes', () => {
  // The blending-mode datasets contain multiple groups (5+ point clouds) and
  // render with software-accelerated WebGL on most CI/test machines, where
  // FPS sits at ~3–10. The default 60s budget is marginal once data loading
  // plus several render passes are added; test.slow() triples it to 180s so
  // we measure correctness, not the test runner's tolerance for slow blits.
  test.slow();

  test('should load dataset with initial blending modes from zarr metadata', async ({ page }) => {
    // `&no-opfs` on every load: this spec never asserts the L2 OPFS tier, and
    // automated Chromium's OPFS stalls systemically (10s per op — issue #1645),
    // starving scene readiness past the test budget. The circuit breaker only
    // helps un-flagged real sessions (it still pays ~3 timeouts per fresh page).
    await page.goto(`/?src=${DATASET}&debug&no-opfs`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    const blendingStates = await readPointsMaterialStates(page);
    expect(blendingStates.length).toBeGreaterThan(0);

    // Every points material must carry the EXACT THREE state for its own
    // `userData.blendingMode` (per getPointBlendingState) — not just
    // membership in the set of plausible blending enums.
    for (const state of blendingStates) {
      expect(
        state.blendingMode,
        `${state.name}: material carries no userData.blendingMode`
      ).toBeTruthy();
      const expected = EXPECTED_STATE[state.blendingMode!];
      expect(expected, `${state.name}: unknown mode "${state.blendingMode}"`).toBeTruthy();
      expect(state.blending, `${state.name}: blending`).toBe(expected.blending);
      expect(state.blendEquation, `${state.name}: blendEquation`).toBe(expected.blendEquation);
      expect(state.blendSrc, `${state.name}: blendSrc`).toBe(expected.blendSrc);
      expect(state.blendDst, `${state.name}: blendDst`).toBe(expected.blendDst);
      expect(state.depthTest, `${state.name}: depthTest`).toBe(expected.depthTest);
      expect(state.transparent, `${state.name}: transparent`).toBe(expected.transparent);
      // Points NEVER depth-write in `normal` — a point sprite stamps a flat
      // depth plane across the whole disc (fringe included), so sorted
      // transparency never depth-writes, regardless of opacity (#1002).
      const expectedDepthWrite = state.blendingMode === 'normal' ? false : expected.depthWrite;
      expect(state.depthWrite, `${state.name}: depthWrite (opacity ${state.opacity})`).toBe(
        expectedDepthWrite
      );
    }
  });

  test('should have different depth behavior for additive vs normal blending', async ({ page }) => {
    // Deterministic fixture: test_points_blending_modes carries one layer
    // per mode, so BOTH additive and normal MUST exist (no silent
    // if-guards — a fixture regression fails loudly here).
    await page.goto(`/?src=${POINTS_BLENDING_FIXTURE}&debug&no-opfs`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    const states = await readPointsMaterialStates(page);
    const additive = states.find((s) => s.name.includes('points_additive'));
    const normal = states.find((s) => s.name.includes('points_normal'));

    expect(additive, 'points_additive mesh must exist in the fixture').toBeTruthy();
    expect(normal, 'points_normal mesh must exist in the fixture').toBeTruthy();

    // Additive blending renders on top: no depth interaction at all.
    expect(additive!.depthWrite).toBe(false);
    expect(additive!.depthTest).toBe(false);

    // Normal blending participates in the depth test.
    expect(normal!.depthTest).toBe(true);
  });

  test('should render without WebGL errors for all blending modes', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&no-opfs`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    // Force multiple renders to flush any deferred errors
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      for (let i = 0; i < 5; i++) debug.renderOnce();
    });
    await waitForNextRender(page);

    const webglErrors = await getWebGLErrors(page);
    expect(webglErrors.length).toBe(0);

    await assertNoConsoleErrors(page);
  });

  test('should handle multiple point clouds with different blending modes simultaneously', async ({
    page,
  }) => {
    await page.goto(`/?src=${MULTI_DATASET}&debug&no-opfs`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    // Count how many distinct blending modes are used
    const blendingModes = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const modes = new Set<number>();

      debug.scene.traverse((obj: any) => {
        // Points are Mesh + nodeType='points'.
        if (obj.userData?.nodeType === 'points' && obj.material) {
          modes.add(obj.material.blending);
        }
      });

      return Array.from(modes);
    });

    // Dataset should have at least one blending mode
    expect(blendingModes.length).toBeGreaterThan(0);

    // No WebGL errors from mixed blending
    const errors = await getWebGLErrors(page);
    expect(errors.length).toBe(0);
  });

  test('@visual visual regression: scene renders with blending applied', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&no-opfs`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);
    await waitForNextRender(page, 5);

    // Take screenshot — this establishes a baseline for blending correctness
    await expect(page).toHaveScreenshot('blending-modes-rendering.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });
});

test.describe('Points blending modes (per-mode material state)', () => {
  // Fixture: five overlapping sunflower-disk point layers on a Venn
  // circle, one per canonical mode, node names literally
  // `points_<mode>`. See generate_points_blending_modes_test() in
  // tests/fixtures/generate_test_data.py.
  test.slow();

  test.beforeAll(async () => {
    const { existsSync } = await import('node:fs');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');
    const specDir = path.dirname(fileURLToPath(import.meta.url));
    const fixtureDir = path.resolve(
      specDir,
      '../../../tests/fixtures/test_points_blending_modes.luxar.zarr'
    );
    if (!existsSync(fixtureDir)) {
      throw new Error(
        `Missing fixture ${fixtureDir} — run \`pnpm test:generate-fixtures\` ` +
          'from packages/luxar-viewer/ first.'
      );
    }
  });

  test('each points_<mode> layer carries the exact per-mode THREE blend state', async ({
    page,
  }) => {
    await page.goto(`/?src=${POINTS_BLENDING_FIXTURE}&debug&no-opfs`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    const states = await readPointsMaterialStates(page);
    expect(states.length).toBe(6);

    for (const [mode, expected] of Object.entries(EXPECTED_STATE)) {
      const state = states.find((s) => s.name.includes(`points_${mode}`));
      expect(state, `points_${mode} mesh not found in scene`).toBeTruthy();
      expect(state!.blendingMode, `points_${mode}: userData.blendingMode`).toBe(mode);
      expect(state!.blending, `points_${mode}: blending`).toBe(expected.blending);
      expect(state!.blendEquation, `points_${mode}: blendEquation`).toBe(expected.blendEquation);
      expect(state!.blendSrc, `points_${mode}: blendSrc`).toBe(expected.blendSrc);
      expect(state!.blendDst, `points_${mode}: blendDst`).toBe(expected.blendDst);
      expect(state!.depthTest, `points_${mode}: depthTest`).toBe(expected.depthTest);
      // Points never depth-write in `normal` (#1002); the EXPECTED_STATE
      // `normal.depthWrite:true` is the generic/line value.
      expect(state!.depthWrite, `points_${mode}: depthWrite`).toBe(
        mode === 'normal' ? false : expected.depthWrite
      );
      expect(state!.transparent, `points_${mode}: transparent`).toBe(expected.transparent);
    }
  });

  test('additive accumulation: the additive/luminous overlap is brighter than either base color', async ({
    page,
  }) => {
    // Pixel discriminator. The fixture makes `points_additive` pure
    // green and `points_luminous` pure red, ADJACENT on the Venn circle
    // so their overlap lens contains only those two layers (blue/cyan/
    // white layers are geometrically excluded). Both render through
    // AdditiveBlending, so lens pixels accumulate red + green — a mixed
    // r&g-high / b-low pixel that NO single layer's base color can
    // produce (every other fixture color carries a high blue channel).
    // No exact color pins: thresholds are loose ratios.
    // ?dpr=1 pins the pixel ratio for deterministic sampling.
    await page.goto(`/?src=${POINTS_BLENDING_FIXTURE}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);
    await waitForNextRender(page, 5);

    // Self-locating sampling (robust to the auto-framed camera pose): a
    // dense grid over the whole canvas. The fixture disks span a large
    // central fraction of the viewport, so the grid is guaranteed to hit
    // every cloud core AND the additive/luminous lens between them.
    // Step 0.0125: the pure-green core is small (the green cloud's outer
    // area is cyan-contaminated by additive overlap) — a measured probe
    // found only ~2 qualifying samples at step 0.025, too tight a margin
    // against framing/antialiasing drift across machines. 4× density
    // keeps ~8 hits while the scan stays instant (~5.5k samples).
    const offsets: Array<[number, number]> = [];
    for (let gx = 0.05; gx <= 0.951; gx += 0.0125) {
      for (let gy = 0.05; gy <= 0.951; gy += 0.0125) {
        offsets.push([gx, gy]);
      }
    }
    const samples = await samplePixelsAt(page, 'canvas#app', offsets, 'framebuffer');

    // Each of the two single-channel clouds renders its own PURE core
    // somewhere (strict dominance ratios exclude the mixed regions, UI
    // chrome grays, and the cyan/blue/white layers)…
    const greenDominant = samples.filter((s) => s.g > 100 && s.g > 2 * s.r && s.g > 2 * s.b);
    const redDominant = samples.filter((s) => s.r > 100 && s.r > 2 * s.g && s.r > 2 * s.b);
    expect(greenDominant.length, 'no pure-green core (points_additive missing?)').toBeGreaterThan(
      0
    );
    expect(redDominant.length, 'no pure-red core (points_luminous missing?)').toBeGreaterThan(0);

    // …and the overlap lens accumulates BOTH channels: a pixel brighter
    // in red than green's base color (red channel 0) and brighter in
    // green than red's base color (green channel 0) can only come from
    // cross-layer additive accumulation. The b < min(r,g)/2 term
    // excludes every other fixture color (cyan/blue/white all carry a
    // high blue channel) and neutral UI grays, so no single layer can
    // satisfy this vacuously.
    const mixed = samples.filter((s) => s.r > 80 && s.g > 80 && s.b < Math.min(s.r, s.g) / 2);
    expect(
      mixed.length,
      'no additive red+green accumulation found in the additive/luminous overlap lens'
    ).toBeGreaterThan(0);
  });
});

test.describe('Blending-mode inheritance (group attr → leaf material + panel)', () => {
  // Fixture: surface_group carries blending_mode='max'; its child leaf
  // child_points OMITS the attr on disk (the Python writer no longer
  // stamps a default), so the viewer must compose the effective mode
  // from the nearest ancestor. See generate_blending_inherited_test().
  test.slow();

  test.beforeAll(async () => {
    const { existsSync } = await import('node:fs');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');
    const specDir = path.dirname(fileURLToPath(import.meta.url));
    const fixtureDir = path.resolve(
      specDir,
      '../../../tests/fixtures/test_blending_inherited.luxar.zarr'
    );
    if (!existsSync(fixtureDir)) {
      throw new Error(
        `Missing fixture ${fixtureDir} — run \`pnpm test:generate-fixtures\` ` +
          'from packages/luxar-viewer/ first.'
      );
    }
  });

  test("child_points material inherits the group's 'max' blend state", async ({ page }) => {
    await page.goto(`/?src=${BLENDING_INHERITED_FIXTURE}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    const states = await readPointsMaterialStates(page);
    const child = states.find((s) => s.name.includes('child_points'));
    expect(child, 'child_points mesh not found in scene').toBeTruthy();

    // The composed effective mode is 'max' (from surface_group) even
    // though the leaf's .zattrs has no blending_mode key. THREE enums:
    // CustomBlending=5, MaxEquation=104, OneFactor=201.
    expect(child!.blendingMode).toBe('max');
    expect(child!.blending).toBe(5);
    expect(child!.blendEquation).toBe(104);
    expect(child!.blendSrc).toBe(201);
    expect(child!.blendDst).toBe(201);
    expect(child!.depthTest).toBe(true);
    expect(child!.depthWrite).toBe(false);
    expect(child!.transparent).toBe(true);
  });

  test("layers panel reports the child layer's composed mode as 'max'", async ({ page }) => {
    await page.goto(`/?src=${BLENDING_INHERITED_FIXTURE}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);
    await openLayersPanel(page);

    const childMode = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const layers = debug?.app?.layersPanel?.layerState?.getLayers() ?? [];
      return layers.find((l: any) => l.name === 'child_points')?.blendingMode ?? null;
    });
    // Panel layer state initializes from the COMPOSED effective attrs,
    // not the leaf's own (absent) attr.
    expect(childMode).toBe('max');
  });
});

test.describe('GSplat normal mode (premultiplied coverage alpha)', () => {
  // Fixture: two large overlapping splats (red behind, green in front,
  // storage order = back-to-front for the default camera) + one small
  // blue reference splat, blending_mode='normal', opacity=0.5. See
  // generate_gsplats_normal_overlap_test() and
  // GSPLAT_DEPTH_SORTING_SPEC.md §3 (Phase 0).
  test.slow();

  // Fail fast with an actionable message instead of a 30 s
  // waitForGSplatsCommitted timeout + opaque 404: this fixture is
  // Python-generated and NOT covered by the Playwright global-setup
  // (which only checks datasets/examples).
  test.beforeAll(async () => {
    const { existsSync } = await import('node:fs');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');
    const specDir = path.dirname(fileURLToPath(import.meta.url));
    for (const name of [
      'test_gsplats_normal_overlap.luxar.zarr',
      'test_gsplats_normal_overlap_reversed.luxar.zarr',
    ]) {
      const fixtureDir = path.resolve(specDir, `../../../tests/fixtures/${name}`);
      if (!existsSync(fixtureDir)) {
        throw new Error(
          `Missing fixture ${fixtureDir} — run \`pnpm test:generate-fixtures\` ` +
            'from packages/luxar-viewer/ first.'
        );
      }
    }
  });

  /** Wait until a gsplats mesh has committed instances. */
  async function waitForGSplatsCommitted(page: import('@playwright/test').Page): Promise<void> {
    await page.waitForFunction(
      () => {
        const debug = (window as unknown as { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
        if (!debug?.scene) return false;
        let committed = false;
        (debug.scene as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
          const o = obj as {
            userData?: { nodeType?: string };
            geometry?: { instanceCount?: number };
          };
          if (o.userData?.nodeType === 'gsplats' && (o.geometry?.instanceCount ?? 0) > 0) {
            committed = true;
          }
        });
        return committed;
      },
      undefined,
      { timeout: 30000 }
    );
  }

  test('material carries the gsplat premultiplied normal state', async ({ page }) => {
    await page.goto(`/?src=${GSPLAT_OVERLAP_FIXTURE}&debug`);
    await waitForLuxarReady(page);
    await waitForGSplatsCommitted(page);

    const state = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let found: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'gsplats' && obj.material && !found) {
          const m = obj.material;
          found = {
            blending: m.blending,
            blendEquation: m.blendEquation,
            blendSrc: m.blendSrc,
            blendDst: m.blendDst,
            blendEquationAlpha: m.blendEquationAlpha,
            transparent: m.transparent,
            depthTest: m.depthTest,
            depthWrite: m.depthWrite,
            premultipliedAlpha: m.premultipliedAlpha,
            blendingMode: m.userData?.blendingMode,
          };
        }
      });
      return found;
    });

    expect(state).not.toBeNull();
    expect(state.blendingMode).toBe('normal');
    // getGSplatNormalBlendingState: CustomBlending(5) + AddEquation(100)
    // + One(201) / OneMinusSrcAlpha(205), symmetric alpha channel.
    expect(state.blending).toBe(5);
    expect(state.blendEquation).toBe(100);
    expect(state.blendSrc).toBe(201);
    expect(state.blendDst).toBe(205);
    expect(state.blendEquationAlpha).toBe(null);
    expect(state.transparent).toBe(true);
    expect(state.depthTest).toBe(true);
    expect(state.depthWrite).toBe(false);
    // The premultipliedAlpha flag must stay OFF (NodeMaterial would
    // auto-inject a second RGB×alpha on the TSL path).
    expect(state.premultipliedAlpha).toBe(false);
  });

  test('TSL path under ?renderer=webgpu carries the same state without GL errors', async ({
    page,
  }) => {
    // In headless CI this runs WebGPURenderer's WebGL2 fallback backend —
    // exactly the bridge where separate alpha-channel blend state trips
    // gl.getError(); the premult-normal state must stay symmetric-clean.
    await page.goto(`/?src=${GSPLAT_OVERLAP_FIXTURE}&renderer=webgpu&debug`);
    await waitForLuxarReady(page);
    await waitForGSplatsCommitted(page);
    await waitForNextRender(page, 5);

    const state = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let found: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'gsplats' && obj.material && !found) {
          const m = obj.material;
          found = {
            blending: m.blending,
            blendSrc: m.blendSrc,
            blendDst: m.blendDst,
            depthWrite: m.depthWrite,
            transparent: m.transparent,
            premultipliedAlpha: m.premultipliedAlpha,
            blendingMode: m.userData?.blendingMode,
          };
        }
      });
      return found;
    });

    expect(state).not.toBeNull();
    expect(state.blendingMode).toBe('normal');
    expect(state.blending).toBe(5); // CustomBlending
    expect(state.blendSrc).toBe(201); // OneFactor
    expect(state.blendDst).toBe(205); // OneMinusSrcAlphaFactor
    expect(state.depthWrite).toBe(false);
    expect(state.transparent).toBe(true);
    expect(state.premultipliedAlpha).toBe(false);

    const webglErrors = await getWebGLErrors(page);
    expect(webglErrors.length).toBe(0);
  });

  test('background splat shows through the overlap (real alpha-over)', async ({ page }) => {
    // Scope: this gate owns only "both splats visible + the background
    // survives the overlap". The exact blend state is asserted by
    // 'material carries the gsplat premultiplied normal state' above, and
    // the draw ordering by the Phase-2 depth-sort test below — a
    // blend-state or sort regression is caught there, not here.
    //
    // ?dpr=1 pins the pixel ratio for deterministic sampling.
    await page.goto(`/?src=${GSPLAT_OVERLAP_FIXTURE}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForGSplatsCommitted(page);
    await waitForNextRender(page, 5);

    // ONE dense capture, then EVERY pixel of the WHOLE frame is classified.
    // No sub-box: the old x ∈ [0.15,0.85] / y ∈ [0.25,0.75] rectangle was a
    // magic crop that wasn't even centred on the content (the splats project
    // at ≈0.34-0.37 of width and ≈0.68-0.71 of height, and float error left
    // the lattice's effective last row at y = 0.70, cutting off nearer the
    // bottom HALF of both discs) and it re-introduced the very framing
    // sensitivity this test exists to remove. Scanning everything is safe
    // because all three discriminators reject neutral pixels — not because UI
    // chrome *cannot* satisfy them, but because the viewer's greys (the rail
    // is ~rgb 42,45,49) cannot. The one non-neutral overlay that could is the
    // scene-identity banner (`changed` ≈rgb 147,40,40 → red-dominant,
    // `unreachable` ≈rgb 138,101,18 → `mixed`, at ~1.9% of the frame — larger
    // than any healthy class), so it is asserted absent first.
    //
    // Dense rather than a sparse lattice because the red-dominant area is a
    // genuine but THIN crescent: the two big splats project nearly
    // concentric (centres ~44 px apart against ~46/54 px screen sigmas) and
    // the nearer green splat is the LARGER on screen, so the background
    // survives only in a rim covering ~0.35% of the frame. The deleted
    // `gx += 0.05` lattice accumulated float error (0.8000000000000002), so
    // it was 14 columns x 10 rows = 140 points over x ∈ [0.15,0.80],
    // y ∈ [0.25,0.70], spaced 64x36 px — and that box clipped the crescent,
    // leaving ≈1 expected hit: a coin-flip-grade oracle, ~60-75% pass odds on
    // a HEALTHY renderer against ~6% under the defect below. It measured its
    // own luck, not the frame.
    //
    // captureCanvasRGBA takes an ELEMENT screenshot, so DOM overlays are
    // composited over the canvas: a raised banner could pass this vacuously.
    expect(
      await page.locator('#luxar-scene-identity-banner').count(),
      'scene-identity banner is up — its non-neutral fill satisfies the colour ' +
        'predicates below, so this gate would pass vacuously'
    ).toBe(0);
    const frame = await captureCanvasRGBA(page, 'canvas#app', 'framebuffer');

    let redDominant = 0;
    let greenDominant = 0;
    let mixed = 0;
    for (let i = 0; i < frame.rgba.length; i += 4) {
      const r = frame.rgba[i];
      const g = frame.rgba[i + 1];
      const b = frame.rgba[i + 2];
      // Three-way dominance: the FORM matches the sibling additive test
      // above, but the brightness floors are deliberately lower (40/30 vs
      // its 100/80) — the crescent's red-dominant pixels bottom out around
      // r ≈ 63, so a 100 floor would erase the very class this test counts.
      // Without the blue term, magenta/cyan would count as red/green.
      if (r > 40 && r > 2 * g && r > 2 * b) redDominant++;
      if (g > 40 && g > 2 * r && g > 2 * b) greenDominant++;
      // The alpha-over discriminator: pre-fix, gsplat 'normal' emitted
      // alpha=1.0, so the front (green) splat fully REPLACED the back
      // (red) splat wherever it covered — no pixel could carry both
      // channels. With premultiplied coverage alpha at opacity 0.5 the
      // overlap composites green over red and both channels survive.
      // `b < min(r,g)/2` is the TIGHTEST term here, not slack: ACES injects
      // blue through its input matrix, so blue runs ≈0.3-0.4 of min(r,g) in
      // the counted ring (≈0.5-0.6 in the excluded bright core). It is still
      // hopeless for a neutral grey (r≈g≈b), and structurally unsatisfiable
      // by ANY single splat — red is (1,0.1,0.1) so b==g, green is
      // (0.1,1,0.1) so b==r, and the blue reference splat is blue-max — so
      // only genuine red-over-green compositing can produce a `mixed` pixel.
      if (r > 30 && g > 30 && b < Math.min(r, g) / 2) mixed++;
    }

    const scanned = frame.width * frame.height;
    // The floors are FRACTIONS of the frame, so the gate is invariant to
    // canvas SCALE and dpr; an ASPECT change re-frames the scene (vertical FOV,
    // and calculateCameraDistance switches between vertical and horizontal
    // fit), so the fractions move and must be re-measured. Both readings below
    // are real headless-Chromium runs of this fixture at 1280x720 / dpr=1
    // (frame = 921,600 px) with the predicates above:
    //   healthy (current renderer): red 3215 = 0.349%, green 18429 = 2.000%,
    //                               mixed 14173 = 1.538%
    //   NEGATIVE CONTROL, the defect this test guards (coverage forced to
    //   1.0 in the LUXAR_NORMAL_PREMULT branch of shader-glsl.ts, so the
    //   front splat replaces the background): red 156 = 0.017%,
    //   green 38164 = 4.141%, mixed 0 = 0.000%
    // The floors are PER-CLASS because the headroom is. Red is a thin rim the
    // defect does NOT zero (healthy 0.349% vs residual 0.017%), so its floor
    // is a geometric-mean compromise inside that narrow window (0.08% = 4.4x
    // under healthy, 4.7x over the residual). Green and mixed have no lower
    // constraint — the defect gives mixed exactly 0.000% and INFLATES green —
    // so 0.5% (4.0x / 3.1x under healthy) costs nothing and there is no reason
    // to leave ~19x slack under them. It buys no sensitivity to GRADED
    // occlusion, though: a measured 3x over-occlusion control (coverage x3 in
    // that same branch) still reads mixed 1.386% / red 0.326% and PASSES —
    // these are hue-DOMINANCE classes, so the overlap ring keeps mixing both
    // channels even when the front splat occludes far harder. This gate
    // deliberately bounds only the total-replacement defect; quantifying
    // graded over-occlusion needs a different measurement (the mixed ring's
    // width, an intensity profile), not this one. Green only checks that the
    // front splat rendered at all (the defect raises it, so it discriminates
    // nothing here); `mixed` is the load-bearing alpha-over discriminator.
    // Retune by re-measuring BOTH sets, not by nudging until green.
    const RED_FLOOR = 0.0008; // thin rim, little room above the defect residual
    const LIT_FLOOR = 0.005; // green/mixed: defect gives 0.000% / 4.141%
    const pct = (n: number) => `${((100 * n) / scanned).toFixed(3)}%`;
    const report =
      `frame ${frame.width}x${frame.height} (${scanned} px): ` +
      `red=${redDominant} (${pct(redDominant)} vs floor ${pct(RED_FLOOR * scanned)}) ` +
      `green=${greenDominant} (${pct(greenDominant)} vs floor ${pct(LIT_FLOOR * scanned)}) ` +
      `mixed=${mixed} (${pct(mixed)} vs floor ${pct(LIT_FLOOR * scanned)})`;
    // Printed on success too, so a passing run records its margin and
    // erosion is visible before the gate goes red.
    console.log(`[gsplat-alpha-over] ${report}`);

    // Both splats render…
    expect(redDominant / scanned, `background (red) splat not visible — ${report}`).toBeGreaterThan(
      RED_FLOOR
    );
    expect(greenDominant / scanned, `front (green) splat not visible — ${report}`).toBeGreaterThan(
      LIT_FLOOR
    );
    // …and the background shows through the overlap (fails pre-fix).
    expect(
      mixed / scanned,
      `no alpha-over overlap — the front splat replaced the background: ${report}`
    ).toBeGreaterThan(LIT_FLOOR);
  });

  test('depth sort applies a back-to-front ordering after load settle (Phase 2)', async ({
    page,
  }) => {
    // The reversed fixture declares its splats front-to-back; the
    // compiler may Morton-reorder storage, so the gate does NOT assume a
    // specific on-disk order. Instead it asserts the applied
    // `aSortedIndex` permutation directly: (a) it departs from identity,
    // and (b) it is back-to-front — view z (from the live camera + the
    // splat-texture centers) is non-decreasing along the ordering. For
    // this scene's camera framing the identity ordering is NOT monotone
    // (the near green splat sits between the two far splats in storage),
    // so pre-Phase-2 the non-identity wait times out — the non-vacuous
    // gate. The overlap pixels are then checked for the correct
    // green-over-red compositing.
    await page.goto(`/?src=${GSPLAT_OVERLAP_REVERSED_FIXTURE}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForGSplatsCommitted(page);

    // The sort lands asynchronously after the commit: wait until the
    // permutation departs from identity.
    await page.waitForFunction(
      () => {
        const debug = (window as unknown as { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
        if (!debug?.scene) return false;
        let sorted = false;
        (debug.scene as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
          const o = obj as {
            userData?: { nodeType?: string; visibleSplatCount?: number };
            geometry?: {
              attributes?: {
                aSortedIndex?: { array?: ArrayLike<number> };
                aSortedIndexB?: { array?: ArrayLike<number> };
              };
              userData?: { sortedIndexSlot?: 0 | 1 };
            };
          };
          if (o.userData?.nodeType !== 'gsplats') return;
          const arr =
            o.geometry?.userData?.sortedIndexSlot === 1
              ? o.geometry?.attributes?.aSortedIndexB?.array
              : o.geometry?.attributes?.aSortedIndex?.array;
          const count = o.userData?.visibleSplatCount ?? 0;
          if (!arr || count < 2) return;
          for (let i = 0; i < count; i++) {
            if (arr[i] !== i) {
              sorted = true;
              return;
            }
          }
        });
        return sorted;
      },
      undefined,
      { timeout: 30000 }
    );

    // Assert the applied permutation is back-to-front: view z of the
    // drawn splats (splat-texture centers under modelView) never
    // decreases along the instance order.
    const monotone = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const results: Array<{ ordering: number[]; viewZs: number[]; ok: boolean }> = [];
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType !== 'gsplats') return;
        const count = obj.userData?.visibleSplatCount ?? 0;
        const arr =
          obj.geometry?.userData?.sortedIndexSlot === 1
            ? obj.geometry?.attributes?.aSortedIndexB?.array
            : obj.geometry?.attributes?.aSortedIndex?.array;
        const texData = obj.geometry?.userData?.elementTexture?.image?.data;
        if (!arr || !texData || count < 2) return;
        const mwi = debug.camera.matrixWorldInverse.elements;
        const mw = obj.matrixWorld.elements;
        // modelView = matrixWorldInverse × matrixWorld (column-major).
        const viewZof = (x: number, y: number, z: number) => {
          const wx = mw[0] * x + mw[4] * y + mw[8] * z + mw[12];
          const wy = mw[1] * x + mw[5] * y + mw[9] * z + mw[13];
          const wz = mw[2] * x + mw[6] * y + mw[10] * z + mw[14];
          return mwi[2] * wx + mwi[6] * wy + mwi[10] * wz + mwi[14];
        };
        const ordering: number[] = [];
        const viewZs: number[] = [];
        let ok = true;
        let prev = -Infinity;
        for (let j = 0; j < count; j++) {
          const idx = arr[j];
          ordering.push(idx);
          const zv = viewZof(texData[idx * 16], texData[idx * 16 + 1], texData[idx * 16 + 2]);
          viewZs.push(zv);
          // Small epsilon: equal-depth splats share a key bucket.
          if (zv < prev - 1e-4) ok = false;
          prev = Math.max(prev, zv);
        }
        results.push({ ordering, viewZs, ok });
      });
      return results;
    });
    expect(monotone.length).toBeGreaterThan(0);
    for (const r of monotone) {
      expect(r.ok, `ordering ${r.ordering} viewZs ${r.viewZs} not back-to-front`).toBe(true);
    }

    await waitForNextRender(page, 5);

    // Visual sanity: the overlap core composites green over red (the
    // correct image; the wrong draw order gives the mirror-image
    // red-dominant core). The overlap's screen position depends on the
    // auto-framing, so project the red/green splat centers to screen
    // coordinates in-page and sample the middle of the segment between
    // them — where the two Gaussians weigh equally and the compositing
    // order alone decides the dominant channel (~2:1).
    const midOffsets = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let offsets: Array<[number, number]> | null = null;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType !== 'gsplats' || offsets) return;
        const texData = obj.geometry?.userData?.elementTexture?.image?.data;
        if (!texData) return;
        const mwi = debug.camera.matrixWorldInverse.elements;
        const pm = debug.camera.projectionMatrix.elements;
        const mw = obj.matrixWorld.elements;
        const toScreen = (x: number, y: number, z: number): [number, number] => {
          const wx = mw[0] * x + mw[4] * y + mw[8] * z + mw[12];
          const wy = mw[1] * x + mw[5] * y + mw[9] * z + mw[13];
          const wz = mw[2] * x + mw[6] * y + mw[10] * z + mw[14];
          const vx = mwi[0] * wx + mwi[4] * wy + mwi[8] * wz + mwi[12];
          const vy = mwi[1] * wx + mwi[5] * wy + mwi[9] * wz + mwi[13];
          const vz = mwi[2] * wx + mwi[6] * wy + mwi[10] * wz + mwi[14];
          const cx = pm[0] * vx + pm[4] * vy + pm[8] * vz + pm[12];
          const cy = pm[1] * vx + pm[5] * vy + pm[9] * vz + pm[13];
          const cw = pm[3] * vx + pm[7] * vy + pm[11] * vz + pm[15];
          return [(cx / cw + 1) / 2, (1 - cy / cw) / 2];
        };
        // Identify red (z≈0, x<1) and green (z≈1) among the first splats.
        let red: [number, number] | null = null;
        let green: [number, number] | null = null;
        for (let i = 0; i < 3; i++) {
          const x = texData[i * 16];
          const z = texData[i * 16 + 2];
          if (z > 0.5) green = toScreen(x, texData[i * 16 + 1], z);
          else if (x < 1.0) red = toScreen(x, texData[i * 16 + 1], z);
        }
        if (red && green) {
          offsets = [];
          for (let t = 0.35; t <= 0.65; t += 0.05) {
            offsets.push([red[0] + (green[0] - red[0]) * t, red[1] + (green[1] - red[1]) * t]);
          }
        }
      });
      return offsets;
    });
    expect(midOffsets).not.toBeNull();
    const samples = await samplePixelsAt(page, 'canvas#app', midOffsets!, 'framebuffer');
    const lit = samples.filter((s) => s.r > 20 || s.g > 20);
    expect(lit.length).toBeGreaterThan(0);
    const greenOverRed = lit.filter((s) => s.g > s.r).length;
    const redOverGreen = lit.filter((s) => s.r > s.g).length;
    expect(greenOverRed).toBeGreaterThan(redOverGreen);
  });

  /** In-page predicate: every gsplats node's applied `aSortedIndex`
   *  permutation is view-z monotone (back-to-front) under the CURRENT
   *  camera pose. Serialized into waitForFunction, so it must be
   *  self-contained. */
  const orderingIsBackToFront = () => {
    const debug = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
    if (!debug?.scene) return false;
    let checked = 0;
    let allOk = true;
    debug.scene.traverse((obj: any) => {
      if (obj.userData?.nodeType !== 'gsplats') return;
      const count = obj.userData?.visibleSplatCount ?? 0;
      const arr =
        obj.geometry?.userData?.sortedIndexSlot === 1
          ? obj.geometry?.attributes?.aSortedIndexB?.array
          : obj.geometry?.attributes?.aSortedIndex?.array;
      const texData = obj.geometry?.userData?.elementTexture?.image?.data;
      if (!arr || !texData || count < 2) return;
      checked++;
      const mwi = debug.camera.matrixWorldInverse.elements;
      const mw = obj.matrixWorld.elements;
      let prev = -Infinity;
      for (let j = 0; j < count; j++) {
        const idx = arr[j];
        const x = texData[idx * 16];
        const y = texData[idx * 16 + 1];
        const z = texData[idx * 16 + 2];
        const wx = mw[0] * x + mw[4] * y + mw[8] * z + mw[12];
        const wy = mw[1] * x + mw[5] * y + mw[9] * z + mw[13];
        const wz = mw[2] * x + mw[6] * y + mw[10] * z + mw[14];
        const zv = mwi[2] * wx + mwi[6] * wy + mwi[10] * wz + mwi[14];
        if (zv < prev - 1e-4) allOk = false;
        prev = Math.max(prev, zv);
      }
    });
    return checked > 0 && allOk;
  };

  test('camera orbit re-sorts — ordering settles back-to-front from every angle (Phase 3)', async ({
    page,
  }) => {
    await page.goto(`/?src=${GSPLAT_OVERLAP_REVERSED_FIXTURE}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForGSplatsCommitted(page);
    // Settle the commit-time sort first (the Phase-2 behavior).
    await page.waitForFunction(orderingIsBackToFront, undefined, { timeout: 30000 });

    // Orbit in three large steps (each far past the 3° threshold). For
    // each: the profiler's Depth Sort pass count MUST increase (a
    // camera-motion re-sort actually dispatched — the non-vacuous gate;
    // pre-Phase-3 no sort ever fires after load settle) and the applied
    // ordering must settle back-to-front under the NEW pose.
    for (const stepDeg of [60, 75, 90]) {
      const sortsBefore: number = await page.evaluate(
        () =>
          (window as unknown as { __luxarDebug?: any }).__luxarDebug
            .getSceneLoader()
            .getProfiler()
            .getDepthSortTimings().count
      );

      await page.evaluate((deg: number) => {
        const debug = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
        const pose = debug.app.getCameraPose();
        const rad = (deg * Math.PI) / 180;
        const [px, py, pz] = pose.position;
        const [tx, , tz] = pose.target;
        // Rotate the camera about the target around +Y (world up).
        const dx = px - tx;
        const dz = pz - tz;
        const nx = tx + dx * Math.cos(rad) + dz * Math.sin(rad);
        const nz = tz - dx * Math.sin(rad) + dz * Math.cos(rad);
        debug.app.setCameraPose({ ...pose, position: [nx, py, nz] });
        // Kick the rAF loop so the per-frame scheduler evaluates.
        debug.renderOnce();
      }, stepDeg);

      await page.waitForFunction(
        (before: number) =>
          (window as unknown as { __luxarDebug?: any }).__luxarDebug
            .getSceneLoader()
            .getProfiler()
            .getDepthSortTimings().count > before,
        sortsBefore,
        { timeout: 15000 }
      );
      await page.waitForFunction(orderingIsBackToFront, undefined, { timeout: 15000 });
    }

    const errors = await getWebGLErrors(page);
    expect(errors).toHaveLength(0);
  });

  test('?depthSort=0 pins the identity ordering across load and camera motion (Phase 3)', async ({
    page,
  }) => {
    await page.goto(`/?src=${GSPLAT_OVERLAP_REVERSED_FIXTURE}&debug&dpr=1&depthSort=0`);
    await waitForLuxarReady(page);
    await waitForGSplatsCommitted(page);
    // Give any (buggy) async sort a chance to land before asserting.
    await waitForNextRender(page, 10);

    const readState = () =>
      page.evaluate(() => {
        const debug = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
        const nodes: Array<{ identity: boolean; count: number }> = [];
        debug.scene.traverse((obj: any) => {
          if (obj.userData?.nodeType !== 'gsplats') return;
          const count = obj.userData?.visibleSplatCount ?? 0;
          const arr =
            obj.geometry?.userData?.sortedIndexSlot === 1
              ? obj.geometry?.attributes?.aSortedIndexB?.array
              : obj.geometry?.attributes?.aSortedIndex?.array;
          if (!arr || count < 1) return;
          let identity = true;
          for (let i = 0; i < count; i++) {
            if (arr[i] !== i) identity = false;
          }
          nodes.push({ identity, count });
        });
        const sorts = debug.getSceneLoader().getProfiler().getDepthSortTimings().count;
        return { nodes, sorts };
      });

    const afterLoad = await readState();
    expect(afterLoad.nodes.length).toBeGreaterThan(0);
    for (const node of afterLoad.nodes) expect(node.identity).toBe(true);
    expect(afterLoad.sorts).toBe(0);

    // Large camera motion must not wake the scheduler either.
    await page.evaluate(() => {
      const debug = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
      const pose = debug.app.getCameraPose();
      const [px, py, pz] = pose.position;
      const [tx, , tz] = pose.target;
      debug.app.setCameraPose({
        ...pose,
        position: [tx - (pz - tz), py, tz + (px - tx)], // 90° about +Y
      });
      debug.renderOnce();
    });
    await waitForNextRender(page, 10);

    const afterOrbit = await readState();
    for (const node of afterOrbit.nodes) expect(node.identity).toBe(true);
    expect(afterOrbit.sorts).toBe(0);
  });
});

test.describe('Points normal mode depth sorting', () => {
  // Fixture: the POINTS twin of the gsplat normal-overlap-reversed
  // scene (three-geometry symmetry — points are depth-sorted too): a
  // back red point, a front green point overlapping it in screen space,
  // and an off-axis blue reference, declared front-first with
  // blending_mode='normal', opacity=0.5. See
  // generate_points_normal_overlap_reversed_test().
  test.slow();

  test.beforeAll(async () => {
    const { existsSync } = await import('node:fs');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');
    const specDir = path.dirname(fileURLToPath(import.meta.url));
    for (const name of [
      'test_points_normal_overlap.luxar.zarr',
      'test_points_normal_overlap_reversed.luxar.zarr',
    ]) {
      const fixtureDir = path.resolve(specDir, `../../../tests/fixtures/${name}`);
      if (!existsSync(fixtureDir)) {
        throw new Error(
          `Missing fixture ${fixtureDir} — run \`pnpm test:generate-fixtures\` ` +
            'from packages/luxar-viewer/ first.'
        );
      }
    }
  });

  /** Wait until a points mesh has committed instances. */
  async function waitForPointsCommitted(page: import('@playwright/test').Page): Promise<void> {
    await page.waitForFunction(
      () => {
        const debug = (window as unknown as { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
        if (!debug?.scene) return false;
        let committed = false;
        (debug.scene as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
          const o = obj as {
            userData?: { nodeType?: string };
            geometry?: { instanceCount?: number };
          };
          if (o.userData?.nodeType === 'points' && (o.geometry?.instanceCount ?? 0) > 0) {
            committed = true;
          }
        });
        return committed;
      },
      undefined,
      { timeout: 30000 }
    );
  }

  test('depth sort applies a back-to-front ordering after load settle (points)', async ({
    page,
  }) => {
    // The points mirror of the gsplat Phase-2 gate. The reversed fixture
    // declares its points front-to-back; the compiler may Morton-reorder
    // storage, so the gate does NOT assume a specific on-disk order.
    // Instead it asserts the applied `aSortedIndex` permutation directly:
    // (a) it departs from identity, and (b) it is back-to-front — view z
    // (from the live camera + the point-texture positions, 12 floats /
    // point) is non-decreasing along the ordering. For this scene's
    // camera framing the identity ordering is NOT monotone (the near
    // green point sits between the two far points in storage), so a
    // points-blind coordinator times out at the non-identity wait — the
    // non-vacuous gate. The overlap pixels are then checked for the
    // correct green-over-red compositing.
    await page.goto(`/?src=${POINTS_OVERLAP_REVERSED_FIXTURE}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForPointsCommitted(page);

    // The sort lands asynchronously after the commit: wait until the
    // permutation departs from identity.
    await page.waitForFunction(
      () => {
        const debug = (window as unknown as { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
        if (!debug?.scene) return false;
        let sorted = false;
        (debug.scene as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
          const o = obj as {
            userData?: { nodeType?: string; visiblePointCount?: number };
            geometry?: {
              attributes?: {
                aSortedIndex?: { array?: ArrayLike<number> };
                aSortedIndexB?: { array?: ArrayLike<number> };
              };
              userData?: { sortedIndexSlot?: 0 | 1 };
            };
          };
          if (o.userData?.nodeType !== 'points') return;
          const arr =
            o.geometry?.userData?.sortedIndexSlot === 1
              ? o.geometry?.attributes?.aSortedIndexB?.array
              : o.geometry?.attributes?.aSortedIndex?.array;
          const count = o.userData?.visiblePointCount ?? 0;
          if (!arr || count < 2) return;
          for (let i = 0; i < count; i++) {
            if (arr[i] !== i) {
              sorted = true;
              return;
            }
          }
        });
        return sorted;
      },
      undefined,
      { timeout: 30000 }
    );

    // Assert the applied permutation is back-to-front: view z of the
    // drawn points (point-texture positions under modelView) never
    // decreases along the instance order. Point texel layout: 3 texels
    // (12 floats) per point, texel0.xyz = position.
    const monotone = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const results: Array<{ ordering: number[]; viewZs: number[]; ok: boolean }> = [];
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType !== 'points') return;
        const count = obj.userData?.visiblePointCount ?? 0;
        const arr =
          obj.geometry?.userData?.sortedIndexSlot === 1
            ? obj.geometry?.attributes?.aSortedIndexB?.array
            : obj.geometry?.attributes?.aSortedIndex?.array;
        const texData = obj.geometry?.userData?.elementTexture?.image?.data;
        if (!arr || !texData || count < 2) return;
        const mwi = debug.camera.matrixWorldInverse.elements;
        const mw = obj.matrixWorld.elements;
        // modelView = matrixWorldInverse × matrixWorld (column-major).
        const viewZof = (x: number, y: number, z: number) => {
          const wx = mw[0] * x + mw[4] * y + mw[8] * z + mw[12];
          const wy = mw[1] * x + mw[5] * y + mw[9] * z + mw[13];
          const wz = mw[2] * x + mw[6] * y + mw[10] * z + mw[14];
          return mwi[2] * wx + mwi[6] * wy + mwi[10] * wz + mwi[14];
        };
        const ordering: number[] = [];
        const viewZs: number[] = [];
        let ok = true;
        let prev = -Infinity;
        for (let j = 0; j < count; j++) {
          const idx = arr[j];
          ordering.push(idx);
          const zv = viewZof(texData[idx * 12], texData[idx * 12 + 1], texData[idx * 12 + 2]);
          viewZs.push(zv);
          // Small epsilon: equal-depth points share a key bucket.
          if (zv < prev - 1e-4) ok = false;
          prev = Math.max(prev, zv);
        }
        results.push({ ordering, viewZs, ok });
      });
      return results;
    });
    expect(monotone.length).toBeGreaterThan(0);
    for (const r of monotone) {
      expect(r.ok, `points ordering ${r.ordering} viewZs ${r.viewZs} not back-to-front`).toBe(true);
    }

    await waitForNextRender(page, 5);

    // Visual sanity: the overlap core composites green over red (the
    // correct image; the wrong draw order gives the mirror-image
    // red-dominant core). The overlap's screen position depends on the
    // auto-framing, so project the red/green point centers to screen
    // coordinates in-page and sample the middle of the segment between
    // them — where the two sprites weigh comparably and the compositing
    // order alone decides the dominant channel.
    const midOffsets = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let offsets: Array<[number, number]> | null = null;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType !== 'points' || offsets) return;
        const texData = obj.geometry?.userData?.elementTexture?.image?.data;
        if (!texData) return;
        const mwi = debug.camera.matrixWorldInverse.elements;
        const pm = debug.camera.projectionMatrix.elements;
        const mw = obj.matrixWorld.elements;
        const toScreen = (x: number, y: number, z: number): [number, number] => {
          const wx = mw[0] * x + mw[4] * y + mw[8] * z + mw[12];
          const wy = mw[1] * x + mw[5] * y + mw[9] * z + mw[13];
          const wz = mw[2] * x + mw[6] * y + mw[10] * z + mw[14];
          const vx = mwi[0] * wx + mwi[4] * wy + mwi[8] * wz + mwi[12];
          const vy = mwi[1] * wx + mwi[5] * wy + mwi[9] * wz + mwi[13];
          const vz = mwi[2] * wx + mwi[6] * wy + mwi[10] * wz + mwi[14];
          const cx = pm[0] * vx + pm[4] * vy + pm[8] * vz + pm[12];
          const cy = pm[1] * vx + pm[5] * vy + pm[9] * vz + pm[13];
          const cw = pm[3] * vx + pm[7] * vy + pm[11] * vz + pm[15];
          return [(cx / cw + 1) / 2, (1 - cy / cw) / 2];
        };
        // Identify red (z≈0, x<1) and green (z≈1) among the first points
        // (point texel stride: 12 floats, texel0.xyz = position).
        let red: [number, number] | null = null;
        let green: [number, number] | null = null;
        for (let i = 0; i < 3; i++) {
          const x = texData[i * 12];
          const z = texData[i * 12 + 2];
          if (z > 0.5) green = toScreen(x, texData[i * 12 + 1], z);
          else if (x < 1.0) red = toScreen(x, texData[i * 12 + 1], z);
        }
        if (red && green) {
          offsets = [];
          for (let t = 0.35; t <= 0.65; t += 0.05) {
            offsets.push([red[0] + (green[0] - red[0]) * t, red[1] + (green[1] - red[1]) * t]);
          }
        }
      });
      return offsets;
    });
    expect(midOffsets).not.toBeNull();
    const samples = await samplePixelsAt(page, 'canvas#app', midOffsets!, 'framebuffer');
    const lit = samples.filter((s) => s.r > 20 || s.g > 20);
    expect(lit.length).toBeGreaterThan(0);
    const greenOverRed = lit.filter((s) => s.g > s.r).length;
    const redOverGreen = lit.filter((s) => s.r > s.g).length;
    expect(greenOverRed).toBeGreaterThan(redOverGreen);

    const webglErrors = await getWebGLErrors(page);
    expect(webglErrors.length).toBe(0);
  });
});

test.describe('Points and Lines volumetric depth sorting', () => {
  // The volumetric arm of the sort gate for the two EMISSIVE non-gsplat
  // types. `needsDepthSort` is `normal ∪ volumetric` and the coordinator
  // judges order-dependence on it uniformly for all four geometry types,
  // but every existing volumetric sort assertion filtered
  // `nodeType === 'gsplats'` — so points and lines could have regressed to
  // unsorted volumetric compositing with the suite still green. These two
  // fixtures are geometry-identical to their `normal` twins, leaving the
  // blending mode as the only variable.
  //
  // Lines are not a redundant copy of points: their centers provider hands
  // the SortWorker SEGMENT MIDPOINTS (commit-lines-geometry.ts) rather than
  // element positions, so it is a genuinely separate path into the same gate.
  //
  // Both fixtures come from tests/fixtures/generate_test_data.py and are
  // covered by the global-setup pre-flight, which fails the whole run by
  // name when a declared fixture is missing or half-written — run
  // `pnpm test:generate-fixtures` first.
  test.slow();

  /**
   * Assert that a reversed-declaration volumetric node reaches a
   * back-to-front ordering: the applied permutation departs from identity
   * AND view z is non-decreasing along it.
   *
   * `stride` is the node's element-texture stride in floats: 12 for points,
   * 24 for lines. The monotonicity check must run on the SAME key the
   * SortWorker was handed, so it reconstructs that key from the texture —
   * texel0.xyz (position) for points, and the mean of texel0.xyz (segment
   * start) and texel1.xyz (segment end) for lines, which is the midpoint the
   * lines centers provider registers.
   */
  async function expectVolumetricBackToFront(
    page: import('@playwright/test').Page,
    fixture: string,
    nodeType: 'points' | 'lines',
    stride: number
  ): Promise<void> {
    await page.goto(`/?src=${fixture}&debug&dpr=1`);
    await waitForLuxarReady(page);

    // Committed instances first — the sort lands asynchronously after.
    await page.waitForFunction(
      (t) => {
        const debug = (window as unknown as { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
        if (!debug?.scene) return false;
        let committed = false;
        (debug.scene as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
          const o = obj as {
            userData?: { nodeType?: string };
            geometry?: { instanceCount?: number };
          };
          if (o.userData?.nodeType === t && (o.geometry?.instanceCount ?? 0) > 0) committed = true;
        });
        return committed;
      },
      nodeType,
      { timeout: 30000 }
    );

    // The mode must actually be volumetric — otherwise a fixture that
    // silently lost its blending_mode would make the sort assertion below
    // pass for the wrong reason (as plain `normal`).
    const modes = await page.evaluate((t) => {
      const debug = (window as any).__luxarDebug;
      const out: string[] = [];
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType !== t) return;
        const mat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
        out.push(mat?.userData?.blendingMode);
      });
      return out;
    }, nodeType);
    expect(modes.length).toBeGreaterThan(0);
    for (const m of modes) expect(m).toBe('volumetric');

    // Non-identity: times out against a coordinator that treats volumetric
    // as order-INDEPENDENT for this node type (the regression this guards).
    await page.waitForFunction(
      (t) => {
        const debug = (window as unknown as { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
        if (!debug?.scene) return false;
        let sorted = false;
        (debug.scene as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
          const o = obj as {
            userData?: {
              nodeType?: string;
              visiblePointCount?: number;
              visibleSegmentCount?: number;
            };
            geometry?: {
              attributes?: {
                aSortedIndex?: { array?: ArrayLike<number> };
                aSortedIndexB?: { array?: ArrayLike<number> };
              };
              userData?: { sortedIndexSlot?: 0 | 1 };
            };
          };
          if (o.userData?.nodeType !== t) return;
          const arr =
            o.geometry?.userData?.sortedIndexSlot === 1
              ? o.geometry?.attributes?.aSortedIndexB?.array
              : o.geometry?.attributes?.aSortedIndex?.array;
          const count = o.userData?.visiblePointCount ?? o.userData?.visibleSegmentCount ?? 0;
          if (!arr || count < 2) return;
          for (let i = 0; i < count; i++) {
            if (arr[i] !== i) {
              sorted = true;
              return;
            }
          }
        });
        return sorted;
      },
      nodeType,
      { timeout: 30000 }
    );

    // …and the permutation it settled on is genuinely back-to-front.
    const monotone = await page.evaluate(
      ({ t, s, mid }) => {
        const debug = (window as any).__luxarDebug;
        const results: Array<{ ordering: number[]; viewZs: number[]; ok: boolean }> = [];
        debug.scene.traverse((obj: any) => {
          if (obj.userData?.nodeType !== t) return;
          const count = obj.userData?.visiblePointCount ?? obj.userData?.visibleSegmentCount ?? 0;
          const arr =
            obj.geometry?.userData?.sortedIndexSlot === 1
              ? obj.geometry?.attributes?.aSortedIndexB?.array
              : obj.geometry?.attributes?.aSortedIndex?.array;
          const texData = obj.geometry?.userData?.elementTexture?.image?.data;
          if (!arr || !texData || count < 2) return;
          const mwi = debug.camera.matrixWorldInverse.elements;
          const mw = obj.matrixWorld.elements;
          const viewZof = (x: number, y: number, z: number) => {
            const wx = mw[0] * x + mw[4] * y + mw[8] * z + mw[12];
            const wy = mw[1] * x + mw[5] * y + mw[9] * z + mw[13];
            const wz = mw[2] * x + mw[6] * y + mw[10] * z + mw[14];
            return mwi[2] * wx + mwi[6] * wy + mwi[10] * wz + mwi[14];
          };
          const ordering: number[] = [];
          const viewZs: number[] = [];
          let ok = true;
          let prev = -Infinity;
          for (let j = 0; j < count; j++) {
            const idx = arr[j];
            ordering.push(idx);
            const b = idx * s;
            // The registered sort key: texel0.xyz, or the texel0/texel1
            // midpoint for lines (see the doc comment above).
            const kx = mid ? (texData[b] + texData[b + 4]) / 2 : texData[b];
            const ky = mid ? (texData[b + 1] + texData[b + 5]) / 2 : texData[b + 1];
            const kz = mid ? (texData[b + 2] + texData[b + 6]) / 2 : texData[b + 2];
            const zv = viewZof(kx, ky, kz);
            viewZs.push(zv);
            // Small epsilon: equal-depth elements share a key bucket.
            if (zv < prev - 1e-4) ok = false;
            prev = Math.max(prev, zv);
          }
          results.push({ ordering, viewZs, ok });
        });
        return results;
      },
      { t: nodeType, s: stride, mid: nodeType === 'lines' }
    );
    expect(monotone.length).toBeGreaterThan(0);
    for (const r of monotone) {
      expect(
        r.ok,
        `${nodeType} volumetric ordering ${r.ordering} viewZs ${r.viewZs} not back-to-front`
      ).toBe(true);
    }

    const webglErrors = await getWebGLErrors(page);
    expect(webglErrors.length).toBe(0);
  }

  test('volumetric POINTS settle on a back-to-front ordering', async ({ page }) => {
    await expectVolumetricBackToFront(page, POINTS_VOLUMETRIC_REVERSED_FIXTURE, 'points', 12);
  });

  test('volumetric LINES settle on a back-to-front ordering', async ({ page }) => {
    await expectVolumetricBackToFront(page, LINES_VOLUMETRIC_REVERSED_FIXTURE, 'lines', 24);
  });
});

test.describe('GSplat volumetric mode (emission–absorption)', () => {
  // Fixture: the overlap geometry (red back splat, green front splat,
  // blue off-axis reference) with blending_mode='volumetric' and
  // absorption=1.0. κ is driven at runtime through the REAL material
  // path (updateAbsorption / applyBlendingMode) so I1 and the
  // darkening test compare frames within one page session (same
  // camera, ?dpr=1). See generate_gsplats_volumetric_test() and
  // VOLUMETRIC_BLENDING_SPEC.md §3.3/§8.
  test.slow();

  test.beforeAll(async () => {
    const { existsSync } = await import('node:fs');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');
    const specDir = path.dirname(fileURLToPath(import.meta.url));
    for (const name of [
      'test_gsplats_volumetric.luxar.zarr',
      'test_gsplats_volumetric_reversed.luxar.zarr',
    ]) {
      const fixtureDir = path.resolve(specDir, `../../../tests/fixtures/${name}`);
      if (!existsSync(fixtureDir)) {
        throw new Error(
          `Missing fixture ${fixtureDir} — run \`pnpm test:generate-fixtures\` ` +
            'from packages/luxar-viewer/ first.'
        );
      }
    }
  });

  /** Wait until a gsplats mesh has committed instances. */
  async function waitCommitted(page: import('@playwright/test').Page): Promise<void> {
    await page.waitForFunction(
      () => {
        const debug = (window as unknown as { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
        if (!debug?.scene) return false;
        let committed = false;
        (debug.scene as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
          const o = obj as {
            userData?: { nodeType?: string };
            geometry?: { instanceCount?: number };
          };
          if (o.userData?.nodeType === 'gsplats' && (o.geometry?.instanceCount ?? 0) > 0) {
            committed = true;
          }
        });
        return committed;
      },
      undefined,
      { timeout: 30000 }
    );
  }

  /** Set κ on every gsplat material through the real update path and re-render. */
  function setAbsorption(page: import('@playwright/test').Page, kappa: number): Promise<void> {
    return page.evaluate((k) => {
      const debug = (window as any).__luxarDebug;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'gsplats' && obj.material?.updateAbsorption) {
          obj.material.updateAbsorption(k);
        }
      });
      debug.renderOnce();
    }, kappa);
  }

  test('material carries the volumetric state, define, and uAbsorption from the zarr attr', async ({
    page,
  }) => {
    await page.goto(`/?src=${GSPLAT_VOLUMETRIC_FIXTURE}&debug`);
    await waitForLuxarReady(page);
    await waitCommitted(page);

    const state = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let found: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'gsplats' && obj.material && !found) {
          const m = obj.material;
          found = {
            blending: m.blending,
            blendEquation: m.blendEquation,
            blendSrc: m.blendSrc,
            blendDst: m.blendDst,
            transparent: m.transparent,
            depthTest: m.depthTest,
            depthWrite: m.depthWrite,
            blendingMode: m.userData?.blendingMode,
            hasVolumetricDefine: !!m.defines && 'LUXAR_VOLUMETRIC' in m.defines,
            absorption: m.uniforms?.uAbsorption?.value,
            projectionMode: m.uniforms?.uProjectionMode?.value,
          };
        }
      });
      return found;
    });

    expect(state).not.toBeNull();
    expect(state.blendingMode).toBe('volumetric');
    expect(state.blending).toBe(EXPECTED_STATE.volumetric.blending);
    expect(state.blendEquation).toBe(EXPECTED_STATE.volumetric.blendEquation);
    expect(state.blendSrc).toBe(EXPECTED_STATE.volumetric.blendSrc);
    expect(state.blendDst).toBe(EXPECTED_STATE.volumetric.blendDst);
    expect(state.depthTest).toBe(EXPECTED_STATE.volumetric.depthTest);
    expect(state.depthWrite).toBe(EXPECTED_STATE.volumetric.depthWrite);
    expect(state.transparent).toBe(EXPECTED_STATE.volumetric.transparent);
    expect(state.hasVolumetricDefine).toBe(true); // GLSL backend
    expect(state.absorption).toBe(1.0); // authored zarr attr reached the uniform
    expect(state.projectionMode).toBe(0); // SUM ray-integral, not peak
  });

  test('I1: κ=0 renders pixel-equal to additive (same session, same camera)', async ({ page }) => {
    // The invariant: with absorption 0, τ=0 ⇒ α=0, S=1 — the
    // One/OneMinusSrcAlpha framebuffer arithmetic degenerates to
    // additive's One+One exactly. The comparison happens in ONE page
    // session (same camera pose, same DPR), switching the material via
    // the real applyBlendingMode path. Per-channel tolerance 2 absorbs
    // TAA/dither noise between frames; the additive limit itself is
    // exact in the blend math (canvas is alpha:false, so RGB readback
    // is unaffected by the differing destination alpha).
    await page.goto(`/?src=${GSPLAT_VOLUMETRIC_FIXTURE}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitCommitted(page);
    await waitForNextRender(page, 5);

    const offsets: Array<[number, number]> = [];
    for (let gx = 0.15; gx <= 0.85; gx += 0.05) {
      for (let gy = 0.25; gy <= 0.75; gy += 0.05) {
        offsets.push([gx, gy]);
      }
    }

    await setAbsorption(page, 0);
    await waitForNextRender(page, 5);
    const volumetricK0 = await samplePixelsAt(page, 'canvas#app', offsets, 'framebuffer');

    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'gsplats' && obj.material?.applyBlendingMode) {
          obj.material.applyBlendingMode('additive');
        }
      });
      debug.renderOnce();
    });
    await waitForNextRender(page, 5);
    const additive = await samplePixelsAt(page, 'canvas#app', offsets, 'framebuffer');

    // Non-vacuous: the scene actually renders content.
    const lit = additive.filter((s) => s.r + s.g + s.b > 30);
    expect(lit.length, 'additive frame rendered black — fixture/camera broke').toBeGreaterThan(0);

    for (let i = 0; i < offsets.length; i++) {
      const a = volumetricK0[i];
      const b = additive[i];
      expect(
        Math.abs(a.r - b.r) + Math.abs(a.g - b.g) + Math.abs(a.b - b.b),
        `sample ${i} at (${offsets[i][0]},${offsets[i][1]}): κ=0 ${JSON.stringify(a)} vs additive ${JSON.stringify(b)}`
      ).toBeLessThanOrEqual(6);
    }
  });

  test('absorption darkens the scene: κ=5 accumulates strictly less light than κ=0', async ({
    page,
  }) => {
    // Emission–absorption bounds accumulated radiance: raising κ both
    // screens each splat's own emission (S(τ)<1) and attenuates what is
    // behind (1−e^(−τ) destination factor). Total sampled luminance
    // must drop measurably — a pure-additive (κ-ignoring) regression
    // would keep the two frames equal.
    await page.goto(`/?src=${GSPLAT_VOLUMETRIC_FIXTURE}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitCommitted(page);
    await waitForNextRender(page, 5);

    const offsets: Array<[number, number]> = [];
    for (let gx = 0.15; gx <= 0.85; gx += 0.05) {
      for (let gy = 0.25; gy <= 0.75; gy += 0.05) {
        offsets.push([gx, gy]);
      }
    }

    await setAbsorption(page, 0);
    await waitForNextRender(page, 5);
    const bright = await samplePixelsAt(page, 'canvas#app', offsets, 'framebuffer');

    await setAbsorption(page, 5);
    await waitForNextRender(page, 5);
    const absorbed = await samplePixelsAt(page, 'canvas#app', offsets, 'framebuffer');

    const sum = (xs: Array<{ r: number; g: number; b: number }>) =>
      xs.reduce((acc, s) => acc + s.r + s.g + s.b, 0);
    const sumBright = sum(bright);
    const sumAbsorbed = sum(absorbed);
    expect(sumBright, 'κ=0 frame rendered black — fixture/camera broke').toBeGreaterThan(1000);
    expect(sumAbsorbed).toBeLessThan(sumBright * 0.9);
  });

  test('depth sort engages for volumetric: reversed fixture settles non-identity + back-to-front', async ({
    page,
  }) => {
    // The volumetric twin of the Phase-2 gate (fail-first evidence at
    // the unit level: depth-sort-coordinator.test.ts fails against an
    // isNormalMode-gated coordinator). The reversed fixture's identity
    // ordering is not back-to-front under the auto-framed camera, so
    // this times out unless needsDepthSort routes volumetric commits
    // through the SortWorker.
    await page.goto(`/?src=${GSPLAT_VOLUMETRIC_REVERSED_FIXTURE}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitCommitted(page);

    await page.waitForFunction(
      () => {
        const debug = (window as unknown as { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
        if (!debug?.scene) return false;
        let sorted = false;
        (debug.scene as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
          const o = obj as {
            userData?: { nodeType?: string; visibleSplatCount?: number };
            geometry?: {
              attributes?: {
                aSortedIndex?: { array?: ArrayLike<number> };
                aSortedIndexB?: { array?: ArrayLike<number> };
              };
              userData?: { sortedIndexSlot?: 0 | 1 };
            };
          };
          if (o.userData?.nodeType !== 'gsplats') return;
          const arr =
            o.geometry?.userData?.sortedIndexSlot === 1
              ? o.geometry?.attributes?.aSortedIndexB?.array
              : o.geometry?.attributes?.aSortedIndex?.array;
          const count = o.userData?.visibleSplatCount ?? 0;
          if (!arr || count < 2) return;
          for (let i = 0; i < count; i++) {
            if (arr[i] !== i) {
              sorted = true;
              return;
            }
          }
        });
        return sorted;
      },
      undefined,
      { timeout: 30000 }
    );

    const monotone = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const results: Array<{ ordering: number[]; ok: boolean }> = [];
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType !== 'gsplats') return;
        const count = obj.userData?.visibleSplatCount ?? 0;
        const arr =
          obj.geometry?.userData?.sortedIndexSlot === 1
            ? obj.geometry?.attributes?.aSortedIndexB?.array
            : obj.geometry?.attributes?.aSortedIndex?.array;
        const texData = obj.geometry?.userData?.elementTexture?.image?.data;
        if (!arr || !texData || count < 2) return;
        const mwi = debug.camera.matrixWorldInverse.elements;
        const mw = obj.matrixWorld.elements;
        const ordering: number[] = [];
        let ok = true;
        let prev = -Infinity;
        for (let j = 0; j < count; j++) {
          const idx = arr[j];
          ordering.push(idx);
          const x = texData[idx * 16];
          const y = texData[idx * 16 + 1];
          const z = texData[idx * 16 + 2];
          const wx = mw[0] * x + mw[4] * y + mw[8] * z + mw[12];
          const wy = mw[1] * x + mw[5] * y + mw[9] * z + mw[13];
          const wz = mw[2] * x + mw[6] * y + mw[10] * z + mw[14];
          const zv = mwi[2] * wx + mwi[6] * wy + mwi[10] * wz + mwi[14];
          if (zv < prev - 1e-4) ok = false;
          prev = Math.max(prev, zv);
        }
        results.push({ ordering, ok });
      });
      return results;
    });
    expect(monotone.length).toBeGreaterThan(0);
    for (const r of monotone) {
      expect(r.ok, `volumetric ordering ${r.ordering} not back-to-front`).toBe(true);
    }

    const webglErrors = await getWebGLErrors(page);
    expect(webglErrors.length).toBe(0);
  });
});

test.describe('GSplat RGBA per-element opacity (occlusion)', () => {
  // Fixture: a bright white back splat, a BLACK high-α (0.95) front splat
  // overlapping it in screen space, and a white reference off-axis. The
  // front splat emits no light — so it only matters through the alpha
  // channel. See generate_gsplats_rgba_occlusion_test() and
  // VOLUMETRIC_BLENDING_SPEC.md §5.4.1.
  test.slow();

  test.beforeAll(async () => {
    const { existsSync } = await import('node:fs');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');
    const specDir = path.dirname(fileURLToPath(import.meta.url));
    const fixtureDir = path.resolve(
      specDir,
      '../../../tests/fixtures/test_gsplats_rgba_occlusion.luxar.zarr'
    );
    if (!existsSync(fixtureDir)) {
      throw new Error(`Missing fixture ${fixtureDir} — run \`pnpm test:generate-fixtures\` first.`);
    }
  });

  async function waitCommitted(page: import('@playwright/test').Page): Promise<void> {
    await page.waitForFunction(
      () => {
        const debug = (window as unknown as { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
        if (!debug?.scene) return false;
        let committed = false;
        (debug.scene as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
          const o = obj as {
            userData?: { nodeType?: string };
            geometry?: { instanceCount?: number };
          };
          if (o.userData?.nodeType === 'gsplats' && (o.geometry?.instanceCount ?? 0) > 0) {
            committed = true;
          }
        });
        return committed;
      },
      undefined,
      { timeout: 30000 }
    );
  }

  test('RGBA colors reach the material as a 4-component layout (uHasElementAlpha=1)', async ({
    page,
  }) => {
    await page.goto(`/?src=${GSPLAT_RGBA_OCCLUSION_FIXTURE}&debug`);
    await waitForLuxarReady(page);
    await waitCommitted(page);

    const hasAlpha = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let v: number | undefined;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'gsplats' && obj.material && v === undefined) {
          v = obj.material.uniforms?.uHasElementAlpha?.value;
        }
      });
      return v;
    });
    // The loader read (N, 4) from the zarr shape and declared it to the
    // material — the gate that enables the volumetric α → optical-depth map.
    expect(hasAlpha).toBe(1);
  });

  test('the black high-α occluder absorbs under κ (its darkening is alpha-driven)', async ({
    page,
  }) => {
    // The front splat is BLACK, so it contributes no emission in either
    // frame — its only effect on the picture is ABSORPTION, which in
    // volumetric is driven by its optical depth τ = κ·opacity·w(α)·rayMass.
    // Raising κ from 0 (τ=0, no occlusion — the additive limit) to 5 turns
    // the black splat into a real occluder that removes the back splat's
    // light where they overlap. Total sampled luminance must drop. A
    // regression that ignored the alpha channel would still darken (rayMass
    // alone), so this is paired with the uHasElementAlpha=1 test and the
    // TSL↔GLSL parity of the w(α) fold to attribute the effect to alpha.
    await page.goto(`/?src=${GSPLAT_RGBA_OCCLUSION_FIXTURE}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitCommitted(page);
    await waitForNextRender(page, 5);

    // Overlap band (screen center) where the black front splat sits over
    // the bright back splat.
    const offsets: Array<[number, number]> = [];
    for (let gx = 0.3; gx <= 0.7; gx += 0.05) {
      for (let gy = 0.35; gy <= 0.65; gy += 0.05) {
        offsets.push([gx, gy]);
      }
    }

    const setAbsorption = (k: number): Promise<void> =>
      page.evaluate((kappa) => {
        const debug = (window as any).__luxarDebug;
        debug.scene.traverse((obj: any) => {
          if (obj.userData?.nodeType === 'gsplats' && obj.material?.updateAbsorption) {
            obj.material.updateAbsorption(kappa);
          }
        });
        debug.renderOnce();
      }, k);

    const sum = (xs: Array<{ r: number; g: number; b: number }>) =>
      xs.reduce((acc, s) => acc + s.r + s.g + s.b, 0);

    await setAbsorption(0); // additive limit — the black splat can't occlude
    await waitForNextRender(page, 5);
    const bright = await samplePixelsAt(page, 'canvas#app', offsets, 'framebuffer');
    const sumBright = sum(bright);

    await setAbsorption(5); // the black splat becomes a real occluder
    await waitForNextRender(page, 5);
    const absorbed = await samplePixelsAt(page, 'canvas#app', offsets, 'framebuffer');
    const sumAbsorbed = sum(absorbed);

    expect(sumBright, 'κ=0 frame rendered black — fixture/camera broke').toBeGreaterThan(1000);
    expect(sumAbsorbed).toBeLessThan(sumBright * 0.9);
  });
});
