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
  samplePixelsAt,
} from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/rendering_modes_example.luxar.zarr';
const MULTI_DATASET = 'http://localhost:9000/datasets/examples/multiple_objects_example.luxar.zarr';
const GSPLAT_OVERLAP_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_normal_overlap.luxar.zarr';

test.describe('Blending Modes', () => {
  // The blending-mode datasets contain multiple groups (5+ point clouds) and
  // render with software-accelerated WebGL on most CI/test machines, where
  // FPS sits at ~3–10. The default 60s budget is marginal once data loading
  // plus several render passes are added; bump to 120s so we measure
  // correctness, not the test runner's tolerance for slow blits.
  test.slow();

  test('should load dataset with initial blending modes from zarr metadata', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    // Get all materials' blending state from the scene
    const blendingStates = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const states: { name: string; blending: number; depthTest: boolean; depthWrite: boolean }[] =
        [];

      debug.scene.traverse((obj: any) => {
        if ((obj.userData?.nodeType === 'points' || obj.type === 'Mesh') && obj.material) {
          states.push({
            name: obj.name || 'unnamed',
            blending: obj.material.blending,
            depthTest: obj.material.depthTest,
            depthWrite: obj.material.depthWrite,
          });
        }
      });

      return states;
    });

    expect(blendingStates.length).toBeGreaterThan(0);

    // All materials should have a valid blending mode (THREE.js enum values)
    // NormalBlending=1, AdditiveBlending=2, CustomBlending=5
    for (const state of blendingStates) {
      expect([1, 2, 5]).toContain(state.blending);
    }
  });

  test('should have different depth behavior for additive vs normal blending', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    // Find additive and normal blended objects
    const modeInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let additive: any = null;
      let normal: any = null;

      debug.scene.traverse((obj: any) => {
        // Points render as `THREE.Mesh + userData.nodeType === 'points'`.
        const isPoints = obj.userData?.nodeType === 'points';
        if (!isPoints || !obj.material) return;

        // THREE.AdditiveBlending = 2
        if (obj.material.blending === 2 && !additive) {
          additive = {
            name: obj.name,
            depthTest: obj.material.depthTest,
            depthWrite: obj.material.depthWrite,
            transparent: obj.material.transparent,
          };
        }
        // THREE.NormalBlending = 1
        if (obj.material.blending === 1 && !normal) {
          normal = {
            name: obj.name,
            depthTest: obj.material.depthTest,
            depthWrite: obj.material.depthWrite,
            transparent: obj.material.transparent,
          };
        }
      });

      return { additive, normal };
    });

    // If both modes exist in this dataset, verify their depth behavior differs
    if (modeInfo.additive && modeInfo.normal) {
      // Additive blending typically has depthTest=false, depthWrite=false
      expect(modeInfo.additive.depthWrite).toBe(false);

      // Normal blending typically has depthTest=true
      expect(modeInfo.normal.depthTest).toBe(true);
    }

    // At least one mode should exist
    expect(modeInfo.additive || modeInfo.normal).toBeTruthy();
  });

  test('should render without WebGL errors for all blending modes', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
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
    await page.goto(`/?src=${MULTI_DATASET}&debug`);
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
    await page.goto(`/?src=${DATASET}&debug`);
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

test.describe('GSplat normal mode (premultiplied coverage alpha)', () => {
  // Fixture: two large overlapping splats (red behind, green in front,
  // storage order = back-to-front for the default camera) + one small
  // blue reference splat, blending_mode='normal', opacity=0.5. See
  // generate_gsplats_normal_overlap_test() and
  // GSPLAT_DEPTH_SORTING_SPEC.md §3 (Phase 0).
  test.slow();

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
    // Suppress the one-time control-rail hint popup: its gray pixels
    // (~rgb 42,45,49) sit inside the sampling grid and would satisfy a
    // naive r>30 && g>30 test — the discriminator below also excludes
    // grays, but keeping the frame clean makes failures readable.
    await page.addInitScript(() => {
      localStorage.setItem('luxar-control-rail-hint-dismissed', '1');
    });
    // ?dpr=1 pins the pixel ratio for deterministic sampling.
    await page.goto(`/?src=${GSPLAT_OVERLAP_FIXTURE}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForGSplatsCommitted(page);
    await waitForNextRender(page, 5);

    // Dense grid over the central region where the two big splats live.
    const offsets: Array<[number, number]> = [];
    for (let gx = 0.15; gx <= 0.85; gx += 0.05) {
      for (let gy = 0.25; gy <= 0.75; gy += 0.05) {
        offsets.push([gx, gy]);
      }
    }
    const samples = await samplePixelsAt(page, 'canvas', offsets);

    const redDominant = samples.filter((s) => s.r > 40 && s.r > 2 * s.g);
    const greenDominant = samples.filter((s) => s.g > 40 && s.g > 2 * s.r);
    // The alpha-over discriminator: pre-fix, gsplat 'normal' emitted
    // alpha=1.0, so the front (green) splat fully REPLACED the back
    // (red) splat wherever it covered — no pixel could carry both
    // channels. With premultiplied coverage alpha at opacity 0.5 the
    // overlap composites green over red and both channels survive.
    // The b < min(r,g)/2 term excludes NEUTRAL pixels (UI chrome,
    // grays): the fixture's red+green overlap has near-zero blue, so a
    // gray popup pixel (r≈g≈b) can never satisfy this vacuously.
    const mixed = samples.filter((s) => s.r > 30 && s.g > 30 && s.b < Math.min(s.r, s.g) / 2);

    // Both splats render…
    expect(redDominant.length).toBeGreaterThan(0);
    expect(greenDominant.length).toBeGreaterThan(0);
    // …and the background shows through the overlap (fails pre-fix).
    expect(mixed.length).toBeGreaterThan(0);
  });
});
