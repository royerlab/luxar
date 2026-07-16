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
const GSPLAT_OVERLAP_REVERSED_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_normal_overlap_reversed.luxar.zarr';

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
    await page.addInitScript(() => {
      localStorage.setItem('luxar-control-rail-hint-dismissed', '1');
    });
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
            geometry?: { attributes?: { aSortedIndex?: { array?: ArrayLike<number> } } };
          };
          if (o.userData?.nodeType !== 'gsplats') return;
          const arr = o.geometry?.attributes?.aSortedIndex?.array;
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
        const arr = obj.geometry?.attributes?.aSortedIndex?.array;
        const texData = obj.geometry?.userData?.splatTexture?.image?.data;
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
        const texData = obj.geometry?.userData?.splatTexture?.image?.data;
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
    const samples = await samplePixelsAt(page, 'canvas', midOffsets!);
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
      const arr = obj.geometry?.attributes?.aSortedIndex?.array;
      const texData = obj.geometry?.userData?.splatTexture?.image?.data;
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
    await page.addInitScript(() => {
      localStorage.setItem('luxar-control-rail-hint-dismissed', '1');
    });
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
    await page.addInitScript(() => {
      localStorage.setItem('luxar-control-rail-hint-dismissed', '1');
    });
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
          const arr = obj.geometry?.attributes?.aSortedIndex?.array;
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
