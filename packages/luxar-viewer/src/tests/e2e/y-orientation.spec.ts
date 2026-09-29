/**
 * Y-orientation pin for `renderToImageData()` across renderer backends.
 *
 * The viewer's renderers use different readback conventions:
 *   - `THREE.WebGLRenderer`: row 0 of an FBO is at the bottom of the
 *     viewport. `readPixelsCompactAsync` flips reads into top-down
 *     order for `ImageData` / canonical-export.
 *   - `WebGPURenderer`: native WebGPU reads top-down; its WebGL2 fallback
 *     reads bottom-up. Both use top-down shader sampling, which is a
 *     separate convention.
 *
 * The test renders the same scene through `?renderer=webgl`,
 * `?renderer=webgpu&webgpuForceWebgl`, and native `?renderer=webgpu`, compares the resulting
 * `renderToImageData()` outputs as per-row luminance profiles
 * (resilient to small canvas-size differences between contexts),
 * and asserts they match. If a future change breaks the orientation
 * contract on either side, the test fails with a clear "much closer
 * when flipped" signal pointing at the side that flipped.
 *
 * @see fullscreen/geometry.ts — UV-flip table keyed on `framebufferYDown`
 * @see hdr/pixel-utils.ts::readPixelsCompactAsync — backend-aware readback
 */

import { test, expect, type Page } from './fixtures';
import { probeWebGPUBackend, waitForLuxarReady, waitForPointsLoaded } from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';
const LABELLED_DATASET =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_labelled_points.luxar.zarr';

const N_BUCKETS = 32;
/**
 * 8/255 ≈ 3% of dynamic range — tolerant of tone-mapping / FXAA
 * precision drift but well below the half-image-flip signature
 * (~13 in practice when the orientation regresses).
 */
const PROFILE_RMSE_TOL = 8;

type Pixels = { width: number; height: number; data: Uint8ClampedArray };

async function exportPixels(page: Page): Promise<Pixels> {
  const result = await page.evaluate(async () => {
    const dbg = (
      window as unknown as {
        __luxarDebug: { postProcessing: { renderToImageData: () => Promise<ImageData> } };
      }
    ).__luxarDebug;
    const img: ImageData = await dbg.postProcessing.renderToImageData();
    return {
      width: img.width,
      height: img.height,
      data: Array.from(img.data) as number[],
    };
  });
  return {
    width: result.width,
    height: result.height,
    data: new Uint8ClampedArray(result.data),
  };
}

function verticalLuminanceProfile(p: Pixels): number[] {
  const profile = new Array<number>(N_BUCKETS).fill(0);
  for (let b = 0; b < N_BUCKETS; b++) {
    const y = Math.min(p.height - 1, Math.floor(((b + 0.5) / N_BUCKETS) * p.height));
    let sum = 0;
    for (let x = 0; x < p.width; x++) {
      const o = (y * p.width + x) * 4;
      sum += 0.2126 * p.data[o] + 0.7152 * p.data[o + 1] + 0.0722 * p.data[o + 2];
    }
    profile[b] = sum / p.width;
  }
  return profile;
}

function profileRMSE(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s / a.length);
}

function flipProfile(p: number[]): number[] {
  return [...p].reverse();
}

async function load(page: Page, url: string): Promise<void> {
  await page.goto(url);
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page);
}

async function assertOrientationParity(
  pageA: Page,
  pageB: Page,
  labelA: string,
  labelB: string
): Promise<void> {
  const a = await exportPixels(pageA);
  const b = await exportPixels(pageB);

  const pa = verticalLuminanceProfile(a);
  const pb = verticalLuminanceProfile(b);
  const direct = profileRMSE(pa, pb);
  const flipped = profileRMSE(pa, flipProfile(pb));

  expect(
    direct,
    `Vertical luminance profile mismatch between ${labelA} (size ${a.width}×${a.height}) and ` +
      `${labelB} (size ${b.width}×${b.height}). RMSE direct=${direct.toFixed(2)}, ` +
      `RMSE Y-flipped=${flipped.toFixed(2)}. If "flipped" is much smaller than "direct", ` +
      'one backend has the wrong Y orientation.'
  ).toBeLessThan(PROFILE_RMSE_TOL);
}

test.describe('Y-orientation contract — renderToImageData cross-backend parity', () => {
  test('WebGL2 vs WebGPURenderer (forceWebGL) match in orientation', async ({ browser }) => {
    const ctxA = await browser.newContext({ viewport: { width: 512, height: 384 } });
    const ctxB = await browser.newContext({ viewport: { width: 512, height: 384 } });
    const pageA = await ctxA.newPage();
    const pageB = await ctxB.newPage();
    try {
      await load(pageA, `/?renderer=webgl&src=${DATASET}&debug`);
      await load(pageB, `/?renderer=webgpu&webgpuForceWebgl&src=${DATASET}&debug`);
      await assertOrientationParity(
        pageA,
        pageB,
        '?renderer=webgl',
        '?renderer=webgpu&webgpuForceWebgl'
      );
    } finally {
      await ctxA.close();
      await ctxB.close();
    }
  });

  test('WebGL2 vs WebGPURenderer (native if available) match in orientation', async ({
    browser,
  }) => {
    const ctxA = await browser.newContext({ viewport: { width: 512, height: 384 } });
    const ctxB = await browser.newContext({ viewport: { width: 512, height: 384 } });
    const pageA = await ctxA.newPage();
    const pageB = await ctxB.newPage();
    try {
      await load(pageA, `/?renderer=webgl&src=${DATASET}&debug`);
      await load(pageB, `/?renderer=webgpu&src=${DATASET}&debug`);
      // Without a real WebGPU adapter (the headless-CI norm),
      // ?renderer=webgpu silently runs the WebGL2 fallback and this
      // test becomes a vacuous duplicate of the forceWebGL one above —
      // while pixel-utils load-bears on this spec pinning the NATIVE
      // readback orientation. Skip explicitly instead, on the same
      // shared gate webgpu-native-smoke.spec.ts uses.
      const { isNative } = await probeWebGPUBackend(pageB);
      test.skip(
        !isNative,
        'No native WebGPU adapter — fallback path is covered by the forceWebGL test'
      );
      await assertOrientationParity(pageA, pageB, '?renderer=webgl', '?renderer=webgpu');
    } finally {
      await ctxA.close();
      await ctxB.close();
    }
  });

  for (const backend of ['native', 'forceWebGL'] as const) {
    test(`${backend} WebGPURenderer picks an off-centre labelled point`, async ({ page }) => {
      const forced = backend === 'forceWebGL' ? '&webgpuForceWebgl' : '';
      await load(page, `/?renderer=webgpu${forced}&src=${LABELLED_DATASET}&debug`);
      if (backend === 'native') {
        test.skip(!(await probeWebGPUBackend(page)).isNative, 'No native WebGPU adapter');
      }

      const target = await page.evaluate(() => {
        const dbg = (window as any).__luxarDebug;
        const camera = dbg.camera;
        const ndc = camera.position.clone().set(5, 5, 0).project(camera);
        const rect = dbg.renderer.domElement.getBoundingClientRect();
        return {
          x: rect.left + ((ndc.x + 1) / 2) * rect.width,
          y: rect.top + ((1 - ndc.y) / 2) * rect.height,
          centreY: rect.top + rect.height / 2,
        };
      });
      expect(Math.abs(target.y - target.centreY)).toBeGreaterThan(15);
      await page.mouse.move(target.x, target.y);
      await expect(page.locator('[data-overlay-name="__hover_text"]')).toHaveText('Point 3', {
        timeout: 10000,
      });
    });
  }
});
