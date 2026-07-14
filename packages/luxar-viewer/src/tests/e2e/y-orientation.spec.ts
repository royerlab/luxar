/**
 * Y-orientation pin for `renderToImageData()` across renderer backends.
 *
 * The viewer's two renderer paths follow opposite framebuffer-Y
 * conventions internally:
 *   - `THREE.WebGLRenderer`: row 0 of an FBO is at the bottom of the
 *     viewport. `readPixelsCompactAsync` flips reads into top-down
 *     order for `ImageData` / canonical-export.
 *   - `WebGPURenderer`: Three.js presents a top-down sampling
 *     convention to TSL `texture(...).sample(uv)` (handled by the
 *     UV-flip in {@link createFullscreenTriangleGeometry}), but
 *     `readRenderTargetPixelsAsync` still returns rows in bottom-up
 *     `gl.readPixels`-compat order on both its real-WebGPU and
 *     WebGL2 backends — verified empirically and pinned by this
 *     spec.
 *
 * The test renders the same scene through `?renderer=webgl` and
 * `?renderer=webgpu&webgpu-force-webgl`, compares the resulting
 * `renderToImageData()` outputs as per-row luminance profiles
 * (resilient to small canvas-size differences between contexts),
 * and asserts they match. If a future change breaks the orientation
 * contract on either side, the test fails with a clear "much closer
 * when flipped" signal pointing at the side that flipped.
 *
 * @see fullscreen/geometry.ts — UV-flip table keyed on `framebufferYDown`
 * @see hdr/pixel-utils.ts::readPixelsCompactAsync — the unconditional flip
 */

import { test, expect, type Page } from './fixtures';
import { waitForLuxarReady, waitForPointsLoaded } from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';

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
      await load(pageB, `/?renderer=webgpu&webgpu-force-webgl&src=${DATASET}&debug`);
      await assertOrientationParity(
        pageA,
        pageB,
        '?renderer=webgl',
        '?renderer=webgpu&webgpu-force-webgl'
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
      // readback orientation. Skip explicitly instead (same pattern as
      // webgpu-native-smoke.spec.ts).
      const isNative = await pageB.evaluate(() => {
        const dbg = (
          window as unknown as {
            __luxarDebug?: {
              app?: { sceneManager?: { capabilities?: { apiSurface?: string } } };
              renderer?: { backend?: { isWebGLBackend?: boolean } };
            };
          }
        ).__luxarDebug;
        return (
          dbg?.app?.sceneManager?.capabilities?.apiSurface === 'webgpu' &&
          dbg?.renderer?.backend?.isWebGLBackend !== true
        );
      });
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
});
