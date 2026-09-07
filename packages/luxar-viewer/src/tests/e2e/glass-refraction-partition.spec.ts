/**
 * `refract_data` glass and the depth partition of the data around it
 * (spec MESH_PHYSICAL_MATERIALS §3.4, Phase 3), pixel-tested on the lens example.
 *
 * Two probes on `mesh_glass_lens_example.luxar.zarr`, each with the refracting lens
 * shown and then hidden through the Layers panel:
 *
 *   - BEHIND — from the baked camera the lens sits in front of the lattice; the pixel at
 *     its centre must CHANGE when the lens is hidden (the glass refracts the data).
 *   - FRONT — from behind the spheres a lattice point lies between the camera and the
 *     lens at the canvas centre; that pixel must NOT change when the lens is hidden (the
 *     partition draws data in front of the glass crisp on top, instead of letting the
 *     glass paint over it). Before the partition this probe read the glass's refracted
 *     image of the lattice, a difference of 60+/255 on both backends.
 *
 * Both renderers run: `?renderer=webgpu` in headless Chromium lands on the WebGL2
 * fallback, which still exercises the TSL guard, the depth-texture Y flip and the
 * four-pass sequence. Native WebGPU is verified out of band with
 * `scripts/ab-webgpu-vs-webgl.mjs` in the system Chrome.
 */

import { test, expect } from './fixtures';
import {
  openLayersPanel,
  placeCameraAt,
  samplePixelsAt,
  waitForLuxarReady,
  waitForNextRender,
  waitForRenderStable,
  type SampledPixel,
} from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/mesh_glass_lens_example.luxar.zarr';
const CANVAS = 'canvas#app';

/** Largest per-channel difference between two samples, 0–255. */
function maxChannelDelta(a: SampledPixel, b: SampledPixel): number {
  return Math.max(Math.abs(a.r - b.r), Math.abs(a.g - b.g), Math.abs(a.b - b.b));
}

/** Toggle the lens layer's eye button (the real user path). */
async function toggleLens(page: Parameters<typeof openLayersPanel>[0]): Promise<void> {
  await openLayersPanel(page);
  const row = page.locator('.luxar-layer-row', {
    has: page.locator('.luxar-layer-row__name', { hasText: /^lens$/ }),
  });
  await expect(row).toHaveCount(1);
  await row.locator('.luxar-layer-row__eye').click();
  await waitForNextRender(page);
  await waitForRenderStable(page);
}

/** Where the refracting lens's centre lands on the canvas, as fractions of its size. */
async function lensCentreOnCanvas(
  page: Parameters<typeof openLayersPanel>[0]
): Promise<[number, number]> {
  return page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    const scene = debug.app.sceneManager.scene;
    let lens: any = null;
    scene.traverse((o: any) => {
      if (o.isMesh && o.material?.userData?.drawAfterEmissive && o.name?.endsWith('lens')) lens = o;
    });
    if (!lens) throw new Error('no refracting lens mesh in the scene');
    if (!lens.geometry.boundingSphere) lens.geometry.computeBoundingSphere();
    const centre = lens.geometry.boundingSphere.center.clone().applyMatrix4(lens.matrixWorld);
    centre.project(debug.camera);
    return [(centre.x + 1) / 2, (1 - centre.y) / 2] as [number, number];
  });
}

async function splitFrames(page: Parameters<typeof openLayersPanel>[0]): Promise<number> {
  return page.evaluate(
    () =>
      (window as any).__luxarDebug.app.sceneManager.postProcessing.refractionSplit?.framesSplit ??
      -1
  );
}

for (const renderer of ['webgl', 'webgpu'] as const) {
  test.describe(`refract_data partition (${renderer})`, () => {
    test.beforeEach(async ({ page }) => {
      await page.goto(
        `/?src=${DATASET}&debug&dpr=1${renderer === 'webgpu' ? '&renderer=webgpu' : ''}`
      );
      await waitForLuxarReady(page);
      // The lens is a mesh layer committed after the lattice; wait for its row.
      await openLayersPanel(page);
      await expect(page.locator('.luxar-layer-row__name', { hasText: /^lens$/ })).toHaveCount(1, {
        timeout: 30000,
      });
      await waitForRenderStable(page, 3, 20000);
    });

    test('the lens refracts the lattice behind it', async ({ page }) => {
      const centre = await lensCentreOnCanvas(page);
      const [withLens] = await samplePixelsAt(page, CANVAS, [centre], 'framebuffer');
      expect(await splitFrames(page)).toBeGreaterThan(0); // the split really ran

      await toggleLens(page);
      const [noLens] = await samplePixelsAt(page, CANVAS, [centre], 'framebuffer');

      // The glass composites something the bare lattice does not have at that pixel
      // (its refracted, magnified image of the rows behind it).
      expect(maxChannelDelta(withLens, noLens)).toBeGreaterThanOrEqual(24);
    });

    test('a lattice point in front of the lens stays crisp and unrefracted', async ({ page }) => {
      // Camera behind the spheres looking back through the lens centre (-3, 0, 1): the
      // lattice point (-3.214, 0, -2) lies on that line, between the camera and the glass.
      const placed = await placeCameraAt(
        page,
        { x: -3.78, y: 0, z: -10 },
        { target: { x: -3, y: 0, z: 1 } }
      );
      expect(placed?.viaOrbitControls).toBe(true);
      await waitForRenderStable(page);
      // A 3x3 neighbourhood of the canvas centre, so a sub-pixel misplacement of the
      // sprite cannot turn the probe into black-vs-black.
      const offsets: Array<[number, number]> = [];
      for (const dx of [-0.002, 0, 0.002])
        for (const dy of [-0.002, 0, 0.002]) offsets.push([0.5 + dx, 0.5 + dy]);

      const withLens = await samplePixelsAt(page, CANVAS, offsets, 'framebuffer');
      await toggleLens(page);
      const noLens = await samplePixelsAt(page, CANVAS, offsets, 'framebuffer');

      // The probe is ON the point: lit with the lens hidden (no vacuous pass)…
      const brightest = Math.max(...noLens.map((p) => p.r + p.g + p.b));
      expect(brightest).toBeGreaterThan(90);
      // …and the glass behind it changes nothing about it.
      const worst = Math.max(...withLens.map((p, i) => maxChannelDelta(p, noLens[i])));
      expect(worst).toBeLessThanOrEqual(24);
    });
  });
}
