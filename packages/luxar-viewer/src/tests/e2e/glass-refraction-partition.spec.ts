/**
 * `refract_data` glass and the depth partition of the data around it
 * (spec MESH_PHYSICAL_MATERIALS §3.4, Phase 3), pixel-tested on the lens example.
 *
 * Two probes on `mesh_glass_lens_example.luxar.zarr`, each with the refracting lens
 * shown and then hidden through the Layers panel:
 *
 *   - BEHIND — from the baked camera the lens sits in front of the lattice; the pixel at
 *     its centre must CHANGE when the lens is hidden (the glass refracts the data).
 *   - FRONT — from behind the spheres the lattice lies between the camera and the lens.
 *     Every lattice point projecting inside the lens disk is sampled with the lens shown
 *     and hidden. With the partition the points are drawn crisp on top of the glass, so
 *     they read the same either way; without it the glass paints its refracted,
 *     magnified image of the lattice over them. Measured on both backends, 22 points:
 *     partition on — mean difference 7/255, one point over 24 (a spot where the glass
 *     refracts the bright neighbouring sphere underneath the point); partition off —
 *     mean 51, nine points over 24, max 207. The thresholds sit between the two.
 *
 * Both renderers run: `?renderer=webgpu` in headless Chromium lands on the WebGL2
 * fallback, which still exercises the TSL guard, the depth-texture Y flip and the
 * four-pass sequence. Native WebGPU is verified out of band with
 * `scripts/ab-webgpu-vs-webgl.mjs` in the system Chrome.
 */

import type { Page } from '@playwright/test';
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

/** The example's lattice: 15 × 15 points on three planes (mesh_glass_lens_example.py). */
const LATTICE_AXIS = Array.from({ length: 15 }, (_, i) => -4.5 + (9 * i) / 14);
const LATTICE_Z = [-5, -3.5, -2];

/** Largest per-channel difference between two samples, 0–255. */
function maxChannelDelta(a: SampledPixel, b: SampledPixel): number {
  return Math.max(Math.abs(a.r - b.r), Math.abs(a.g - b.g), Math.abs(a.b - b.b));
}

/** Toggle the lens layer's eye button (the real user path). */
async function toggleLens(page: Page): Promise<void> {
  await openLayersPanel(page);
  const row = page.locator('.luxar-layer-row', {
    has: page.locator('.luxar-layer-row__name', { hasText: /^lens$/ }),
  });
  await expect(row).toHaveCount(1);
  await row.locator('.luxar-layer-row__eye').click();
  await waitForNextRender(page);
  await waitForRenderStable(page);
}

/**
 * Screen positions, as canvas fractions, of the refracting lens's centre and of every
 * lattice point that projects inside 80% of its disk (the rim is excluded so a sprite
 * straddling the silhouette cannot blur the verdict).
 */
async function lensAndLatticeOnCanvas(
  page: Page,
  lattice: { axis: number[]; z: number[] }
): Promise<{ centre: [number, number]; inside: Array<[number, number]> }> {
  return page.evaluate(
    ({ axis, zs }) => {
      const debug = (window as any).__luxarDebug;
      const scene = debug.app.sceneManager.scene;
      const cam = debug.camera;
      let lens: any = null;
      scene.traverse((o: any) => {
        if (o.isMesh && o.material?.userData?.drawAfterEmissive && o.name?.endsWith('lens')) {
          lens = o;
        }
      });
      if (!lens) throw new Error('no refracting lens mesh in the scene');
      if (!lens.geometry.boundingSphere) lens.geometry.computeBoundingSphere();
      const centreWorld = lens.geometry.boundingSphere.center
        .clone()
        .applyMatrix4(lens.matrixWorld);
      const radius: number = lens.geometry.boundingSphere.radius;
      const toFraction = (v: any): [number, number] => {
        const p = v.clone().project(cam);
        return [(p.x + 1) / 2, (1 - p.y) / 2];
      };
      const centre = toFraction(centreWorld);
      const edge = centreWorld.clone();
      edge.y += radius;
      const [ex, ey] = toFraction(edge);
      const canvas = document.querySelector('canvas#app') as HTMLCanvasElement;
      const aspect = canvas.clientWidth / canvas.clientHeight;
      // Distances in canvas fractions, corrected for the aspect so the disk is round.
      const dist = (a: [number, number], b: [number, number]) =>
        Math.hypot((a[0] - b[0]) * aspect, a[1] - b[1]);
      const rim = dist([ex, ey], centre);
      const inside: Array<[number, number]> = [];
      const v = centreWorld.clone();
      for (const x of axis) {
        for (const y of axis) {
          for (const z of zs) {
            v.set(x, y, z);
            const f = toFraction(v);
            if (dist(f, centre) < rim * 0.8) inside.push(f);
          }
        }
      }
      return { centre, inside };
    },
    { axis: lattice.axis, zs: lattice.z }
  );
}

async function splitFrames(page: Page): Promise<number> {
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
      const { centre } = await lensAndLatticeOnCanvas(page, { axis: LATTICE_AXIS, z: LATTICE_Z });
      const [withLens] = await samplePixelsAt(page, CANVAS, [centre], 'framebuffer');
      expect(await splitFrames(page)).toBeGreaterThan(0); // the split really ran

      await toggleLens(page);
      const [noLens] = await samplePixelsAt(page, CANVAS, [centre], 'framebuffer');

      // The glass composites something the bare lattice does not have at that pixel
      // (its refracted, magnified image of the rows behind it).
      expect(maxChannelDelta(withLens, noLens)).toBeGreaterThanOrEqual(24);
    });

    test('lattice points in front of the lens stay crisp and unrefracted', async ({ page }) => {
      // Camera behind the spheres looking back through the lens centre (-3, 0, 1): the
      // whole lattice (z from -5 to -2) lies between the camera and the glass.
      const placed = await placeCameraAt(
        page,
        { x: -3.78, y: 0, z: -10 },
        { target: { x: -3, y: 0, z: 1 } }
      );
      expect(placed?.viaOrbitControls).toBe(true);
      await waitForRenderStable(page);
      const { inside } = await lensAndLatticeOnCanvas(page, { axis: LATTICE_AXIS, z: LATTICE_Z });
      // Enough points for the statistics to mean something (22 at this pose).
      expect(inside.length).toBeGreaterThanOrEqual(12);

      const withLens = await samplePixelsAt(page, CANVAS, inside, 'framebuffer');
      await toggleLens(page);
      const noLens = await samplePixelsAt(page, CANVAS, inside, 'framebuffer');

      // The probes are ON points: all lit with the lens hidden (no vacuous pass)…
      for (const p of noLens) expect(p.r + p.g + p.b).toBeGreaterThan(300);
      // …and the glass behind them changes almost nothing: mean well under the
      // unpartitioned 51, and at most a couple of points where the glass itself
      // brightens under a point (the refracted image of the neighbouring sphere).
      const deltas = withLens.map((p, i) => maxChannelDelta(p, noLens[i]));
      const mean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
      expect(mean).toBeLessThanOrEqual(20);
      expect(deltas.filter((d) => d > 24).length).toBeLessThanOrEqual(3);
    });
  });
}
