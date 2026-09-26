/**
 * A restored camera pose keeps its up vector, in orbit and in fly mode.
 *
 * Two independent ways the up used to be dropped:
 * - orbit/ortho: `restoreCamera` set `camera.up` but not the orientation, and
 *   the controls' `reinitialize()` reads the orientation from
 *   `camera.quaternion`, so they rebuilt the PREVIOUS roll;
 * - fly: re-targeting (`setTarget` → `lookAtSmooth`) built the look-at with a
 *   hard-coded world +Y, so a Z-up scene or an authored roll was replaced.
 *
 * The check restores poses whose up is not +Y (a Z-up view and a 45° roll),
 * lets the loop run, and reads the camera's up back.
 */

import type { Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForPointsLoaded } from './helpers';

const DATASET =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_points_blending_modes.luxar.zarr';

const S = Math.SQRT1_2;
const UPS: Array<[string, [number, number, number]]> = [
  ['Z-up', [0, 0, 1]],
  ['45° roll', [0, S, S]],
];

async function restoredUp(
  page: Page,
  mode: 'orbit' | 'fly',
  up: [number, number, number]
): Promise<number[]> {
  return page.evaluate(
    async ({ m, u }) => {
      const dbg = (window as any).__luxarDebug;
      const app = dbg.app;
      const frame = () => new Promise((r) => requestAnimationFrame(() => r(null)));
      app.sceneManager.setControlType(m);
      for (let i = 0; i < 3; i++) await frame();
      const framed = app.getCameraPose();
      const dist = Math.hypot(
        framed.position[0] - framed.target[0],
        framed.position[1] - framed.target[1],
        framed.position[2] - framed.target[2]
      );
      // Look along -X, so both test ups are perpendicular to the view.
      const t = framed.target;
      app.setCameraPose({ ...framed, position: [t[0] + dist, t[1], t[2]], target: t, up: u });
      for (let i = 0; i < 10; i++) await frame();
      return app.getCameraPose().up as number[];
    },
    { m: mode, u: up }
  );
}

test.beforeEach(async ({ page }) => {
  await page.goto(`/?src=${DATASET}&debug&renderer=webgl`);
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page);
});

for (const mode of ['orbit', 'fly'] as const) {
  for (const [label, up] of UPS) {
    test(`${mode}: a restored ${label} pose keeps its up`, async ({ page }) => {
      const got = await restoredUp(page, mode, up);
      const norm = Math.hypot(got[0], got[1], got[2]);
      const dot = (got[0] * up[0] + got[1] * up[1] + got[2] * up[2]) / norm;
      expect(dot, `up read back as [${got.map((v) => v.toFixed(3)).join(', ')}]`).toBeGreaterThan(
        0.999
      );
    });
  }
}
