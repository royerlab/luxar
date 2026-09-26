/**
 * Every frame of a camera flight renders with the clipping planes of its own
 * pose.
 *
 * Dynamic clipping derives near/far from the camera position each frame, in a
 * per-frame callback registered at init. A flight moves the camera in a
 * per-frame callback registered when the flight STARTS, i.e. after it. With
 * callbacks run in registration order the flight moved the camera after
 * clipping had already read it, so each flight frame rendered with the
 * previous frame's near/far: stale by one frame for the whole flight. Frame
 * phases now run camera writers first.
 *
 * The check records, at every render during a dolly flight, the camera
 * position and the far plane the frame actually used. Afterwards it
 * recomputes the far plane dynamic clipping gives for each recorded position,
 * and requires every frame to match its own pose (within clipping's own 0.1%
 * update threshold). The flight is steep enough that one frame of staleness
 * is more than ten times that tolerance.
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForPointsLoaded } from './helpers';

// A store with metadata bounds: dynamic clipping needs them (a synthetic
// injected scene has none, and clipping leaves its planes alone).
const DATASET =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_points_blending_modes.luxar.zarr';

interface Trace {
  frames: Array<{ position: number[]; far: number }>;
  reference: number[];
  dynamicClipping: boolean;
}

test('each flight frame renders with the far plane of its own pose', async ({ page }) => {
  await page.setViewportSize({ width: 640, height: 480 });
  await page.goto(`/?src=${DATASET}&debug&renderer=webgl`);
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page);

  const trace: Trace = await page.evaluate(async () => {
    const dbg = (window as any).__luxarDebug;
    const app = dbg.app;
    const sm = app.sceneManager;
    const frame = () => new Promise((r) => requestAnimationFrame(() => r(null)));
    for (let i = 0; i < 10; i++) await frame();

    // From the auto-framed pose, start four times further out along the view
    // direction and dolly in to 0.3 of the framing distance.
    const framed = app.getCameraPose();
    const at = (k: number) =>
      framed.target.map((t: number, i: number) => t + k * (framed.position[i] - t));
    const start = { ...framed, position: at(4) };
    app.setCameraPose(start);
    for (let i = 0; i < 5; i++) await frame();

    const frames: Array<{ position: number[]; far: number }> = [];
    const pp = sm.postProcessing;
    const render = pp.render.bind(pp);
    pp.render = (...args: unknown[]) => {
      frames.push({ position: sm.camera.position.toArray(), far: sm.camera.far });
      return render(...args);
    };
    try {
      await app.flyTo({ ...start, position: at(0.3) }, { durationMs: 3000, easing: 'linear' });
    } finally {
      pp.render = render;
    }

    // What dynamic clipping gives each recorded position. Force the update
    // past clipping's 0.1% change threshold by moving the plane off first.
    const reference = frames.map((f) => {
      sm.camera.position.fromArray(f.position);
      sm.camera.far = 1;
      sm.camera.updateProjectionMatrix();
      sm.updateDynamicClippingPlanes();
      return sm.camera.far as number;
    });
    return { frames, reference, dynamicClipping: sm.getDynamicClippingState().enabled as boolean };
  });

  expect(trace.dynamicClipping, 'the check needs dynamic clipping on').toBe(true);
  expect(trace.frames.length, 'too few flight frames to judge').toBeGreaterThan(10);
  // Non-vacuous: the flight moves the far plane by well over the tolerance
  // between consecutive frames, so a one-frame-stale plane would show.
  const steps = trace.reference.slice(1).map((r, i) => Math.abs(r - trace.reference[i]) / r);
  expect(Math.max(...steps)).toBeGreaterThan(0.01);

  const worst = Math.max(
    ...trace.frames.map((f, i) => Math.abs(f.far - trace.reference[i]) / trace.reference[i])
  );
  expect(worst).toBeLessThan(0.002);
});
