/**
 * Mobile: orbit touch gestures own the canvas.
 *
 * The audit's headline measurement was a two-finger pinch that left the camera
 * untouched and set `visualViewport.scale` to 5 — the browser took the gesture
 * as page zoom. These specs assert the inverse under device emulation: the
 * camera moves, the page does not.
 */

import { test, expect } from '../fixtures';
import { waitForLuxarReady, waitForPointsLoaded } from '../helpers';
import {
  cameraPose,
  canvasCentre,
  distance,
  doubleTap,
  dragFinger,
  inputProbe,
  pinch,
  pinchThenDragSurvivor,
  quaternionAngle,
  twist,
} from './touch-helpers';

const DATASET = 'http://localhost:9000/datasets/examples/build_example_manual.luxar.zarr';

test.describe('mobile gestures (orbit)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page);
    await page.waitForTimeout(500);
  });

  test('device emulation presents a coarse, non-hovering pointer', async ({ page }) => {
    const probe = await inputProbe(page);
    expect(probe.coarse).toBe(true);
    expect(probe.hoverNone).toBe(true);
    expect(probe.anyHover).toBe(false);
    expect(probe.touchPoints).toBeGreaterThan(0);
    expect(probe.pageScale).toBe(1);
  });

  test('one-finger drag rotates the camera', async ({ page }) => {
    const c = await canvasCentre(page);
    const before = await cameraPose(page);
    await dragFinger(page, c, { x: c.x + 120, y: c.y + 40 });
    await page.waitForTimeout(600);
    const after = await cameraPose(page);
    expect(quaternionAngle(before.quaternion, after.quaternion)).toBeGreaterThan(0.05);
  });

  test('pinch dollies the camera and does NOT zoom the page', async ({ page }) => {
    const c = await canvasCentre(page);
    const before = await cameraPose(page);
    await pinch(page, c, 40, 130);
    await page.waitForTimeout(900);
    const after = await cameraPose(page);
    const probe = await inputProbe(page);
    expect(probe.pageScale).toBe(1);
    // Pinch-out zooms IN: the camera moves toward the target.
    expect(distance(after.position, after.target)).toBeLessThan(
      distance(before.position, before.target)
    );
  });

  test('two-finger twist rolls the view without dollying', async ({ page }) => {
    const c = await canvasCentre(page);
    const before = await cameraPose(page);
    await twist(page, c, 90, 0, Math.PI / 2);
    await page.waitForTimeout(900);
    const after = await cameraPose(page);
    expect(quaternionAngle(before.quaternion, after.quaternion)).toBeGreaterThan(0.2);
    const dBefore = distance(before.position, before.target);
    const dAfter = distance(after.position, after.target);
    expect(Math.abs(dAfter - dBefore) / dBefore).toBeLessThan(0.05);
  });

  test('lifting one finger of a pinch continues as a rotate with the other', async ({ page }) => {
    const c = await canvasCentre(page);
    const before = await cameraPose(page);
    await page.evaluate(() => {
      const events: Array<{ type: string; touches: number[]; changed: number[] }> = [];
      for (const type of ['touchend', 'touchmove']) {
        document.addEventListener(type, (event) => {
          const touchEvent = event as TouchEvent;
          events.push({
            type,
            touches: Array.from(touchEvent.touches, (touch) => touch.identifier),
            changed: Array.from(touchEvent.changedTouches, (touch) => touch.identifier),
          });
        });
      }
      (window as unknown as { __mobileTouchEvents: typeof events }).__mobileTouchEvents = events;
    });
    await pinchThenDragSurvivor(page, c, 60, { x: c.x + 100, y: c.y + 60 });
    await page.waitForTimeout(600);
    const after = await cameraPose(page);
    const events = await page.evaluate(
      () =>
        (
          window as unknown as {
            __mobileTouchEvents: Array<{ type: string; touches: number[]; changed: number[] }>;
          }
        ).__mobileTouchEvents
    );
    const releaseIndex = events.findIndex((event) => event.type === 'touchend');
    expect(events[releaseIndex]).toEqual({ type: 'touchend', touches: [1], changed: [2] });
    expect(events[releaseIndex + 1]).toMatchObject({ type: 'touchmove', touches: [1] });
    expect(quaternionAngle(before.quaternion, after.quaternion)).toBeGreaterThan(0.05);
    expect((await inputProbe(page)).pageScale).toBe(1);
  });

  test('double-tap re-frames the scene after a pinch dollied it', async ({ page }) => {
    const c = await canvasCentre(page);
    const home = await cameraPose(page);
    // A pinch changes the camera DISTANCE (an orbit drag would not: it keeps
    // the fit distance, so a re-fit could legitimately land where it started).
    await pinch(page, c, 40, 130);
    await page.waitForTimeout(900);
    const moved = await cameraPose(page);
    const dollied = distance(home.position, moved.position);
    expect(dollied).toBeGreaterThan(1);
    await doubleTap(page, c);
    await page.waitForTimeout(1500); // recenter flies the camera
    const after = await cameraPose(page);
    expect((await inputProbe(page)).pageScale).toBe(1);
    // The re-fit backs the camera out of the pinched pose to the framing
    // distance (a fit lands at the FIT distance, which is not the load-time
    // auto-frame position — so the proof is not "back to home")…
    expect(distance(moved.position, after.position)).toBeGreaterThan(dollied * 0.5);
    expect(distance(after.position, after.target)).toBeGreaterThan(
      distance(moved.position, moved.target)
    );
    // …and it lands exactly where the recenter command itself lands: the fit
    // is idempotent, so a direct recenter afterwards must not move the camera.
    await page.evaluate(() =>
      (
        window as unknown as { __luxarDebug: { app: { recenterCamera(): void } } }
      ).__luxarDebug.app.recenterCamera()
    );
    await page.waitForTimeout(1500);
    const refit = await cameraPose(page);
    expect(distance(after.position, refit.position)).toBeLessThan(1e-2);
  });
});
