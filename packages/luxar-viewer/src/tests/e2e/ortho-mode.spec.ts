/**
 * E2E Tests for Orthographic Camera Mode and Scale Bar
 *
 * These tests verify:
 * - Switching to ortho mode via V key (orbit -> fly -> ortho)
 * - Scene content preservation across mode switches
 * - Zoom behavior in orthographic mode
 * - Scale bar visibility and label with B key
 * - Scale bar updates on zoom
 * - Clean round-trip back to orbit mode
 *
 * Dataset: scene_dimensions_example.luxar.zarr (has physical units: um)
 */

import { test, expect } from './fixtures';
import type { Page } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  getLuxarState,
  waitForNextRender,
  focusCanvas,
  getWebGLErrors,
  assertNoConsoleErrors,
} from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/scene_dimensions_example.luxar.zarr';

/**
 * Cycle V twice (orbit → fly → ortho) and assert the mode landed.
 * The 'switch to ortho mode via V key' test intentionally does NOT use
 * this helper — it asserts each intermediate mode step-by-step.
 */
async function switchToOrtho(page: Page): Promise<void> {
  await page.keyboard.press('v');
  await waitForNextRender(page);
  await page.keyboard.press('v');
  await waitForNextRender(page);
  const controlType = await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    return debug.controls?.getControlType?.();
  });
  expect(controlType).toBe('ortho');
}

/**
 * Dispatch `count` synthetic wheel notches at the canvas center. The
 * orbit-controls wheel handler requires client coordinates — a
 * coordinate-less WheelEvent is silently ignored (the root cause of the
 * old PERMANENT SKIPs in this file).
 */
async function wheelAtCanvasCenter(page: Page, deltaY: number, count = 1): Promise<void> {
  await page.evaluate(
    ({ dy, n }) => {
      const canvas = document.querySelector('canvas');
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      for (let i = 0; i < n; i++) {
        canvas.dispatchEvent(
          new WheelEvent('wheel', {
            deltaY: dy,
            clientX: rect.left + rect.width / 2,
            clientY: rect.top + rect.height / 2,
            bubbles: true,
            cancelable: true,
          })
        );
      }
    },
    { dy: deltaY, n: count }
  );
}

/**
 * Put the (perspective) camera into an OFF-AXIS, non-default pose with a
 * non-default up and a nudged FOV, then re-derive the orbit state. Required
 * for the #774 round-trip tests to actually discriminate: the auto-framed pose
 * is an exact front view (center + (0,0,D), up +Y, default FOV), which the
 * OLD front-view ortho reset reproduces — so before/after would match even
 * against the pre-fix code. An off-axis pose + off-default FOV breaks that.
 */
async function putCameraOffAxis(page: Page): Promise<void> {
  await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    const cam = debug.camera;
    const orbit = debug.controls.getControls(); // active LuxarOrbitControls
    const c = orbit.target; // content center = current pivot
    const cx = c.x;
    const cy = c.y;
    const cz = c.z;
    // Distinct offsets on all three axes → genuinely off-axis; non-default up.
    cam.position.set(cx + 37, cy + 29, cz + 43);
    cam.up.set(0.1, 0.95, 0.2);
    cam.lookAt(cx, cy, cz);
    cam.updateMatrixWorld();
    orbit.reinitialize();
    orbit.update();
    // Nudge FOV off the default so a FOV-restore regression is observable.
    debug.app.components.sceneManager.updateFOV(40);
  });
  await waitForNextRender(page);
}

test.describe('Orthographic Camera Mode', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page);
    await focusCanvas(page);
  });

  test('should switch to ortho mode via V key', async ({ page }) => {
    // Start in orbit mode
    const initialType = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.controls?.getControlType?.();
    });
    expect(initialType).toBe('orbit');

    // Press V once: orbit -> fly
    await page.keyboard.press('v');
    await waitForNextRender(page);

    const afterFirstV = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.controls?.getControlType?.();
    });
    expect(afterFirstV).toBe('fly');

    // Press V again: fly -> ortho
    await page.keyboard.press('v');
    await waitForNextRender(page);

    const afterSecondV = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.controls?.getControlType?.();
    });
    expect(afterSecondV).toBe('ortho');
  });

  test('should preserve scene content when switching to ortho', async ({ page }) => {
    // Get point count before switching
    const stateBefore = await getLuxarState(page);
    const pointsBefore = stateBefore.totalPoints;
    expect(pointsBefore).toBeGreaterThan(0);

    // Switch to ortho mode (V twice: orbit -> fly -> ortho)
    await switchToOrtho(page);

    // Get point count after switching
    const stateAfter = await getLuxarState(page);
    const pointsAfter = stateAfter.totalPoints;

    // Point count should be unchanged
    expect(pointsAfter).toBe(pointsBefore);
  });

  test('should zoom in ortho mode via scroll wheel', async ({ page }) => {
    // Previously PERMANENT SKIP with a misdiagnosis: the wheel path works
    // fine in headless Chromium, but TWO test-harness bugs masked it —
    //   1. `__luxarDebug.camera` was a stale snapshot of the perspective
    //      camera taken at install time; after the V-key ortho swap the
    //      test watched the abandoned camera's zoom (forever 1.0). Now a
    //      live getter (see core/app/debug/debug-interface.ts).
    //   2. The synthetic WheelEvent carried no clientX/clientY, so the
    //      handler's pointer-anchored zoom path ignored it. Dispatching
    //      at the canvas center works.
    // Switch to ortho mode (V twice)
    await switchToOrtho(page);

    // Read initial camera zoom
    const initialZoom = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.camera?.zoom;
    });
    expect(typeof initialZoom).toBe('number');

    // Scroll to zoom in
    await wheelAtCanvasCenter(page, -500);
    await waitForNextRender(page, 3);

    // Force render to apply damping
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      for (let i = 0; i < 5; i++) {
        debug?.renderOnce?.();
      }
    });
    // Damping settle past the synchronous renderOnce loop — see
    // mouse-interactions.spec.ts for the same shape.
    await page.waitForTimeout(300);

    // Read zoom after scroll
    const newZoom = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.camera?.zoom;
    });

    // Zoom should have changed (scrolling up = zoom in = higher zoom value)
    expect(newZoom).not.toBe(initialZoom);
  });

  test('should show scale bar with B key', async ({ page }) => {
    // Press B to toggle scale bar
    await page.keyboard.press('b');
    await waitForNextRender(page);

    // Check scale bar is visible
    const scaleBarVisible = await page.evaluate(() => {
      const bar = document.querySelector('.luxar-scale-bar');
      return bar !== null && getComputedStyle(bar).display !== 'none';
    });
    expect(scaleBarVisible).toBe(true);

    // Check scale bar label contains text
    const labelText = await page.evaluate(() => {
      const label = document.querySelector('.luxar-scale-bar__label');
      return label?.textContent?.trim() ?? '';
    });
    expect(labelText.length).toBeGreaterThan(0);
  });

  test('should update scale bar on zoom', async ({ page }) => {
    // Previously PERMANENT SKIP — same misdiagnosis as 'should zoom in
    // ortho mode via scroll wheel' (see that test): wheel works when the
    // event carries client coordinates.
    // Switch to ortho mode for predictable zoom behavior
    await switchToOrtho(page);

    // Show scale bar
    await page.keyboard.press('b');
    await waitForNextRender(page);

    // Read initial label text
    const initialLabel = await page.evaluate(() => {
      const label = document.querySelector('.luxar-scale-bar__label');
      return label?.textContent?.trim() ?? '';
    });
    expect(initialLabel.length).toBeGreaterThan(0);

    // Zoom in far enough to leave the current 1-2-5 label bucket:
    // 15 notches ≈ 10x+ zoom, several buckets away.
    await wheelAtCanvasCenter(page, -500, 15);
    await waitForNextRender(page, 3);

    // The zoom damps in over many frames and the scale-bar label updates
    // from a paint-driven observer, so poll (driving frames each probe)
    // instead of guessing a fixed settle window.
    await expect
      .poll(
        async () => {
          await page.evaluate(() => {
            const debug = (window as any).__luxarDebug;
            for (let i = 0; i < 5; i++) debug?.renderOnce?.();
          });
          return page.evaluate(() => {
            const label = document.querySelector('.luxar-scale-bar__label');
            return label?.textContent?.trim() ?? '';
          });
        },
        { timeout: 10000 }
      )
      .not.toBe(initialLabel);
  });

  test('should switch back to orbit cleanly', async ({ page }) => {
    // Switch to ortho: orbit -> fly -> ortho
    await switchToOrtho(page);

    // Switch back: ortho -> orbit
    await page.keyboard.press('v');
    await waitForNextRender(page);

    const backToOrbit = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.controls?.getControlType?.();
    });
    expect(backToOrbit).toBe('orbit');

    // Verify no WebGL errors
    const glErrors = await getWebGLErrors(page);
    expect(glErrors).toEqual([]);

    // Verify no console errors
    await assertNoConsoleErrors(page);
  });

  // #774: cycling control modes with NO other interaction must be a no-op for
  // the view. A full orbit → fly → ortho → orbit round trip (three V presses)
  // must land the camera back at its exact starting pose, FOV and pivot.
  test('round trip orbit→fly→ortho→orbit preserves camera pose, fov and target', async ({
    page,
  }) => {
    const snapshot = async () =>
      page.evaluate(() => {
        const debug = (window as any).__luxarDebug;
        const cam = debug.camera;
        cam.updateMatrixWorld();
        const target = debug.controls.getFocusTarget();
        return {
          position: cam.position.toArray(),
          quaternion: cam.quaternion.toArray(),
          up: cam.up.toArray(),
          fov: cam.isPerspectiveCamera ? cam.fov : null,
          target: [target.x, target.y, target.z],
          type: debug.controls.getControlType(),
        };
      });

    // Off-axis, non-default pose + FOV BEFORE the cycle (see helper docstring):
    // this is what makes the test fail against the pre-fix front-view reset.
    await putCameraOffAxis(page);

    const before = await snapshot();
    expect(before.type).toBe('orbit');
    expect(before.fov).not.toBeNull();

    // orbit → fly → ortho → orbit (three presses).
    for (let i = 0; i < 3; i++) {
      await page.keyboard.press('v');
      await waitForNextRender(page);
    }

    const after = await snapshot();
    expect(after.type).toBe('orbit');

    const EPS = 1e-3;
    for (let i = 0; i < 3; i++) {
      expect(Math.abs(after.position[i] - before.position[i])).toBeLessThan(EPS);
      expect(Math.abs(after.target[i] - before.target[i])).toBeLessThan(EPS);
      expect(Math.abs(after.up[i] - before.up[i])).toBeLessThan(EPS);
    }
    for (let i = 0; i < 4; i++) {
      expect(Math.abs(after.quaternion[i] - before.quaternion[i])).toBeLessThan(EPS);
    }
    expect(after.fov).toBeCloseTo(before.fov as number, 3);
  });

  // #774: apparent framing must be invariant across the round trip. Project an
  // OFF-AXIS world point (inside the point cloud, NOT on the view axis) to NDC
  // before and after cycling modes and assert it lands at the same screen
  // location — an on-axis point is NDC-invariant under any dolly/zoom/FOV error,
  // so only an off-axis point can catch a sideways shift or zoom drift.
  test('round trip keeps an off-axis world point at the same NDC (no shift / zoom drift)', async ({
    page,
  }) => {
    // Project an off-axis point through the live camera: ndc = P * V * worldPoint.
    const ndcOfPoint = async () =>
      page.evaluate(() => {
        const debug = (window as any).__luxarDebug;
        const cam = debug.camera;
        cam.updateMatrixWorld();
        cam.updateProjectionMatrix();
        const p = cam.projectionMatrix.elements; // column-major
        const v = cam.matrixWorldInverse.elements; // column-major
        const mul = (m: number[], x: number[]) => {
          const o = [0, 0, 0, 0];
          for (let row = 0; row < 4; row++) {
            let s = 0;
            for (let col = 0; col < 4; col++) s += m[col * 4 + row] * x[col];
            o[row] = s;
          }
          return o;
        };
        // Off-axis probe point inside the scene extent.
        const eye = mul(v, [30, 20, 10, 1]);
        const clip = mul(p, eye);
        return [clip[0] / clip[3], clip[1] / clip[3]];
      });

    // Off-axis CAMERA pose too — an off-axis probe alone can't catch the drift
    // if the camera itself is a front view (the old ortho reset reproduces it).
    await putCameraOffAxis(page);

    const before = await ndcOfPoint();

    for (let i = 0; i < 3; i++) {
      await page.keyboard.press('v');
      await waitForNextRender(page);
    }

    const after = await ndcOfPoint();
    expect(Math.abs(after[0] - before[0])).toBeLessThan(2e-3);
    expect(Math.abs(after[1] - before[1])).toBeLessThan(2e-3);
  });
});
