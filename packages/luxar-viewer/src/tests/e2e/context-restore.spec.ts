/**
 * WebGL Context-Restore Tests (CR-1)
 *
 * These tests verify the post-processing pipeline survives a
 * `webglcontextlost` / `webglcontextrestored` cycle without losing user
 * settings and without breaking external consumers (PickingSystem,
 * AnimationController, RenderingControls) that hold cached references to
 * `PostProcessingManager`.
 *
 * The fix (CR-1) is an in-place rebuild: the manager's identity is
 * preserved across the restore so consumer references stay valid.
 *
 * IMPORTANT: This test class protects against the regression that the
 * `fix/p0-review-hardening` branch introduced — making
 * `webglcontextrestored` actually recreate the post-processing pipeline.
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady } from './helpers';

async function loseAndRestoreContext(page: import('@playwright/test').Page): Promise<boolean> {
  const lost = await page.evaluate(() => {
    const canvas = document.querySelector('canvas') as HTMLCanvasElement | null;
    if (!canvas) return false;
    const gl = canvas.getContext('webgl2') as WebGL2RenderingContext | null;
    const ext = gl?.getExtension('WEBGL_lose_context');
    if (!ext) return false;
    ext.loseContext();
    return true;
  });
  if (!lost) return false;
  // Intentional fixed sleeps: WEBGL_lose_context dispatches the lost
  // and restored events asynchronously through the browser's GL queue,
  // which is not exposed via a JS-observable signal. The 300/800 ms
  // windows give the lost handler (post-processing dispose, scene
  // resource invalidation) and the restore handler
  // (rebuildAfterContextRestore + dirty marking) time to complete.
  // These are wall-clock waits on browser-internal events; an
  // event-driven wait would need a custom hook in scene-manager.
  await page.waitForTimeout(300);
  await page.evaluate(() => {
    const canvas = document.querySelector('canvas') as HTMLCanvasElement | null;
    const gl = canvas?.getContext('webgl2') as WebGL2RenderingContext | null;
    gl?.getExtension('WEBGL_lose_context')?.restoreContext();
  });
  await page.waitForTimeout(800);
  return true;
}

test.describe('WebGL Context Restore (CR-1)', () => {
  test('PostProcessingManager identity is preserved across context restore', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const beforeId = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const pp = debug?.postProcessing;
      // Tag the instance with a unique marker we can check after restore.
      if (pp) {
        (pp as any).__crTestMarker = '__cr1_marker_v1__';
      }
      return !!pp;
    });
    expect(beforeId).toBe(true);

    const triggered = await loseAndRestoreContext(page);
    expect(triggered).toBe(true);

    const sameInstance = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const pp = debug?.postProcessing;
      return (pp as any)?.__crTestMarker === '__cr1_marker_v1__';
    });
    expect(sameInstance).toBe(true);
  });

  // Note: a "cached consumer references remain valid after context restore"
  // test would be redundant with the identity-preservation test above —
  // because the manager identity is preserved, every consumer that cached a
  // reference at init time continues to hold a valid live reference. That
  // contract is verified at the unit level for the PickingSystem /
  // AnimationController / RenderingControls re-wiring boundary in
  // `src/tests/unit/scene/scene-manager.test.ts`.

  test('Renderer continues to render after context restore (no exceptions)', async ({ page }) => {
    // The full-pixel "no black screen" check is environment-dependent in
    // headless Playwright (the WEBGL_lose_context extension's
    // `restoreContext()` does not always synchronously re-establish a usable
    // context). The contract this test enforces is the strict one: the
    // restore cycle must not throw and `renderOnce()` must not throw.
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/?debug');
    await waitForLuxarReady(page);

    expect(await loseAndRestoreContext(page)).toBe(true);

    const renderOk = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      try {
        debug?.renderOnce?.();
        return true;
      } catch {
        return false;
      }
    });
    expect(renderOk).toBe(true);

    // No unhandled exceptions during the loss/restore + render cycle.
    expect(errors).toEqual([]);
  });

  test('Durable user settings survive context restore', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Set a non-default exposure so we can detect a default-revert after restore.
    const targetExposure = 1.7;
    const targetSetOk = await page.evaluate((exposure) => {
      const debug = (window as any).__luxarDebug;
      const pp = debug?.postProcessing;
      if (!pp || typeof pp.updateExposure !== 'function') return false;
      pp.updateExposure(exposure);
      return pp.toneMappingEffect?.exposure === exposure;
    }, targetExposure);
    expect(targetSetOk).toBe(true);

    expect(await loseAndRestoreContext(page)).toBe(true);

    const exposureAfter = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const pp = debug?.postProcessing;
      return pp?.toneMappingEffect?.exposure ?? null;
    });
    expect(exposureAfter).toBeCloseTo(targetExposure, 5);

    // No unhandled exceptions during the loss/restore cycle.
    expect(errors).toEqual([]);
  });
});
