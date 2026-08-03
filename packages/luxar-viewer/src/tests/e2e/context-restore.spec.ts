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
  // Context-restore is a WebGL-specific concern: it tests the
  // `webglcontextlost` / `webglcontextrestored` canvas event cycle
  // and the `WEBGL_lose_context` extension that fires them. Neither
  // applies under the default `WebGPURenderer` — Three's WebGPU
  // backend handles `device.lost` internally and the scene-manager
  // skips constructing `WebGLContextRecovery` when
  // `caps.apiSurface !== 'webgl2'`. Run these tests under
  // `VITE_LUXAR_USE_LEGACY_WEBGL=1` for coverage of the GLSL path.
  test.beforeEach(async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);
    const api = await page.evaluate(
      () =>
        ((window as any).__luxarDebug?.app?.sceneManager?.capabilities?.apiSurface as
          string | undefined) ?? 'unknown'
    );
    test.skip(
      api !== 'webgl2',
      `context-restore tests require the WebGL path (caps.apiSurface=${api}); ` +
        'run with ?renderer=webgl to exercise this suite.'
    );
  });

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

  test('PickingSystem survives context restore (instance reachable, no exceptions)', async ({
    page,
  }) => {
    // Locks in the picking-system rebuild path on context restore (see
    // state-lifecycle-reliability.md §T4). We do not assert a specific
    // pick result — that requires a guaranteed-labeled dataset and a
    // hit-test pixel position the headless GPU agrees on. Instead we
    // assert the weaker but reliable invariant: the picking-system
    // remains reachable through the app after restore AND a renderOnce
    // following a synthetic mousemove does not throw. A regression that
    // forgot to re-register picking shadow nodes shows up as either a
    // dangling reference or an exception from the pick render pass.
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const beforeRestore = await page.evaluate(() => {
      const debug = (window as unknown as { __luxarDebug?: { app?: unknown } }).__luxarDebug;
      const app = debug?.app as { pickingSystem?: unknown } | undefined;
      return Boolean(app?.pickingSystem);
    });
    // Picking-system is only constructed when at least one node has
    // labels. If the default scene has none, skip the deeper assertion.
    test.skip(!beforeRestore, 'No picking system in default scene; nothing to validate');

    expect(await loseAndRestoreContext(page)).toBe(true);

    const afterRestore = await page.evaluate(() => {
      const debug = (
        window as unknown as { __luxarDebug?: { app?: unknown; renderOnce?: () => void } }
      ).__luxarDebug;
      const app = debug?.app as { pickingSystem?: unknown } | undefined;
      if (!app?.pickingSystem) return { reachable: false, rendered: false };
      try {
        debug?.renderOnce?.();
        return { reachable: true, rendered: true };
      } catch {
        return { reachable: true, rendered: false };
      }
    });
    expect(afterRestore.reachable).toBe(true);
    expect(afterRestore.rendered).toBe(true);

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
      return typeof pp.getExposure === 'function' && pp.getExposure() === exposure;
    }, targetExposure);
    expect(targetSetOk).toBe(true);

    expect(await loseAndRestoreContext(page)).toBe(true);

    const exposureAfter = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const pp = debug?.postProcessing;
      return typeof pp?.getExposure === 'function' ? pp.getExposure() : null;
    });
    expect(exposureAfter).toBeCloseTo(targetExposure, 5);

    // No unhandled exceptions during the loss/restore cycle.
    expect(errors).toEqual([]);
  });
});
