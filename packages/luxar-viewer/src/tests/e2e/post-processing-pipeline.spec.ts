/**
 * E2E Tests for Post-Processing Pipeline
 *
 * These tests verify:
 * - Post-processing pipeline existence and structure
 * - Cinematic mode toggle (vignette + chromatic lens distortion)
 * - Exposure control via renderingControls API
 * - WebGL stability after effect toggles
 * - Rapid toggle stress test
 * - Visual regression with default post-processing
 *
 * Dataset: build_example_structured.luxar.zarr (3D, reliable point count)
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  getLuxarState,
  waitForNextRender,
  focusCanvas,
  getWebGLErrors,
  waitForRenderStable,
} from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';

test.describe('Post-Processing Pipeline', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page);
    await focusCanvas(page);
  });

  test('should have functional post-processing pipeline', async ({ page }) => {
    // Verify postProcessing object exists on debug interface
    const hasPostProcessing = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return !!debug.postProcessing;
    });
    expect(hasPostProcessing).toBe(true);

    // Verify renderingControls.settings has bloomStrength and exposure
    const settingsCheck = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const settings = debug.renderingControls?.settings;
      if (!settings) return null;
      return {
        hasBloomStrength: 'bloomStrength' in settings,
        hasExposure: 'exposure' in settings,
        bloomStrengthType: typeof settings.bloomStrength,
        exposureType: typeof settings.exposure,
      };
    });

    expect(settingsCheck).not.toBeNull();
    expect(settingsCheck!.hasBloomStrength).toBe(true);
    expect(settingsCheck!.hasExposure).toBe(true);
    expect(settingsCheck!.bloomStrengthType).toBe('number');
    expect(settingsCheck!.exposureType).toBe('number');
  });

  test('should toggle cinematic mode with C key', async ({ page }) => {
    // Get initial cinematic state
    const initialState = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const settings = debug.renderingControls?.settings;
      return {
        vignetteEnabled: settings?.vignetteEnabled ?? false,
        chromaticLensDistortionEnabled: settings?.chromaticLensDistortionEnabled ?? false,
      };
    });

    // Initially cinematic mode should be off
    expect(initialState.vignetteEnabled).toBe(false);
    expect(initialState.chromaticLensDistortionEnabled).toBe(false);

    // Press C to enable cinematic mode
    await page.keyboard.press('c');
    await waitForNextRender(page);

    const afterEnable = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const settings = debug.renderingControls?.settings;
      return {
        vignetteEnabled: settings?.vignetteEnabled ?? false,
        chromaticLensDistortionEnabled: settings?.chromaticLensDistortionEnabled ?? false,
      };
    });

    expect(afterEnable.vignetteEnabled).toBe(true);
    expect(afterEnable.chromaticLensDistortionEnabled).toBe(true);

    // Press C again to disable cinematic mode
    await page.keyboard.press('c');
    await waitForNextRender(page);

    const afterDisable = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const settings = debug.renderingControls?.settings;
      return {
        vignetteEnabled: settings?.vignetteEnabled ?? false,
        chromaticLensDistortionEnabled: settings?.chromaticLensDistortionEnabled ?? false,
      };
    });

    expect(afterDisable.vignetteEnabled).toBe(false);
    expect(afterDisable.chromaticLensDistortionEnabled).toBe(false);
  });

  test('should change exposure via post-processing API and mirror UI settings', async ({
    page,
  }) => {
    // Read initial exposure
    const initialExposure = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.renderingControls?.settings?.exposure;
    });
    expect(typeof initialExposure).toBe('number');

    // Set exposure through the real post-processing API; mirror the
    // rendering-controls setting so UI state and shader state stay in sync.
    const applied = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const pp = debug.postProcessing;
      const settings = debug.renderingControls?.settings;
      if (!pp || typeof pp.updateExposure !== 'function') return false;
      pp.updateExposure(2.0);
      if (settings) settings.exposure = 2.0;
      return typeof pp.getExposure === 'function' && pp.getExposure() === 2.0;
    });
    expect(applied).toBe(true);

    await waitForNextRender(page);

    // Read back exposure from both UI settings and shader state.
    const newExposure = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        setting: debug.renderingControls?.settings?.exposure,
        shader: debug.postProcessing?.getExposure?.(),
      };
    });

    expect(newExposure.setting).toBe(2.0);
    expect(newExposure.shader).toBe(2.0);
    expect(newExposure.setting).not.toBe(initialExposure);
  });

  test('should render without WebGL errors after effect toggles', async ({ page }) => {
    // Toggle bloom off/on through the real manager API.
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      debug.postProcessing?.setBloomEnabled?.(false);
      const settings = debug.renderingControls?.settings;
      if (settings) settings.bloomEnabled = false;
    });
    await waitForNextRender(page);

    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      debug.postProcessing?.setBloomEnabled?.(true, 0.5);
      const settings = debug.renderingControls?.settings;
      if (settings) {
        settings.bloomEnabled = true;
        settings.bloomStrength = 0.5;
      }
    });
    await waitForNextRender(page);

    // Toggle FXAA off/on through the real manager API.
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const settings = debug.renderingControls?.settings;
      const next = !(settings?.fxaaEnabled ?? false);
      debug.postProcessing?.setFXAAEnabled?.(next);
      if (settings && 'fxaaEnabled' in settings) settings.fxaaEnabled = next;
    });
    await waitForNextRender(page);

    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const settings = debug.renderingControls?.settings;
      const next = !(settings?.fxaaEnabled ?? false);
      debug.postProcessing?.setFXAAEnabled?.(next);
      if (settings && 'fxaaEnabled' in settings) settings.fxaaEnabled = next;
    });
    await waitForNextRender(page);

    // Toggle cinematic on/off
    await page.keyboard.press('c');
    await waitForNextRender(page);
    await page.keyboard.press('c');
    await waitForNextRender(page);

    // Check for WebGL errors
    const glErrors = await getWebGLErrors(page);
    expect(glErrors).toEqual([]);
  });

  test('should survive rapid effect toggling', async ({ page }) => {
    // Toggle cinematic mode (C key) 10 times rapidly. The 50 ms pacing
    // is intentional: the test exercises the rapid-toggle race window
    // where successive enable/disable transitions land in the same
    // animation frame batch. Replacing this with a tighter loop changes
    // the failure mode being exercised.
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('c');
      await page.waitForTimeout(50);
    }

    // Wait for renders to settle
    await waitForNextRender(page, 3);

    // Verify the viewer did not crash: getState should still work
    const state = await getLuxarState(page);
    expect(state).toBeDefined();
    expect(state.initialized).toBe(true);

    // Verify no WebGL errors accumulated
    const glErrors = await getWebGLErrors(page);
    expect(glErrors).toEqual([]);
  });

  test('@visual visual regression: scene with default post-processing', async ({ page }) => {
    // waitForRenderStable already drives the wait off the renderer frame
    // counter; an additional fixed sleep would be redundant.
    await waitForRenderStable(page, 5);

    await expect(page).toHaveScreenshot('post-processing-default.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });
});
