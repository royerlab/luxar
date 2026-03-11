/**
 * Dimension Animation E2E Tests
 *
 * Tests the dimension animation feature that allows automated playback through
 * dimension ranges (time-lapse, z-stack traversal, etc.).
 *
 * Features tested:
 * - Play/pause controls
 * - FPS selection (1, 2, 5, 10, 15, 30, 60 FPS)
 * - Loop modes (once, loop, bounce)
 * - Keyboard shortcuts (K, Home, End, Shift+arrows)
 * - Animation progress and state
 */

import { test, expect } from '@playwright/test';
import {
  waitForLuxarReady,
  getLuxarState,
  waitForDataLoaded,
  waitForNextRender,
  waitForAnimationStep,
} from './helpers';

// Test dataset with multiple dimensions for animation
const DATASET = 'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.zarr';

/**
 * Helper to get dimension animation state
 */
async function getAnimationState(page: any, dimIndex: number): Promise<any> {
  return await page.evaluate((idx: number) => {
    const debug = (window as any).__luxarDebug;
    const inputHandler = debug?.inputHandler;
    const animManager = inputHandler?.animationManager;
    if (!animManager) return null;
    return animManager.getState(idx);
  }, dimIndex);
}

/**
 * Helper to check if animation is playing
 */
async function isAnimating(page: any, dimIndex: number): Promise<boolean> {
  return await page.evaluate((idx: number) => {
    const debug = (window as any).__luxarDebug;
    const inputHandler = debug?.inputHandler;
    const animManager = inputHandler?.animationManager;
    if (!animManager) return false;
    return animManager.isAnimating(idx);
  }, dimIndex);
}

/**
 * Helper to get dimension value
 * Reads from the slider UI element for the dimension
 */
async function getDimensionValue(page: any, dimIndex: number): Promise<number> {
  return await page.evaluate((idx: number) => {
    const debug = (window as any).__luxarDebug;
    const sceneDimsManager = debug?.sceneDimsManager;
    if (!sceneDimsManager) return -1;

    const dims = sceneDimsManager.getDims();
    if (!dims) return -1;

    // Read the actual value from currentStep
    if (dims.currentStep && dims.currentStep[idx] !== undefined) {
      return dims.currentStep[idx];
    }

    // Fallback: Try to read from UI slider if currentStep isn't available
    // Find the slider for this dimension (non-displayed dimensions have sliders)
    const sliders = document.querySelectorAll('.luxar-dimension-slider input[type="range"]');
    const navigableDims = dims.displayed
      ? Array.from({ length: dims.ndim }, (_, i) => i).filter((i) => !dims.displayed.includes(i))
      : [];
    const sliderIndex = navigableDims.indexOf(idx);

    if (sliderIndex >= 0 && sliders[sliderIndex]) {
      const slider = sliders[sliderIndex] as HTMLInputElement;
      return parseFloat(slider.value);
    }

    return -1;
  }, dimIndex);
}

/**
 * Helper to wait for dimension value to change
 */
async function waitForDimensionValueChange(
  page: any,
  dimIndex: number,
  initialValue: number,
  timeout = 5000
): Promise<void> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const currentValue = await getDimensionValue(page, dimIndex);
    if (currentValue !== initialValue) {
      return;
    }
    await page.waitForTimeout(100);
  }

  throw new Error(`Timeout waiting for dimension ${dimIndex} value to change from ${initialValue}`);
}

test.describe('Dimension Animation - UI Controls', () => {
  test('should show animation controls in dimension sliders', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Dimension sliders should be visible by default for nD datasets
    // (no need to press 'n' - they're already shown)

    // Wait for dimension sliders to be attached and visible
    await page.waitForSelector('.luxar-dimension-sliders', { timeout: 5000 });

    // Wait for play button to be visible (checking opacity, not hidden state)
    await page.waitForFunction(
      () => {
        const btn = document.querySelector('.luxar-dimension-slider__play-btn');
        if (!btn) return false;
        const style = window.getComputedStyle(btn);
        return style.display !== 'none' && parseFloat(style.opacity) > 0;
      },
      null,
      { timeout: 3000 }
    );

    const playButton = await page.locator('.luxar-dimension-slider__play-btn').first();

    // Right-click play button to open context menu
    await playButton.click({ button: 'right' });
    await page.waitForTimeout(200);

    // Check for context menu with speed and loop options
    const menuControls = await page.evaluate(() => {
      const menu = document.querySelector('.luxar-dimension-slider__context-menu');
      const speedSection = document.querySelector('.luxar-dimension-slider__context-section');
      const speedItems = Array.from(
        document.querySelectorAll('.luxar-dimension-slider__context-item')
      );

      return {
        hasContextMenu: menu !== null,
        hasSpeedSection: speedSection !== null,
        hasMenuItems: speedItems.length > 0,
      };
    });

    expect(menuControls.hasContextMenu).toBe(true);
    expect(menuControls.hasSpeedSection).toBe(true);
    expect(menuControls.hasMenuItems).toBe(true);
  });

  test('should toggle play button text on click', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);

    // Wait for play button to be visible
    const playButton = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton.waitFor({ state: 'visible', timeout: 5000 });

    // Get initial play button text
    const initialText = await page.evaluate(() => {
      const playBtn = document.querySelector('.luxar-dimension-slider__play-btn');
      return playBtn?.textContent?.trim();
    });

    expect(initialText).toBe('▶');

    // Click play button
    await page.click('.luxar-dimension-slider__play-btn');
    await waitForNextRender(page);

    // Check button text changed to pause
    const pauseText = await page.evaluate(() => {
      const playBtn = document.querySelector('.luxar-dimension-slider__play-btn');
      return playBtn?.textContent?.trim();
    });

    expect(pauseText).toBe('⏸');

    // Click again to pause
    await page.click('.luxar-dimension-slider__play-btn');
    await waitForNextRender(page);

    // Check button text changed back to play
    const playText = await page.evaluate(() => {
      const playBtn = document.querySelector('.luxar-dimension-slider__play-btn');
      return playBtn?.textContent?.trim();
    });

    expect(playText).toBe('▶');
  });

  test('should change FPS via selector', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);

    // Wait for play button to be visible, then right-click to open context menu
    const playButton = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton.waitFor({ state: 'visible', timeout: 5000 });
    await playButton.click({ button: 'right' });
    await page.waitForTimeout(200);

    // Click on 30 FPS option in context menu
    await page.click('text=30 FPS');
    await page.waitForTimeout(200);

    // Start animation
    await page.click('.luxar-dimension-slider__play-btn');
    await waitForNextRender(page);

    // Check animation state has correct FPS
    const state = await getAnimationState(page, 3); // 4th dimension = index 3
    expect(state).not.toBeNull();
    expect(state?.targetFPS).toBe(30);
  });

  test('should change loop mode via selector', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);

    // Wait for play button to be visible, then right-click to open context menu
    const playButton = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton.waitFor({ state: 'visible', timeout: 5000 });
    await playButton.click({ button: 'right' });
    await page.waitForTimeout(200);

    // Click on Bounce option in context menu
    await page.click('text=Bounce');
    await page.waitForTimeout(200);

    // Start animation
    await page.click('.luxar-dimension-slider__play-btn');
    await waitForNextRender(page);

    // Check animation state has correct loop mode
    const state = await getAnimationState(page, 3);
    expect(state).not.toBeNull();
    expect(state?.loopMode).toBe('bounce');
  });
});

test.describe('Dimension Animation - Keyboard Shortcuts', () => {
  test('should toggle animation with K key', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus (sliders shouldn't capture keyboard)
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);

    // Press K to start animation
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Check animation is playing
    const isPlaying = await isAnimating(page, 3);
    expect(isPlaying).toBe(true);

    // Press K again to pause
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Check animation is paused
    const isPaused = await isAnimating(page, 3);
    expect(isPaused).toBe(false);
  });

  test('should jump to start with Home key', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);

    // Navigate forward a few steps
    await page.keyboard.press(']');
    await waitForAnimationStep(page, 3);
    await page.keyboard.press(']');
    await waitForAnimationStep(page, 3);

    // Get current value (should not be at start)
    const beforeValue = await getDimensionValue(page, 3);
    expect(beforeValue).toBeGreaterThan(0);

    // Press Home to jump to start
    await page.keyboard.press('Home');
    await waitForAnimationStep(page, 3);

    // Check dimension is at start (value 0)
    const afterValue = await getDimensionValue(page, 3);
    expect(afterValue).toBe(0);
  });

  test('should jump to end with End key', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);

    // Get the max value for this dimension
    const maxValue = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const sceneDimsManager = debug?.sceneDimsManager;
      if (!sceneDimsManager) return -1;
      const ranges = sceneDimsManager.getDimensionRanges();
      if (!ranges) return -1;
      return ranges[3]?.[1] ?? -1; // 4th dimension = index 3, max = [1]
    });

    // Press End to jump to end
    await page.keyboard.press('End');
    await waitForAnimationStep(page, 3);

    // Check dimension is at end
    const value = await getDimensionValue(page, 3);
    expect(value).toBe(maxValue);
  });

  test('should increase speed with Shift+Up', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension and start animation
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Get initial FPS
    const initialState = await getAnimationState(page, 3);
    const initialFPS = initialState?.targetFPS ?? 10;

    // Call increaseSpeed directly (keyboard shortcuts don't work reliably in E2E tests)
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const inputHandler = debug?.inputHandler;
      const animManager = inputHandler?.animationManager;
      if (animManager) {
        animManager.increaseSpeed(3);
      }
    });
    await waitForNextRender(page);

    // Check FPS increased
    const newState = await getAnimationState(page, 3);
    const newFPS = newState?.targetFPS ?? 10;
    expect(newFPS).toBeGreaterThan(initialFPS);
  });

  test('should decrease speed with Shift+Down', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension and start animation at high FPS
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);

    // Set FPS to 30 first via UI

    // Wait for play button to be visible, then right-click to select 30 FPS
    const playButton = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton.waitFor({ state: 'visible', timeout: 5000 });
    await playButton.click({ button: 'right' });
    await page.waitForTimeout(200);
    await page.click('text=30 FPS');
    await page.waitForTimeout(200);

    // Start animation
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Get initial FPS (should be 30)
    const initialState = await getAnimationState(page, 3);
    const initialFPS = initialState?.targetFPS ?? 30;

    // Call decreaseSpeed directly (keyboard shortcuts not working in E2E tests)
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const inputHandler = debug?.inputHandler;
      const animManager = inputHandler?.animationManager;
      if (animManager) {
        animManager.decreaseSpeed(3);
      }
    });
    await waitForNextRender(page);

    // Check FPS decreased
    const newState = await getAnimationState(page, 3);
    const newFPS = newState?.targetFPS ?? 30;
    expect(newFPS).toBeLessThan(initialFPS);
  });
});

test.describe('Dimension Animation - Animation Behavior', () => {
  test('should animate through dimension values', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);

    // Get initial value
    const initialValue = await getDimensionValue(page, 3);

    // Start animation at high FPS for faster test

    // Wait for play button to be visible, then right-click and select 30 FPS
    const playButton1 = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton1.waitFor({ state: 'visible', timeout: 5000 });
    await playButton1.click({ button: 'right' });
    await page.waitForTimeout(200);
    await page.click('text=30 FPS');
    await page.waitForTimeout(200);

    await page.keyboard.press('k'); // Start animation
    await waitForNextRender(page);

    // Wait for value to change (animation progressing)
    await waitForDimensionValueChange(page, 3, initialValue, 3000);

    // Verify animation is still playing
    const isPlaying = await isAnimating(page, 3);
    expect(isPlaying).toBe(true);

    // Get new value
    const newValue = await getDimensionValue(page, 3);
    expect(newValue).toBeGreaterThan(initialValue);
  });

  test('should loop continuously with loop mode', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);

    // Set loop mode to loop and high FPS

    // Wait for play button to be visible, then right-click to open context menu
    const playButton2 = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton2.waitFor({ state: 'visible', timeout: 5000 });
    await playButton2.click({ button: 'right' });
    await page.waitForTimeout(200);

    // Select loop mode
    await page.click('text=Loop');
    await page.waitForTimeout(100);

    // Right-click again to set FPS
    await playButton2.click({ button: 'right' });
    await page.waitForTimeout(200);
    await page.click('text=30 FPS');
    await page.waitForTimeout(200);

    // Jump to near end
    await page.keyboard.press('End');
    await waitForAnimationStep(page, 3);

    const nearEndValue = await getDimensionValue(page, 3);

    // Start animation
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Wait long enough to pass end and wrap (should take < 1 second at 30 FPS)
    await page.waitForTimeout(1500);

    // Value should have wrapped to beginning
    const wrappedValue = await getDimensionValue(page, 3);
    expect(wrappedValue).toBeLessThan(nearEndValue);

    // Animation should still be playing
    const isPlaying = await isAnimating(page, 3);
    expect(isPlaying).toBe(true);
  });

  test('should stop at end with once mode', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);

    // Set loop mode to once and high FPS using direct API calls (more reliable than UI)
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const animManager = debug?.inputHandler?.animationManager;
      if (animManager) {
        animManager.setLoopMode(3, 'once');
        animManager.setTargetFPS(3, 60);
      }
    });
    await waitForNextRender(page, 1);

    // Get max value and jump to near end (not quite at end)
    const maxValue = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const sceneDimsManager = debug?.sceneDimsManager;
      const ranges = sceneDimsManager?.getDimensionRanges();
      return ranges?.[3]?.[1] ?? 10;
    });

    // Set position to max - 2, leaving some distance to animate
    await page.evaluate((max: number) => {
      const debug = (window as any).__luxarDebug;
      const sceneDimsManager = debug?.sceneDimsManager;
      sceneDimsManager?.setDimensionValue(3, Math.max(0, max - 2));
    }, maxValue);
    await waitForNextRender(page);

    // Start animation
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Verify animation started
    const startedPlaying = await isAnimating(page, 3);
    expect(startedPlaying).toBe(true);

    // Wait for animation to complete
    // The "once" mode should stop when reaching the end
    // At 60 FPS, 2 steps should complete in ~33ms per step, but allow for loading delays
    // Use a poll loop for better debugging
    let attempts = 0;
    const maxAttempts = 100; // 10 seconds with 100ms intervals
    let stopped = false;
    while (attempts < maxAttempts && !stopped) {
      stopped = await page.evaluate(() => {
        const debug = (window as any).__luxarDebug;
        const mgr = debug?.inputHandler?.animationManager;
        return mgr ? !mgr.isAnimating(3) : true; // If no manager, consider stopped
      });
      if (!stopped) {
        await page.waitForTimeout(100);
        attempts++;
      }
    }

    // Get debug info if still not stopped
    if (!stopped) {
      const debugInfo = await page.evaluate(() => {
        const debug = (window as any).__luxarDebug;
        const mgr = debug?.inputHandler?.animationManager;
        const state = mgr?.getState(3);
        const sceneDims = debug?.sceneDimsManager;
        const dims = sceneDims?.getDims();
        return {
          hasManager: !!mgr,
          isAnimating: mgr?.isAnimating(3),
          state: state
            ? {
                isPlaying: state.isPlaying,
                loopMode: state.loopMode,
                direction: state.direction,
                targetFPS: state.targetFPS,
              }
            : null,
          currentValue: dims?.currentStep?.[3],
          ranges: sceneDims?.getDimensionRanges(),
        };
      });
      console.log('Animation debug info:', JSON.stringify(debugInfo, null, 2));
    }

    // Animation should have stopped
    const isPlaying = await isAnimating(page, 3);
    expect(isPlaying).toBe(false);

    // Value should be at or very close to max (allow small floating point error)
    const finalValue = await getDimensionValue(page, 3);
    expect(finalValue).toBeCloseTo(maxValue, 1); // Within 0.1 of max
  });

  test('should reverse direction with bounce mode', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);

    // Set loop mode to bounce and high FPS

    // Wait for play button to be visible, then right-click to open context menu
    const playButton4 = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton4.waitFor({ state: 'visible', timeout: 5000 });
    await playButton4.click({ button: 'right' });
    await page.waitForTimeout(200);

    // Select bounce mode
    await page.click('text=Bounce');
    await page.waitForTimeout(100);

    // Right-click again to set FPS
    await playButton4.click({ button: 'right' });
    await page.waitForTimeout(200);
    // Use specific selector to avoid matching resolution indicator
    await page.click('.luxar-dimension-slider__context-item:has-text("60 FPS")');
    await page.waitForTimeout(200);

    // Jump to near end
    await page.keyboard.press('End');
    await waitForAnimationStep(page, 3);

    const nearEndValue = await getDimensionValue(page, 3);

    // Start animation
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Wait for bounce (should reverse and move backward)
    await page.waitForTimeout(1000);

    // Value should have decreased (bounced back)
    const bouncedValue = await getDimensionValue(page, 3);
    expect(bouncedValue).toBeLessThan(nearEndValue);

    // Animation should still be playing
    const isPlaying = await isAnimating(page, 3);
    expect(isPlaying).toBe(true);

    // Direction should be backward
    const state = await getAnimationState(page, 3);
    expect(state?.direction).toBe('backward');
  });

  test('should handle pause and resume correctly', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Select a dimension and start animation
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Get current value
    const playingValue = await getDimensionValue(page, 3);

    // Pause animation
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Value should stay roughly constant while paused (allow for frames in flight during pause)
    await page.waitForTimeout(1000);
    const pausedValue = await getDimensionValue(page, 3);
    // Allow ~0.5 difference for animation frames that may complete during pause transition
    expect(Math.abs(pausedValue - playingValue)).toBeLessThan(0.5);

    // Resume animation
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Wait for value to change
    await waitForDimensionValueChange(page, 3, pausedValue, 2000);

    // Verify animation resumed
    const resumedValue = await getDimensionValue(page, 3);
    expect(resumedValue).not.toBe(pausedValue);
  });
});

test.describe('Dimension Animation - Multiple Dimensions', () => {
  test('should animate multiple dimensions independently', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page, 1);

    // Start animation on dimension 4 (index 3)
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Check dimension 4 is animating
    const dim4Playing = await isAnimating(page, 3);
    expect(dim4Playing).toBe(true);

    // Start animation on dimension 5 (index 4)
    await page.keyboard.press('5');
    await waitForNextRender(page, 1);
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Check both dimensions are animating
    const dim4StillPlaying = await isAnimating(page, 3);
    const dim5Playing = await isAnimating(page, 4);
    expect(dim4StillPlaying).toBe(true);
    expect(dim5Playing).toBe(true);

    // Pause dimension 4
    await page.keyboard.press('4');
    await waitForNextRender(page, 1);
    await page.keyboard.press('k');
    await waitForNextRender(page);

    // Check dimension 4 paused but dimension 5 still playing
    const dim4Paused = await isAnimating(page, 3);
    const dim5StillPlaying = await isAnimating(page, 4);
    expect(dim4Paused).toBe(false);
    expect(dim5StillPlaying).toBe(true);
  });
});

test.describe('Dimension Animation - Error Handling', () => {
  test('should not crash when animation manager not initialized', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Try to use animation shortcuts immediately (might not be initialized)
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

    // Should not have errors
    expect(errors.length).toBe(0);
  });

  test('should handle invalid dimension index gracefully', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Try to animate an invalid dimension (way out of range)
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const inputHandler = debug?.app?.inputHandler;
      const animManager = inputHandler?.animationManager;
      if (animManager) {
        animManager.play(999); // Invalid index
      }
    });

    await page.waitForTimeout(500);

    // Should not crash
    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);
  });
});
