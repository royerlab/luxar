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
import { waitForLuxarReady, getLuxarState, waitForDataLoaded } from './helpers';

// Test dataset with multiple dimensions for animation
const DATASET = 'http://localhost:9000/packages/luxar/examples/dimension_sliders_5d_example.zarr';

/**
 * Helper to get dimension animation state
 */
async function getAnimationState(page: any, dimIndex: number): Promise<any> {
  return await page.evaluate((idx: number) => {
    const debug = (window as any).__luxarDebug;
    const inputHandler = debug?.app?.inputHandler;
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
    const inputHandler = debug?.app?.inputHandler;
    const animManager = inputHandler?.animationManager;
    if (!animManager) return false;
    return animManager.isAnimating(idx);
  }, dimIndex);
}

/**
 * Helper to get dimension value
 */
async function getDimensionValue(page: any, dimIndex: number): Promise<number> {
  return await page.evaluate((idx: number) => {
    const debug = (window as any).__luxarDebug;
    const sceneDimsManager = debug?.app?.sceneDimsManager;
    if (!sceneDimsManager) return -1;
    const dims = sceneDimsManager.getDims();
    if (!dims) return -1;
    return dims.currentStep[idx];
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

    // Open dimension sliders
    await page.keyboard.press('n');

    // Wait for play button to be visible
    const playButton = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton.waitFor({ state: 'visible', timeout: 5000 });

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

    // Open dimension sliders
    await page.keyboard.press('n');
    await page.waitForTimeout(300);

    // Select a dimension
    await page.keyboard.press('4');
    await page.waitForTimeout(100);

    // Get initial play button text
    const initialText = await page.evaluate(() => {
      const playBtn = document.querySelector('.luxar-dimension-slider__play-btn');
      return playBtn?.textContent?.trim();
    });

    expect(initialText).toBe('▶');

    // Click play button
    await page.click('.luxar-dimension-slider__play-btn');
    await page.waitForTimeout(200);

    // Check button text changed to pause
    const pauseText = await page.evaluate(() => {
      const playBtn = document.querySelector('.luxar-dimension-slider__play-btn');
      return playBtn?.textContent?.trim();
    });

    expect(pauseText).toBe('⏸');

    // Click again to pause
    await page.click('.luxar-dimension-slider__play-btn');
    await page.waitForTimeout(200);

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

    // Open dimension sliders
    await page.keyboard.press('n');
    await page.waitForTimeout(300);

    // Select a dimension
    await page.keyboard.press('4');
    await page.waitForTimeout(100);

    // Right-click play button to open context menu
    const playButton = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton.click({ button: 'right' });
    await page.waitForTimeout(200);

    // Click on 30 FPS option in context menu
    await page.click('text=30 FPS');
    await page.waitForTimeout(200);

    // Start animation
    await page.click('.luxar-dimension-slider__play-btn');
    await page.waitForTimeout(200);

    // Check animation state has correct FPS
    const state = await getAnimationState(page, 3); // 4th dimension = index 3
    expect(state).not.toBeNull();
    expect(state?.targetFPS).toBe(30);
  });

  test('should change loop mode via selector', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Open dimension sliders
    await page.keyboard.press('n');
    await page.waitForTimeout(300);

    // Select a dimension
    await page.keyboard.press('4');
    await page.waitForTimeout(100);

    // Right-click play button to open context menu
    const playButton = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton.click({ button: 'right' });
    await page.waitForTimeout(200);

    // Click on Bounce option in context menu
    await page.click('text=Bounce');
    await page.waitForTimeout(200);

    // Start animation
    await page.click('.luxar-dimension-slider__play-btn');
    await page.waitForTimeout(200);

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

    // Select a dimension
    await page.keyboard.press('4');
    await page.waitForTimeout(100);

    // Press K to start animation
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

    // Check animation is playing
    const isPlaying = await isAnimating(page, 3);
    expect(isPlaying).toBe(true);

    // Press K again to pause
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

    // Check animation is paused
    const isPaused = await isAnimating(page, 3);
    expect(isPaused).toBe(false);
  });

  test('should jump to start with Home key', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Select a dimension
    await page.keyboard.press('4');
    await page.waitForTimeout(100);

    // Navigate forward a few steps
    await page.keyboard.press(']');
    await page.waitForTimeout(200);
    await page.keyboard.press(']');
    await page.waitForTimeout(200);

    // Get current value (should not be at start)
    const beforeValue = await getDimensionValue(page, 3);
    expect(beforeValue).toBeGreaterThan(0);

    // Press Home to jump to start
    await page.keyboard.press('Home');
    await page.waitForTimeout(200);

    // Check dimension is at start (value 0)
    const afterValue = await getDimensionValue(page, 3);
    expect(afterValue).toBe(0);
  });

  test('should jump to end with End key', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Select a dimension
    await page.keyboard.press('4');
    await page.waitForTimeout(100);

    // Get the max value for this dimension
    const maxValue = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const sceneDimsManager = debug?.app?.sceneDimsManager;
      if (!sceneDimsManager) return -1;
      const ranges = sceneDimsManager.getDimensionRanges();
      if (!ranges) return -1;
      return ranges[3]?.[1] ?? -1; // 4th dimension = index 3, max = [1]
    });

    // Press End to jump to end
    await page.keyboard.press('End');
    await page.waitForTimeout(200);

    // Check dimension is at end
    const value = await getDimensionValue(page, 3);
    expect(value).toBe(maxValue);
  });

  test('should increase speed with Shift+Up', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Select a dimension and start animation
    await page.keyboard.press('4');
    await page.waitForTimeout(100);
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

    // Get initial FPS
    const initialState = await getAnimationState(page, 3);
    const initialFPS = initialState?.targetFPS ?? 10;

    // Press Shift+Up to increase speed
    await page.keyboard.press('Shift+ArrowUp');
    await page.waitForTimeout(200);

    // Check FPS increased
    const newState = await getAnimationState(page, 3);
    const newFPS = newState?.targetFPS ?? 10;
    expect(newFPS).toBeGreaterThan(initialFPS);
  });

  test('should decrease speed with Shift+Down', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Select a dimension and start animation at high FPS
    await page.keyboard.press('4');
    await page.waitForTimeout(100);

    // Set FPS to 30 first via UI
    await page.keyboard.press('n'); // Open sliders
    await page.waitForTimeout(200);

    // Right-click play button and select 30 FPS
    const playButton = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton.click({ button: 'right' });
    await page.waitForTimeout(200);
    await page.click('text=30 FPS');
    await page.waitForTimeout(200);

    // Start animation
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

    // Get initial FPS (should be 30)
    const initialState = await getAnimationState(page, 3);
    const initialFPS = initialState?.targetFPS ?? 30;

    // Press Shift+Down to decrease speed
    await page.keyboard.press('Shift+ArrowDown');
    await page.waitForTimeout(200);

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

    // Select a dimension
    await page.keyboard.press('4');
    await page.waitForTimeout(100);

    // Get initial value
    const initialValue = await getDimensionValue(page, 3);

    // Start animation at high FPS for faster test
    await page.keyboard.press('n'); // Open sliders
    await page.waitForTimeout(200);

    // Right-click play button and select 30 FPS
    const playButton1 = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton1.click({ button: 'right' });
    await page.waitForTimeout(200);
    await page.click('text=30 FPS');
    await page.waitForTimeout(200);

    await page.keyboard.press('k'); // Start animation
    await page.waitForTimeout(200);

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

    // Select a dimension
    await page.keyboard.press('4');
    await page.waitForTimeout(100);

    // Set loop mode to loop and high FPS
    await page.keyboard.press('n'); // Open sliders
    await page.waitForTimeout(200);

    // Right-click play button to open context menu
    const playButton2 = await page.locator('.luxar-dimension-slider__play-btn').first();
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
    await page.waitForTimeout(200);

    const nearEndValue = await getDimensionValue(page, 3);

    // Start animation
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

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

    // Select a dimension
    await page.keyboard.press('4');
    await page.waitForTimeout(100);

    // Set loop mode to once and high FPS
    await page.keyboard.press('n'); // Open sliders
    await page.waitForTimeout(200);

    // Right-click play button to open context menu
    const playButton3 = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton3.click({ button: 'right' });
    await page.waitForTimeout(200);

    // Select once mode
    await page.click('text=Once');
    await page.waitForTimeout(100);

    // Right-click again to set FPS
    await playButton3.click({ button: 'right' });
    await page.waitForTimeout(200);
    await page.click('text=60 FPS');
    await page.waitForTimeout(200);

    // Jump to near end (not quite at end)
    const maxValue = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const sceneDimsManager = debug?.app?.sceneDimsManager;
      const ranges = sceneDimsManager?.getDimensionRanges();
      return ranges?.[3]?.[1] ?? 10;
    });

    await page.evaluate((max: number) => {
      const debug = (window as any).__luxarDebug;
      const sceneDimsManager = debug?.app?.sceneDimsManager;
      sceneDimsManager?.setDimensionValue(3, Math.max(0, max - 2));
    }, maxValue);
    await page.waitForTimeout(200);

    // Start animation
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

    // Wait for animation to complete (should stop at max)
    await page.waitForTimeout(1500);

    // Animation should have stopped
    const isPlaying = await isAnimating(page, 3);
    expect(isPlaying).toBe(false);

    // Value should be at max
    const finalValue = await getDimensionValue(page, 3);
    expect(finalValue).toBe(maxValue);
  });

  test('should reverse direction with bounce mode', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Select a dimension
    await page.keyboard.press('4');
    await page.waitForTimeout(100);

    // Set loop mode to bounce and high FPS
    await page.keyboard.press('n'); // Open sliders
    await page.waitForTimeout(200);

    // Right-click play button to open context menu
    const playButton4 = await page.locator('.luxar-dimension-slider__play-btn').first();
    await playButton4.click({ button: 'right' });
    await page.waitForTimeout(200);

    // Select bounce mode
    await page.click('text=Bounce');
    await page.waitForTimeout(100);

    // Right-click again to set FPS
    await playButton4.click({ button: 'right' });
    await page.waitForTimeout(200);
    await page.click('text=60 FPS');
    await page.waitForTimeout(200);

    // Jump to near end
    await page.keyboard.press('End');
    await page.waitForTimeout(200);

    const nearEndValue = await getDimensionValue(page, 3);

    // Start animation
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

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

    // Select a dimension and start animation
    await page.keyboard.press('4');
    await page.waitForTimeout(100);
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

    // Get current value
    const playingValue = await getDimensionValue(page, 3);

    // Pause animation
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

    // Value should stay constant while paused
    await page.waitForTimeout(1000);
    const pausedValue = await getDimensionValue(page, 3);
    expect(pausedValue).toBe(playingValue);

    // Resume animation
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

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

    // Start animation on dimension 4 (index 3)
    await page.keyboard.press('4');
    await page.waitForTimeout(100);
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

    // Check dimension 4 is animating
    const dim4Playing = await isAnimating(page, 3);
    expect(dim4Playing).toBe(true);

    // Start animation on dimension 5 (index 4)
    await page.keyboard.press('5');
    await page.waitForTimeout(100);
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

    // Check both dimensions are animating
    const dim4StillPlaying = await isAnimating(page, 3);
    const dim5Playing = await isAnimating(page, 4);
    expect(dim4StillPlaying).toBe(true);
    expect(dim5Playing).toBe(true);

    // Pause dimension 4
    await page.keyboard.press('4');
    await page.waitForTimeout(100);
    await page.keyboard.press('k');
    await page.waitForTimeout(200);

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
