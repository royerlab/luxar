/**
 * Comprehensive E2E Tests for Keyboard Input System
 *
 * Tests the unified binding registration system including:
 * - Fly controls with all modifier combinations (WASD, Shift+W, Alt+W)
 * - keyupHandler feature (Shift release, movement stop)
 * - Toggle shortcuts (C, P, I, M, F, N, O)
 * - Browser shortcuts preservation (Cmd+R, Cmd+F)
 * - Context passthrough (global shortcuts in fly mode)
 * - Escape key behavior
 *
 * This test suite provides critical coverage for the input system refactoring
 * that migrated from dual-track architecture to unified binding system.
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady, getLuxarState } from './helpers';

// Test datasets
const DATASETS = {
  sliders5D: 'http://localhost:9000/packages/luxar/examples/dimension_sliders_5d_example.zarr',
};

test.describe('Keyboard Input System - Fly Controls', () => {
  test('should move forward with W key in fly mode', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Switch to fly mode
    await page.keyboard.press('v');
    await page.keyboard.press('v'); // orbit → arcball → fly
    await page.waitForTimeout(300);

    // Get initial camera position
    const initialState = await getLuxarState(page);
    const initialZ = initialState.camera.position.z;

    // Press and hold W for forward movement
    await page.keyboard.down('w');
    await page.waitForTimeout(200); // Move for 200ms
    await page.keyboard.up('w');

    // Get new camera position
    const finalState = await getLuxarState(page);
    const finalZ = finalState.camera.position.z;

    // Camera should have moved forward (Z decreased)
    expect(finalZ).toBeLessThan(initialZ);
  });

  test('should move faster with Shift+W (speed boost)', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Switch to fly mode
    await page.keyboard.press('v');
    await page.keyboard.press('v');
    await page.waitForTimeout(300);

    // Test 1: Normal W speed
    const initial1 = await getLuxarState(page);
    const startZ1 = initial1.camera.position.z;

    await page.keyboard.down('w');
    await page.waitForTimeout(200);
    await page.keyboard.up('w');
    await page.waitForTimeout(100);

    const final1 = await getLuxarState(page);
    const normalDistance = Math.abs(final1.camera.position.z - startZ1);

    // Reset position for fair comparison
    await page.keyboard.press('f'); // Recenter
    await page.waitForTimeout(500);

    // Test 2: Shift+W speed
    const initial2 = await getLuxarState(page);
    const startZ2 = initial2.camera.position.z;

    await page.keyboard.down('Shift');
    await page.keyboard.down('w');
    await page.waitForTimeout(200);
    await page.keyboard.up('w');
    await page.keyboard.up('Shift');
    await page.waitForTimeout(100);

    const final2 = await getLuxarState(page);
    const boostDistance = Math.abs(final2.camera.position.z - startZ2);

    // Shift+W should move faster than normal W
    expect(boostDistance).toBeGreaterThan(normalDistance);
  });

  test('should move vertically with Alt+W and Alt+S', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Switch to fly mode
    await page.keyboard.press('v');
    await page.keyboard.press('v');
    await page.waitForTimeout(300);

    // Get initial Y position
    const initialState = await getLuxarState(page);
    const initialY = initialState.camera.position.y;

    // Press Alt+W to move up
    await page.keyboard.down('Alt');
    await page.keyboard.down('w');
    await page.waitForTimeout(200);
    await page.keyboard.up('w');
    await page.keyboard.up('Alt');
    await page.waitForTimeout(100);

    // Get Y position after moving up
    const afterUpState = await getLuxarState(page);
    const afterUpY = afterUpState.camera.position.y;

    // Y should have increased (moved up)
    expect(afterUpY).toBeGreaterThan(initialY);

    // Press Alt+S to move down
    await page.keyboard.down('Alt');
    await page.keyboard.down('s');
    await page.waitForTimeout(400); // Move longer to get back below initial
    await page.keyboard.up('s');
    await page.keyboard.up('Alt');
    await page.waitForTimeout(100);

    // Get Y position after moving down
    const afterDownState = await getLuxarState(page);
    const afterDownY = afterDownState.camera.position.y;

    // Y should have decreased (moved down) from the up position
    expect(afterDownY).toBeLessThan(afterUpY);
  });

  test('should stop movement when W key is released', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Switch to fly mode with inertia OFF (immediate stop)
    await page.keyboard.press('v');
    await page.keyboard.press('v');
    await page.waitForTimeout(300);

    // Disable inertia for predictable stop behavior
    await page.keyboard.press('i');
    await page.waitForTimeout(200);

    // Press W to start movement
    await page.keyboard.down('w');
    await page.waitForTimeout(200);

    // Release W
    await page.keyboard.up('w');
    await page.waitForTimeout(100);

    // Get position after release
    const pos1 = await getLuxarState(page);
    const z1 = pos1.camera.position.z;

    // Wait and check if still moving
    await page.waitForTimeout(200);
    const pos2 = await getLuxarState(page);
    const z2 = pos2.camera.position.z;

    // Position should be stable (not changing significantly)
    // With inertia off and high damping, movement should stop quickly
    const movement = Math.abs(z2 - z1);
    expect(movement).toBeLessThan(0.1); // Very little movement after release
  });

  test('should support strafe movement with A and D keys', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Switch to fly mode
    await page.keyboard.press('v');
    await page.keyboard.press('v');
    await page.waitForTimeout(300);

    // Get initial X position
    const initialState = await getLuxarState(page);
    const initialX = initialState.camera.position.x;

    // Press A to strafe left
    await page.keyboard.down('a');
    await page.waitForTimeout(200);
    await page.keyboard.up('a');
    await page.waitForTimeout(100);

    const afterLeftState = await getLuxarState(page);
    const afterLeftX = afterLeftState.camera.position.x;

    // X position should have changed (strafed)
    expect(Math.abs(afterLeftX - initialX)).toBeGreaterThan(0.1);
  });
});

test.describe('Keyboard Input System - keyupHandler Feature', () => {
  test('should not double-trigger toggle actions on key release', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Get initial cinematic mode state (should be off)
    const initialCinematic = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug?.app?.renderingControls?.getCinematicMode?.() || false;
    });

    // Press C to toggle cinematic mode ON
    await page.keyboard.down('c');
    await page.waitForTimeout(200);

    // Check state while key is held
    const whilePressedCinematic = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug?.app?.renderingControls?.getCinematicMode?.() || false;
    });

    // Release C
    await page.keyboard.up('c');
    await page.waitForTimeout(200);

    // Check state after release
    const afterReleaseCinematic = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug?.app?.renderingControls?.getCinematicMode?.() || false;
    });

    // State should toggle from initial (OFF) to ON when pressed
    expect(whilePressedCinematic).toBe(!initialCinematic);

    // State should STAY the same after release (not toggle back)
    expect(afterReleaseCinematic).toBe(whilePressedCinematic);
  });

  test('Shift key should not toggle panels on release', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Get all panel states before
    const statesBefore = await page.evaluate(() => {
      return {
        help: !!document.getElementById('help-overlay'),
        rendering: !!document.querySelector('.luxar-gui'),
        performance: !!document.querySelector('.stats'),
      };
    });

    // Press and release Shift
    await page.keyboard.down('Shift');
    await page.waitForTimeout(100);
    await page.keyboard.up('Shift');
    await page.waitForTimeout(200);

    // Get panel states after
    const statesAfter = await page.evaluate(() => {
      return {
        help: !!document.getElementById('help-overlay'),
        rendering: !!document.querySelector('.luxar-gui'),
        performance: !!document.querySelector('.stats'),
      };
    });

    // No panels should have toggled
    expect(statesAfter.help).toBe(statesBefore.help);
    expect(statesAfter.rendering).toBe(statesBefore.rendering);
    expect(statesAfter.performance).toBe(statesBefore.performance);
  });
});

test.describe('Keyboard Input System - Toggle Shortcuts', () => {
  test('C key should toggle cinematic mode persistently', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Press C to toggle ON
    await page.keyboard.press('c');
    await page.waitForTimeout(300);

    const afterFirstPress = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      // Check if any cinematic effects are enabled
      const rendering = debug?.app?.renderingControls?.getSettings?.();
      return rendering?.vignetteEnabled || rendering?.chromaticLensDistortionEnabled || false;
    });

    // Should have toggled ON (or at least changed state)
    // Press C again to toggle OFF
    await page.keyboard.press('c');
    await page.waitForTimeout(300);

    const afterSecondPress = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const rendering = debug?.app?.renderingControls?.getSettings?.();
      return rendering?.vignetteEnabled || rendering?.chromaticLensDistortionEnabled || false;
    });

    // Should have toggled back to opposite state
    expect(afterSecondPress).toBe(!afterFirstPress);
  });

  test('P key should toggle performance stats panel', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Check initial stats visibility
    const initialVisible = await page.evaluate(() => {
      const stats = document.querySelector('.stats') as HTMLElement;
      return stats && stats.style.display !== 'none';
    });

    // Toggle stats
    await page.keyboard.press('p');
    await page.waitForTimeout(300);

    const afterToggle = await page.evaluate(() => {
      const stats = document.querySelector('.stats') as HTMLElement;
      return stats && stats.style.display !== 'none';
    });

    // Should have toggled
    expect(afterToggle).toBe(!initialVisible);

    // Toggle back
    await page.keyboard.press('p');
    await page.waitForTimeout(300);

    const afterSecondToggle = await page.evaluate(() => {
      const stats = document.querySelector('.stats') as HTMLElement;
      return stats && stats.style.display !== 'none';
    });

    // Should be back to initial state
    expect(afterSecondToggle).toBe(initialVisible);
  });

  test('I key should toggle inertial mode in fly mode', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Switch to fly mode
    await page.keyboard.press('v');
    await page.keyboard.press('v');
    await page.waitForTimeout(300);

    // Get initial inertial mode state
    const initialInertial = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const controls = debug?.app?.controls?.getFlyControls?.();
      return controls?.inertialMode ?? null;
    });

    // Should have a valid boolean state
    expect(typeof initialInertial).toBe('boolean');

    // Toggle inertial mode
    await page.keyboard.press('i');
    await page.waitForTimeout(300);

    const afterToggle = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const controls = debug?.app?.controls?.getFlyControls?.();
      return controls?.inertialMode ?? null;
    });

    // Should have toggled
    expect(afterToggle).toBe(!initialInertial);
  });

  test('F key should recenter camera on scene', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Move camera away from center
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      debug.camera.position.set(100, 100, 100);
    });
    await page.waitForTimeout(100);

    // Press F to recenter
    await page.keyboard.press('f');
    await page.waitForTimeout(1000); // Wait for smooth animation

    const state = await getLuxarState(page);
    const distance = Math.sqrt(
      state.camera.position.x ** 2 + state.camera.position.y ** 2 + state.camera.position.z ** 2
    );

    // Camera should be closer to origin after recentering
    expect(distance).toBeLessThan(50); // Much closer than 100,100,100
  });

  test('N key should toggle dimension sliders (if nD dataset)', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Wait for dimension sliders to initialize
    await page.waitForTimeout(1000);

    // Check initial slider visibility (not existence - the element exists but may be hidden)
    const initialVisible = await page.evaluate(() => {
      const sliders = document.querySelector('.luxar-dimension-sliders');
      if (!sliders) return false;
      // Check if actually visible (not display: none)
      return window.getComputedStyle(sliders).display !== 'none';
    });

    // For 5D dataset, sliders should start visible
    expect(initialVisible).toBe(true);

    // Toggle sliders (hide them)
    await page.keyboard.press('n');
    await page.waitForTimeout(300);

    const afterFirstToggle = await page.evaluate(() => {
      const sliders = document.querySelector('.luxar-dimension-sliders');
      if (!sliders) return false;
      return window.getComputedStyle(sliders).display !== 'none';
    });

    // After toggle, should be hidden
    expect(afterFirstToggle).toBe(false);

    // Toggle again (show them)
    await page.keyboard.press('n');
    await page.waitForTimeout(300);

    const afterSecondToggle = await page.evaluate(() => {
      const sliders = document.querySelector('.luxar-dimension-sliders');
      if (!sliders) return false;
      return window.getComputedStyle(sliders).display !== 'none';
    });

    // After second toggle, should be visible again
    expect(afterSecondToggle).toBe(true);
  });
});

test.describe('Keyboard Input System - Context Passthrough', () => {
  test('H key should work in fly mode (passthrough)', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Switch to fly mode
    await page.keyboard.press('v');
    await page.keyboard.press('v');
    await page.waitForTimeout(300);

    // Verify we're in fly mode
    const controlType = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug?.app?.controls?.getControlType?.() || null;
    });
    expect(controlType).toBe('fly');

    // Press H to show help (should work via passthrough to NAVIGATION context)
    await page.keyboard.press('h');
    await page.waitForTimeout(500);

    // Check if help is visible
    const helpVisible = await page.evaluate(() => {
      const helpOverlay = document.querySelector('#help-overlay, .help-overlay');
      return !!helpOverlay;
    });

    // Help should appear even in fly mode
    expect(helpVisible).toBe(true);
  });

  test('WASD keys should be blocked in orbit mode', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Ensure we're in orbit mode
    const controlType = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug?.app?.controls?.getControlType?.() || null;
    });

    if (controlType === 'fly') {
      // Switch back to orbit
      await page.keyboard.press('v');
      await page.waitForTimeout(300);
    }

    // Get camera position
    const initialState = await getLuxarState(page);
    const initialPos = initialState.camera.position;

    // Try to press W (should be blocked in orbit mode)
    await page.keyboard.down('w');
    await page.waitForTimeout(200);
    await page.keyboard.up('w');
    await page.waitForTimeout(100);

    const finalState = await getLuxarState(page);
    const finalPos = finalState.camera.position;

    // Camera should NOT have moved (WASD blocked in orbit mode)
    const distance = Math.sqrt(
      (finalPos.x - initialPos.x) ** 2 +
        (finalPos.y - initialPos.y) ** 2 +
        (finalPos.z - initialPos.z) ** 2
    );
    expect(distance).toBeLessThan(0.01); // Essentially no movement
  });
});

test.describe('Keyboard Input System - Browser Shortcuts Protection', () => {
  test('Cmd+R should not be blocked by R key binding', async ({ page, browserName }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Listen for navigation (page reload)
    page.on('framenavigated', () => {
      // Navigation occurred - test will verify this with timeout
    });

    // Determine modifier key based on platform
    const modifierKey = browserName === 'webkit' ? 'Meta' : 'Control';

    // Press Cmd+R or Ctrl+R (should trigger page refresh, not be blocked)
    await page.keyboard.press(`${modifierKey}+r`);
    await page.waitForTimeout(1000);

    // Navigation should have been attempted (not blocked)
    // Note: In test environment, reload might be blocked for other reasons,
    // but the key point is that our input handler didn't block it
    // We verify by checking that the rendering controls didn't toggle
    const renderingToggled = await page.evaluate(() => {
      // If our binding blocked the shortcut, rendering controls would show
      // If browser shortcut worked, page would reload (rendering controls reset)
      const panel = document.querySelector('.luxar-gui');
      return panel && (panel as HTMLElement).style.display !== 'none';
    });

    // Rendering controls should NOT have toggled (browser shortcut took precedence)
    expect(renderingToggled).toBe(false);
  });

  test('Ctrl+1 should not be blocked by 1 key binding', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);
    await page.waitForTimeout(1000);

    // Get current dimension state
    const initialDimension = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug?.app?.sceneDimsManager?.getSelectedDimension?.() || null;
    });

    // Press Ctrl+1 (browser tab switch, should NOT select dimension)
    await page.keyboard.press('Control+1');
    await page.waitForTimeout(300);

    const afterCtrl1 = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug?.app?.sceneDimsManager?.getSelectedDimension?.() || null;
    });

    // Dimension should NOT have changed (Ctrl+1 should be allowed to browser)
    expect(afterCtrl1).toBe(initialDimension);
  });
});

test.describe('Keyboard Input System - Escape Key', () => {
  test('Escape should close help overlay', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Show help
    await page.keyboard.press('h');
    await page.waitForTimeout(300);

    // Verify help is visible
    const helpVisible = await page.evaluate(() => {
      return !!document.getElementById('help-overlay');
    });
    expect(helpVisible).toBe(true);

    // Press Escape to close
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);

    // Verify help is closed
    const helpClosed = await page.evaluate(() => {
      return !!document.getElementById('help-overlay');
    });
    expect(helpClosed).toBe(false);
  });
});

test.describe('Keyboard Input System - Modifier Combinations', () => {
  test('should handle Shift+Alt+W (fast vertical movement)', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Switch to fly mode
    await page.keyboard.press('v');
    await page.keyboard.press('v');
    await page.waitForTimeout(300);

    // Get initial Y position
    const initialState = await getLuxarState(page);
    const initialY = initialState.camera.position.y;

    // Press Shift+Alt+W for fast vertical movement
    await page.keyboard.down('Shift');
    await page.keyboard.down('Alt');
    await page.keyboard.down('w');
    await page.waitForTimeout(200);
    await page.keyboard.up('w');
    await page.keyboard.up('Alt');
    await page.keyboard.up('Shift');
    await page.waitForTimeout(100);

    const finalState = await getLuxarState(page);
    const finalY = finalState.camera.position.y;

    // Should have moved up significantly (faster than Alt+W alone)
    const movement = finalY - initialY;
    expect(movement).toBeGreaterThan(0.5); // Significant upward movement
  });

  test('Arrow keys should work in fly mode for camera rotation', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Switch to fly mode
    await page.keyboard.press('v');
    await page.keyboard.press('v');
    await page.waitForTimeout(300);

    // Get initial camera rotation
    const initialState = await getLuxarState(page);
    const initialRotation = initialState.camera.rotation;

    // Press ArrowUp to look up
    await page.keyboard.down('ArrowUp');
    await page.waitForTimeout(200);
    await page.keyboard.up('ArrowUp');
    await page.waitForTimeout(100);

    const finalState = await getLuxarState(page);
    const finalRotation = finalState.camera.rotation;

    // Camera rotation should have changed
    const rotationChanged =
      Math.abs(finalRotation.x - initialRotation.x) > 0.01 ||
      Math.abs(finalRotation.y - initialRotation.y) > 0.01;

    expect(rotationChanged).toBe(true);
  });
});

test.describe('Keyboard Input System - Case Sensitivity', () => {
  test('uppercase H should work same as lowercase h', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.sliders5D}&debug`);
    await waitForLuxarReady(page);

    // Press uppercase H (Shift+h)
    await page.keyboard.press('H');
    await page.waitForTimeout(300);

    // Help should appear (case-insensitive)
    const helpVisible = await page.evaluate(() => {
      return !!document.getElementById('help-overlay');
    });

    expect(helpVisible).toBe(true);
  });
});
