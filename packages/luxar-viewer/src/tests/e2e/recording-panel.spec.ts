/**
 * E2E tests for the Recording Panel (screenshot & video capture)
 *
 * Tests:
 * - Panel toggle via T key
 * - Quick screenshot via G key
 * - Panel UI structure (button, mode toggle, advanced options)
 * - Screenshot capture flow
 * - Video recording confirmation dialog
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForNextRender,
  dismissDatasetBrowser,
  assertNoConsoleErrors,
} from './helpers';

test.describe('Recording Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);
    // Dismiss the dataset browser modal so keyboard shortcuts reach the app
    await dismissDatasetBrowser(page);
  });

  // Every test that interacts with the viewer should fail on JS exceptions.
  // Without this, a renderer/recorder crash can be silent while the panel
  // toggle still works.
  test.afterEach(async ({ page }) => {
    await assertNoConsoleErrors(page);
  });

  test('should toggle recording panel with T key', async ({ page }) => {
    // Panel should not be visible initially
    const panelBefore = await page.evaluate(() => {
      return (window as any).__luxarDebug.recordingPanel?.isVisible() ?? false;
    });
    expect(panelBefore).toBe(false);

    // Press T to show panel
    await page.keyboard.press('t');
    await waitForNextRender(page);

    const panelAfterShow = await page.evaluate(() => {
      return (window as any).__luxarDebug.recordingPanel?.isVisible() ?? false;
    });
    expect(panelAfterShow).toBe(true);

    // Verify panel DOM is visible
    const panelElement = await page.locator('.luxar-gui.luxar-recording-panel');
    await expect(panelElement).toBeVisible();

    // Press T again to hide
    await page.keyboard.press('t');
    await waitForNextRender(page);

    const panelAfterHide = await page.evaluate(() => {
      return (window as any).__luxarDebug.recordingPanel?.isVisible() ?? false;
    });
    expect(panelAfterHide).toBe(false);
  });

  test('should close recording panel with Escape', async ({ page }) => {
    // Open panel
    await page.keyboard.press('t');
    await waitForNextRender(page);

    const isVisible = await page.evaluate(() => {
      return (window as any).__luxarDebug.recordingPanel?.isVisible() ?? false;
    });
    expect(isVisible).toBe(true);

    // Press Escape to close
    await page.keyboard.press('Escape');
    await waitForNextRender(page);

    const isVisibleAfter = await page.evaluate(() => {
      return (window as any).__luxarDebug.recordingPanel?.isVisible() ?? false;
    });
    expect(isVisibleAfter).toBe(false);
  });

  test('should have correct panel structure', async ({ page }) => {
    // Open panel
    await page.keyboard.press('t');
    await waitForNextRender(page);

    // Verify title
    const titleText = await page
      .locator('.luxar-gui.luxar-recording-panel .luxar-gui__title')
      .textContent();
    expect(titleText).toContain('Recording');

    // Verify capture button exists with red styling
    const captureBtn = page.locator('.luxar-recording-btn');
    await expect(captureBtn).toBeVisible();

    // Verify mode dropdown exists
    const modeSelect = await page.evaluate(() => {
      const panel = document.querySelector('.luxar-gui.luxar-recording-panel');
      const selects = panel?.querySelectorAll('select');
      if (!selects) return null;
      const selectArr = Array.from(selects);
      for (let i = 0; i < selectArr.length; i++) {
        const options = Array.from(selectArr[i].options).map((o: HTMLOptionElement) => o.text);
        if (options.includes('Image') && options.includes('Video') && options.includes('Turntable'))
          return options;
      }
      return null;
    });
    expect(modeSelect).toBeTruthy();

    // Verify Advanced Options folder exists (closed by default)
    const advancedFolder = await page.evaluate(() => {
      const panel = document.querySelector('.luxar-gui.luxar-recording-panel');
      const folders = Array.from(panel?.querySelectorAll('.luxar-gui__folder-title') ?? []);
      for (let i = 0; i < folders.length; i++) {
        if (folders[i].textContent?.includes('Advanced')) return true;
      }
      return false;
    });
    expect(advancedFolder).toBe(true);
  });

  test('should trigger screenshot with G key', async ({ page }) => {
    // Set up download listener
    const downloadPromise = page.waitForEvent('download', { timeout: 10000 });

    // Press G for quick screenshot
    await page.keyboard.press('g');

    // Wait for download to start
    const download = await downloadPromise;
    const filename = download.suggestedFilename();

    // Verify filename format
    expect(filename).toMatch(/^luxar-capture-\d{4}-\d{2}-\d{2}-\d{6}\.\w+$/);
  });

  test('should trigger screenshot from panel button', async ({ page }) => {
    // Open panel
    await page.keyboard.press('t');
    await waitForNextRender(page);

    // The panel now defaults to Video mode, where the action button starts a
    // recording (confirmation dialog) rather than downloading a still. Select
    // Image mode first so the button captures a screenshot.
    await page.evaluate(() => {
      const panel = document.querySelector('.luxar-gui.luxar-recording-panel');
      const selects = Array.from(panel?.querySelectorAll('select') ?? []);
      for (const select of selects) {
        const options = Array.from(select.options) as HTMLOptionElement[];
        const imageOption = options.find((o: HTMLOptionElement) => o.text === 'Image');
        if (imageOption) {
          select.value = imageOption.value;
          select.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }
    });
    await waitForNextRender(page);

    // Set up download listener
    const downloadPromise = page.waitForEvent('download', { timeout: 10000 });

    // Click the capture button
    await page.locator('.luxar-recording-btn button').click();

    // Wait for download
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/^luxar-capture-.*\.webp$/);
  });

  test('should show confirmation dialog for video recording', async ({ page }) => {
    // Open panel
    await page.keyboard.press('t');
    await waitForNextRender(page);

    // Switch to video mode
    await page.evaluate(() => {
      const panel = document.querySelector('.luxar-gui.luxar-recording-panel');
      const selects = Array.from(panel?.querySelectorAll('select') ?? []);
      for (let i = 0; i < selects.length; i++) {
        const select = selects[i];
        const options = Array.from(select.options) as HTMLOptionElement[];
        const videoOption = options.find((o: HTMLOptionElement) => o.text === 'Video');
        if (videoOption) {
          select.value = videoOption.value;
          select.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }
    });
    await waitForNextRender(page);

    // Click capture button (should show confirmation dialog)
    await page.locator('.luxar-recording-btn button').click();
    await waitForNextRender(page);

    // Verify confirmation dialog appears
    const dialogTitle = await page.locator('.luxar-recording-confirm__title').textContent();
    expect(dialogTitle).toContain('Start Video Recording');

    // Verify Escape keybinding info is shown
    const keybinding = await page.locator('.luxar-recording-confirm__keybinding').textContent();
    expect(keybinding).toContain('Escape');

    // Cancel the dialog
    await page.locator('[data-action="cancel"]').click();
    await waitForNextRender(page);

    // Dialog should be gone
    const dialogGone = await page.locator('.luxar-recording-confirm').count();
    expect(dialogGone).toBe(0);
  });

  test('should show recording indicator during video recording', async ({ page }) => {
    // Start recording via debug interface (skip the dialog).
    //
    // Two fixes layered here:
    //
    // 1. `showConfirmationDialog` lives on `panel.session` after the
    //    recording-panel refactor (was on `panel` when this test was
    //    written). Overriding the wrong object left the real dialog
    //    blocking the recording, so the indicator never appeared.
    //
    // 2. `startVideoRecording()` resolves only when MediaRecorder.onstop
    //    fires (i.e. when recording stops). We must NOT await it here —
    //    doing so blocks page.evaluate() forever since the test stops
    //    recording via an Escape key press below, which cannot fire
    //    while evaluate is still pending. Fire-and-forget; the indicator
    //    polling below observes the start.
    await page.evaluate(() => {
      const panel = (window as any).__luxarDebug.recordingPanel;
      (panel as any).mode = 'video';
      (panel as any).session.showConfirmationDialog = () => Promise.resolve(true);
      void panel.startVideoRecording();
    });

    // Recording indicator should appear; expect.toBeVisible() polls
    // automatically with its own retry-with-timeout, so a fixed sleep is
    // redundant.
    const indicator = page.locator('.luxar-recording-indicator');
    await expect(indicator).toBeVisible({ timeout: 5000 });

    // Verify it shows REC text
    const recText = await indicator.locator('.luxar-recording-indicator__text').textContent();
    expect(recText).toBe('REC');

    // Stop recording — indicator is removed asynchronously after
    // MediaRecorder.stop() resolves; expect.not.toBeVisible polls.
    await page.keyboard.press('Escape');
    await expect(indicator).not.toBeVisible({ timeout: 5000 });
  });

  test('should have Show Panels toggle in Advanced Options', async ({ page }) => {
    // Open panel
    await page.keyboard.press('t');
    await waitForNextRender(page);

    // Expand the Advanced Options folder (collapsed by default)
    const advancedFolder = page
      .locator('.luxar-gui.luxar-recording-panel .luxar-gui__folder-title')
      .filter({ hasText: 'Advanced' });
    if ((await advancedFolder.count()) > 0) {
      await advancedFolder.click();
      await waitForNextRender(page);
    }

    // Check for Show Panels checkbox (label class is luxar-gui__controller-name)
    const hasShowPanels = await page.evaluate(() => {
      const panel = document.querySelector('.luxar-gui.luxar-recording-panel');
      const labels = Array.from(panel?.querySelectorAll('.luxar-gui__controller-name') ?? []);
      return labels.some((el) => el.textContent?.includes('Show Panels'));
    });
    expect(hasShowPanels).toBe(true);
  });
});
