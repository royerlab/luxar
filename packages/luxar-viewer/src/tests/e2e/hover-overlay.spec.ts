/**
 * E2E Tests for Hover Overlay DOM rendering
 *
 * These tests verify that `OverlayManager.updateHoverContent` actually
 * mutates the overlay DOM and that CSS transitions on the resulting
 * element behave the way they need to in a real browser. The browser
 * path is the only thing that exercises real CSS — jsdom unit tests
 * verify the DOM mutations but not the cascade.
 *
 * The probe bypasses GPU picking entirely by calling
 * `__luxarDebug.getOverlayManager().updateHoverContent({...})` directly.
 * That isolates the overlay-rendering side of the pipeline, which is
 * exactly the layer where the recent visible_range + fade trap lived.
 *
 * The fixture is registered at runtime via `loadOverlays(...)` because
 * `tests/fixtures/generate_test_data.py` has no labeled-points fixture
 * yet. A follow-up fixture would let a sibling spec also exercise the
 * picking → label → overlay path end-to-end.
 *
 * Dataset: build_example_structured.luxar.zarr — any 3D fixture works; the
 * scene content is irrelevant to the probe.
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForPointsLoaded, assertNoConsoleErrors } from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';

/**
 * Inject a synthetic hover overlay via `OverlayManager.loadOverlays`.
 * `loadOverlays` appends to its internal maps, so calling it post-load
 * adds the probe alongside whatever the scene already declared.
 */
async function registerProbeOverlay(page: import('@playwright/test').Page): Promise<void> {
  await page.evaluate(() => {
    const debug = (window as unknown as { __luxarDebug: { getOverlayManager: () => unknown } })
      .__luxarDebug;
    const mgr = debug.getOverlayManager() as {
      loadOverlays: (cfgs: unknown[], baseUrl: string) => Promise<void>;
    } | null;
    if (!mgr) throw new Error('OverlayManager not available on __luxarDebug');
    return mgr.loadOverlays(
      [
        {
          name: '__probe_hover',
          type: 'overlay_text',
          position: [0.5, 0.5],
          opacity: 1.0,
          anchor: 'center',
          transition: 'none',
          transition_duration: 0,
          interactive: false,
          z_index: 999,
          hover: true,
          text: '{hover_label}',
          font: 'sans',
          font_size: 0.02,
          color: 'white',
          background: 'rgba(0,0,0,0.8)',
          padding: 0.005,
        },
      ],
      ''
    );
  });
}

test.describe('Hover Overlay DOM rendering', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page);
    await registerProbeOverlay(page);
  });

  test.afterEach(async ({ page }) => {
    await assertNoConsoleErrors(page);
  });

  test('updateHoverContent renders the label and clears it on null', async ({ page }) => {
    const probe = page.locator('[data-overlay-name="__probe_hover"]');

    // Initial state: hover overlays start at opacity 0.
    await expect(probe).toHaveCount(1);
    await expect(probe).toHaveCSS('opacity', '0');

    // Show: opacity flips to 1 and the label appears in the DOM.
    await page.evaluate(() => {
      const mgr = (
        window as unknown as { __luxarDebug: { getOverlayManager: () => unknown } }
      ).__luxarDebug.getOverlayManager() as {
        updateHoverContent: (r: unknown) => void;
      };
      mgr.updateHoverContent({
        label: 'PROBE-LABEL',
        nodeName: '/probe',
        elementIndex: 7,
      });
    });

    await expect(probe).toHaveCSS('opacity', '1');
    await expect(probe).toHaveText('PROBE-LABEL');

    // Clear: opacity returns to 0. (Text content is not cleared by
    // updateHoverContent(null) — it's just faded out; that's by design
    // for the fade transition.)
    await page.evaluate(() => {
      const mgr = (
        window as unknown as { __luxarDebug: { getOverlayManager: () => unknown } }
      ).__luxarDebug.getOverlayManager() as {
        updateHoverContent: (r: unknown) => void;
      };
      mgr.updateHoverContent(null);
    });

    await expect(probe).toHaveCSS('opacity', '0');
  });

  test('updateHoverContent re-renders when the label changes', async ({ page }) => {
    const probe = page.locator('[data-overlay-name="__probe_hover"]');

    await page.evaluate(() => {
      const mgr = (
        window as unknown as { __luxarDebug: { getOverlayManager: () => unknown } }
      ).__luxarDebug.getOverlayManager() as {
        updateHoverContent: (r: unknown) => void;
      };
      mgr.updateHoverContent({ label: 'first', nodeName: '/probe', elementIndex: 0 });
    });
    await expect(probe).toHaveText('first');

    await page.evaluate(() => {
      const mgr = (
        window as unknown as { __luxarDebug: { getOverlayManager: () => unknown } }
      ).__luxarDebug.getOverlayManager() as {
        updateHoverContent: (r: unknown) => void;
      };
      mgr.updateHoverContent({ label: 'second', nodeName: '/probe', elementIndex: 1 });
    });
    await expect(probe).toHaveText('second');
  });
});
