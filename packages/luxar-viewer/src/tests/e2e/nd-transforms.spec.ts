/**
 * E2E test: nD Transforms (inverse-query approach)
 *
 * Verifies that nd_transform on groups correctly shifts the viewer's
 * slice query so points appear at the right world-space time values.
 *
 * Test fixture: test_nd_transforms.luxar.zarr
 *   - GroupA: 50 red points at time=0 (no nd_transform)
 *   - GroupB: 50 blue points at local time=0, nd_transform={"Time": {"offset": 5}}
 *
 * Expected behavior:
 *   - At world time=0: ~50 points visible (GroupA only)
 *   - At world time=5: ~50 points visible (GroupB only, shifted by offset)
 *   - At world time=3: ~0 points (neither group has data here)
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, getLuxarState, waitForDataLoaded, waitForNextRender } from './helpers';

// Fixture URL — served by the Python HTTP server (port 9000) from project root
const FIXTURE_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';
const DATASET = `${FIXTURE_BASE}/test_nd_transforms.luxar.zarr`;

test.describe('nD Transforms', () => {
  test('should load scene with nd_transform groups', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Scene should be initialized
    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);

    // Should have some points visible (at time=0, GroupA's 50 points)
    expect(state.totalPoints).toBeGreaterThan(0);

    // No page errors
    expect(errors).toEqual([]);
  });

  test('should show GroupA points at time=0 and GroupB points at time=5', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Click canvas to ensure focus for keyboard input
    await page.click('canvas');
    await waitForNextRender(page, 2);

    // Get initial state at time=0
    const stateAtTime0 = await getLuxarState(page);
    const pointsAtTime0 = stateAtTime0.totalPoints;

    // GroupA has 50 points at time=0, GroupB has none (its data is at world time=5)
    // We expect approximately 50 points (could be slightly different due to tolerance)
    expect(pointsAtTime0).toBeGreaterThan(0);
    expect(pointsAtTime0).toBeLessThanOrEqual(100); // At most 100 (both groups)

    // Navigate to time=5 (press '4' to select Time dim, then ']' 5 times)
    await page.keyboard.press('4'); // Select Time dimension (4th dim, index 3)
    await waitForNextRender(page, 2);

    for (let i = 0; i < 5; i++) {
      await page.keyboard.press(']');
      await waitForNextRender(page, 2);
    }

    // Wait for the post-navigation re-load (worker projection + chunk
    // fetches) to settle. waitForDataLoaded watches state.isLoading,
    // which is the actual signal here.
    await waitForDataLoaded(page);
    await waitForNextRender(page, 2);

    const stateAtTime5 = await getLuxarState(page);
    const pointsAtTime5 = stateAtTime5.totalPoints;

    // At time=5: GroupB's data should be visible (50 blue points)
    // GroupA's data is at time=0, so it should NOT be visible
    // If nd_transform works: we see ~50 points (GroupB)
    // If nd_transform is broken: we see 0 points (GroupB's local time=0 doesn't match world time=5)
    expect(pointsAtTime5).toBeGreaterThan(0);
  });

  test('should have no points at intermediate time values', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    await page.click('canvas');
    await waitForNextRender(page, 2);

    // Navigate to time=3 (neither group has data here)
    await page.keyboard.press('4');
    await waitForNextRender(page, 2);

    for (let i = 0; i < 3; i++) {
      await page.keyboard.press(']');
      await waitForNextRender(page, 2);
    }

    // Same as above — wait for the post-navigation worker round-trip
    // rather than a fixed sleep.
    await waitForDataLoaded(page);
    await waitForNextRender(page, 2);

    const state = await getLuxarState(page);
    // At time=3: no data from either group
    // GroupA has data at time=0 only
    // GroupB has data at world time=5 only
    expect(state.totalPoints).toBe(0);
  });
});
