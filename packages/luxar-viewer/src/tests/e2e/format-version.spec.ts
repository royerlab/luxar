/**
 * Format-version policy, end to end (root header → loader → overlay).
 *
 * The policy itself is pinned by the shared case table
 * (`src/tests/unit/data/format-version.test.ts` ⇔
 * `luxar/typing_utils/tests/test_format_version.py`). This spec asserts the
 * USER-VISIBLE consequences on real stores whose root headers were patched by
 * `tests/fixtures/generate_test_data.py::generate_format_version_fixtures`:
 *
 * - a legacy 0.1 root (`luxar_version` only — what every published pre-0.2
 *   store, incl. the DESI Zenodo record, looks like) renders silently;
 * - a newer-major (`9.9`), unparsable (`abc`) and older-unsupported (`0.0`)
 *   root each reach the error overlay, which names the offending version.
 *
 * Needs `pnpm test:generate-fixtures` (the Playwright pre-flight enforces it).
 */

import { test, expect, ALLOW_CONSOLE_ERRORS } from './fixtures';
import { waitForLuxarReady, waitForPointsLoaded, assertNoConsoleErrors } from './helpers';

const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';

const FIXTURES = {
  legacy: `${FIXTURES_BASE}/test_legacy_scene_v0_1.luxar.zarr`,
  newerMajor: `${FIXTURES_BASE}/test_scene_newer_major.luxar.zarr`,
  unparsable: `${FIXTURES_BASE}/test_scene_unparsable_version.luxar.zarr`,
  olderUnsupported: `${FIXTURES_BASE}/test_scene_older_unsupported.luxar.zarr`,
};

const ERROR_TEXT = '#luxar-error-message-text';

test.describe.configure({ timeout: 120000 });

test.describe('Format version policy', () => {
  test('a legacy 0.1 scene (luxar_version only) renders silently', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.legacy}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1);
    await assertNoConsoleErrors(page);

    // No overlay, no format toast: the legacy key is a SUPPORTED arm.
    await expect(page.locator(ERROR_TEXT)).toHaveCount(0);
    const toastMentionsFormat = await page.evaluate(() =>
      Array.from(document.querySelectorAll('.luxar-toast, [class*="toast"]')).some((el) =>
        /format/i.test(el.textContent ?? '')
      )
    );
    expect(toastMentionsFormat).toBe(false);
  });

  const refused: Array<[string, string, string]> = [
    ['newer-major', FIXTURES.newerMajor, '9.9'],
    ['unparsable', FIXTURES.unparsable, 'abc'],
    ['older-unsupported', FIXTURES.olderUnsupported, '0.0'],
  ];

  for (const [arm, url, version] of refused) {
    test(`a ${arm} scene (${version}) shows the error overlay naming the version`, async ({
      page,
    }, testInfo) => {
      testInfo.annotations.push({
        type: ALLOW_CONSOLE_ERRORS,
        description: 'A refused format version is reported through console.error by design.',
      });

      await page.goto(`/?src=${url}&debug`);

      const overlay = page.locator(ERROR_TEXT);
      await overlay.waitFor({ state: 'visible', timeout: 60000 });
      const text = (await overlay.textContent()) ?? '';
      expect(text).toContain('Unsupported format_version');
      expect(text).toContain(version);

      // Nothing was drawn: the loader threw before any node was built.
      const drawn = await page.evaluate(() => {
        const debug = (window as unknown as { __luxarDebug?: { getState?: () => unknown } })
          .__luxarDebug;
        const state = debug?.getState?.() as { totalPoints?: number } | undefined;
        return state?.totalPoints ?? 0;
      });
      expect(drawn).toBe(0);
    });
  }
});
