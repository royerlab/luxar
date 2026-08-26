/**
 * Zipped-store E2E: a `.luxar.zarr.zip` read over HTTP range requests.
 *
 * The positive path had no browser coverage until now, for a structural reason
 * rather than an oversight: the suite's data server is `python3 -m http.server`,
 * which ignores `Range` and answers 200 with the whole body, so the zipped path
 * could not run at all. `playwright.config.ts` now also boots
 * `tools/range-http-server.py` on 9001, rooted at the same directory, and these
 * two tests use the pair to pin BOTH halves of the contract:
 *
 * - over a range-capable server, the archive renders its scene;
 * - over one that ignores `Range`, it fails LOUDLY instead of quietly rendering
 *   an empty scene, which is the failure mode that matters — a viewer showing
 *   nothing looks like a dataset with nothing in it.
 *
 * The equivalence assertion is the point of the first test. "The zip rendered
 * something" is not evidence: the zipped benchmark once reported a spectacular
 * 273 ms / 4-request result that turned out to be a stale dev server timing an
 * EMPTY scene. So the archive is compared against its own directory twin —
 * one scene packaged two ways by `generate_zipped_archive_test()` — served from
 * the SAME origin, leaving the container format as the only variable.
 */

import { test, expect, ALLOW_CONSOLE_ERRORS } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  getLuxarState,
  assertNoConsoleErrors,
} from './helpers';

/** Range-capable origin (206 + `Content-Range`). Only this one can serve the archive. */
const RANGE_BASE = 'http://localhost:9001/packages/luxar-viewer/tests/fixtures';
/** The suite's default origin: `python3 -m http.server`, which ignores `Range` entirely. */
const NO_RANGE_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';

const ARCHIVE = 'test_zipped_archive.luxar.zarr.zip';
const DIRECTORY = 'test_zipped_archive.luxar.zarr';

/** What `generate_zipped_archive_test()` writes: 4 nodes x 64 points. */
const EXPECTED_POINTS = 256;
const EXPECTED_NODES = 4;

// Two full scene loads per test plus WebGL init; the file-scope precedent for
// this value is test-fixtures-rendering.spec.ts, which documents why 60 s is
// too tight to attribute a failure to a phase.
test.describe.configure({ timeout: 120000 });

test.describe('Zipped store (.zarr.zip)', () => {
  test('renders an archive identically to its directory twin', async ({ page }) => {
    await page.goto(`/?src=${RANGE_BASE}/${ARCHIVE}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1);
    await assertNoConsoleErrors(page);

    const zipped = await getLuxarState(page);
    expect(zipped.totalPoints).toBe(EXPECTED_POINTS);
    expect(zipped.pointClouds.length).toBe(EXPECTED_NODES);

    // The control. Same scene, same server, same origin — only the container
    // differs, so an equal count cannot be explained by anything else.
    await page.goto(`/?src=${RANGE_BASE}/${DIRECTORY}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1);

    const unzipped = await getLuxarState(page);
    expect(zipped.totalPoints).toBe(unzipped.totalPoints);
    expect(zipped.pointClouds.length).toBe(unzipped.pointClouds.length);
  });

  test('fails loudly when the server ignores Range, rather than rendering empty', async ({
    page,
  }, testInfo) => {
    // Annotated on THIS test only — the positive one above keeps the gate, and
    // the console error here is the assertion's subject rather than a defect.
    testInfo.annotations.push({
      type: ALLOW_CONSOLE_ERRORS,
      description: 'The Range-support diagnostic under test is emitted as console.error.',
    });

    // Captured through Playwright rather than `getConsoleMessages`, which reads
    // the IN-PAGE interceptor at `__luxarDebug.consoleInterceptor`. A fatal
    // archive fault aborts application startup, so that object never exists and
    // the helper answers with empty buckets — a legitimate "no interceptor"
    // reply that is indistinguishable here from "no error was reported". For
    // the same reason there is no `waitForLuxarReady` below: on this path the
    // app deliberately never becomes ready.
    const consoleText: string[] = [];
    page.on('console', (message) => consoleText.push(message.text()));
    page.on('pageerror', (error) => consoleText.push(String(error)));

    // Port 9000 answers 200 with the whole archive for every window the reader
    // asks for. Silently accepting that is the dangerous outcome, because those
    // bytes decode into nothing and the viewer then looks merely empty.
    await page.goto(`/?src=${NO_RANGE_BASE}/${ARCHIVE}&debug`);

    // The diagnosis has to REACH someone, so this asserts on the remedy text
    // rather than on any thrown value: telling the user what to do about it is
    // the error's entire purpose.
    await expect
      .poll(() => consoleText.some((line) => /honours HTTP Range requests/i.test(line)), {
        message:
          'expected a Range-support diagnostic in the console; ' +
          'a zipped store served without 206 must not fail silently',
        timeout: 45000,
      })
      .toBe(true);

    // And nothing may be drawn: an unreadable archive that renders zero points
    // while reporting success is precisely the failure this guards.
    const totalPoints = await page.evaluate(() => {
      const debug = (window as { __luxarDebug?: { getState?: () => { totalPoints?: number } } })
        .__luxarDebug;
      return debug?.getState?.()?.totalPoints ?? 0;
    });
    expect(totalPoints).toBe(0);
  });
});
