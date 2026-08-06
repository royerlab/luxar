/**
 * Shared Playwright fixture that auto-asserts no console errors after
 * every E2E test.
 *
 * The previous setup left it to each spec to remember to call
 * `assertNoConsoleErrors(page)`. 28 of 40 specs forgot — exactly the
 * surface where console errors are the first symptom of regression.
 * This fixture rolls the helper's "should be called in EVERY E2E test"
 * docstring into the harness so it runs whether the spec author
 * remembered or not.
 *
 * Specs that genuinely tolerate certain errors annotate the test:
 *
 * ```ts
 * import { test } from './fixtures';
 *
 * test('intentionally errors when the URL is bad', async ({ page }) => {
 *   test.info().annotations.push({
 *     type: 'allow-console-errors',
 *     description: 'Bad-URL recovery surfaces a console.error by design.',
 *   });
 *   // ... test body ...
 * });
 * ```
 *
 * The annotation type is checked verbatim — typos turn into hard
 * failures rather than silent opt-outs.
 *
 * @module tests/e2e/fixtures
 */

import { test as base } from '@playwright/test';
import { assertNoConsoleErrors } from './helpers';

/** Annotation type that opts a spec out of the auto console-error check. */
export const ALLOW_CONSOLE_ERRORS = 'allow-console-errors';

/**
 * Console error patterns the auto-fixture treats as environmental
 * flakiness rather than test failures.
 *
 * Every E2E spec uses this fixture. Specs that also make their own
 * explicit `assertNoConsoleErrors(page)` call keep
 * that explicit contract (it runs strictly with no allow-list and
 * fails first if anything unexpected appears); the auto-fixture
 * covers cases where the spec author forgot to add the explicit
 * call. Headless-browser environmental noise (WebGL context loss
 * under GPU pressure, intermittent fetch failures during teardown)
 * is filtered here so it doesn't drown out real regressions.
 *
 * Keep the list narrow — it's safer to add a per-spec annotation than
 * to silence a broad pattern globally.
 */
export const DEFAULT_ALLOWED_CONSOLE_ERRORS: RegExp[] = [
  // Headless-Chromium occasionally drops the WebGL context under GPU
  // memory pressure mid-run; the viewer's recovery path logs but
  // continues. Real context-loss bugs surface as test-result divergence
  // (black canvas, wrong frame counts) the spec catches separately.
  /WebGL context lost/,

  // Network-level 4xx/5xx surfaced by the browser as
  // "Failed to load resource: the server responded with a status of N…".
  // The viewer probes optional resources during normal scene loading
  // (zarr's .zattrs/.zgroup/zarr.json detection chain, optional overlays,
  // optional chunk indices, fallback PROPFIND for directory listing).
  // Each miss is a benign 404/501 that the loader's try/catch handles
  // and continues from. Real load failures surface as application-level
  // errors (LoaderError toast, broken renders) which tests assert on
  // directly. A wholly-wrong dataset path also fails through `expect`s
  // on point counts or canvas state, not via this allow-list.
  //
  // The 4xx/5xx pattern is broad and can mask unrelated
  // missing-asset regressions (a missing JS bundle would also match).
  // Set `LUXAR_E2E_STRICT_CONSOLE=1` to disable it and keep only the
  // WebGL-context-loss allow — useful when hardening a smoke run
  // that asserts no missing assets.
  ...(process.env.LUXAR_E2E_STRICT_CONSOLE
    ? []
    : [/Failed to load resource: the server responded with a status of (4\d\d|50[12])/]),
];

/**
 * Extended `test` fixture: drop-in replacement for `@playwright/test`'s
 * `test`. Specs that import from this module get auto console-error
 * checking after each test.
 *
 * The fixture subscribes to Playwright's own `console` and
 * `pageerror` page events in addition to reading the viewer's debug
 * interceptor. Both signal sources are merged and filtered against
 * the same allow-list before the assertion fires; errors fired
 * before the viewer's debug interceptor installs (loading the wrong
 * asset, pre-init ReferenceErrors) and uncaught exceptions surfaced
 * via `pageerror` are still caught.
 */
export const test = base.extend({
  page: async ({ page }, use, testInfo) => {
    // Capture Playwright-native console errors + uncaught exceptions
    // for the duration of the test. The viewer's own debug interceptor
    // captures things differently (formatted, filtered) and is
    // already polled by `assertNoConsoleErrors`; combining the two
    // gives full coverage.
    const captured: { kind: string; text: string }[] = [];
    const onConsole = (msg: import('@playwright/test').ConsoleMessage): void => {
      if (msg.type() === 'error') {
        captured.push({ kind: 'console.error', text: msg.text() });
      }
    };
    const onPageError = (err: Error): void => {
      captured.push({ kind: 'pageerror', text: err.message });
    };
    page.on('console', onConsole);
    page.on('pageerror', onPageError);

    await use(page);

    // Listeners must come off after the test or they leak across tests
    // sharing the same browser context.
    page.off('console', onConsole);
    page.off('pageerror', onPageError);

    // Skip the assertion if the spec opted out via annotation.
    const annotated = testInfo.annotations.some((a) => a.type === ALLOW_CONSOLE_ERRORS);
    if (annotated) return;

    // Skip if the test already failed — the test's own error is the
    // signal we care about; piling on a console-error message would
    // bury it.
    if (testInfo.status === 'failed' || testInfo.status === 'timedOut') return;

    // Filter Playwright-captured errors against the allow-list and
    // surface real ones via the standard assertion path. Then run
    // the existing in-app interceptor check (it has its own
    // formatting + per-call allow-list logic).
    const filtered = captured.filter(
      (e) => !DEFAULT_ALLOWED_CONSOLE_ERRORS.some((re) => re.test(e.text))
    );
    if (filtered.length > 0) {
      const summary = filtered.map((e) => `[${e.kind}] ${e.text}`).join('\n  ');
      throw new Error(
        `Unexpected console / page errors during test:\n  ${summary}\n(captured by Playwright page events; DEFAULT_ALLOWED_CONSOLE_ERRORS did not match)`
      );
    }
    await assertNoConsoleErrors(page, DEFAULT_ALLOWED_CONSOLE_ERRORS);
  },
});

export { expect } from '@playwright/test';
/** Re-export of Playwright's core page/locator/browser types so specs import everything from this fixture module. */
export type { Page, Locator, Browser, BrowserContext } from '@playwright/test';
