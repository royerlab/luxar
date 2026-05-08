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
 * Extended `test` fixture: drop-in replacement for `@playwright/test`'s
 * `test`. Specs that import from this module get auto console-error
 * checking after each test.
 */
export const test = base.extend({
  page: async ({ page }, use, testInfo) => {
    await use(page);

    // Skip the assertion if the spec opted out via annotation.
    const annotated = testInfo.annotations.some((a) => a.type === ALLOW_CONSOLE_ERRORS);
    if (annotated) return;

    // Skip if the test already failed — the test's own error is the
    // signal we care about; piling on a console-error message would
    // bury it.
    if (testInfo.status === 'failed' || testInfo.status === 'timedOut') return;

    await assertNoConsoleErrors(page);
  },
});

export { expect } from '@playwright/test';
export type { Page, Locator, Browser, BrowserContext } from '@playwright/test';
