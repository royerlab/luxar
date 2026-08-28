/**
 * Shared Playwright fixture that auto-asserts no console errors after
 * every E2E test.
 *
 * The previous setup left it to each spec to remember an explicit
 * `assertNoConsoleErrors(page)` call, and most never did — exactly the
 * surface where console errors are the first symptom of regression.
 * This fixture moves that check into the harness, taken from Playwright's
 * own `console` / `pageerror` page events, so it runs whether the spec
 * author remembered or not. (67 specs today; 59 import `test` from here,
 * the other 8 import `@playwright/test` directly and get no teardown at
 * all. 42 of those 59 still make no explicit call of their own.)
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
 * ONE gate, and it is the Playwright-side one (#1760): `page.on('console')`
 * where `msg.type() === 'error'`, plus `page.on('pageerror')`. That is
 * deliberately the wider of the two sources available. It sees errors fired
 * BEFORE the viewer's in-page console interceptor installs (a mistyped asset,
 * a pre-init `ReferenceError`), it sees uncaught exceptions via `pageerror`
 * and browser-generated errors the app never routed through `console`, it
 * accumulates into an unbounded array rather than the interceptor's ring
 * buffer (which evicts at `DEFAULT_MAX_BUFFER_SIZE`), and it survives
 * navigation, which resets the in-page buffer. Everything the in-page buffer
 * holds under `errors` reaches it too: every write into that buffer goes
 * through the interceptor's private `captureMessage`, reachable only from its
 * five `patch()` closures, each of which re-emits through
 * `this.originalConsole.<method>` (`src/utils/console-interceptor.ts`).
 *
 * So the fixture does NOT read the in-page buffer. It used to, via
 * `assertNoConsoleErrors` — which consults only `messages.errors`, i.e. a
 * second, narrower opinion on the verdict just rendered above — at the price
 * of a `page.evaluate` round trip a saturated main thread can withhold for
 * minutes (#1651/#1746/#1747/#1760). A spec that wants the in-page buffer
 * specifically — for its `warnings` / `logs` buckets, which have no
 * Playwright-side gate, or for a stricter no-allow-list check — still calls
 * `assertNoConsoleErrors` / `getConsoleMessages` itself, and several do.
 *
 * @module tests/e2e/fixtures
 */

import { test as base } from '@playwright/test';
import { EXAMPLE_DATASETS_STALE_ENV } from '../../../tools/example-fixture-freshness';

/** Annotation type that opts a spec out of the auto console-error check. */
export const ALLOW_CONSOLE_ERRORS = 'allow-console-errors';

/**
 * One error the fixture captured from a Playwright page event: `kind` names
 * which event it came from (`console.error` or `pageerror`), `text` is the
 * message the allow-list is matched against.
 */
export interface CapturedConsoleError {
  kind: string;
  text: string;
}

/**
 * Console error patterns the auto-fixture treats as environmental
 * flakiness rather than test failures.
 *
 * The 59 of the 67 E2E specs that import `test` from here use this
 * fixture; the other 8 import `@playwright/test` directly.
 * Specs that also make their own
 * explicit `assertNoConsoleErrors(page)` call keep
 * that explicit contract (it reads the viewer's in-page buffer — which several
 * other helpers and specs still read too — and a bare call passes no
 * allow-list at all, though two call sites do pass their own);
 * the auto-fixture
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
 * The gate's whole decision, as a pure function: which captured entries
 * survive the allow-list.
 *
 * Split out of the teardown closure — which only a Playwright runner can
 * execute — so the decision itself is unit-testable: see
 * `src/tests/unit/tests/e2e-fixture-console-gate.test.ts`. (Importing this
 * module from vitest is fine; `base.extend` needs no runner at import time.)
 *
 * Each entry is matched on its OWN `text`, never on a joined summary: a
 * pattern must not be able to forgive an unrelated error just because an
 * allowed one happened to land next to it. `kind` is carried for the report
 * only, so a `pageerror` is judged exactly like a `console.error`.
 *
 * @param captured - Errors collected from `page.on('console'|'pageerror')`
 * @param allowed - Patterns to forgive, normally {@link DEFAULT_ALLOWED_CONSOLE_ERRORS}
 * @returns The entries no pattern matched, in capture order
 */
export function unexpectedConsoleErrors(
  captured: readonly CapturedConsoleError[],
  allowed: readonly RegExp[]
): CapturedConsoleError[] {
  return captured.filter((entry) => !allowed.some((pattern) => pattern.test(entry.text)));
}

export function staleExampleDatasetFailureWarning(
  status: string | undefined,
  examplesAreStale: boolean
): string | undefined {
  if (!examplesAreStale || (status !== 'failed' && status !== 'timedOut')) return undefined;
  return 'Example datasets are stale. If this spec reads datasets/examples, run "make run-examples" from the repository root.';
}

/**
 * Extended `test` fixture: drop-in replacement for `@playwright/test`'s
 * `test`. Specs that import from this module get auto console-error
 * checking after each test.
 *
 * The check subscribes to Playwright's own `console` and `pageerror` page
 * events for the duration of the test and filters what it caught through
 * {@link unexpectedConsoleErrors}. It does not read the viewer's in-page
 * interceptor buffer — see the module docblock for why that is the wider
 * source and not merely the cheaper one.
 *
 * The listeners come off in a `finally` that runs AFTER the gate, so that both
 * opt-out paths (the annotation, and a test that already failed) detach them
 * too, and so an error emitted once the test body has ended is still counted.
 * That second guarantee is currently vacuous — nothing between `use(page)`
 * returning and the filter yields to the event loop — but it is what made the
 * old shape wrong (it detached BEFORE an `await assertNoConsoleErrors`) and
 * what keeps this one safe if an `await` is ever reintroduced here.
 */
export const test = base.extend({
  page: async ({ page }, use, testInfo) => {
    // Capture Playwright-native console errors + uncaught exceptions
    // for the duration of the test.
    const captured: CapturedConsoleError[] = [];
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

    try {
      await use(page);

      const staleExamplesWarning = staleExampleDatasetFailureWarning(
        testInfo.status,
        process.env[EXAMPLE_DATASETS_STALE_ENV] === '1'
      );
      if (staleExamplesWarning) console.warn(`\n⚠️  ${staleExamplesWarning}\n`);

      // Skip the assertion if the spec opted out via annotation.
      const annotated = testInfo.annotations.some((a) => a.type === ALLOW_CONSOLE_ERRORS);
      if (annotated) return;

      // Skip if the test already failed — the test's own error is the
      // signal we care about; piling on a console-error message would
      // bury it.
      if (testInfo.status === 'failed' || testInfo.status === 'timedOut') return;

      const unexpected = unexpectedConsoleErrors(captured, DEFAULT_ALLOWED_CONSOLE_ERRORS);
      if (unexpected.length > 0) {
        const summary = unexpected.map((e) => `[${e.kind}] ${e.text}`).join('\n  ');
        throw new Error(
          `Unexpected console / page errors during test:\n  ${summary}\n(captured by Playwright page events; DEFAULT_ALLOWED_CONSOLE_ERRORS did not match)`
        );
      }
    } finally {
      // Listeners must come off after the test or they leak across tests
      // sharing the same browser context. In a `finally` so the two early
      // returns and the throw above all detach, and AFTER the gate so nothing
      // emitted past the end of the test can land with the gate blind.
      page.off('console', onConsole);
      page.off('pageerror', onPageError);
    }
  },
});

export { expect } from '@playwright/test';
/** Re-export of Playwright's core page/locator/browser types so specs import everything from this fixture module. */
export type { Page, Locator, Browser, BrowserContext } from '@playwright/test';
