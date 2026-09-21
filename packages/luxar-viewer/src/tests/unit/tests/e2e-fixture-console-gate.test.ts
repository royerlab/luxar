/**
 * The shared E2E fixture's console-error gate (#1760).
 *
 * `src/tests/e2e/fixtures.ts` fails a test when Playwright's own
 * `page.on('console')` / `page.on('pageerror')` events delivered an error the
 * allow-list does not forgive. That decision used to be a `captured.filter`
 * buried in the fixture's teardown closure, where it could only be exercised
 * by running the E2E suite in a browser; it is now the exported pure function
 * `unexpectedConsoleErrors`, which is what this file pins. The teardown's
 * surrounding plumbing (the `use(page)` await, the annotation opt-out, the
 * already-failed skip, the `finally` that detaches the listeners) still needs
 * a Playwright runner and is not covered here.
 *
 * Two properties carry the weight, because both were live hazards in the code
 * this replaced. First, matching is PER ENTRY: the teardown joins the
 * survivors into one report string, and a gate that matched patterns against
 * such a join would let one forgiven 404 pardon a real `TypeError` sitting
 * next to it. Second, a `pageerror` is judged exactly like a `console.error` —
 * it is the only source for an uncaught exception the app never routed through
 * `console`, so it must not be special-cased out.
 *
 * NO `@vitest-environment` docblock: the subjects are plain objects and
 * regexes — nothing here touches the DOM — so the `node` default (#1634) is
 * right and a jsdom opt-in would buy nothing.
 */
import { afterEach, describe, expect, it } from 'vitest';
import {
  DEFAULT_ALLOWED_CONSOLE_ERRORS,
  exampleDatasetFailureWarning,
  unexpectedConsoleErrors,
  type CapturedConsoleError,
} from '../../e2e/fixtures';

/** A captured `console.error`, as the fixture's `onConsole` listener records it. */
function consoleError(text: string): CapturedConsoleError {
  return { kind: 'console.error', text };
}

/** A captured uncaught exception, as the fixture's `onPageError` listener records it. */
function pageError(text: string): CapturedConsoleError {
  return { kind: 'pageerror', text };
}

describe('exampleDatasetFailureWarning', () => {
  it.each(['failed', 'timedOut'] as const)(
    'adds an actionable reminder to a %s spec when examples are stale',
    (status) => {
      expect(exampleDatasetFailureWarning(status, 'stale')).toBe(
        'Example datasets are stale. If this spec reads datasets/examples, run "make run-examples" from the repository root.'
      );
    }
  );

  // The case that went silent and cost nine specs a bare 45 s readiness
  // timeout each, with the page showing only "Unable to Load Dataset".
  it('names ungenerated examples as the cause of a failure', () => {
    expect(exampleDatasetFailureWarning('timedOut', 'missing')).toBe(
      'Example datasets have not been generated, so every spec that reads datasets/examples fails on a bare readiness timeout. To fix, run "make run-examples" from the repository root.'
    );
  });

  it('hedges rather than asserts when freshness could not be checked', () => {
    expect(exampleDatasetFailureWarning('failed', 'unavailable')).toBe(
      'Example dataset freshness could not be verified. If this spec reads datasets/examples, run "make run-examples" from the repository root.'
    );
  });

  it('stays silent for passing specs and current examples', () => {
    expect(exampleDatasetFailureWarning('passed', 'stale')).toBeUndefined();
    expect(exampleDatasetFailureWarning('passed', 'missing')).toBeUndefined();
    expect(exampleDatasetFailureWarning('failed', undefined)).toBeUndefined();
  });
});

describe('unexpectedConsoleErrors', () => {
  it('drops an entry an allow pattern matches', () => {
    const captured = [consoleError('WebGL context lost after 40 frames')];

    // Red against a gate that returned `captured` unfiltered: the whole point
    // of the allow-list is that headless-Chromium GPU noise does not fail a
    // run. Also red if the filter matched `entry.kind` instead of `entry.text`
    // — no pattern in the list can match `'console.error'`.
    expect(unexpectedConsoleErrors(captured, DEFAULT_ALLOWED_CONSOLE_ERRORS)).toEqual([]);
  });

  it('keeps an entry no pattern matches', () => {
    const boom = consoleError("TypeError: Cannot read properties of null (reading 'geometry')");

    // Red against a dropped `!` in `!allowed.some(...)` — the inverted gate
    // would forgive exactly the errors it exists to catch — and red against
    // any `return []` short-circuit.
    expect(unexpectedConsoleErrors([boom], DEFAULT_ALLOWED_CONSOLE_ERRORS)).toEqual([boom]);
  });

  it('yields nothing for an empty capture', () => {
    // The healthy path — every passing test in the suite reaches it — so it
    // must be exactly empty and must not throw. Red against the classic
    // empty-array bug in this shape, a `reduce` with no seed
    // (`TypeError: Reduce of empty array with no initial value`), and against
    // any sentinel entry manufactured when nothing was captured.
    expect(unexpectedConsoleErrors([], DEFAULT_ALLOWED_CONSOLE_ERRORS)).toEqual([]);
    expect(unexpectedConsoleErrors([], [])).toEqual([]);
  });

  it('applies the allow-list per entry, not to a joined string', () => {
    const allowed = consoleError(
      'Failed to load resource: the server responded with a status of 404 (Not Found)'
    );
    const real = consoleError('[❌] [Loader] chunk decode failed: unexpected end of stream');

    // The allow-list is spelled out locally rather than taken from
    // `DEFAULT_ALLOWED_CONSOLE_ERRORS`: that export drops the 404 pattern under
    // `LUXAR_E2E_STRICT_CONSOLE` (a supported knob — `test:e2e:smoke:strict`
    // sets it), so anyone with it exported would see this assertion go red for
    // reasons that have nothing to do with the property below.
    const allowList = [
      /WebGL context lost/,
      /Failed to load resource: the server responded with a status of (4\d\d|50[12])/,
    ];

    // THE mutation this file exists for. A gate that tested each pattern
    // against `captured.map(e => e.text).join('\n')` — the shape the teardown
    // uses for its REPORT — finds the 404 pattern matching that blob and
    // forgives everything, real decode failure included. Both orders are
    // asserted because a `.join()` gate is order-insensitive while the real
    // one is not: with only one ordering, a mutant could still look right.
    expect(unexpectedConsoleErrors([allowed, real], allowList)).toEqual([real]);
    expect(unexpectedConsoleErrors([real, allowed], allowList)).toEqual([real]);
  });

  it('judges a pageerror exactly like a console.error', () => {
    const uncaught = pageError("TypeError: Cannot read properties of undefined (reading 'x')");
    const allowedUncaught = pageError('WebGL context lost');

    // Red against any `kind`-sensitive special case. Dropping `pageerror`
    // entries would silence the fixture's only channel for an uncaught
    // exception that never went through `console`; passing them through
    // unfiltered would fail runs on the very GPU noise the allow-list covers.
    expect(unexpectedConsoleErrors([uncaught], DEFAULT_ALLOWED_CONSOLE_ERRORS)).toEqual([uncaught]);
    expect(unexpectedConsoleErrors([allowedUncaught], DEFAULT_ALLOWED_CONSOLE_ERRORS)).toEqual([]);
  });

  it('tries every pattern, and preserves capture order among survivors', () => {
    const first = consoleError('first real failure');
    const second = pageError('second real failure');
    const allowed = consoleError('WebGL context lost');

    // Red against a gate that consulted only `allowed[0]` (or only the last
    // pattern): here the forgiving pattern is the SECOND of two.
    expect(
      unexpectedConsoleErrors([allowed], [/nothing matches this/, /WebGL context lost/])
    ).toEqual([]);
    // Order matters for the thrown report: the reader is looking for the
    // FIRST thing that went wrong, so a gate that reversed or sorted its
    // survivors would misattribute the cause.
    expect(
      unexpectedConsoleErrors([first, allowed, second], DEFAULT_ALLOWED_CONSOLE_ERRORS)
    ).toEqual([first, second]);
  });

  it('does not read LUXAR_E2E_STRICT_CONSOLE itself', () => {
    // The env knob belongs to `DEFAULT_ALLOWED_CONSOLE_ERRORS`, evaluated once
    // at module load, and must NOT be re-read here: a gate that consulted the
    // environment per call would answer differently from the allow-list it was
    // handed, so a spec passing its own patterns could not predict the result.
    const fourOhFour = consoleError(
      'Failed to load resource: the server responded with a status of 404 (Not Found)'
    );
    const strictAllowList = [/WebGL context lost/];

    process.env.LUXAR_E2E_STRICT_CONSOLE = '1';
    expect(unexpectedConsoleErrors([fourOhFour], strictAllowList)).toEqual([fourOhFour]);

    delete process.env.LUXAR_E2E_STRICT_CONSOLE;
    expect(unexpectedConsoleErrors([fourOhFour], strictAllowList)).toEqual([fourOhFour]);
  });

  it('leaves the caller’s array untouched', () => {
    const captured = [consoleError('WebGL context lost'), consoleError('real failure')];

    // Red against an in-place `splice`-style filter. The fixture reuses
    // `captured` for nothing today, but a mutating gate would make the
    // function unusable from a spec that keeps its own capture buffer.
    unexpectedConsoleErrors(captured, DEFAULT_ALLOWED_CONSOLE_ERRORS);
    expect(captured).toHaveLength(2);
  });
});

describe('DEFAULT_ALLOWED_CONSOLE_ERRORS', () => {
  it('stays narrow, and forgives only the two documented classes', () => {
    // A guard on the list itself: it is the one place a broad pattern would
    // silence the gate for all 59 fixture-importing specs at once. An upper
    // bound rather than an equality, so it holds under
    // `LUXAR_E2E_STRICT_CONSOLE` too, where the 404 pattern drops out and only
    // the WebGL-context-loss one remains.
    expect(DEFAULT_ALLOWED_CONSOLE_ERRORS.length).toBeLessThanOrEqual(2);

    const unrelated = [
      consoleError("TypeError: Cannot read properties of null (reading 'geometry')"),
      consoleError('[❌] [SpatialIndex] query returned 0 of 12000 expected points'),
      consoleError('THREE.WebGLProgram: Shader Error 0:'),
      pageError('ReferenceError: LuxarViewer is not defined'),
    ];
    expect(unexpectedConsoleErrors(unrelated, DEFAULT_ALLOWED_CONSOLE_ERRORS)).toEqual(unrelated);
  });
});

// The one test that writes `LUXAR_E2E_STRICT_CONSOLE` must not strip an
// ambient value: vitest reuses a worker process across files, so a plain
// `delete` here would silently un-set the knob for every later file in it.
const strictConsoleBeforeTests = process.env.LUXAR_E2E_STRICT_CONSOLE;

afterEach(() => {
  if (strictConsoleBeforeTests === undefined) {
    delete process.env.LUXAR_E2E_STRICT_CONSOLE;
  } else {
    process.env.LUXAR_E2E_STRICT_CONSOLE = strictConsoleBeforeTests;
  }
});
