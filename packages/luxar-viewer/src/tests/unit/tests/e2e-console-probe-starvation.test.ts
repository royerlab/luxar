/**
 * The starved-probe discrimination the shared E2E fixture relies on (#1760).
 *
 * Measured cause: when the viewer's main thread saturates, the CDP
 * `Runtime.evaluate` channel starves independently of Playwright's own
 * `page.on('console')` stream — reproduced as 20 consecutive 5 s
 * `page.evaluate(() => 1)` timeouts (95 s, not one round trip serviced) while
 * 19 console messages arrived through Playwright in the same window. Because
 * the fixture's Playwright-side error gate has already run by then, its
 * teardown must RECORD that starvation, not fail the test; a direct caller
 * must still fail. The whole decision hangs on `getConsoleMessages` throwing a
 * distinguishable TYPE, which is what this file pins.
 *
 * `isConsoleProbeUnanswered` is pure, and the fixture's own consumption of it
 * cannot be unit-tested (importing `./fixtures` runs `base.extend` outside a
 * Playwright runner), so the seam tested here is exactly the type contract
 * between the two.
 *
 * NO `@vitest-environment` docblock: nothing below touches the DOM — the
 * subjects are a thrown value and a fake `page` — so the `node` default (#1634)
 * is right and a jsdom opt-in would buy nothing.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Page } from '@playwright/test';
import {
  ConsoleProbeUnansweredError,
  describeConsoleProbeUnanswered,
  getConsoleMessages,
  isConsoleProbeUnanswered,
} from '../../e2e/helpers';

/** A `page` whose `evaluate` never settles — the starved round trip. */
function starvedPage(): Page {
  return { evaluate: () => new Promise(() => {}) } as unknown as Page;
}

/** A `page` whose `evaluate` answers immediately with `value`. */
function answeringPage(value: unknown): Page {
  return { evaluate: async () => value } as unknown as Page;
}

afterEach(() => {
  vi.useRealTimers();
});

describe('getConsoleMessages starvation type', () => {
  it('rejects with ConsoleProbeUnansweredError when the evaluate never answers', async () => {
    vi.useFakeTimers();

    const settled = getConsoleMessages(starvedPage(), 5000).then(
      () => 'resolved',
      (e: unknown) => e
    );
    await vi.advanceTimersByTimeAsync(5000);
    const outcome = await settled;

    // A bare `Error` here (the pre-#1760 behaviour) leaves the fixture no way
    // to tell starvation from a real console-error verdict, so it would have
    // to keep failing the test.
    expect(outcome).toBeInstanceOf(ConsoleProbeUnansweredError);
    expect(isConsoleProbeUnanswered(outcome)).toBe(true);
    // Attribution is what makes the failure actionable for a DIRECT caller:
    // the message must still name the probe and its deadline.
    expect((outcome as Error).message).toContain('never answered');
    expect((outcome as Error).message).toContain('5000 ms');
    // The deadline is carried as data, not only spelled into the message, so
    // the fixture's annotation does not have to re-parse it.
    expect((outcome as ConsoleProbeUnansweredError).timeout).toBe(5000);
    expect((outcome as Error).name).toBe('ConsoleProbeUnansweredError');
  });

  it('resolves with the buckets — and does not throw — when the page answers', async () => {
    const buckets = {
      errors: ['[❌] [Loader] boom'],
      warnings: ['[⚠️] [Cache] evicted'],
      logs: ['[ℹ️] hello'],
      all: ['[❌] [Loader] boom', '[⚠️] [Cache] evicted', '[ℹ️] hello'],
    };

    // An ANSWERED probe must still hand its contents back verbatim: the
    // starvation type must not have been wired into the success path (e.g. by
    // throwing on a non-empty `errors` bucket, which is
    // `assertNoConsoleErrors`' job, not this helper's).
    await expect(getConsoleMessages(answeringPage(buckets), 5000)).resolves.toEqual(buckets);
  });
});

describe('isConsoleProbeUnanswered', () => {
  it('is true for the starvation error and false for a real console-error verdict', () => {
    const starved = new ConsoleProbeUnansweredError('probe unanswered within 45000 ms', 45000);
    // The exact text `assertNoConsoleErrors` throws. If the fixture forgave
    // this, every spec's console-error gate would become a silent pass — the
    // one outcome the whole design exists to prevent.
    const realVerdict = new Error(
      'Console errors detected: 2 errors.\nFirst error: [❌] [Loader] decode failed'
    );

    expect(isConsoleProbeUnanswered(starved)).toBe(true);
    expect(isConsoleProbeUnanswered(realVerdict)).toBe(false);
  });

  it('is false for non-Error thrown values', () => {
    // Playwright and third-party code can reject with anything; a predicate
    // that reached into `.name`/`.message` unguarded would throw here instead
    // of answering, taking the teardown down with it.
    expect(isConsoleProbeUnanswered(undefined)).toBe(false);
    expect(isConsoleProbeUnanswered(null)).toBe(false);
    expect(isConsoleProbeUnanswered('the page never answered the console-buffer probe')).toBe(
      false
    );
    expect(isConsoleProbeUnanswered({ name: 'ConsoleProbeUnansweredError', timeout: 45000 })).toBe(
      false
    );
  });

  it('is false for an Error that merely QUOTES the probe text', () => {
    // Mutation strength: the discrimination must be by type, not by string
    // match. A helper that wrapped or re-threw the probe failure as a plain
    // `Error` with the same wording — or an unrelated failure that quotes it,
    // e.g. a spec asserting on log contents — has to stay fatal, because
    // nothing about it establishes that only the in-page channel starved.
    const impostor = new Error(
      'getConsoleMessages: the page never answered the console-buffer probe within 45000 ms — ' +
        'its main thread is saturated and starving the evaluate round trip.'
    );

    expect(isConsoleProbeUnanswered(impostor)).toBe(false);

    // Faking the marker fields on a plain Error is not enough either.
    const namedImpostor = new Error('probe unanswered');
    namedImpostor.name = 'ConsoleProbeUnansweredError';
    expect(isConsoleProbeUnanswered(namedImpostor)).toBe(false);
  });
});

describe('describeConsoleProbeUnanswered', () => {
  it('names the deadline and why the test is not being failed', () => {
    const text = describeConsoleProbeUnanswered(
      new ConsoleProbeUnansweredError('unanswered', 45000)
    );

    // The annotation and the warn line share this string; a reader who finds
    // it in a report needs the deadline (to tell it from a short mid-test
    // probe) and the reason it was survivable (the Playwright-side gate).
    expect(text).toContain('45000 ms');
    expect(text).toMatch(/pageerror/);
    expect(text).toMatch(/#1760/);
  });

  it('reports the deadline it was actually given, not a hardcoded default', () => {
    const text = describeConsoleProbeUnanswered(
      new ConsoleProbeUnansweredError('unanswered', 8000)
    );

    expect(text).toContain('8000 ms');
    expect(text).not.toContain('45000');
  });
});
