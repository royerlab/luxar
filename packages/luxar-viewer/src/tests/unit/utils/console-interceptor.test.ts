/**
 * ConsoleInterceptor ring-buffer correctness tests.
 *
 * Regression coverage for the off-by-one bug where the buffer would grow
 * to `maxBufferSize + 1` once it filled and the index-0 slot would never
 * be overwritten. After the fix, length stays at `maxBufferSize` and the
 * oldest messages are dropped FIFO.
 */

import { describe, expect, it, beforeEach, afterEach } from 'vitest';
import { consoleInterceptor } from '../../../utils/console-interceptor';

describe('ConsoleInterceptor ring buffer', () => {
  beforeEach(() => {
    // Importing the module is side-effect-free; patch explicitly so
    // console.log calls below are captured.
    consoleInterceptor.patch();
    consoleInterceptor.clearBuffer();
  });

  afterEach(() => {
    consoleInterceptor.dispose();
  });

  it('appends messages while under capacity (args + type both recorded)', () => {
    // utils.md W4 / Phase E61 strengthening: pair the args[0] assertion
    // with a type-tag check. A mutant that always sets type to a fixed
    // value regardless of source (e.g. always 'log' even for warn/error)
    // would have slipped through with the args-only check.
    console.log('a');
    console.log('b');
    console.log('c');

    const messages = consoleInterceptor.getBufferedMessages();
    expect(messages.map((m) => m.args[0])).toEqual(['a', 'b', 'c']);
    expect(messages.map((m) => m.type)).toEqual(['log', 'log', 'log']);
  });

  it('caps buffer at maxBufferSize when more messages are pushed', () => {
    const max = consoleInterceptor.getStats().maxSize;
    const total = max + 5;

    for (let i = 0; i < total; i++) {
      console.log(`msg-${i}`);
    }

    const messages = consoleInterceptor.getBufferedMessages();
    expect(messages.length).toBe(max);
    expect(messages[0].args[0]).toBe(`msg-${total - max}`);
    expect(messages[messages.length - 1].args[0]).toBe(`msg-${total - 1}`);
  });

  it('returns messages in chronological order after wrap', () => {
    const max = consoleInterceptor.getStats().maxSize;
    const total = max + 100;

    for (let i = 0; i < total; i++) {
      console.log(`m-${i}`);
    }

    const messages = consoleInterceptor.getBufferedMessages();
    expect(messages.length).toBe(max);

    // Indices in the surviving messages must be strictly increasing —
    // no stranded index-0 slot from the old off-by-one bug.
    for (let i = 1; i < messages.length; i++) {
      const prev = Number((messages[i - 1].args[0] as string).slice(2));
      const curr = Number((messages[i].args[0] as string).slice(2));
      expect(curr).toBe(prev + 1);
    }
  });

  it('reports total length matching getBufferedMessages length', () => {
    const max = consoleInterceptor.getStats().maxSize;

    for (let i = 0; i < max + 50; i++) {
      console.log(`x-${i}`);
    }

    const stats = consoleInterceptor.getStats();
    const messages = consoleInterceptor.getBufferedMessages();
    expect(stats.total).toBe(messages.length);
    expect(stats.total).toBe(max);
  });
});

describe('ConsoleInterceptor opt-in patching (embedability)', () => {
  it('does not patch console.* until patch() is called (verifies actual identity, not just isPatched flag)', () => {
    // utils.md W3 / Phase E61 strengthening: previously the test only
    // toggled the `isPatched` flag and asserted it. A mutant that
    // flipped `this.isIntercepting = true` but never actually replaced
    // `console.log` would pass. Capture the original method reference
    // and assert that patch() replaces it AND dispose() restores it.
    consoleInterceptor.dispose();
    const originalLog = console.log;
    expect(consoleInterceptor.isPatched).toBe(false);
    expect(console.log).toBe(originalLog);

    consoleInterceptor.patch();
    expect(consoleInterceptor.isPatched).toBe(true);
    expect(console.log).not.toBe(originalLog); // actually replaced

    consoleInterceptor.dispose();
    expect(consoleInterceptor.isPatched).toBe(false);
    expect(console.log).toBe(originalLog); // restored
  });

  it('captures messages while patched and stops on dispose()', () => {
    consoleInterceptor.dispose();
    consoleInterceptor.clearBuffer();

    // Pre-patch: console.log does NOT land in the buffer.
    console.log('pre-patch');
    expect(consoleInterceptor.getBufferedMessages().length).toBe(0);

    consoleInterceptor.patch();
    consoleInterceptor.clearBuffer();

    console.log('captured-1');
    console.log('captured-2');
    expect(consoleInterceptor.getBufferedMessages().map((m) => m.args[0])).toEqual([
      'captured-1',
      'captured-2',
    ]);

    consoleInterceptor.dispose();
    consoleInterceptor.clearBuffer();

    // Post-dispose: console.log no longer goes through the interceptor.
    console.log('post-dispose');
    expect(consoleInterceptor.getBufferedMessages().length).toBe(0);
  });
});

describe('ConsoleInterceptor stack capture', () => {
  beforeEach(() => {
    consoleInterceptor.patch();
    consoleInterceptor.clearBuffer();
  });

  afterEach(() => {
    consoleInterceptor.dispose();
  });

  const lastMessage = () => {
    const msgs = consoleInterceptor.getBufferedMessages();
    return msgs[msgs.length - 1];
  };

  it('captures the stack of an Error passed AFTER the message string', () => {
    // Every `log.*` call formats its message into args[0] as a STRING and passes
    // the error behind it, so looking only at args[0] could never find one.
    const err = new Error('boom');
    console.error('[❌] [Cache] failed', err);

    expect(lastMessage().stack).toBe(err.stack);
  });

  it('captures a stack on the warn path too', () => {
    // ~30 `log.warning(…, error)` sites pass a real Error; without this they had
    // neither a message (pre-fix formatters) nor any trace to fall back on.
    const err = new Error('cache write failed');
    console.warn('[⚠️] [Cache] OPFSStore failed', err);

    expect(lastMessage().stack).toBe(err.stack);
  });

  it('does NOT fabricate a stack when no Error was passed', () => {
    // This used to synthesize `new Error().stack` whenever the message merely
    // contained the word "error", producing a plausible-looking trace rooted
    // inside the interceptor — worse than no stack, because a bug-report reader
    // would follow it.
    console.error('an error happened, but no Error object was passed');

    expect(lastMessage().stack).toBeUndefined();
  });

  it('prefers the first Error when several args carry stacks', () => {
    const first = new Error('first');
    const second = new Error('second');
    console.error('msg', first, second);

    expect(lastMessage().stack).toBe(first.stack);
  });

  it('does not throw (and still buffers) when an arg has a throwing stack accessor', () => {
    // extractStack runs inside the patched console.warn/error BEFORE the
    // original console call — if it threw, the warning itself would vanish and
    // the calling code (typically already inside a catch block) would throw.
    const hostile = new Error('boom');
    Object.defineProperty(hostile, 'stack', {
      get() {
        throw new Error('hostile stack getter');
      },
    });

    // Stub the pass-through targets: Node's own console.warn/error ALSO read
    // `.stack` when printing (util.inspect) and would throw on their own —
    // what's under test is only the interceptor's capture path.
    const orig = consoleInterceptor.getOriginalConsole();
    const { warn: origWarn, error: origError } = orig;
    orig.warn = () => {};
    orig.error = () => {};
    try {
      expect(() => console.warn('[⚠️] [Cache] failed', hostile)).not.toThrow();
      expect(() => console.error('[❌] [Cache] failed', hostile)).not.toThrow();
    } finally {
      orig.warn = origWarn;
      orig.error = origError;
    }

    const msgs = consoleInterceptor.getBufferedMessages();
    expect(msgs.length).toBe(2);
    expect(msgs[0].stack).toBeUndefined();
    expect(msgs[1].stack).toBeUndefined();
  });
});
