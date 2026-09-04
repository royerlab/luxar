/**
 * ConsoleInterceptor ring-buffer correctness tests.
 *
 * Regression coverage for the off-by-one bug where the buffer would grow
 * to `maxBufferSize + 1` once it filled and the index-0 slot would never
 * be overwritten. After the fix, length stays at `maxBufferSize` and the
 * oldest messages are dropped FIFO.
 */

import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { runInNewContext } from 'node:vm';
import { consoleInterceptor, disposeConsoleInterceptor } from '../../../utils/console-interceptor';

type ConsoleMethod = 'log' | 'warn' | 'error' | 'info' | 'debug';
const CONSOLE_METHODS: readonly ConsoleMethod[] = ['log', 'warn', 'error', 'info', 'debug'];

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

  it('prefers a real Error over an earlier duck-typed stack carrier', () => {
    // A plain `{ stack }` object satisfies getErrorStack, so a single-pass scan
    // would record the context string ahead of the real Error's trace.
    const realError = new Error('the real failure');
    console.error('failed', { stack: 'just some context string' }, realError);

    expect(lastMessage().stack).toBe(realError.stack);
  });

  it('prefers a CROSS-REALM Error over an earlier duck-typed stack carrier', () => {
    // A cross-realm Error fails `instanceof Error`, so the priority pass must
    // detect it via the [[Class]] brand (isGenuineError) or the context string
    // wins again.
    const crossRealm = runInNewContext('new Error("the real failure")') as Error;
    expect((crossRealm as unknown) instanceof Error).toBe(false);
    console.error('failed', { stack: 'just some context string' }, crossRealm);

    expect(lastMessage().stack).toBe(crossRealm.stack);
  });

  it('prefers a real Error over an earlier FULL-TRIPLE context object', () => {
    // A context bag carrying all three of name/message/stack satisfies the
    // wide isErrorLike (it renders as an Error), but it must NOT satisfy the
    // stack-precedence pass — only a genuine Error may outrank the real trace.
    const realError = new Error('the real failure');
    console.error(
      'failed',
      { name: 'Context', message: 'decode phase', stack: 'context stack' },
      realError
    );

    expect(lastMessage().stack).toBe(realError.stack);
  });

  it('falls back to a duck-typed stack carrier when no real Error is present', () => {
    // Some Firefox DOMExceptions carry a stack without reporting as an Error.
    console.error('failed', { stack: 'at somewhere' });

    expect(lastMessage().stack).toBe('at somewhere');
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

  it('does not throw on a revoked Proxy arg and still finds a later Error stack', () => {
    // The real-Error preference pass evaluates `instanceof`, which walks
    // [[GetPrototypeOf]] and throws for a revoked Proxy. That throw must be
    // contained (same never-throw contract as above), and the real Error behind
    // the Proxy must still be found.
    const { proxy, revoke } = Proxy.revocable({}, {});
    revoke();
    const err = new Error('boom');

    // Node's own console.error inspects the revoked Proxy and throws on its
    // own; stub the pass-through so only the interceptor's capture is tested.
    const orig = consoleInterceptor.getOriginalConsole();
    const origError = orig.error;
    orig.error = () => {};
    try {
      expect(() => console.error('cleanup failed', proxy, err)).not.toThrow();
    } finally {
      orig.error = origError;
    }

    expect(lastMessage().stack).toBe(err.stack);
  });
});

describe('ConsoleInterceptor info/debug capture', () => {
  beforeEach(() => {
    consoleInterceptor.patch();
    consoleInterceptor.clearBuffer();
  });

  afterEach(() => {
    consoleInterceptor.dispose();
  });

  // `log`, `warn` and `error` were already covered; `info` and `debug` are the
  // two overrides nothing exercised. They matter because the type tag is what
  // the debug console filters on — an override that captured the wrong tag, or
  // dropped the pass-through, would be invisible to every other test here.
  it.each([
    ['info', () => console.info('hello', 1)],
    ['debug', () => console.debug('hello', 1)],
  ])('captures console.%s under its own type tag', (type, emit) => {
    emit();

    const messages = consoleInterceptor.getBufferedMessages();
    expect(messages).toHaveLength(1);
    expect(messages[0].type).toBe(type);
    expect(messages[0].args).toEqual(['hello', 1]);
  });

  it.each(['info', 'debug'] as const)('still forwards console.%s to the host', (type) => {
    // The interceptor is a TEE, not a sink: swallowing output would hide every
    // message from the real devtools console.
    const original = consoleInterceptor.getOriginalConsole();
    const spy = vi.spyOn(original, type).mockImplementation(() => {});
    try {
      console[type]('passed through', 42);
      expect(spy).toHaveBeenCalledWith('passed through', 42);
    } finally {
      spy.mockRestore();
    }
  });

  it('counts info and debug separately in getStats()', () => {
    console.info('i');
    console.debug('d');
    console.debug('d2');

    const stats = consoleInterceptor.getStats();
    expect(stats.types.info).toBe(1);
    expect(stats.types.debug).toBe(2);
  });
});

describe('ConsoleInterceptor listeners', () => {
  beforeEach(() => {
    consoleInterceptor.patch();
    consoleInterceptor.clearBuffer();
  });

  afterEach(() => {
    consoleInterceptor.dispose();
  });

  it('notifies a registered listener with the buffered message', () => {
    const seen: Array<{ type: string; args: unknown[] }> = [];
    const listener = (m: { type: string; args: unknown[] }) =>
      seen.push({ type: m.type, args: m.args });
    consoleInterceptor.addListener(listener);
    try {
      console.warn('watch me');
    } finally {
      consoleInterceptor.removeListener(listener);
    }

    expect(seen).toHaveLength(1);
    expect(seen[0].type).toBe('warn');
    expect(seen[0].args).toEqual(['watch me']);
  });

  it('stops notifying after removeListener', () => {
    const listener = vi.fn();
    consoleInterceptor.addListener(listener);
    console.log('first');
    consoleInterceptor.removeListener(listener);
    console.log('second');

    expect(listener).toHaveBeenCalledTimes(1);
  });

  it('a throwing listener neither breaks console nor starves the others', () => {
    // The per-listener catch is what stops one bad subscriber (a half-torn-down
    // UI panel, say) from taking out console.* process-wide.
    const thrower = () => {
      throw new Error('listener blew up');
    };
    const healthy = vi.fn();
    consoleInterceptor.addListener(thrower);
    consoleInterceptor.addListener(healthy);

    // The failure is reported through the ORIGINAL console.error so it cannot
    // recurse back into the interceptor; silence it to keep the test output
    // clean, and assert it actually reported rather than swallowing.
    const original = consoleInterceptor.getOriginalConsole();
    const errorSpy = vi.spyOn(original, 'error').mockImplementation(() => {});
    try {
      expect(() => console.log('still fine')).not.toThrow();
      expect(healthy).toHaveBeenCalledTimes(1);
      expect(errorSpy).toHaveBeenCalledWith('Error in console listener:', expect.any(Error));
    } finally {
      errorSpy.mockRestore();
      consoleInterceptor.removeListener(thrower);
      consoleInterceptor.removeListener(healthy);
    }

    // And the message still reached the buffer despite the throw.
    expect(consoleInterceptor.getBufferedMessages().at(-1)?.args).toEqual(['still fine']);
  });
});

describe('ConsoleInterceptor singleton disposal', () => {
  beforeEach(() => {
    disposeConsoleInterceptor();
  });

  afterEach(() => {
    disposeConsoleInterceptor();
  });

  it('reports isPatched across the patch/dispose cycle', () => {
    expect(consoleInterceptor.isPatched).toBe(false);
    consoleInterceptor.patch();
    expect(consoleInterceptor.isPatched).toBe(true);
    consoleInterceptor.dispose();
    expect(consoleInterceptor.isPatched).toBe(false);
  });

  it('un-intercepts the host console when disposed WHILE PATCHED', () => {
    // The app dispose pipeline's contract: a mount/unmount cycle must leave the
    // host console usable and no longer feeding the buffer. Nothing tested the
    // while-patched path, which is the only one with work to do.
    //
    // Asserted FUNCTIONALLY, not by reference identity. The module stores a
    // bound copy of `console.log`, so dispose restores that copy rather than the
    // exact original object — deliberate, because an unbound console method
    // throws "Illegal invocation" in some hosts.
    const patchedMarker = { captured: 0 };
    consoleInterceptor.patch();
    consoleInterceptor.clearBuffer();
    const patchedMethods = Object.fromEntries(
      CONSOLE_METHODS.map((method) => [method, console[method]])
    ) as Record<ConsoleMethod, typeof console.log>;

    console.log('while patched');
    patchedMarker.captured = consoleInterceptor.getBufferedMessages().length;
    expect(patchedMarker.captured).toBe(1);

    disposeConsoleInterceptor();

    // The override is gone …
    for (const method of CONSOLE_METHODS) {
      expect(console[method], `console.${method} remained patched`).not.toBe(
        patchedMethods[method]
      );
    }
    // … the console still works …
    expect(() => console.log('after dispose')).not.toThrow();
    // … and nothing is being captured any more.
    expect(consoleInterceptor.getBufferedMessages()).toEqual([]);
  });

  it('restores the same console methods across repeated singleton lifecycles', () => {
    consoleInterceptor.patch();
    disposeConsoleInterceptor();
    const restoredMethods = Object.fromEntries(
      CONSOLE_METHODS.map((method) => [method, console[method]])
    ) as Record<ConsoleMethod, typeof console.log>;

    consoleInterceptor.patch();
    disposeConsoleInterceptor();

    for (const method of CONSOLE_METHODS) {
      expect(console[method], `console.${method} changed across lifecycles`).toBe(
        restoredMethods[method]
      );
    }
  });

  it('is idempotent and side-effect-free when never patched', () => {
    const pristine = console.log;

    expect(() => disposeConsoleInterceptor()).not.toThrow();
    expect(() => disposeConsoleInterceptor()).not.toThrow();

    expect(console.log).toBe(pristine);
  });

  it('hands out a FRESH interceptor after disposal', () => {
    consoleInterceptor.patch();
    console.log('belongs to the old instance');
    expect(consoleInterceptor.getBufferedMessages().length).toBeGreaterThan(0);

    disposeConsoleInterceptor();

    // The proxy lazily rebuilds the singleton, so the buffer must be empty
    // rather than carrying the previous instance's messages.
    expect(consoleInterceptor.getBufferedMessages()).toEqual([]);
    expect(consoleInterceptor.isPatched).toBe(false);
  });
});

describe('ConsoleInterceptor singleton proxy traps', () => {
  afterEach(() => {
    disposeConsoleInterceptor();
  });

  it('forwards `in` checks to the live instance', () => {
    expect('patch' in consoleInterceptor).toBe(true);
    expect('nonExistentMember' in consoleInterceptor).toBe(false);
  });

  it('forwards writes through to the live instance', () => {
    // REGRESSION GUARD. The trap used to pass `receiver` to `Reflect.set`,
    // which completes the write with CreateDataProperty(receiver, …) — landing
    // the value on the proxy's empty `{}` target instead of the instance, and
    // still reporting success. Reading the same property through the proxy then
    // returned `undefined` from the live instance.
    (consoleInterceptor as unknown as { probe: number }).probe = 1;

    expect((consoleInterceptor as unknown as { probe: number }).probe).toBe(1);
  });
});
