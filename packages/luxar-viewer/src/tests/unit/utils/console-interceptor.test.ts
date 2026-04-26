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

  it('appends messages while under capacity', () => {
    console.log('a');
    console.log('b');
    console.log('c');

    const messages = consoleInterceptor.getBufferedMessages();
    expect(messages.map((m) => m.args[0])).toEqual(['a', 'b', 'c']);
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
  it('does not patch console.* until patch() is called', () => {
    // Start from a clean slate so this test does not rely on previous order.
    consoleInterceptor.dispose();
    expect(consoleInterceptor.isPatched).toBe(false);

    consoleInterceptor.patch();
    expect(consoleInterceptor.isPatched).toBe(true);

    consoleInterceptor.dispose();
    expect(consoleInterceptor.isPatched).toBe(false);
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
