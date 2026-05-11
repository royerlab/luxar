/**
 * Unit tests for the project log utility.
 *
 * Each method is a thin wrapper around console.log / console.warn /
 * console.error with a `[emoji] [Module] message` prefix. We spy on the
 * console methods to verify the format and routing for each helper.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { log, formatLog, createModuleLogger, LogEmoji, Modules } from '../../../utils/log';

describe('formatLog', () => {
  it('produces "[emoji] [Module] message"', () => {
    expect(formatLog('🚀', 'Luxar', 'starting')).toBe('[🚀] [Luxar] starting');
  });

  it('preserves embedded brackets in the message', () => {
    expect(formatLog('ℹ️', 'X', 'foo [bar]')).toBe('[ℹ️] [X] foo [bar]');
  });
});

describe('log routing', () => {
  let logSpy: ReturnType<typeof vi.spyOn>;
  let warnSpy: ReturnType<typeof vi.spyOn>;
  let errorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
  });

  afterEach(() => {
    logSpy.mockRestore();
    warnSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it('log.info goes to console.log with the INFO emoji', () => {
    log.info('Mod', 'hello');
    expect(logSpy).toHaveBeenCalledTimes(1);
    expect(logSpy.mock.calls[0][0] as string).toBe(`[${LogEmoji.INFO}] [Mod] hello`);
  });

  it('log.success goes to console.log with the SUCCESS emoji', () => {
    log.success('Mod', 'done');
    expect(logSpy.mock.calls[0][0] as string).toBe(`[${LogEmoji.SUCCESS}] [Mod] done`);
  });

  it('log.error goes to console.ERROR with the ERROR emoji', () => {
    log.error('Mod', 'boom');
    expect(errorSpy).toHaveBeenCalledTimes(1);
    expect(errorSpy.mock.calls[0][0] as string).toBe(`[${LogEmoji.ERROR}] [Mod] boom`);
  });

  it('log.warning goes to console.WARN with the WARNING emoji', () => {
    log.warning('Mod', 'careful');
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0] as string).toBe(`[${LogEmoji.WARNING}] [Mod] careful`);
  });

  it('action helpers (load/update/query/data) tag with their action emoji', () => {
    log.load('Mod', 'fetching');
    log.update('Mod', 'updating');
    log.query('Mod', 'querying');
    log.data('Mod', 'crunching');

    const messages = logSpy.mock.calls.map((c: unknown[]) => c[0] as string);
    expect(messages).toEqual([
      `[${LogEmoji.LOAD}] [Mod] fetching`,
      `[${LogEmoji.UPDATE}] [Mod] updating`,
      `[${LogEmoji.QUERY}] [Mod] querying`,
      `[${LogEmoji.DATA}] [Mod] crunching`,
    ]);
  });

  it('log.custom uses the supplied emoji', () => {
    log.custom('🎯', 'Mod', 'aim');
    expect(logSpy.mock.calls[0][0] as string).toBe('[🎯] [Mod] aim');
  });

  it('log.raw passes through the formatted message untouched', () => {
    log.raw('already-formatted-message');
    expect(logSpy.mock.calls[0][0] as string).toBe('already-formatted-message');
  });

  it('forwards trailing args through to console (objects, errors, primitives)', () => {
    const obj = { a: 1 };
    const err = new Error('details');
    log.info('Mod', 'with extras', obj, err, 42);
    const call = logSpy.mock.calls[0];
    expect(call[1]).toBe(obj);
    expect(call[2]).toBe(err);
    expect(call[3]).toBe(42);
  });
});

describe('createModuleLogger', () => {
  let logSpy: ReturnType<typeof vi.spyOn>;
  let warnSpy: ReturnType<typeof vi.spyOn>;
  let errorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
  });

  afterEach(() => {
    logSpy.mockRestore();
    warnSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it('binds every method to the supplied module name', () => {
    const ml = createModuleLogger('MyMod');

    ml.log('plain');
    ml.info('info');
    ml.success('done');
    ml.error('boom');
    ml.warning('careful');
    ml.load('fetching');
    ml.update('updating');
    ml.query('querying');
    ml.data('crunching');
    ml.custom('🎯', 'aim');

    // log + info + success + load + update + query + data + custom each
    // route through console.log. error and warning route to console.error
    // and console.warn respectively. Total console.log calls: 8.
    const allLogMessages = logSpy.mock.calls.map((c: unknown[]) => c[0] as string);
    expect(allLogMessages.length).toBe(8);
    for (const m of allLogMessages) {
      expect(m).toContain('[MyMod]');
    }

    expect(errorSpy.mock.calls[0][0] as string).toContain('[MyMod] boom');
    expect(warnSpy.mock.calls[0][0] as string).toContain('[MyMod] careful');
  });

  it('uses Modules.* values as module strings', () => {
    const ml = createModuleLogger(Modules.LUXAR);
    ml.info('hi');
    expect(logSpy.mock.calls[0][0] as string).toContain('[Luxar]');
  });
});
