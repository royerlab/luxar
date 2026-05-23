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

  // [utils.md/O2][P10] Replaced a single it() that bundled 8 console.log
  // assertions with an it.each — each console.log-routed method is one row,
  // and the two console.warn/error-routed methods are pinned in dedicated
  // tests below. A method that silently switches console routes (e.g.
  // `info` becoming `warn`) now produces a single failing case instead of
  // a confusing aggregate.
  describe('createModuleLogger — every method binds the supplied module name', () => {
    it.each([
      ['log', 'plain'],
      ['info', 'info'],
      ['success', 'done'],
      ['load', 'fetching'],
      ['update', 'updating'],
      ['query', 'querying'],
      ['data', 'crunching'],
    ] as const)('%s routes through console.log and tags the module', (method, message) => {
      const ml = createModuleLogger('MyMod');
      (ml[method as keyof typeof ml] as (m: string) => void)(message);

      // Exactly one console.log call, tagged with the module name.
      expect(logSpy.mock.calls.length).toBe(1);
      expect(logSpy.mock.calls[0][0] as string).toContain('[MyMod]');
      // The user-supplied message must reach the console.
      expect(logSpy.mock.calls[0][0] as string).toContain(message);
    });

    it('custom() also routes through console.log and tags the module', () => {
      const ml = createModuleLogger('MyMod');
      ml.custom('🎯', 'aim');
      expect(logSpy.mock.calls.length).toBe(1);
      expect(logSpy.mock.calls[0][0] as string).toContain('[MyMod]');
    });

    it('error() routes through console.error and tags the module', () => {
      const ml = createModuleLogger('MyMod');
      ml.error('boom');
      expect(errorSpy.mock.calls.length).toBe(1);
      expect(errorSpy.mock.calls[0][0] as string).toContain('[MyMod] boom');
    });

    it('warning() routes through console.warn and tags the module', () => {
      const ml = createModuleLogger('MyMod');
      ml.warning('careful');
      expect(warnSpy.mock.calls.length).toBe(1);
      expect(warnSpy.mock.calls[0][0] as string).toContain('[MyMod] careful');
    });
  });

  it('uses Modules.* values as module strings', () => {
    const ml = createModuleLogger(Modules.LUXAR);
    ml.info('hi');
    expect(logSpy.mock.calls[0][0] as string).toContain('[Luxar]');
  });
});
