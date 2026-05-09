/**
 * Unit tests for the post-processing effect-disposal helpers.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  safeDisposeEffect,
  safeDisposeEffects,
  safeRemoveAndDisposePass,
} from '../../../../rendering/post-processing/effect-disposal';

describe('safeDisposeEffect', () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  });
  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('returns true and is a no-op for null/undefined effect', () => {
    expect(safeDisposeEffect(null, 'x')).toBe(true);
    expect(safeDisposeEffect(undefined, 'x')).toBe(true);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('returns true and is a no-op when effect lacks dispose()', () => {
    expect(safeDisposeEffect({} as never, 'x')).toBe(true);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('calls dispose() and returns true when present', () => {
    const dispose = vi.fn();
    expect(safeDisposeEffect({ dispose }, 'Bloom')).toBe(true);
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('catches a throwing dispose(), warns with the supplied name, returns false', () => {
    const dispose = vi.fn(() => {
      throw new Error('boom');
    });
    expect(safeDisposeEffect({ dispose }, 'BloomRebuild')).toBe(false);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    const [msg] = warnSpy.mock.calls[0] as string[];
    expect(msg).toContain('BloomRebuild');
    expect(msg).toContain('boom');
  });
});

describe('safeDisposeEffects', () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  });
  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('returns 0 for an empty list', () => {
    expect(safeDisposeEffects([])).toBe(0);
  });

  it('disposes each effect in order and counts successes', () => {
    const order: string[] = [];
    const a = { dispose: () => order.push('a') };
    const b = { dispose: () => order.push('b') };
    const c = { dispose: () => order.push('c') };
    const count = safeDisposeEffects([
      [a, 'A'],
      [b, 'B'],
      [c, 'C'],
    ]);
    expect(order).toEqual(['a', 'b', 'c']);
    expect(count).toBe(3);
  });

  it('does not let a throwing effect block subsequent disposals', () => {
    const order: string[] = [];
    const a = {
      dispose: () => {
        order.push('a');
        throw new Error('a-fail');
      },
    };
    const b = { dispose: () => order.push('b') };
    const c = {
      dispose: () => {
        order.push('c');
        throw new Error('c-fail');
      },
    };
    const count = safeDisposeEffects([
      [a, 'A'],
      [b, 'B'],
      [c, 'C'],
    ]);
    expect(order).toEqual(['a', 'b', 'c']);
    expect(count).toBe(1); // only b succeeded
    expect(warnSpy).toHaveBeenCalledTimes(2);
  });

  it('skips null/undefined entries without contributing to the success count', () => {
    const a = { dispose: vi.fn() };
    const count = safeDisposeEffects([
      [null, 'Null'],
      [undefined, 'Undef'],
      [{} as never, 'NoDispose'],
      [a, 'A'],
    ]);
    expect(count).toBe(1);
    expect(a.dispose).toHaveBeenCalledTimes(1);
  });
});

describe('safeRemoveAndDisposePass (Phase 21A)', () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  });
  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('no-op when pass is null/undefined; composer untouched', () => {
    const composer = { removePass: vi.fn() };
    expect(safeRemoveAndDisposePass(composer, null, 'p')).toBe(true);
    expect(safeRemoveAndDisposePass(composer, undefined, 'p')).toBe(true);
    expect(composer.removePass).not.toHaveBeenCalled();
  });

  it('removes pass from composer then calls dispose()', () => {
    const dispose = vi.fn();
    const pass = { dispose };
    const composer = { removePass: vi.fn() };
    expect(safeRemoveAndDisposePass(composer, pass, 'effectPass')).toBe(true);
    expect(composer.removePass).toHaveBeenCalledWith(pass);
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it('with no composer, only dispose() is invoked', () => {
    const dispose = vi.fn();
    expect(safeRemoveAndDisposePass(null, { dispose }, 'p')).toBe(true);
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it('logs a warning if composer.removePass throws but still disposes', () => {
    const dispose = vi.fn();
    const pass = { dispose };
    const composer = {
      removePass: vi.fn(() => {
        throw new Error('remove-fail');
      }),
    };
    expect(safeRemoveAndDisposePass(composer, pass, 'effectPass')).toBe(true);
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalled();
    const [msg] = warnSpy.mock.calls[0] as string[];
    expect(msg).toContain('effectPass');
  });

  it('returns false when dispose throws (after removePass succeeds)', () => {
    const dispose = vi.fn(() => {
      throw new Error('dispose-fail');
    });
    const pass = { dispose };
    const composer = { removePass: vi.fn() };
    expect(safeRemoveAndDisposePass(composer, pass, 'effectPass')).toBe(false);
    expect(composer.removePass).toHaveBeenCalledTimes(1);
  });
});
