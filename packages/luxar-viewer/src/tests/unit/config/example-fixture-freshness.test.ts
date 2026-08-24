import { describe, expect, it, vi } from 'vitest';
import { checkExampleFixtureFreshness } from '../../../../tools/example-fixture-freshness';

describe('checkExampleFixtureFreshness', () => {
  it('reports current fixtures when the checker succeeds', () => {
    const checker = vi.fn();

    expect(checkExampleFixtureFreshness('/checkout', checker)).toBe('current');
    expect(checker).toHaveBeenCalledOnce();
    expect(checker).toHaveBeenCalledWith('/checkout');
  });

  it('reports stale fixtures when the checker exits non-zero', () => {
    const checker = vi.fn(() => {
      throw { status: 1 };
    });

    expect(checkExampleFixtureFreshness('/checkout', checker)).toBe('stale');
  });

  it('reports an unavailable checker when it cannot be launched', () => {
    const checker = vi.fn(() => {
      throw new Error('spawn hatch ENOENT');
    });

    expect(checkExampleFixtureFreshness('/checkout', checker)).toBe('unavailable');
  });
});
