import { describe, expect, it } from 'vitest';

import type { ViewState } from '../../../../data/data-loader-types';
import { withPassDirectives } from '../../../../data/loaders/pass-directives';

const VIEW: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 3],
  tolerance: [0, 0, 0, 0],
};

describe('withPassDirectives', () => {
  it('returns the SAME view state when the pass carries no directive', () => {
    expect(withPassDirectives(VIEW, {})).toBe(VIEW);
  });

  it('injects the playback budget and pinned depth into a copy', () => {
    const out = withPassDirectives(VIEW, { frameBudgetMs: 12, ladderDepth: 'auto' });
    expect(out).not.toBe(VIEW);
    expect(out).toMatchObject({ ...VIEW, frameBudgetMs: 12, ladderDepth: 'auto' });
    expect(VIEW.frameBudgetMs).toBeUndefined();
  });

  it('injects a lone pinned depth too', () => {
    expect(withPassDirectives(VIEW, { ladderDepth: 2 }).ladderDepth).toBe(2);
  });
});
