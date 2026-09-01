import { describe, expect, it } from 'vitest';
import { canLadderRegrow } from '../../../../data/scene-loader/commit/stamp-view-version';

describe('canLadderRegrow', () => {
  it('returns false only after an unsliced progressive ladder completes', () => {
    expect(canLadderRegrow({ loader: { hasMoreLODs: false } }, 3)).toBe(false);
    expect(canLadderRegrow({ loader: { hasMoreLODs: true } }, 3)).toBe(true);
    expect(canLadderRegrow({ loader: { hasMoreLODs: false } }, 4)).toBe(true);
  });

  it('keeps non-progressive leaves reusable', () => {
    expect(canLadderRegrow({ loader: {} }, 3)).toBe(true);
    expect(canLadderRegrow(undefined, 3)).toBe(true);
  });
});
