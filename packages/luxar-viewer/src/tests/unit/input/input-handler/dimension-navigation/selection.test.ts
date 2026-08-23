import { describe, expect, it } from 'vitest';
import type { SimpleDims } from '../../../../../types/dims';
import { describeNavigableKeys } from '../../../../../input/input-handler/dimension-navigation/selection';

const dims: SimpleDims = {
  ndim: 5,
  displayed: [0, 1, 2],
  currentStep: [0, 0, 0, 7, 0],
};

describe('describeNavigableKeys', () => {
  it('lists digit slots using normalized dimension names', () => {
    expect(describeNavigableKeys(4, dims, ['X', 'Y', 'Z', 'Frame', 'Channel'])).toBe(
      'Dimension key 5 is unavailable. Use 1 for Frame or 2 for Channel.'
    );
  });

  it('uses the same fallback labels as the dimension panel', () => {
    expect(describeNavigableKeys(4, dims, ['Dim 0', 'Dim 1', 'Dim 2', 'Dim 3', 'Dim 4'])).toBe(
      'Dimension key 5 is unavailable. Use 1 for Dim 3 or 2 for Dim 4.'
    );
  });

  it('describes scenes without navigable dimensions accurately', () => {
    expect(describeNavigableKeys(4, null, [])).toBe(
      'Dimension key 5 is unavailable. This scene has no navigable dimensions.'
    );
    expect(describeNavigableKeys(4, { ...dims, displayed: [0, 1, 2, 3, 4] }, [])).toBe(
      'Dimension key 5 is unavailable. This scene has no navigable dimensions.'
    );
  });
});
