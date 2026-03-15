import { describe, expect, it } from 'vitest';

import { type ArrayMetadata } from '../../../data/array-decoder';
import { RangeLoader } from '../../../data/loaders/range-loader';

describe('RangeLoader.detectEncoding', () => {
  it('detects array_ref when target is present without name', () => {
    const attrs: ArrayMetadata = {
      encoding: {
        target: '/SharedNode/colors',
      },
    };

    expect(RangeLoader.detectEncoding(attrs)).toBe('array_ref');
  });

  it('detects array_ref when name is explicit', () => {
    const attrs: ArrayMetadata = {
      encoding: {
        name: 'array_ref',
        target: '/SharedNode/colors',
      },
    };

    expect(RangeLoader.detectEncoding(attrs)).toBe('array_ref');
  });
});
