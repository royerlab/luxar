import { describe, expect, it } from 'vitest';
import {
  compactGSplatLabelIds,
  gsplatLabelAt,
  projectGSplatLabelIndices,
} from '../../../../data/gsplats/label-channel';

describe('GSplats categorical label channel', () => {
  it('maps uint64 ids exactly through compact GPU indices', () => {
    const aboveFloatPrecision = 9007199254740993n;
    const channel = compactGSplatLabelIds(
      new BigUint64Array([aboveFloatPrecision, 7n, aboveFloatPrecision]),
      { [aboveFloatPrecision.toString()]: 'large id', '7': 'seven' }
    );

    expect(Array.from(channel.indices)).toEqual([2, 1, 2]);
    expect(gsplatLabelAt(channel, 0)).toEqual({
      id: aboveFloatPrecision.toString(),
      name: 'large id',
    });
  });

  it('compacts labels with the same source indices as visible splats', () => {
    expect(
      Array.from(projectGSplatLabelIndices(new Uint32Array([1, 2, 3]), new Uint32Array([2, 0]), 2))
    ).toEqual([3, 1]);
  });

  it('copies labels on the identity projection path', () => {
    const source = new Uint32Array([1, 2, 3]);
    const projected = projectGSplatLabelIndices(source, undefined, 2);

    source[0] = 9;

    expect(Array.from(projected)).toEqual([1, 2]);
    expect(projected.buffer).not.toBe(source.buffer);
  });

  it('fails loudly when projection references a missing source label', () => {
    expect(() =>
      projectGSplatLabelIndices(new Uint32Array([1, 2]), new Uint32Array([2]), 1)
    ).toThrow(/source index 2.*2 label indices/);
  });

  it('fails loudly when the vocabulary cannot interpret an id', () => {
    expect(() => compactGSplatLabelIds(new Uint8Array([4]), { '3': 'three' })).toThrow(
      /label id 4.*label_vocabulary/
    );
  });
});
