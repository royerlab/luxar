/**
 * Direct tests for the shared progressive-loader concatenation helpers
 * (`data/loaders/progressive/concat-helpers.ts`), focused on the
 * LADDER-DTYPE CONTRACT: `TypedArray.set` converts by VALUE, not
 * semantics, so mixed dtypes across LOD levels would silently corrupt
 * (Uint8 0..255 written into a Float32 0..1 merge = 255× values;
 * Float32 into Uint8 truncates). The helpers fail fast instead.
 */

import { describe, it, expect } from 'vitest';
import {
  concatOptionalField,
  concatRequiredField,
} from '../../../../../data/loaders/progressive/concat-helpers';

type Part = { data: Float32Array | Uint8Array; n: number };
const count = (p: Part) => p.n;

describe('concat-helpers ladder-dtype contract', () => {
  it('concatenates same-dtype parts (dtype preserved, values in order)', () => {
    const parts: Part[] = [
      { data: new Uint8Array([1, 2]), n: 2 },
      { data: new Uint8Array([3]), n: 1 },
    ];
    const out = concatRequiredField(parts, (p) => p.data, count, 1, 'x');
    expect(out).toBeInstanceOf(Uint8Array);
    expect(Array.from(out as Uint8Array)).toEqual([1, 2, 3]);
  });

  it('throws a descriptive error on mixed dtypes (required field)', () => {
    const parts: Part[] = [
      { data: new Uint8Array([255, 255]), n: 2 },
      { data: new Float32Array([0.5]), n: 1 },
    ];
    expect(() => concatRequiredField(parts, (p) => p.data, count, 1, 'colors')).toThrow(
      /level 1 carries 'colors' as Float32Array but level 0 uses Uint8Array/
    );
  });

  it('throws on mixed dtypes through the optional-field path too', () => {
    const parts: Part[] = [
      { data: new Float32Array([0.5]), n: 1 },
      { data: new Uint8Array([255]), n: 1 },
    ];
    expect(() => concatOptionalField(parts, (p) => p.data, count, 1, 'scalars')).toThrow(
      /'scalars' as Uint8Array but level 0 uses Float32Array/
    );
  });

  it('optional field still drops on mixed PRESENCE without touching dtypes', () => {
    const parts: Array<{ data: Uint8Array | null; n: number }> = [
      { data: new Uint8Array([1]), n: 1 },
      { data: null, n: 1 },
    ];
    expect(
      concatOptionalField(
        parts,
        (p) => p.data,
        (p) => p.n,
        1,
        'colors'
      )
    ).toBeUndefined();
  });
});
