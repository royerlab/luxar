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
  concatColorsWhiteFilled,
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

  it('respects perItem (e.g. 3-vector positions)', () => {
    const parts: Part[] = [
      { data: new Float32Array([0, 0, 0]), n: 1 },
      { data: new Float32Array([1, 1, 1, 2, 2, 2]), n: 2 },
    ];
    const out = concatRequiredField(parts, (p) => p.data, count, 3, 'positions');
    expect(out).toHaveLength(9);
    expect(Array.from(out as Float32Array).slice(3)).toEqual([1, 1, 1, 2, 2, 2]);
  });

  it('optional field concatenates when every part carries it', () => {
    const parts: Part[] = [
      { data: new Float32Array([1]), n: 1 },
      { data: new Float32Array([2]), n: 1 },
    ];
    const out = concatOptionalField(parts, (p) => p.data, count, 1, 'scalars');
    expect(Array.from(out as Float32Array)).toEqual([1, 2]);
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

/**
 * A part with no rows contributes nothing, so it gets no vote on which
 * optional attributes the merged result carries. The canonical empty payloads
 * (`createEmptyPointsData` / `createEmptyLinesData`) OMIT their optional
 * fields rather than emitting zero-length arrays, so without this rule a
 * single slice-culled level of an additive ladder stripped colors / radii /
 * sharpness / scalars from every OTHER level too (#1456).
 */
describe('concat-helpers zero-row abstainer rule', () => {
  type Opt = { data?: Float32Array | Uint8Array | null; n: number };
  const n = (p: Opt) => p.n;

  it('a zero-row part missing the field does not veto it for the others', () => {
    const parts: Opt[] = [{ n: 0 }, { data: new Uint8Array([7, 8]), n: 2 }];
    const out = concatOptionalField(parts, (p) => p.data, n, 1, 'colors');
    expect(out).toBeInstanceOf(Uint8Array);
    expect(Array.from(out as Uint8Array)).toEqual([7, 8]);
  });

  it('a zero-row part in the MIDDLE neither vetoes nor shifts the rows', () => {
    const parts: Opt[] = [
      { data: new Float32Array([1, 2]), n: 2 },
      { n: 0 },
      { data: new Float32Array([3]), n: 1 },
    ];
    const out = concatOptionalField(parts, (p) => p.data, n, 1, 'scalars');
    expect(Array.from(out as Float32Array)).toEqual([1, 2, 3]);
  });

  it('a zero-row part cannot impose its dtype on the merge', () => {
    // It copies nothing, so comparing its array against the real levels' could
    // only ever produce a spurious throw.
    const parts: Opt[] = [
      { data: new Float32Array(0), n: 0 },
      { data: new Uint8Array([9]), n: 1 },
    ];
    const out = concatOptionalField(parts, (p) => p.data, n, 1, 'colors');
    expect(out).toBeInstanceOf(Uint8Array);
    expect(Array.from(out as Uint8Array)).toEqual([9]);
  });

  it('all-zero-row parts fall back to all-or-nothing: undefined when absent', () => {
    // The degenerate case `[].every(...) === true` must not invent a field.
    // This is exactly the wholly-empty ladder, and it yields what it always did.
    const parts: Opt[] = [{ n: 0 }, { n: 0 }];
    expect(concatOptionalField(parts, (p) => p.data, n, 1, 'colors')).toBeUndefined();
  });

  it('a dtype mismatch still names the REAL ladder level past an abstainer', () => {
    // The abstainer rule hands `concatRequiredField` a filtered subset, so
    // without an index remap this would blame level 1 (and compare against
    // "level 0") for a mismatch that actually sits at level 2 — a corrupt-store
    // diagnostic pointing at the wrong level is worse than none.
    const parts: Opt[] = [
      { n: 0 },
      { data: new Float32Array([0.5]), n: 1 },
      { data: new Uint8Array([255]), n: 1 },
    ];
    expect(() => concatOptionalField(parts, (p) => p.data, n, 1, 'colors')).toThrow(
      /level 2 carries 'colors' as Uint8Array but level 1 uses Float32Array/
    );
  });

  it('all-zero-row parts that DO carry the field still yield a zero-length array', () => {
    const parts: Opt[] = [
      { data: new Uint8Array(0), n: 0 },
      { data: new Uint8Array(0), n: 0 },
    ];
    const out = concatOptionalField(parts, (p) => p.data, n, 1, 'colors');
    expect(out).toBeInstanceOf(Uint8Array);
    expect(out).toHaveLength(0);
  });
});

describe('concatColorsWhiteFilled', () => {
  type Part = { n: number; colors?: Uint8Array | Float32Array | null; k?: 3 | 4 };
  const get = (p: Part) => ({ colors: p.colors, components: p.k });
  const count = (p: Part) => p.n;

  it.fails('asserts each coloured part against its OWN declared layout, naming the level', () => {
    // A 2-element RGBA buffer declared RGB: 8 values for 2 x 3.
    expect(() =>
      concatColorsWhiteFilled<Part>(
        [
          { n: 1, colors: new Uint8Array(3) },
          { n: 2, colors: new Uint8Array(8) },
        ],
        get,
        count,
        't'
      )
    ).toThrow(/t \(LOD level 1\): colors length 8 does not match/);
  });

  it('is null when no part carries colours', () => {
    expect(concatColorsWhiteFilled<Part>([{ n: 2 }, { n: 1 }], get, count, 't')).toBeNull();
  });

  it('white-fills a colourless part at the dtype full scale', () => {
    const out = concatColorsWhiteFilled<Part>(
      [{ n: 1, colors: new Uint8Array([1, 2, 3]) }, { n: 2 }],
      get,
      count,
      't'
    )!;
    expect(Array.from(out.colors)).toEqual([1, 2, 3, 255, 255, 255, 255, 255, 255]);
    expect(out.colorComponents).toBe(3);
  });

  it('fills an RGBA ladder opaque and keeps the float dtype', () => {
    const out = concatColorsWhiteFilled<Part>(
      [{ n: 1 }, { n: 1, colors: new Float32Array([0.1, 0.2, 0.3, 0.4]), k: 4 }],
      get,
      count,
      't'
    )!;
    expect(out.colors).toBeInstanceOf(Float32Array);
    expect(Array.from(out.colors.subarray(0, 4))).toEqual([1, 1, 1, 1]);
    expect(out.colorComponents).toBe(4);
  });

  it('lets a zero-row part abstain from the dtype and layout checks', () => {
    const out = concatColorsWhiteFilled<Part>(
      [
        { n: 0, colors: new Float32Array(0) },
        { n: 1, colors: new Uint8Array([9, 9, 9]) },
      ],
      get,
      count,
      't'
    )!;
    expect(out.colors).toBeInstanceOf(Uint8Array);
    expect(Array.from(out.colors)).toEqual([9, 9, 9]);
  });

  it('rejects mixed dtypes and layouts, naming the level', () => {
    expect(() =>
      concatColorsWhiteFilled<Part>(
        [
          { n: 1, colors: new Uint8Array(3) },
          { n: 1, colors: new Float32Array(3) },
        ],
        get,
        count,
        't'
      )
    ).toThrow(/t: mixed color dtypes .*level 1/);
    expect(() =>
      concatColorsWhiteFilled<Part>(
        [
          { n: 1, colors: new Uint8Array(3) },
          { n: 1, colors: new Uint8Array(4), k: 4 },
        ],
        get,
        count,
        't'
      )
    ).toThrow(/mixed color layouts .*level 1: 4 vs 3 components/);
  });
});
