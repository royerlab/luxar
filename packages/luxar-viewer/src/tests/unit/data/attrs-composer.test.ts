import { describe, it, expect, test } from 'vitest';
import * as fc from 'fast-check';
import {
  composeAttrs,
  collectAncestorAttrs,
  getEffectiveAttrs,
} from '../../../data/attrs-composer';
import type { SceneNode } from '../../../data/data-loader-types';

describe('composeAttrs', () => {
  it('returns identity for empty chain', () => {
    const e = composeAttrs([]);
    expect(e.opacity).toBe(1);
    expect(e.absorption).toBe(1);
    expect(e.gamma).toBe(1);
    expect(e.intensity).toBe(1);
    expect(e.offset).toBe(0);
    expect(e.blending_mode).toBe('additive');
  });

  it('multiplies opacity/gamma/intensity through the chain', () => {
    const e = composeAttrs([
      { opacity: 0.5, intensity: 2.0, gamma: 1.2 },
      { opacity: 0.5, intensity: 3.0, gamma: 1.5 },
    ]);
    expect(e.opacity).toBeCloseTo(0.25, 5);
    expect(e.intensity).toBeCloseTo(6.0, 5);
    expect(e.gamma).toBeCloseTo(1.8, 5);
  });

  it('multiplies absorption through the chain (identity 1, unclamped above, floored at 0)', () => {
    // Volumetric κ composes like opacity/intensity — ancestors scale it;
    // κ = 0 at ANY level zeroes the subtree (pure-additive look).
    expect(composeAttrs([{ absorption: 2.0 }, { absorption: 0.5 }]).absorption).toBeCloseTo(
      1.0,
      6
    );
    expect(composeAttrs([{ opacity: 0.5 }]).absorption).toBe(1); // unset ⇒ identity
    expect(composeAttrs([{ absorption: 4 }, { absorption: 4 }]).absorption).toBe(16); // no upper clamp
    expect(composeAttrs([{ absorption: 3 }, { absorption: 0 }]).absorption).toBe(0);
    expect(composeAttrs([{ absorption: -2 }]).absorption).toBe(0); // floored, never negative
  });

  it('sums offsets through the chain', () => {
    const e = composeAttrs([{ offset: 0.1 }, { offset: -0.05 }, { offset: 0.2 }]);
    expect(e.offset).toBeCloseTo(0.25, 5);
  });

  it('uses nearest-set blending_mode (later overrides earlier)', () => {
    expect(composeAttrs([{ blending_mode: 'normal' }]).blending_mode).toBe('normal');
    expect(
      composeAttrs([{ blending_mode: 'normal' }, { blending_mode: 'max' }]).blending_mode
    ).toBe('max');
    // Unset child keeps parent's choice
    expect(composeAttrs([{ blending_mode: 'normal' }, { opacity: 0.5 }]).blending_mode).toBe(
      'normal'
    );
  });

  it("normalizes an unknown winning blending_mode to 'normal'", () => {
    // The composed mode is the material chokepoint — a malformed zarr
    // attr must come out canonical, not leak an arbitrary string.
    expect(composeAttrs([{ blending_mode: 'compose-bogus' }]).blending_mode).toBe('normal');
    // Empty string is malformed-authored (the Python validator rejects it),
    // NOT "unset" — it must take the unknown-mode path, not the default.
    expect(composeAttrs([{ blending_mode: '' }]).blending_mode).toBe('normal');
    expect(
      composeAttrs([{ blending_mode: 'additive' }, { blending_mode: 'compose-bogus-2' }])
        .blending_mode
    ).toBe('normal');
  });

  it("composes to 'additive' when no level in the chain sets a mode", () => {
    expect(composeAttrs([]).blending_mode).toBe('additive');
    expect(composeAttrs([{ opacity: 0.5 }, { gamma: 2 }]).blending_mode).toBe('additive');
    expect(composeAttrs([{ blending_mode: undefined }]).blending_mode).toBe('additive');
  });

  it('clamps opacity to [0, 1]', () => {
    expect(composeAttrs([{ opacity: 2.0 }]).opacity).toBe(1);
    expect(composeAttrs([{ opacity: -0.5 }]).opacity).toBe(0);
  });

  it('clamps gamma to [0.1, 10]', () => {
    expect(composeAttrs([{ gamma: 100 }]).gamma).toBe(10);
    expect(composeAttrs([{ gamma: 0.01 }]).gamma).toBe(0.1);
  });

  it('treats unset fields as identity (not 0)', () => {
    const e = composeAttrs([{ opacity: 0.5 }, {}]);
    expect(e.opacity).toBeCloseTo(0.5, 5);
    expect(e.intensity).toBe(1);
    expect(e.offset).toBe(0);
  });
});

describe('collectAncestorAttrs / getEffectiveAttrs', () => {
  const root: SceneNode = {
    path: '',
    type: 'scene',
    hasSpatialIndex: false,
    attrs: {},
    children: [
      {
        path: 'grp',
        type: 'group',
        hasSpatialIndex: false,
        attrs: { opacity: 0.5, intensity: 2.0 },
        children: [
          {
            path: 'grp/pts',
            type: 'points',
            hasSpatialIndex: true,
            attrs: { opacity: 0.5, offset: 0.1 },
          },
        ],
      },
    ],
  };

  it('walks from root to the target and collects each node attrs', () => {
    const chain = collectAncestorAttrs(root, 'grp/pts');
    expect(chain.length).toBe(2);
    expect(chain[0].opacity).toBe(0.5);
    expect(chain[1].offset).toBe(0.1);
  });

  it('composes an effective opacity of 0.25 for a 0.5×0.5 chain', () => {
    const e = getEffectiveAttrs(root, 'grp/pts');
    expect(e.opacity).toBeCloseTo(0.25, 5);
    expect(e.intensity).toBeCloseTo(2.0, 5);
    expect(e.offset).toBeCloseTo(0.1, 5);
  });

  it('returns identity for a path that does not exist', () => {
    const e = getEffectiveAttrs(root, 'missing/node');
    expect(e.opacity).toBe(1);
  });

  // Regression: the real SceneLoader uses leading-slash paths ('/', '/grp',
  // '/grp/pts'). A naive path-segment join would produce '//grp' and miss
  // every child. Verify the leading-slash format walks correctly.
  describe('with scene-loader leading-slash paths', () => {
    const slashRoot: SceneNode = {
      path: '/',
      type: 'scene',
      hasSpatialIndex: false,
      attrs: {},
      children: [
        {
          path: '/RedCloud',
          type: 'points',
          hasSpatialIndex: true,
          attrs: { layer: true, opacity: 0.8, gamma: 2.0 },
        },
        {
          path: '/CompositeLayer',
          type: 'group',
          hasSpatialIndex: false,
          attrs: { layer: true, opacity: 0.5 },
          children: [
            {
              path: '/CompositeLayer/GreenPart',
              type: 'points',
              hasSpatialIndex: true,
              attrs: { opacity: 0.5 },
            },
          ],
        },
      ],
    };

    it('walks to a direct child of the slash-root', () => {
      const chain = collectAncestorAttrs(slashRoot, '/RedCloud');
      expect(chain.length).toBe(1);
      expect(chain[0].opacity).toBe(0.8);
      expect(chain[0].gamma).toBe(2.0);
    });

    it('walks to a nested child (group → data leaf)', () => {
      const chain = collectAncestorAttrs(slashRoot, '/CompositeLayer/GreenPart');
      expect(chain.length).toBe(2);
      expect(chain[0].opacity).toBe(0.5);
      expect(chain[1].opacity).toBe(0.5);
    });

    it('composes effective opacity of 0.25 for slash-formatted paths', () => {
      const e = getEffectiveAttrs(slashRoot, '/CompositeLayer/GreenPart');
      expect(e.opacity).toBeCloseTo(0.25, 5);
    });

    it('returns identity for an unknown slash path', () => {
      const e = getEffectiveAttrs(slashRoot, '/does/not/exist');
      expect(e.opacity).toBe(1);
    });
  });
});

// [data.md/H6][P12] composeAttrs — algebraic monoid invariants over arbitrary
// root-to-leaf chains. Pins the spec:
//   * opacity / gamma / intensity compose multiplicatively
//   * offset composes additively
//   * blending_mode is nearest-set (right-biased)
//   * empty chain is the identity
//   * appending an all-undefined record is a no-op (identity element)
//   * opacity is clamped to [0,1]; gamma to [0.1, 10]; intensity to [0, +inf)
//
// These are the algebraic invariants that mutations in the production source
// (e.g. swapping + with * on offset, dropping a clamp, biasing wrong
// direction on blending_mode) would violate.
// fast-check's fc.float requires Math.fround()-clamped bounds.
const f = (x: number) => Math.fround(x);
const attrArb: fc.Arbitrary<{
  opacity?: number;
  absorption?: number;
  gamma?: number;
  intensity?: number;
  offset?: number;
  blending_mode?: string;
}> = fc.record({
  opacity: fc.option(fc.float({ min: f(0.01), max: f(1), noNaN: true, noDefaultInfinity: true }), {
    nil: undefined,
  }),
  absorption: fc.option(
    fc.float({ min: f(0), max: f(8), noNaN: true, noDefaultInfinity: true }),
    { nil: undefined }
  ),
  gamma: fc.option(fc.float({ min: f(0.5), max: f(2), noNaN: true, noDefaultInfinity: true }), {
    nil: undefined,
  }),
  intensity: fc.option(fc.float({ min: f(0.1), max: f(4), noNaN: true, noDefaultInfinity: true }), {
    nil: undefined,
  }),
  offset: fc.option(fc.float({ min: f(-1), max: f(1), noNaN: true, noDefaultInfinity: true }), {
    nil: undefined,
  }),
  // Only the five canonical modes: composeAttrs normalizes the winning
  // string (unknown → 'normal'), so probing invented modes would test
  // the normalizer, not the right-bias — covered separately below.
  blending_mode: fc.option(fc.constantFrom('normal', 'additive', 'max', 'opaque', 'luminous', 'volumetric'), {
    nil: undefined,
  }),
});

describe('composeAttrs — algebraic invariants (data.md H6)', () => {
  test('empty chain returns the documented identity', () => {
    const e = composeAttrs([]);
    expect(e).toEqual({
      opacity: 1.0,
      absorption: 1.0,
      gamma: 1.0,
      intensity: 1.0,
      offset: 0.0,
      blending_mode: 'additive',
    });
  });

  test('appending an all-undefined record is the monoid identity (no-op)', () => {
    fc.assert(
      fc.property(fc.array(attrArb, { maxLength: 8 }), (chain) => {
        const before = composeAttrs(chain);
        const after = composeAttrs([...chain, {}]);
        // All numeric fields equal; blending_mode unchanged.
        expect(after.opacity).toBeCloseTo(before.opacity, 6);
        expect(after.absorption).toBeCloseTo(before.absorption, 6);
        expect(after.gamma).toBeCloseTo(before.gamma, 6);
        expect(after.intensity).toBeCloseTo(before.intensity, 6);
        expect(after.offset).toBeCloseTo(before.offset, 6);
        expect(after.blending_mode).toBe(before.blending_mode);
      })
    );
  });

  test('opacity is the clamped product of set values', () => {
    fc.assert(
      fc.property(
        fc.array(fc.float({ min: f(0.01), max: f(1), noNaN: true, noDefaultInfinity: true }), {
          maxLength: 6,
        }),
        (xs) => {
          const e = composeAttrs(xs.map((x) => ({ opacity: x })));
          const raw = xs.reduce((acc, x) => acc * x, 1);
          // Clamped to [0, 1] — but xs are already in [0, 1], so the product is too.
          expect(e.opacity).toBeCloseTo(Math.max(0, Math.min(1, raw)), 4);
        }
      )
    );
  });

  test('offset is the additive sum of set offsets', () => {
    fc.assert(
      fc.property(
        fc.array(fc.float({ min: f(-1), max: f(1), noNaN: true, noDefaultInfinity: true }), {
          maxLength: 8,
        }),
        (xs) => {
          const e = composeAttrs(xs.map((x) => ({ offset: x })));
          const sum = xs.reduce((acc, x) => acc + x, 0);
          expect(e.offset).toBeCloseTo(sum, 3);
        }
      )
    );
  });

  test('blending_mode is right-biased (later wins)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom('normal', 'additive', 'max', 'opaque', 'luminous', 'volumetric'), {
          minLength: 1,
          maxLength: 6,
        }),
        (modes) => {
          const e = composeAttrs(modes.map((m) => ({ blending_mode: m })));
          expect(e.blending_mode).toBe(modes[modes.length - 1]);
        }
      )
    );
  });

  test('opacity clamps to [0, 1] under any input', () => {
    fc.assert(
      fc.property(
        fc.array(fc.float({ min: f(-2), max: f(4), noNaN: true, noDefaultInfinity: true }), {
          maxLength: 4,
        }),
        (xs) => {
          const e = composeAttrs(xs.map((x) => ({ opacity: x })));
          expect(e.opacity).toBeGreaterThanOrEqual(0);
          expect(e.opacity).toBeLessThanOrEqual(1);
        }
      )
    );
  });

  test('gamma clamps to [0.1, 10] under any input', () => {
    fc.assert(
      fc.property(
        fc.array(fc.float({ min: f(0.001), max: f(100), noNaN: true, noDefaultInfinity: true }), {
          maxLength: 4,
        }),
        (xs) => {
          const e = composeAttrs(xs.map((x) => ({ gamma: x })));
          expect(e.gamma).toBeGreaterThanOrEqual(0.1);
          expect(e.gamma).toBeLessThanOrEqual(10);
        }
      )
    );
  });

  test('intensity is clamped at 0 (cannot be negative)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.float({ min: f(-2), max: f(4), noNaN: true, noDefaultInfinity: true }), {
          maxLength: 4,
        }),
        (xs) => {
          const e = composeAttrs(xs.map((x) => ({ intensity: x })));
          expect(e.intensity).toBeGreaterThanOrEqual(0);
        }
      )
    );
  });

  test('absorption is floored at 0 under any input (like intensity)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.float({ min: f(-2), max: f(4), noNaN: true, noDefaultInfinity: true }), {
          maxLength: 4,
        }),
        (xs) => {
          const e = composeAttrs(xs.map((x) => ({ absorption: x })));
          expect(e.absorption).toBeGreaterThanOrEqual(0);
        }
      )
    );
  });

  test('absorption is the unclamped-above product of set values, floored at 0', () => {
    fc.assert(
      fc.property(
        fc.array(fc.float({ min: f(0), max: f(8), noNaN: true, noDefaultInfinity: true }), {
          maxLength: 6,
        }),
        (xs) => {
          const e = composeAttrs(xs.map((x) => ({ absorption: x })));
          const raw = xs.reduce((acc, x) => acc * x, 1);
          expect(e.absorption).toBeCloseTo(raw, 3);
          expect(e.absorption).toBeGreaterThanOrEqual(0);
        }
      )
    );
  });
});
