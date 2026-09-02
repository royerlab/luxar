import { describe, it, expect, test } from 'vitest';
import * as fc from 'fast-check';
import {
  composeAttrs,
  collectAncestorAttrs,
  getEffectiveAttrs,
} from '../../../data/attrs-composer';
import { applyEffectiveAttrs } from '../../../data/scene-loader/view-state/effective-attrs';
import type { SceneNode } from '../../../data/data-loader-types';
import { defaultBlendingMode } from '../../../types/geometry-capabilities';
import { GEOMETRY_TYPES } from '../../../types/format-contract';

describe('composeAttrs', () => {
  it('returns identity for empty chain', () => {
    const e = composeAttrs([]);
    expect(e.opacity).toBe(1);
    expect(e.absorption).toBe(1);
    expect(e.gamma).toBe(1);
    expect(e.intensity).toBe(1);
    expect(e.offset).toBe(0);
    expect(e.blending_mode).toBeUndefined();
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
    expect(composeAttrs([{ absorption: 2.0 }, { absorption: 0.5 }]).absorption).toBeCloseTo(1.0, 6);
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

  it('leaves blending_mode undefined when no level in the chain sets a mode', () => {
    // An unset chain composes to `undefined` — each consumer applies its own
    // per-type default (mesh → opaque, emissive → additive; spec §6.3).
    expect(composeAttrs([]).blending_mode).toBeUndefined();
    expect(composeAttrs([{ opacity: 0.5 }, { gamma: 2 }]).blending_mode).toBeUndefined();
    expect(composeAttrs([{ blending_mode: undefined }]).blending_mode).toBeUndefined();
  });

  it('uses nearest-set join (later overrides earlier)', () => {
    // Lines-only, and compositing for the same reason blending_mode is: the
    // author sets `join` once on a partition / LOD wrapper and every internal
    // child must inherit it (COMPOSITING_ATTRS in core/group/compositing.py).
    // Without this the wrapper's `join="none"` never reached the leaf and
    // every part rendered mitred — the opposite of what was asked.
    expect(composeAttrs([{ join: 'none' }]).join).toBe('none');
    expect(composeAttrs([{ join: 'none' }, { join: 'miter' }]).join).toBe('miter');
    // Unset child keeps the wrapper's choice — the partition case.
    expect(composeAttrs([{ join: 'none' }, { opacity: 0.5 }]).join).toBe('none');
  });

  it('leaves join undefined when no level in the chain sets one', () => {
    // Unset must stay `undefined` so `createLinesNode` applies
    // DEFAULT_LINE_JOIN rather than a value invented here.
    expect(composeAttrs([]).join).toBeUndefined();
    expect(composeAttrs([{ opacity: 0.5 }, { gamma: 2 }]).join).toBeUndefined();
    expect(composeAttrs([{ join: undefined }]).join).toBeUndefined();
  });

  it('passes a malformed join through unnormalized', () => {
    // Deliberately NOT validated here: `createLinesNode` runs the winning
    // value through `parseLineJoinStyle` and warns once. Normalizing in both
    // places would warn twice and hide the authored spelling from the log.
    expect(composeAttrs([{ join: 'mitre' }]).join).toBe('mitre');
    expect(composeAttrs([{ join: '' }]).join).toBe('');
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
  absorption: fc.option(fc.float({ min: f(0), max: f(8), noNaN: true, noDefaultInfinity: true }), {
    nil: undefined,
  }),
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
  blending_mode: fc.option(
    fc.constantFrom('normal', 'additive', 'max', 'opaque', 'luminous', 'volumetric'),
    {
      nil: undefined,
    }
  ),
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
      blending_mode: undefined,
      join: undefined,
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

describe('composeAttrs — an unset blending chain stays undefined (spec §6.3)', () => {
  it('reports undefined rather than inventing `additive`', () => {
    // The regression #1272 fixed, from the other side. `normalizeBlendingMode(undefined)`
    // used to return `'additive'` right here, which collapsed "nobody set a mode" into
    // "somebody set additive" — and that made `createMeshNode`'s `?? 'opaque'` DEAD CODE,
    // so every unstamped mesh rendered additive with §6.3's asymmetry inert. Leaving it
    // undefined is what lets each consumer apply its own per-type default.
    expect(composeAttrs([{}, {}]).blending_mode).toBeUndefined();
  });

  it('keeps nearest-setter-wins intact', () => {
    // The half that must NOT change: a mesh under `group(blending_mode="additive")`
    // still renders additive, which is why §6.3 forbids the writer from stamping a mode.
    expect(composeAttrs([{ blending_mode: 'additive' }, {}]).blending_mode).toBe('additive');
    expect(composeAttrs([{}, { blending_mode: 'max' }]).blending_mode).toBe('max');
    expect(
      composeAttrs([{ blending_mode: 'additive' }, { blending_mode: 'normal' }]).blending_mode
    ).toBe('normal');
  });

  it('still normalizes an unknown authored mode instead of reporting undefined', () => {
    // A garbage string is a DIFFERENT case from an absent one: someone tried to set
    // something, so it goes to `'normal'` with a one-time warning, where absence leaves
    // the per-type default to the consumer. Conflating them would silently turn a typo
    // into a mesh's `opaque`.
    expect(composeAttrs([{ blending_mode: 'nonsense' }]).blending_mode).toBe('normal');
  });

  it('defaults per GEOMETRY TYPE, mesh being the only non-additive one', () => {
    // Driven off the vocabulary so a fifth geometry type joins automatically. The claim
    // is that mesh is the ONLY exception — a second non-additive default should be a
    // deliberate decision, made at the table in `geometry-capabilities.ts`.
    const exceptions = GEOMETRY_TYPES.filter((t) => defaultBlendingMode(t) !== 'additive');
    expect(exceptions).toEqual(['mesh']);
    expect(defaultBlendingMode('mesh')).toBe('opaque');
  });

  it('falls back to additive for a non-geometry type', () => {
    // A group carries no default of its own — its descendants decide.
    expect(defaultBlendingMode('group')).toBe('additive');
    expect(defaultBlendingMode(undefined)).toBe('additive');
  });
});

// [#1600] `colormap` composes root→leaf, nearest-setter-wins, with its custom
// LUT bytes travelling as part of the SAME record.
//
// Before the fix it was not composed at all: `applyEffectiveAttrs` let it
// survive only inside the raw `...node.attrs` spread, so every consumer read
// the node's OWN value and an ancestor-authored palette never arrived. The
// Python half of the same bug manufactured a per-leaf `'gray'` that would have
// out-competed the ancestor even once composition existed.
describe('composeAttrs — colormap (#1600)', () => {
  it('is undefined for an unset chain', () => {
    expect(composeAttrs([]).colormap).toBeUndefined();
    expect(composeAttrs([{ opacity: 0.5 }, { gamma: 2 }]).colormap).toBeUndefined();
    expect(composeAttrs([{ opacity: 0.5 }]).customLutBytes).toBeUndefined();
  });

  it('inherits an ancestor palette when the leaf sets none', () => {
    expect(composeAttrs([{ colormap: 'plasma' }, { opacity: 0.5 }]).colormap).toBe('plasma');
  });

  it('lets the leaf override an ancestor palette (nearest-setter-wins)', () => {
    expect(composeAttrs([{ colormap: 'plasma' }, { colormap: 'inferno' }]).colormap).toBe(
      'inferno'
    );
  });

  it('takes the LAST setter across a three-level chain', () => {
    expect(
      composeAttrs([{ colormap: 'gray' }, { colormap: 'plasma' }, { opacity: 0.5 }]).colormap
    ).toBe('plasma');
  });

  it('carries the custom LUT bytes down with an inherited palette', () => {
    const lut = new Uint8Array([1, 2, 3]);
    const e = composeAttrs([{ colormap: 'custom', customLutBytes: lut }, { opacity: 0.5 }]);
    expect(e.colormap).toBe('custom');
    expect(e.customLutBytes).toBe(lut);
  });

  it('never pairs one node’s name with another node’s bytes', () => {
    const ancestorLut = new Uint8Array([9, 9, 9]);
    // A leaf that names a BUILTIN palette must not inherit the ancestor's
    // custom bytes — `getColormapTexture('viridis', <other LUT>)` would paint
    // the ancestor's palette under the leaf's name.
    const e = composeAttrs([
      { colormap: 'custom', customLutBytes: ancestorLut },
      { colormap: 'viridis' },
    ]);
    expect(e.colormap).toBe('viridis');
    expect(e.customLutBytes).toBeUndefined();

    // ...and the reverse: a leaf's own custom bytes win outright.
    const leafLut = new Uint8Array([4, 5, 6]);
    const e2 = composeAttrs([
      { colormap: 'custom', customLutBytes: ancestorLut },
      { colormap: 'custom', customLutBytes: leafLut },
    ]);
    expect(e2.customLutBytes).toBe(leafLut);
  });
});

describe('applyEffectiveAttrs — colormap reaches the consumer record (#1600)', () => {
  const lut = new Uint8Array([7, 7, 7, 7]);

  /** group authors a custom palette; the gsplats leaf authors none. */
  const graph: SceneNode = {
    path: '/',
    type: 'scene',
    hasSpatialIndex: false,
    attrs: {},
    children: [
      {
        path: '/layer',
        type: 'group',
        hasSpatialIndex: false,
        attrs: { layer: true, colormap: 'custom', customLutBytes: lut },
        children: [
          {
            path: '/layer/gs',
            type: 'gsplats',
            hasSpatialIndex: true,
            // `has_colors` on purpose: see the assertion below.
            attrs: { has_colors: true, opacity: 0.5 },
          },
          {
            path: '/layer/pts',
            type: 'points',
            hasSpatialIndex: true,
            attrs: { has_scalars: false },
          },
        ],
      },
    ],
  };

  it('hands a gsplats leaf the ancestor palette AND its LUT bytes', () => {
    const leaf = graph.children![0].children![0];
    const eff = applyEffectiveAttrs(graph, leaf);
    expect(eff.colormap).toBe('custom');
    expect(eff.customLutBytes).toBe(lut);
  });

  it('applies the inherited palette to a gsplats leaf that has its OWN colors', () => {
    // The deliberate semantics (#1600): a gsplats leaf is always
    // colormap-capable — its amplitude IS the scalar — so an ancestor's
    // palette overrides per-splat colours there, exactly as the Layers panel's
    // `applyColormap` fan-out already does. Points/lines/mesh gate on
    // `has_scalars` instead, so an inherited palette is a no-op for them
    // without a scalar channel (asserted below).
    const leaf = graph.children![0].children![0];
    expect(leaf.attrs.has_colors).toBe(true);
    expect(applyEffectiveAttrs(graph, leaf).colormap).toBe('custom');
  });

  it('still hands a scalar-less points leaf the palette, for its own gate to refuse', () => {
    const pts = graph.children![0].children![1];
    const eff = applyEffectiveAttrs(graph, pts);
    expect(eff.colormap).toBe('custom');
    // `createPointsMaterial` requires `has_scalars` before it builds a LUT, so
    // composition stays type-agnostic and the consumer owns the decision.
    expect(eff.has_scalars).toBe(false);
  });

  it('leaves an unset chain without a colormap', () => {
    const bare: SceneNode = {
      path: '/',
      type: 'scene',
      hasSpatialIndex: false,
      attrs: {},
      children: [{ path: '/gs', type: 'gsplats', hasSpatialIndex: true, attrs: {} }],
    };
    expect(applyEffectiveAttrs(bare, bare.children![0]).colormap).toBeUndefined();
  });
});

// `depth_level` — the authored cross-layer draw order
// (docs/guides/specs/LAYER_DEPTH_LEVEL_SPEC.md). Composes nearest-setter-wins
// like `blending_mode`, and its `undefined` is load-bearing: only an AUTHORED
// level suppresses the renderer's containment rule, so "nobody set one" must
// stay distinguishable from "someone set 0" (spec D2/D3).
describe('composeAttrs — depth_level', () => {
  it('is undefined for an unset chain', () => {
    expect(composeAttrs([]).depth_level).toBeUndefined();
    expect(composeAttrs([{ opacity: 0.5 }, { gamma: 2 }]).depth_level).toBeUndefined();
  });

  it('inherits an ancestor level when the leaf sets none', () => {
    expect(composeAttrs([{ depth_level: 20 }, { opacity: 0.5 }]).depth_level).toBe(20);
  });

  it('lets the leaf override an ancestor level (nearest-setter-wins)', () => {
    expect(composeAttrs([{ depth_level: 20 }, { depth_level: 30 }]).depth_level).toBe(30);
  });

  it('takes the LAST setter across a three-level chain', () => {
    expect(
      composeAttrs([{ depth_level: 10 }, { depth_level: 20 }, { opacity: 0.5 }]).depth_level
    ).toBe(20);
  });

  // An authored 0 is a real band, not an absence. If these two ever agree, the
  // renderer can no longer tell an explicit level from a default and every
  // legacy store silently loses its containment ordering.
  it('preserves an authored 0 as distinct from unset', () => {
    expect(composeAttrs([{ depth_level: 0 }]).depth_level).toBe(0);
    expect(composeAttrs([{}]).depth_level).toBeUndefined();
  });

  it('carries a negative level (bands are ordered, not counted)', () => {
    expect(composeAttrs([{ depth_level: -5 }]).depth_level).toBe(-5);
  });
});

// The raw-attrs → ComposableAttrs hop is an explicit ALLOWLIST
// (`toComposable`), so a field missing from it writes cleanly, reads cleanly and
// does nothing. These go through `getEffectiveAttrs`, which is the only path
// that exercises it.
describe('depth_level survives the raw-attrs allowlist', () => {
  const graph = (rootAttrs: Record<string, unknown>, leafAttrs: Record<string, unknown>) => ({
    path: '',
    type: 'group',
    attrs: rootAttrs,
    children: [{ path: '/gs', type: 'gsplats', hasSpatialIndex: true, attrs: leafAttrs }],
  });

  it('reaches EffectiveAttrs from a leaf', () => {
    const g = graph({}, { depth_level: 30 });
    expect(getEffectiveAttrs(g as never, '/gs').depth_level).toBe(30);
  });

  it('reaches EffectiveAttrs from an ancestor group', () => {
    const g = graph({ depth_level: 10 }, {});
    expect(getEffectiveAttrs(g as never, '/gs').depth_level).toBe(10);
  });

  // Tolerant read against a strict write: the Python writer refuses anything
  // but an int, but a hand-edited store can carry junk, and a non-finite level
  // would poison the band comparator for the whole frame. Treated as ABSENT so
  // the node falls back to the inferred ordering rather than an arbitrary band.
  it.each([
    ['a string', 'front'],
    ['null', null],
    ['NaN', Number.NaN],
    ['Infinity', Number.POSITIVE_INFINITY],
    ['-Infinity', Number.NEGATIVE_INFINITY],
  ])('treats %s as unset', (_label, value) => {
    const g = graph({}, { depth_level: value });
    expect(getEffectiveAttrs(g as never, '/gs').depth_level).toBeUndefined();
  });

  it('applyEffectiveAttrs puts the composed level on the consumer record', () => {
    const g = graph({ depth_level: 10 }, {});
    expect(applyEffectiveAttrs(g as never, g.children[0] as never).depth_level).toBe(10);
  });
});
