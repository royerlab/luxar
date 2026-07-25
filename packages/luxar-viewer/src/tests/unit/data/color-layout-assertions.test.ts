/**
 * assertColorLayout — the layout-OMISSION guard (all three geometries).
 *
 * Every layout-FLIP hazard has a guard (accumulator configure throw,
 * progressive-concat mixed-layout throw, commit append parity conjunct),
 * but an RGBA array whose producer forgot to declare `colorComponents`
 * satisfies every `≥ count·3` minimum check (4N ≥ 3N) and silently
 * mis-strides every element after the first. These tests pin the strict
 * chokepoint checks that turn the omission into a loud throw
 * (volumetric double-check follow-up).
 */
import { describe, it, expect } from 'vitest';
import { assertColorLayout } from '../../../data/loaders/color-loader';
import { projectPointsTo3D } from '../../../data/points/projection';

describe('assertColorLayout (shared helper)', () => {
  it('accepts exact RGB and RGBA layouts, and absent colors', () => {
    expect(() => assertColorLayout(new Float32Array(12), 4, 3, 'ctx')).not.toThrow();
    expect(() => assertColorLayout(new Float32Array(16), 4, 4, 'ctx')).not.toThrow();
    expect(() => assertColorLayout(null, 4, 3, 'ctx')).not.toThrow();
    expect(() => assertColorLayout(undefined, 4, 4, 'ctx')).not.toThrow();
    expect(() => assertColorLayout(new Float32Array(0), 0, 3, 'ctx')).not.toThrow();
  });

  it('throws on the OMISSION pattern: RGBA-sized colors declared as RGB', () => {
    // 4 elements × 4 channels = 16 floats, but colorComponents omitted
    // (defaulting to 3) → 16 ≠ 12. The old minimum checks passed this.
    expect(() => assertColorLayout(new Float32Array(16), 4, 3, 'ctx')).toThrow(
      /colors length 16 does not match count 4 × colorComponents 3/
    );
  });

  it('throws on short colors too (subsumes the old minimum check)', () => {
    expect(() => assertColorLayout(new Float32Array(9), 4, 3, 'ctx')).toThrow(/does not match/);
  });
});

describe('projectPointsTo3D — strict color layout at the projection entry', () => {
  const view = { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] };

  it('rejects an RGBA-sized colors array with UNDECLARED colorComponents (the silent mis-stride)', () => {
    const n = 4;
    expect(() =>
      projectPointsTo3D(
        {} as never, // wasm — never reached: the layout throw fires in validation
        new Float32Array(n * 3),
        new Float32Array(n * 4), // RGBA-sized …
        new Float32Array(n),
        null,
        view as never,
        [{ start: 0, end: n }] as never,
        {} as never
        // … but colorComponents omitted → defaults to 3 → must THROW,
        // not mis-stride (the pre-strict minimum check accepted this).
      )
    ).toThrow(/colors length 16 does not match count 4 × colorComponents 3/);
  });
});
