/**
 * The slab rule for sound nodes — a discrete story dim admits half a step, a
 * row outside the slab is silent, `extend_to_all` widens, and a descriptor
 * without positions is the caller's "always audible".
 */

import { describe, it, expect } from 'vitest';
import {
  buildSoundBaseViewState,
  buildWaypointViewState,
  computeRowAudibility,
  displayedXYZ,
} from '../../../audio/audibility';
import type { SimpleDims } from '../../../types/dims';

function storyDims(step: number): SimpleDims {
  return {
    ndim: 4,
    currentStep: [step, 0, 0, 0],
    displayed: [1, 2, 3],
    metadata: [
      { name: 'story', unit: '', scale: 1, discrete: true, step: 1, range: [0, 2], display: false },
      { name: 'x', unit: 'um', scale: 1, display: true },
      { name: 'y', unit: 'um', scale: 1, display: true },
      { name: 'z', unit: 'um', scale: 1, display: true },
    ],
  };
}

const rows = Float32Array.from([
  0,
  1,
  2,
  3, // story 0
  1,
  4,
  5,
  6, // story 1
  2,
  7,
  8,
  9, // story 2
]);
const desc = { path: '/sounds/x', positions: rows, nPositions: 3, ndim: 4 };

describe('buildSoundBaseViewState', () => {
  it('uses the dims snapshot and a half-step tolerance on a discrete hidden dim', () => {
    const vs = buildSoundBaseViewState(storyDims(1));
    expect(vs.displayDims).toEqual([1, 2, 3]);
    expect(vs.slicePosition).toEqual([1, 0, 0, 0]);
    expect(vs.tolerance[0]).toBeGreaterThan(0);
    expect(vs.tolerance[0]).toBeLessThan(1);
  });
});

describe('computeRowAudibility', () => {
  it('marks only the row whose story matches the slice', () => {
    const out = new Uint8Array(3);
    const n = computeRowAudibility(
      desc,
      undefined,
      buildSoundBaseViewState(storyDims(1)),
      null,
      out
    );
    expect(n).toBe(1);
    expect(Array.from(out)).toEqual([0, 1, 0]);
  });

  it('is silent everywhere when no row matches', () => {
    const out = new Uint8Array(3);
    const dims = storyDims(1);
    dims.currentStep = [7, 0, 0, 0];
    expect(computeRowAudibility(desc, undefined, buildSoundBaseViewState(dims), null, out)).toBe(0);
  });

  it('extend_to_all over the story dim makes every row live', () => {
    const out = new Uint8Array(3);
    const n = computeRowAudibility(
      desc,
      ['story'],
      buildSoundBaseViewState(storyDims(1)),
      null,
      out
    );
    expect(n).toBe(3);
  });

  it('a descriptor without positions reports zero rows (caller treats as always audible)', () => {
    const out = new Uint8Array(1);
    const none = { path: '/s', positions: null, nPositions: 0, ndim: 4 };
    expect(
      computeRowAudibility(none, undefined, buildSoundBaseViewState(storyDims(0)), null, out)
    ).toBe(0);
  });
});

describe('displayedXYZ', () => {
  it('picks the displayed columns of a row and pads missing axes with 0', () => {
    expect(displayedXYZ(rows, 1, 4, [1, 2, 3])).toEqual([4, 5, 6]);
    expect(displayedXYZ(rows, 2, 4, [3, 1])).toEqual([9, 7, 0]);
  });
});

describe('waypoint membership (the `when` clause as a slab)', () => {
  it('an exact value admits ±0.5 like the overlay rule; unnamed hidden dims admit anything', () => {
    const vs = buildWaypointViewState({ story: 1 }, storyDims(2));
    expect(vs.slicePosition[0]).toBe(1);
    expect(vs.tolerance[0]).toBe(0.5);
    const out = new Uint8Array(3);
    expect(
      computeRowAudibility(
        desc,
        undefined,
        buildWaypointViewState({ story: 1 }, storyDims(2)),
        null,
        out
      )
    ).toBe(1);
    expect(Array.from(out)).toEqual([0, 1, 0]);
  });

  it('a range takes its midpoint and half-width', () => {
    const out = new Uint8Array(3);
    expect(
      computeRowAudibility(
        desc,
        undefined,
        buildWaypointViewState({ story: [1, 2] }, storyDims(0)),
        null,
        out
      )
    ).toBe(2);
    expect(Array.from(out)).toEqual([0, 1, 1]);
  });

  it('a dimension the scene does not have is skipped, so every row belongs', () => {
    const out = new Uint8Array(3);
    expect(
      computeRowAudibility(
        desc,
        undefined,
        buildWaypointViewState({ nope: 7 }, storyDims(0)),
        null,
        out
      )
    ).toBe(3);
  });

  it('extend_to_all on the named dimension makes every row belong', () => {
    const out = new Uint8Array(3);
    expect(
      computeRowAudibility(
        desc,
        ['story'],
        buildWaypointViewState({ story: 1 }, storyDims(0)),
        null,
        out
      )
    ).toBe(3);
  });
});
