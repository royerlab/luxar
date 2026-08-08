/**
 * The indexed (mesh) depth-sort apply — `rendering/depth-sort-coordinator/triangle-ordering.ts`.
 *
 * Two things are being pinned here, and only one of them is "does it permute":
 *
 * 1. **The result is a PERMUTATION of the source, with winding intact.** A
 *    triangle's three indices must stay in their authored order inside the
 *    triple (swapping two of them reverses winding and turns a single-sided
 *    surface inside out), while the triples themselves move.
 * 2. **Each apply is independent of the last.** The apply reads the canonical
 *    source, never the live buffer. Reading the buffer would COMPOSE
 *    permutations — a bug that is invisible on the first sort and produces a
 *    scrambled surface on the second, which is exactly the shape that survives
 *    a naive test.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import * as THREE from 'three';
import {
  acknowledgeTriangleOrderingDraw,
  cancelAllTriangleOrderingApplies,
  cancelTriangleOrderingApply,
  computeFaceCentroids,
  writeSortedTriangleOrdering,
} from '../../../rendering/depth-sort-coordinator/triangle-ordering';

afterEach(() => {
  // Module-scoped acknowledgement map: a test that leaves an entry behind would
  // otherwise let the next one's `onAbandoned` fire from a stale geometry.
  cancelAllTriangleOrderingApplies();
});

/** Four triangles over eight vertices, with `capacity` slack past the visible set. */
function makeGeometry(
  triples: number[],
  { capacity = triples.length, uint16 = false } = {}
): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  const array = uint16 ? new Uint16Array(capacity) : new Uint32Array(capacity);
  array.set(triples);
  geometry.setIndex(new THREE.BufferAttribute(array, 1, false));
  geometry.setDrawRange(0, triples.length);
  return geometry;
}

/** The drawn prefix as an array of triples, for multiset comparisons. */
function drawnTriples(geometry: THREE.BufferGeometry, faceCount: number): string[] {
  const a = geometry.index!.array;
  const out: string[] = [];
  for (let f = 0; f < faceCount; f++) out.push(`${a[f * 3]},${a[f * 3 + 1]},${a[f * 3 + 2]}`);
  return out;
}

const SOURCE = new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6]);

describe('computeFaceCentroids', () => {
  it('averages the three referenced vertices, per component', () => {
    // Four vertices — v0 (0,0,0), v1 (3,0,0), v2 (0,6,0), v3 (0,0,9). Deliberately
    // asymmetric so a transposed or component-swapped read shows up.
    const position = new Float32Array([0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 9]);
    const centers = computeFaceCentroids(position, new Uint32Array([0, 1, 2, 1, 2, 3]), 2);
    expect(Array.from(centers)).toEqual([1, 2, 0, 1, 2, 3]);
  });

  it('reads only the first `faceCount` triples', () => {
    // The index array can be longer than the visible set; the tail is stale and
    // must not contribute a center the worker would then sort.
    const position = new Float32Array([0, 0, 0, 3, 3, 3, 9, 9, 9]);
    const centers = computeFaceCentroids(position, new Uint32Array([0, 0, 0, 2, 2, 2]), 1);
    expect(centers).toHaveLength(3);
    expect(Array.from(centers)).toEqual([0, 0, 0]);
  });

  it('allocates fresh each call (the buffer is transferred and detached)', () => {
    const position = new Float32Array([1, 1, 1]);
    const indices = new Uint32Array([0, 0, 0]);
    expect(computeFaceCentroids(position, indices, 1)).not.toBe(
      computeFaceCentroids(position, indices, 1)
    );
  });
});

describe('writeSortedTriangleOrdering — the permutation itself', () => {
  it('reorders whole triples and preserves winding within each', () => {
    const geometry = makeGeometry([...SOURCE]);
    // Reverse the face order.
    const written = writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([3, 2, 1, 0]), 4);
    expect(written).toBe(12);
    // Source faces are (0,1,2) (3,4,5) (6,7,0) (2,4,6); reversed, each triple
    // still reads left-to-right as authored.
    expect(Array.from(geometry.index!.array)).toEqual([2, 4, 6, 6, 7, 0, 3, 4, 5, 0, 1, 2]);
  });

  it('is a permutation of the source triples — same multiset, none lost or doubled', () => {
    const geometry = makeGeometry([...SOURCE]);
    const before = drawnTriples(geometry, 4);
    writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([2, 0, 3, 1]), 4);
    expect(drawnTriples(geometry, 4).sort()).toEqual([...before].sort());
  });

  it('does not COMPOSE with the previous ordering (each apply reads the source)', () => {
    // The bug this exists for: permuting the LIVE index buffer instead of the
    // canonical source. The first sort looks perfect either way; the second
    // silently applies σ∘π and scrambles the surface.
    const geometry = makeGeometry([...SOURCE]);
    const pi = new Uint32Array([1, 2, 3, 0]);
    writeSortedTriangleOrdering(geometry, SOURCE, pi, 4);
    writeSortedTriangleOrdering(geometry, SOURCE, pi, 4);
    // Applying the same ordering twice must be idempotent, not π².
    expect(drawnTriples(geometry, 4)).toEqual(['3,4,5', '6,7,0', '2,4,6', '0,1,2']);
  });

  it('leaves the capacity tail past the drawn prefix untouched', () => {
    // The tail holds stale indices by design (`applyMeshIndices` bounds the draw
    // with `drawRange`); writing into it would be pure wasted bandwidth.
    const geometry = makeGeometry([...SOURCE], { capacity: 18 });
    geometry.index!.array.set([99, 99, 99, 99, 99, 99], 12);
    writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([3, 2, 1, 0]), 4);
    expect(Array.from(geometry.index!.array.slice(12))).toEqual([99, 99, 99, 99, 99, 99]);
  });

  it('registers exactly one update range over the drawn prefix', () => {
    const geometry = makeGeometry([...SOURCE], { capacity: 18 });
    const attr = geometry.index!;
    // A pending range from the commit that produced this ordering — the same
    // span, so folding it must not widen or lose anything.
    attr.addUpdateRange(0, 12);
    const versionBefore = attr.version;
    writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([1, 0, 3, 2]), 4);
    expect(attr.updateRanges).toEqual([{ start: 0, count: 12 }]);
    // `needsUpdate` is a WRITE-ONLY setter in three (it bumps `version`); reading
    // it back gives `undefined`, so the version counter is the observable.
    expect(attr.version).toBe(versionBefore + 1);
  });

  it('does not flag an upload when the ordering is rejected', () => {
    // The counterpart of the above: a rejected write must not cost a re-upload of
    // the whole prefix on the next frame.
    const geometry = makeGeometry([...SOURCE]);
    geometry.setDrawRange(0, 6);
    const attr = geometry.index!;
    const versionBefore = attr.version;
    writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([3, 2, 1, 0]), 4);
    expect(attr.version).toBe(versionBefore);
  });

  it('narrows correctly into a Uint16 index buffer', () => {
    // `createMeshIndexAttribute` picks Uint16 under 65536 vertices, so the write
    // has to be dtype-agnostic — the source is always Uint32.
    const geometry = makeGeometry([...SOURCE], { uint16: true });
    writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([3, 2, 1, 0]), 4);
    expect(geometry.index!.array).toBeInstanceOf(Uint16Array);
    expect(drawnTriples(geometry, 4)).toEqual(['2,4,6', '6,7,0', '3,4,5', '0,1,2']);
  });
});

describe('writeSortedTriangleOrdering — rejections leave the buffer untouched', () => {
  const cases: [string, () => { geometry: THREE.BufferGeometry; call: () => number }][] = [
    [
      'no index attribute (a placeholder that never committed)',
      () => {
        const geometry = new THREE.BufferGeometry();
        return {
          geometry,
          call: () =>
            writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([0, 1, 2, 3]), 4),
        };
      },
    ],
    [
      'a zero face count',
      () => {
        const geometry = makeGeometry([...SOURCE]);
        return {
          geometry,
          call: () => writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array(0), 0),
        };
      },
    ],
    [
      'an ordering shorter than the face count',
      () => {
        const geometry = makeGeometry([...SOURCE]);
        return {
          geometry,
          call: () => writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([0, 1]), 4),
        };
      },
    ],
    [
      'a source shorter than the face count',
      () => {
        const geometry = makeGeometry([...SOURCE]);
        return {
          geometry,
          call: () =>
            writeSortedTriangleOrdering(
              geometry,
              SOURCE.subarray(0, 6),
              new Uint32Array([0, 1, 2, 3]),
              4
            ),
        };
      },
    ],
    [
      'an index buffer too small for the face count',
      () => {
        const geometry = makeGeometry([0, 1, 2], { capacity: 3 });
        geometry.setDrawRange(0, 12);
        return {
          geometry,
          call: () =>
            writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([0, 1, 2, 3]), 4),
        };
      },
    ],
    [
      'a drawRange describing a different face set (a slice move raced the sort)',
      () => {
        const geometry = makeGeometry([...SOURCE]);
        geometry.setDrawRange(0, 6);
        return {
          geometry,
          call: () =>
            writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([3, 2, 1, 0]), 4),
        };
      },
    ],
  ];

  for (const [name, build] of cases) {
    it(`rejects ${name}`, () => {
      const { geometry, call } = build();
      const before = geometry.index ? Array.from(geometry.index.array) : null;
      expect(call()).toBe(0);
      expect(geometry.index ? Array.from(geometry.index.array) : null).toEqual(before);
    });
  }

  it('does not take ownership of the caller-s session on a rejection', () => {
    // The contract `writeSortedIndexOrdering` sets: neither hook fires unless the
    // write is ACCEPTED, so the caller keeps its profiler session and closes it.
    const geometry = makeGeometry([...SOURCE]);
    geometry.setDrawRange(0, 6);
    const onApplied = vi.fn();
    const onAbandoned = vi.fn();
    writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([3, 2, 1, 0]), 4, {
      onApplied,
      onAbandoned,
    });
    acknowledgeTriangleOrderingDraw(geometry);
    expect(onApplied).not.toHaveBeenCalled();
    expect(onAbandoned).not.toHaveBeenCalled();
  });
});

describe('writeSortedTriangleOrdering — the apply lifecycle (issue #713)', () => {
  it('reports applied only after a draw, and only once', () => {
    const geometry = makeGeometry([...SOURCE]);
    const onApplied = vi.fn();
    const onAbandoned = vi.fn();
    writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([3, 2, 1, 0]), 4, {
      onApplied,
      onAbandoned,
    });
    // Written but not drawn: the update range has not been consumed yet.
    expect(onApplied).not.toHaveBeenCalled();

    acknowledgeTriangleOrderingDraw(geometry);
    expect(onApplied).toHaveBeenCalledTimes(1);

    // A second render of the same geometry must not re-close the session.
    acknowledgeTriangleOrderingDraw(geometry);
    expect(onApplied).toHaveBeenCalledTimes(1);
    expect(onAbandoned).not.toHaveBeenCalled();
  });

  it('abandons an ordering superseded before it was ever drawn', () => {
    const geometry = makeGeometry([...SOURCE]);
    const first = { onApplied: vi.fn(), onAbandoned: vi.fn() };
    const second = { onApplied: vi.fn(), onAbandoned: vi.fn() };
    writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([3, 2, 1, 0]), 4, first);
    writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([0, 1, 2, 3]), 4, second);
    // Exactly one hook per accepted ordering: the first never reached a frame.
    expect(first.onAbandoned).toHaveBeenCalledTimes(1);
    expect(first.onApplied).not.toHaveBeenCalled();

    acknowledgeTriangleOrderingDraw(geometry);
    expect(second.onApplied).toHaveBeenCalledTimes(1);
    expect(second.onAbandoned).not.toHaveBeenCalled();
  });

  it('abandons on cancel, and the cancel is idempotent', () => {
    const geometry = makeGeometry([...SOURCE]);
    const hooks = { onApplied: vi.fn(), onAbandoned: vi.fn() };
    writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([3, 2, 1, 0]), 4, hooks);
    cancelTriangleOrderingApply(geometry);
    cancelTriangleOrderingApply(geometry);
    expect(hooks.onAbandoned).toHaveBeenCalledTimes(1);
    // A draw after the cancel must not resurrect the session.
    acknowledgeTriangleOrderingDraw(geometry);
    expect(hooks.onApplied).not.toHaveBeenCalled();
  });

  it('sweeps every pending acknowledgement on teardown', () => {
    const a = makeGeometry([...SOURCE]);
    const b = makeGeometry([...SOURCE]);
    const hooksA = { onApplied: vi.fn(), onAbandoned: vi.fn() };
    const hooksB = { onApplied: vi.fn(), onAbandoned: vi.fn() };
    writeSortedTriangleOrdering(a, SOURCE, new Uint32Array([3, 2, 1, 0]), 4, hooksA);
    writeSortedTriangleOrdering(b, SOURCE, new Uint32Array([1, 0, 3, 2]), 4, hooksB);
    cancelAllTriangleOrderingApplies();
    expect(hooksA.onAbandoned).toHaveBeenCalledTimes(1);
    expect(hooksB.onAbandoned).toHaveBeenCalledTimes(1);
  });

  it('accepts an ordering with no callbacks at all', () => {
    // The profiler is optional; a session-less apply must still write.
    const geometry = makeGeometry([...SOURCE]);
    expect(writeSortedTriangleOrdering(geometry, SOURCE, new Uint32Array([3, 2, 1, 0]), 4)).toBe(
      12
    );
    expect(() => acknowledgeTriangleOrderingDraw(geometry)).not.toThrow();
  });
});
