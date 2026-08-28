/**
 * Tests for `repairSortedIndexForCount` and `writeSortedIndexOrderingLive`.
 *
 * These are the two writers that between them remove the unsorted frame an
 * nD re-slice used to draw. The invariant that matters for both is the one
 * `commit-gsplats-geometry.ts` names: what reaches the GPU must be a
 * PERMUTATION of `[0, count)`. Anything else draws one element twice and
 * another never — the exact corruption double-buffering exists to prevent.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  repairSortedIndexForCount,
  writeSortedIndexOrderingLive,
  getActiveSortedIndexAttribute,
} from '../../../rendering/element-storage';

/** A geometry with the real two-buffer ordering pair, per attachElementStorage. */
function makeGeometry(capacity: number): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute(
    'aSortedIndex',
    new THREE.InstancedBufferAttribute(new Uint32Array(capacity), 1)
  );
  geometry.setAttribute(
    'aSortedIndexB',
    new THREE.InstancedBufferAttribute(new Uint32Array(capacity), 1)
  );
  return geometry;
}

function setActive(geometry: THREE.InstancedBufferGeometry, values: number[]): void {
  const arr = getActiveSortedIndexAttribute(geometry)!.array as Uint32Array;
  arr.set(values, 0);
}

function readActive(geometry: THREE.InstancedBufferGeometry, count: number): number[] {
  const arr = getActiveSortedIndexAttribute(geometry)!.array as Uint32Array;
  return Array.from(arr.subarray(0, count));
}

/** The property every case must satisfy. */
function expectPermutationOf(values: number[], count: number): void {
  expect(values).toHaveLength(count);
  expect([...values].sort((a, b) => a - b)).toEqual(Array.from({ length: count }, (_, i) => i));
}

describe('repairSortedIndexForCount', () => {
  it('keeps the surviving prefix in order and appends the new tail on a GROW', () => {
    const g = makeGeometry(16);
    setActive(g, [3, 0, 2, 1]);
    repairSortedIndexForCount(g, 4, 7);
    // The four survivors keep their relative order; 4,5,6 are new.
    expect(readActive(g, 7)).toEqual([3, 0, 2, 1, 4, 5, 6]);
    expectPermutationOf(readActive(g, 7), 7);
  });

  it('drops the out-of-range entries and compacts on a SHRINK', () => {
    const g = makeGeometry(16);
    setActive(g, [5, 1, 4, 0, 3, 2]);
    repairSortedIndexForCount(g, 6, 3);
    // 5 and 4 and 3 are past the new count; 1, 0, 2 survive in order.
    expect(readActive(g, 3)).toEqual([1, 0, 2]);
    expectPermutationOf(readActive(g, 3), 3);
  });

  it('reproduces the permutation unchanged when the count did not move', () => {
    const g = makeGeometry(16);
    setActive(g, [3, 0, 2, 1]);
    repairSortedIndexForCount(g, 4, 4);
    expect(readActive(g, 4)).toEqual([3, 0, 2, 1]);
  });

  it('still yields a permutation from a MALFORMED input', () => {
    // A partially applied chunked ordering is a reachable state — which is
    // why cancelSortedIndexOrderingApply exists. Duplicates and out-of-range
    // entries must degrade to a valid permutation, never to a double-draw.
    const g = makeGeometry(16);
    setActive(g, [2, 2, 2, 99, 0, 99]);
    repairSortedIndexForCount(g, 6, 6);
    const out = readActive(g, 6);
    expectPermutationOf(out, 6);
    // The salvageable information is kept and kept first.
    expect(out.slice(0, 2)).toEqual([2, 0]);
  });

  it('yields a permutation across a randomised sweep of count changes', () => {
    for (let trial = 0; trial < 300; trial++) {
      const prevCount = 1 + Math.floor(Math.random() * 40);
      const count = 1 + Math.floor(Math.random() * 40);
      const g = makeGeometry(64);
      // A real (shuffled) permutation of [0, prevCount).
      const perm = Array.from({ length: prevCount }, (_, i) => i);
      for (let i = perm.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [perm[i], perm[j]] = [perm[j], perm[i]];
      }
      setActive(g, perm);
      repairSortedIndexForCount(g, prevCount, count);
      expectPermutationOf(readActive(g, count), count);
    }
  });

  it('registers a [0, count) update range so the result is uploaded', () => {
    const g = makeGeometry(16);
    setActive(g, [3, 0, 2, 1]);
    const attr = getActiveSortedIndexAttribute(g)!;
    attr.clearUpdateRanges();
    const versionBefore = attr.version;
    repairSortedIndexForCount(g, 4, 6);
    // `needsUpdate` is a setter-only property on BufferAttribute (reading it
    // yields undefined), so the observable signal is the version bump.
    expect(attr.version).toBeGreaterThan(versionBefore);
    expect(attr.updateRanges).toEqual([{ start: 0, count: 6 }]);
  });

  it('does NOT re-home the geometry on slot 0', () => {
    // Unlike a full identity write. A repair only fires when the tenant,
    // geometry and buffers are unchanged, so the slot/uniform pairing is
    // already established; normalising would discard the very permutation
    // being repaired.
    const g = makeGeometry(16);
    g.userData.sortedIndexSlot = 1;
    setActive(g, [3, 0, 2, 1]); // writes into aSortedIndexB
    repairSortedIndexForCount(g, 4, 5);
    expect(g.userData.sortedIndexSlot).toBe(1);
    expect(
      Array.from((g.getAttribute('aSortedIndexB').array as Uint32Array).subarray(0, 5))
    ).toEqual([3, 0, 2, 1, 4]);
    // The inactive buffer was left alone.
    expect(
      Array.from((g.getAttribute('aSortedIndex').array as Uint32Array).subarray(0, 5))
    ).toEqual([0, 0, 0, 0, 0]);
  });

  it('beats storage order as a depth prior — the reason it exists', () => {
    // Mirrors the live measurement: elements drift a little between slices,
    // the resident count changes, and the question is whether the previous
    // permutation or storage order is the better stand-in for one frame.
    // Storage order here is spatially coherent (a Hilbert-ish sweep), which
    // is what the real loader writes — not a random order.
    const N = 2000;
    const zOf = (i: number) => Math.sin(i * 0.37) * 50; // storage order != depth order
    const prev = Array.from({ length: N }, (_, i) => i).sort((a, b) => zOf(a) - zOf(b));

    const g = makeGeometry(N);
    setActive(g, prev);
    const count = N - 40; // the slice shed a few elements
    repairSortedIndexForCount(g, N, count);
    const repaired = readActive(g, count);
    const identity = Array.from({ length: count }, (_, i) => i);

    const score = (ordering: number[]) => {
      const slot = new Int32Array(count);
      ordering.forEach((element, position) => (slot[element] = position));
      let ok = 0;
      let used = 0;
      for (let a = 0; a < count; a += 7) {
        for (let b = a + 1; b < count; b += 101) {
          if (zOf(a) === zOf(b)) continue;
          used++;
          if (zOf(a) < zOf(b) === slot[a] < slot[b]) ok++;
        }
      }
      return ok / used;
    };

    expect(score(repaired)).toBeGreaterThan(0.99);
    expect(score(identity)).toBeLessThan(0.75);
    // Mutation guard: identity is what the code did before, and it is worse.
    expect(score(repaired)).toBeGreaterThan(score(identity));
  });
});

describe('writeSortedIndexOrderingLive', () => {
  it('publishes a complete ordering immediately, without a slot flip', () => {
    const g = makeGeometry(8);
    const written = writeSortedIndexOrderingLive(g, new Uint32Array([2, 0, 3, 1]), 4);
    expect(written).toBe(4);
    expect(readActive(g, 4)).toEqual([2, 0, 3, 1]);
    expect(g.userData.sortedIndexSlot ?? 0).toBe(0);
    const attr = getActiveSortedIndexAttribute(g)!;
    expect(attr.version).toBeGreaterThan(0);
    expect(attr.updateRanges).toEqual([{ start: 0, count: 4 }]);
  });

  it('rejects a truncated ordering rather than clamping it', () => {
    const g = makeGeometry(8);
    setActive(g, [7, 6, 5, 4]);
    expect(writeSortedIndexOrderingLive(g, new Uint32Array([1, 0]), 4)).toBe(0);
    expect(readActive(g, 4)).toEqual([7, 6, 5, 4]); // untouched
  });

  it('rejects a count larger than the buffers', () => {
    const g = makeGeometry(4);
    expect(writeSortedIndexOrderingLive(g, new Uint32Array(9), 9)).toBe(0);
  });

  it('rejects a non-positive or fractional count', () => {
    const g = makeGeometry(8);
    for (const bad of [0, -1, 2.5, NaN]) {
      expect(writeSortedIndexOrderingLive(g, new Uint32Array([0, 1, 2, 3]), bad)).toBe(0);
    }
  });
});
