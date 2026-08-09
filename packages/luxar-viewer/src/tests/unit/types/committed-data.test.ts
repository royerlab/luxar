/**
 * Unit tests for the `committedData` stamp accessors
 * (`types/committed-data.ts`).
 *
 * The stamp is a load-bearing contract: presence + identity is the
 * memoized-concat no-op key, and ABSENCE is the LOD demotion signal.
 * Pinned here: `hasCommittedData` tests `!== undefined`, and
 * `clearCommittedData` DELETES the property (never assigns `undefined`) —
 * the on-object representation is `mesh.userData.committedData`.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  hasCommittedData,
  getCommittedData,
  setCommittedData,
  clearCommittedData,
  invalidateCommittedDataStamp,
  getElementIdMap,
  setElementIdMap,
} from '../../../types/committed-data';

describe('committed-data accessors', () => {
  it('has/get/set/clear round-trip', () => {
    const mesh = new THREE.Mesh();
    const ref = { positions: new Float32Array(3) };

    // Fresh mesh: never committed.
    expect(hasCommittedData(mesh)).toBe(false);
    expect(getCommittedData(mesh)).toBeUndefined();

    setCommittedData(mesh, ref);
    expect(hasCommittedData(mesh)).toBe(true);
    // Identity, not equality — the no-op key is the exact reference.
    expect(getCommittedData(mesh)).toBe(ref);
    // On-object representation: the userData slot (the storage contract
    // the coordinator/commit tests assert directly).
    expect(mesh.userData.committedData).toBe(ref);

    clearCommittedData(mesh);
    expect(hasCommittedData(mesh)).toBe(false);
    expect(getCommittedData(mesh)).toBeUndefined();
  });

  it('clearCommittedData uses DELETE semantics (property removed, not set to undefined)', () => {
    const mesh = new THREE.Mesh();
    setCommittedData(mesh, 'data');
    clearCommittedData(mesh);
    // `delete`, not `= undefined`: absence is the demotion signal and the
    // key must not linger (e.g. `'committedData' in userData` sweeps).
    expect('committedData' in mesh.userData).toBe(false);
  });

  it('clearCommittedData is idempotent on a never-committed object', () => {
    const mesh = new THREE.Mesh();
    expect(() => clearCommittedData(mesh)).not.toThrow();
    expect(hasCommittedData(mesh)).toBe(false);
  });

  it('overwriting the stamp replaces the reference', () => {
    const mesh = new THREE.Mesh();
    const a = { id: 'a' };
    const b = { id: 'b' };
    setCommittedData(mesh, a);
    setCommittedData(mesh, b);
    expect(getCommittedData(mesh)).toBe(b);
    expect(getCommittedData(mesh)).not.toBe(a);
  });
});

describe('elementIdMap sibling stamp (issue #1423)', () => {
  it('get/set round-trip via the userData slot', () => {
    const mesh = new THREE.Mesh();
    expect(getElementIdMap(mesh)).toBeUndefined();

    const map = new Uint32Array([2048, 2049, 4096]);
    setElementIdMap(mesh, map);
    expect(getElementIdMap(mesh)).toBe(map);
    expect((mesh.userData as { elementIdMap?: unknown }).elementIdMap).toBe(map);
  });

  it('setElementIdMap(undefined) DELETES the key (no stale map survives a mapless commit)', () => {
    const mesh = new THREE.Mesh();
    setElementIdMap(mesh, new Uint32Array([1, 2, 3]));
    setElementIdMap(mesh, undefined);
    expect(getElementIdMap(mesh)).toBeUndefined();
    expect('elementIdMap' in mesh.userData).toBe(false);
  });

  it('reads as absent when the stored value is not a Uint32Array', () => {
    const mesh = new THREE.Mesh();
    (mesh.userData as { elementIdMap?: unknown }).elementIdMap = [1, 2, 3];
    expect(getElementIdMap(mesh)).toBeUndefined();
  });

  it('clearCommittedData drops the map too', () => {
    // The map describes the very buffers whose stamp is being dropped, so it
    // must not outlive them — a stale map is a silently WRONG label, strictly
    // worse than falling back to the slot.
    const mesh = new THREE.Mesh();
    setCommittedData(mesh, { id: 'a' });
    setElementIdMap(mesh, new Uint32Array([2048, 2049]));

    clearCommittedData(mesh);

    expect(getElementIdMap(mesh)).toBeUndefined();
    expect('elementIdMap' in mesh.userData).toBe(false);
    expect(hasCommittedData(mesh)).toBe(false);
  });

  it('invalidateCommittedDataStamp KEEPS the map (geometry stays resident)', () => {
    // The blending-mode switch clears the no-op stamp purely to defeat the
    // memoized-concat fast path; the buffers stay on the GPU and pickable
    // until the async reprocess commits, so the map still describes them.
    // Dropping it there would resolve every hover in that window through the
    // raw slot — the wrong label this map exists to prevent.
    const mesh = new THREE.Mesh();
    const map = new Uint32Array([2048, 2049]);
    setCommittedData(mesh, { id: 'a' });
    setElementIdMap(mesh, map);

    invalidateCommittedDataStamp(mesh);

    expect(hasCommittedData(mesh)).toBe(false);
    expect('committedData' in mesh.userData).toBe(false);
    expect(getElementIdMap(mesh)).toBe(map);
  });
});
