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
