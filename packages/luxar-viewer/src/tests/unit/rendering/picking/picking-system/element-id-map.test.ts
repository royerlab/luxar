/**
 * Unit tests for `resolveOnDiskElementId` (issue #1421).
 *
 * The helper translates the visible-buffer storage slot a pick vote reports
 * into the on-disk element index the per-element label CSR is keyed by. Every
 * path that cannot produce a confident answer must fall back to the slot
 * unchanged — picking runs on the hover path and must never throw.
 *
 * The map is read off the MESH-level `elementIdMap` stamp (issue #1423 moved
 * it there from the committed payload, which can be a SliceCache-owned
 * snapshot that must never be mutated).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { resolveOnDiskElementId } from '../../../../../rendering/picking/picking-system/element-id-map';
import {
  setCommittedData,
  setElementIdMap,
  clearCommittedData,
} from '../../../../../types/committed-data';

function makeNode(): THREE.Object3D {
  return new THREE.Object3D();
}

describe('resolveOnDiskElementId', () => {
  it('returns the slot unchanged when the node has no committed data', () => {
    expect(resolveOnDiskElementId(makeNode(), 7)).toBe(7);
  });

  it('returns the slot unchanged when a committed node published no map', () => {
    const node = makeNode();
    setCommittedData(node, { positions: new Float32Array(9), pointCount: 3 });
    expect(resolveOnDiskElementId(node, 2)).toBe(2);
  });

  it('maps the slot through a published elementIdMap stamp', () => {
    const node = makeNode();
    setElementIdMap(node, new Uint32Array([2048, 2049, 4096]));
    expect(resolveOnDiskElementId(node, 0)).toBe(2048);
    expect(resolveOnDiskElementId(node, 2)).toBe(4096);
  });

  it('ignores an elementIds field on the committed payload (the stamp is the only source)', () => {
    const node = makeNode();
    setCommittedData(node, { elementIds: new Uint32Array([2048, 2049, 4096]) });
    expect(resolveOnDiskElementId(node, 0)).toBe(0);
  });

  it('returns the slot unchanged when the slot is outside the map', () => {
    const node = makeNode();
    setElementIdMap(node, new Uint32Array([2048, 2049]));
    expect(resolveOnDiskElementId(node, 5)).toBe(5);
    expect(resolveOnDiskElementId(node, -1)).toBe(-1);
  });

  it('returns the slot unchanged (no throw) for a non-Uint32Array stamp', () => {
    const node = makeNode();
    // A plain array looks index-able but is not the published contract.
    (node.userData as { elementIdMap?: unknown }).elementIdMap = [2048, 2049];
    expect(() => resolveOnDiskElementId(node, 0)).not.toThrow();
    expect(resolveOnDiskElementId(node, 0)).toBe(0);
  });

  it('returns the slot unchanged when the committed data is not an object', () => {
    const node = makeNode();
    setCommittedData(node, 'not-a-payload');
    expect(resolveOnDiskElementId(node, 3)).toBe(3);
  });

  it('falls back to the slot once clearCommittedData drops the map', () => {
    // A blending-mode switch clears the pair; a map left behind would describe
    // geometry that is no longer vouched for — a silently WRONG label.
    const node = makeNode();
    setCommittedData(node, { pointCount: 3 });
    setElementIdMap(node, new Uint32Array([2048, 2049, 4096]));
    expect(resolveOnDiskElementId(node, 1)).toBe(2049);
    clearCommittedData(node);
    expect(resolveOnDiskElementId(node, 1)).toBe(1);
  });
});
