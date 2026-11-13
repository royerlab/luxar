/**
 * Device orderings (#3162): a GPU-side producer writes the INACTIVE ordering
 * buffer itself, and the same staged-apply state machine flips, raises held
 * draws, acknowledges and abandons it exactly as it does a host array.
 *
 * The host-mirror invariant is pinned here too: once a device write has filled
 * a buffer, its host array is stale, so the next host write must upload the
 * whole prefix instead of splicing a slice into the device-written order.
 */
import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  acknowledgeSortedIndexOrderingDraw,
  activeSortedIndexSlot,
  attachElementStorage,
  cancelAllSortedIndexOrderingApplies,
  getActiveSortedIndexAttribute,
  holdSortedIndexDrawForAppend,
  isSortedIndexHostArrayCurrent,
  repairSortedIndexHostArray,
  pumpSortedIndexOrderingApply,
  releaseSortedIndexDrawHold,
  writeDeviceSortedIndexOrdering,
  writeSortedIndexIdentity,
  writeSortedIndexOrdering,
  type DeviceOrderingWrite,
} from '../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../rendering/element-texture-layout';
import { configureRendererBackendPolicy } from '../../../rendering/renderer-backend-policy';

const CAPACITY = 64;

function makeGeometry(count = CAPACITY): THREE.InstancedBufferGeometry {
  configureRendererBackendPolicy('webgpu-native');
  const geometry = new THREE.InstancedBufferGeometry();
  attachElementStorage(geometry, CAPACITY, POINT_TEXTURE_LAYOUT);
  writeSortedIndexIdentity(geometry, count);
  geometry.instanceCount = count;
  return geometry;
}

/** A device writer that records its target and stands in for the GPU result. */
function deviceWrite(
  count: number,
  ok = true
): DeviceOrderingWrite & { targets: THREE.InstancedBufferAttribute[] } {
  const targets: THREE.InstancedBufferAttribute[] = [];
  return {
    count,
    targets,
    write: (target) => {
      targets.push(target);
      return ok;
    },
  };
}

function lifecycle() {
  return { onApplied: vi.fn(), onAbandoned: vi.fn() };
}

const inactiveName = (g: THREE.InstancedBufferGeometry) =>
  activeSortedIndexSlot(g) === 0 ? 'aSortedIndexB' : 'aSortedIndex';

afterEach(() => {
  cancelAllSortedIndexOrderingApplies();
  configureRendererBackendPolicy('webgl');
});

describe('device orderings', () => {
  it('writes the inactive buffer, flips, and applies after the draw', () => {
    const geometry = makeGeometry();
    const inactive = geometry.getAttribute(
      inactiveName(geometry)
    ) as THREE.InstancedBufferAttribute;
    const versionBefore = inactive.version;
    const op = deviceWrite(CAPACITY);
    const hooks = lifecycle();

    expect(writeDeviceSortedIndexOrdering(geometry, op, hooks)).toBe(CAPACITY);
    expect(op.targets).toEqual([]); // staging alone writes nothing

    const result = pumpSortedIndexOrderingApply(geometry);
    expect(result).toEqual({ more: false, flipped: true, stalled: false });
    expect(op.targets).toEqual([inactive]);
    expect(getActiveSortedIndexAttribute(geometry)).toBe(inactive);
    // The device owns the buffer contents: no host upload may be scheduled.
    expect(inactive.version).toBe(versionBefore);
    expect(hooks.onApplied).not.toHaveBeenCalled();

    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(hooks.onApplied).toHaveBeenCalledTimes(1);
    expect(hooks.onAbandoned).not.toHaveBeenCalled();
  });

  it('abandons an ordering whose device write fails, without flipping', () => {
    const geometry = makeGeometry();
    const hooks = lifecycle();
    writeDeviceSortedIndexOrdering(geometry, deviceWrite(CAPACITY, false), hooks);

    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({
      more: false,
      flipped: false,
      stalled: false,
    });
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(hooks.onAbandoned).toHaveBeenCalledTimes(1);
    expect(hooks.onApplied).not.toHaveBeenCalled();
  });

  it('treats a throwing device writer as a failed write, not a stranded apply', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const geometry = makeGeometry();
    const hooks = lifecycle();
    const throwing: DeviceOrderingWrite = {
      count: CAPACITY,
      write: () => {
        throw new Error('encoder exploded');
      },
    };
    writeDeviceSortedIndexOrdering(geometry, throwing, hooks);
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({
      more: false,
      flipped: false,
      stalled: false,
    });
    expect(hooks.onAbandoned).toHaveBeenCalledTimes(1);
    expect(pumpSortedIndexOrderingApply(geometry).more).toBe(false);
    vi.restoreAllMocks();
  });

  it('starts the held ordering when a failed device write is dropped', () => {
    const geometry = makeGeometry();
    const failed = lifecycle();
    writeDeviceSortedIndexOrdering(geometry, deviceWrite(CAPACITY, false), failed);
    const next = deviceWrite(CAPACITY);
    writeDeviceSortedIndexOrdering(geometry, next, lifecycle());

    expect(pumpSortedIndexOrderingApply(geometry).more).toBe(true);
    expect(failed.onAbandoned).toHaveBeenCalledTimes(1);
    expect(pumpSortedIndexOrderingApply(geometry).flipped).toBe(true);
    expect(next.targets).toHaveLength(1);
  });

  it('holds behind a host stream and supersedes a held host ordering', () => {
    const geometry = makeGeometry();
    const host = new Uint32Array(CAPACITY).map((_, i) => CAPACITY - 1 - i);
    writeSortedIndexOrdering(geometry, host, CAPACITY);
    const heldHost = lifecycle();
    writeSortedIndexOrdering(geometry, host.slice(), CAPACITY, heldHost);
    const op = deviceWrite(CAPACITY);
    writeDeviceSortedIndexOrdering(geometry, op, lifecycle());

    // The newer device ordering replaced the held host one.
    expect(heldHost.onAbandoned).toHaveBeenCalledTimes(1);
    expect(pumpSortedIndexOrderingApply(geometry)).toMatchObject({ more: true, flipped: true });
    expect(op.targets).toEqual([]);
    geometry.getAttribute('aSortedIndex').clearUpdateRanges(); // Initial identity was drawn.
    getActiveSortedIndexAttribute(geometry)!.clearUpdateRanges(); // Host ordering was drawn.
    expect(pumpSortedIndexOrderingApply(geometry)).toMatchObject({ more: false, flipped: true });
    expect(op.targets).toHaveLength(1);
  });

  it('raises a held append draw on the flip', () => {
    const geometry = makeGeometry(32);
    expect(holdSortedIndexDrawForAppend(geometry, 32, 48)).toBe(32);
    geometry.instanceCount = 32;
    writeDeviceSortedIndexOrdering(geometry, deviceWrite(48), lifecycle());
    pumpSortedIndexOrderingApply(geometry);
    expect(geometry.instanceCount).toBe(48);
  });

  it('waits for an earlier host upload before writing that GPU slot', () => {
    const geometry = makeGeometry(32);
    const slot0 = getActiveSortedIndexAttribute(geometry)!;
    slot0.clearUpdateRanges(); // The initial identity was drawn.
    writeSortedIndexIdentity(geometry, 48); // Commit while culled: tail upload remains pending.
    expect(slot0.updateRanges).toEqual([{ start: 32, count: 16 }]);

    writeDeviceSortedIndexOrdering(geometry, deviceWrite(48), lifecycle());
    expect(pumpSortedIndexOrderingApply(geometry).flipped).toBe(true); // B
    const next = deviceWrite(48);
    const requestFrame = vi.fn();
    writeDeviceSortedIndexOrdering(geometry, next, { requestFrame });
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({
      more: true,
      flipped: false,
      stalled: true,
    });
    expect(next.targets).toEqual([]);
    expect(slot0.updateRanges).toEqual([{ start: 32, count: 16 }]);

    slot0.clearUpdateRanges(); // The next drawn frame updates this existing buffer.
    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(slot0.updateRanges).toEqual([]);
    expect(requestFrame).toHaveBeenCalledTimes(1);
    expect(pumpSortedIndexOrderingApply(geometry).flipped).toBe(true);
    expect(next.targets).toEqual([slot0]);
  });

  it('resumes when the first WebGPU upload leaves its ranges set', () => {
    const geometry = makeGeometry(32);
    const slot0 = getActiveSortedIndexAttribute(geometry)!;
    writeSortedIndexIdentity(geometry, 48);
    writeDeviceSortedIndexOrdering(geometry, deviceWrite(48));
    expect(pumpSortedIndexOrderingApply(geometry).flipped).toBe(true);
    const next = deviceWrite(48);
    const requestFrame = vi.fn();
    writeDeviceSortedIndexOrdering(geometry, next, { requestFrame });
    expect(pumpSortedIndexOrderingApply(geometry).stalled).toBe(true);
    expect(slot0.updateRanges).toEqual([{ start: 0, count: 48 }]);

    // Three creates the buffer from the full array without clearing ranges.
    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(slot0.updateRanges).toEqual([]);
    expect(requestFrame).toHaveBeenCalledTimes(1);
    expect(pumpSortedIndexOrderingApply(geometry).flipped).toBe(true);
    expect(next.targets).toEqual([slot0]);
  });

  it('rejects a count it cannot stage, firing no hooks', () => {
    const geometry = makeGeometry();
    for (const count of [0, -1, 2.5, CAPACITY + 1]) {
      const hooks = lifecycle();
      expect(writeDeviceSortedIndexOrdering(geometry, deviceWrite(count), hooks)).toBe(0);
      expect(hooks.onAbandoned).not.toHaveBeenCalled();
    }
    expect(pumpSortedIndexOrderingApply(geometry).flipped).toBe(false);
  });
});

describe('host writes after a device write', () => {
  /**
   * A geometry whose slot-0 buffer was identity-memoised by a host write and
   * then overwritten by a device write (two device orderings: B, then A).
   */
  function deviceWrittenSlot0(count: number): THREE.InstancedBufferGeometry {
    const geometry = makeGeometry(count);
    getActiveSortedIndexAttribute(geometry)!.clearUpdateRanges(); // Initial identity was drawn.
    for (let k = 0; k < 2; k++) {
      writeDeviceSortedIndexOrdering(geometry, deviceWrite(count), lifecycle());
      pumpSortedIndexOrderingApply(geometry);
    }
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    return geometry;
  }

  it('repair the grown population in full instead of splicing a tail on', () => {
    const geometry = deviceWrittenSlot0(32);
    // With an identity memo over [0, 32) the release would upload only the
    // new tail [32, 48), leaving the device-written order underneath it.
    holdSortedIndexDrawForAppend(geometry, 32, 48);
    expect(releaseSortedIndexDrawHold(geometry)).toBe(true);
    expect(getActiveSortedIndexAttribute(geometry)!.updateRanges).toEqual([
      { start: 0, count: 48 },
    ]);
  });

  it('write identity in full', () => {
    const geometry = deviceWrittenSlot0(CAPACITY);
    writeSortedIndexIdentity(geometry, CAPACITY);
    expect(getActiveSortedIndexAttribute(geometry)!.updateRanges).toEqual([
      { start: 0, count: CAPACITY },
    ]);
  });

  it('extend an identity prefix again once a host write has restored it', () => {
    const geometry = deviceWrittenSlot0(CAPACITY);
    writeSortedIndexIdentity(geometry, 32);
    const slot0 = getActiveSortedIndexAttribute(geometry)!;
    slot0.clearUpdateRanges();
    writeSortedIndexIdentity(geometry, CAPACITY);
    expect(slot0.updateRanges).toEqual([{ start: 32, count: CAPACITY - 32 }]);
  });
});

describe('a host array a device write left behind', () => {
  const isPermutation = (a: ArrayLike<number>, n: number) => {
    const seen = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (a[i] >= n || seen[a[i]]) return false;
      seen[a[i]] = 1;
    }
    return true;
  };

  /**
   * Slot B's host array holds a 32-element order; the node then grows to 48
   * and two device sorts flip A then B, so the drawn slot B's host array is a
   * permutation of the OLD population: not an ordering of what it draws.
   */
  function stale(): THREE.InstancedBufferGeometry {
    const geometry = makeGeometry(32);
    const reversed = new Uint32Array(32).map((_, i) => 31 - i);
    writeSortedIndexOrdering(geometry, reversed, 32);
    pumpSortedIndexOrderingApply(geometry);
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    // Both attribute uploads completed in the draw after the host sort.
    geometry.getAttribute('aSortedIndex').clearUpdateRanges();
    getActiveSortedIndexAttribute(geometry)!.clearUpdateRanges();
    holdSortedIndexDrawForAppend(geometry, 32, 48);
    for (let k = 0; k < 2; k++) {
      writeDeviceSortedIndexOrdering(geometry, deviceWrite(48), lifecycle());
      pumpSortedIndexOrderingApply(geometry);
    }
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    expect(geometry.instanceCount).toBe(48);
    return geometry;
  }

  it('is reported, so it is never used to seed a grown geometry', () => {
    const geometry = stale();
    expect(isSortedIndexHostArrayCurrent(geometry)).toBe(false);
    const arr = getActiveSortedIndexAttribute(geometry)!.array as Uint32Array;
    expect(isPermutation(arr, 48)).toBe(false);
  });

  it('is repaired into a permutation of the drawn population before a restore', () => {
    const geometry = stale();
    repairSortedIndexHostArray(geometry);
    const arr = getActiveSortedIndexAttribute(geometry)!.array as Uint32Array;
    expect(isPermutation(arr, 48)).toBe(true);
    expect(isSortedIndexHostArrayCurrent(geometry)).toBe(true);
  });

  it('is current again after a host write from index 0', () => {
    const geometry = stale();
    writeSortedIndexIdentity(geometry, 48);
    expect(isSortedIndexHostArrayCurrent(geometry)).toBe(true);
  });
});
