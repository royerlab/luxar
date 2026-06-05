/**
 * Unit tests for `invalidateRenderObjectFor` and the soft-dispose
 * contract with `MaterialManager`.
 *
 * The helper exists to evict Three's cached `RenderObject` when the
 * GPU buffer pool rebuilds a geometry's `InstancedInterleavedBuffer`.
 * It does so by dispatching a `'dispose'` event on the mesh's
 * material. `MaterialManager` also listens for that event to
 * unregister torn-down materials — without the `SOFT_DISPOSE_FLAG`
 * coordination, every buffer-pool grow would silently un-cache the
 * still-alive material and stop propagating global camera updates.
 *
 * These tests pin the soft-dispose contract: the listener MUST skip
 * registry/cache cleanup when the flag is set, and the flag MUST be
 * cleared after the dispatch returns (success or exception).
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';

import { invalidateRenderObjectFor } from '../../../../data/scene-loader/commit/invalidate-render-object';
import { SOFT_DISPOSE_FLAG } from '../../../../rendering/material-manager';

describe('invalidateRenderObjectFor', () => {
  it('dispatches a "dispose" event on the mesh material', () => {
    const material = new THREE.MeshBasicMaterial();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);

    const listener = vi.fn();
    material.addEventListener('dispose', listener);

    invalidateRenderObjectFor(mesh);

    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener.mock.calls[0][0].type).toBe('dispose');
  });

  it('sets the SOFT_DISPOSE_FLAG on the material during dispatch and clears it afterward', () => {
    const material = new THREE.MeshBasicMaterial();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);

    let flagDuringDispatch: boolean | undefined;
    material.addEventListener('dispose', () => {
      flagDuringDispatch = (material as unknown as Record<symbol, boolean | undefined>)[
        SOFT_DISPOSE_FLAG
      ];
    });

    invalidateRenderObjectFor(mesh);

    expect(flagDuringDispatch).toBe(true);
    // After the dispatch returns, the flag must be cleared.
    expect(
      (material as unknown as Record<symbol, boolean | undefined>)[SOFT_DISPOSE_FLAG]
    ).toBeUndefined();
  });

  it('clears the SOFT_DISPOSE_FLAG even if a listener throws', () => {
    const material = new THREE.MeshBasicMaterial();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);

    material.addEventListener('dispose', () => {
      throw new Error('boom');
    });

    expect(() => invalidateRenderObjectFor(mesh)).toThrow('boom');
    // The flag must be cleared by the `finally` even after the throw.
    expect(
      (material as unknown as Record<symbol, boolean | undefined>)[SOFT_DISPOSE_FLAG]
    ).toBeUndefined();
  });

  // NOTE: null/undefined `mesh.material` is not tested — `THREE.Mesh.material`
  // is typed `Material | Material[]` (never null), and every call site
  // (commit-{points,lines,gsplats}-geometry) passes a NodeFactory-built mesh
  // that always has a real material. A null material is a type-impossible
  // state, so there is no behavioral contract to pin here.

  it('dispatches "dispose" on every entry when the mesh uses a material array', () => {
    const m1 = new THREE.MeshBasicMaterial();
    const m2 = new THREE.MeshBasicMaterial();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry());
    mesh.material = [m1, m2];

    const l1 = vi.fn();
    const l2 = vi.fn();
    m1.addEventListener('dispose', l1);
    m2.addEventListener('dispose', l2);

    invalidateRenderObjectFor(mesh);

    expect(l1).toHaveBeenCalledTimes(1);
    expect(l2).toHaveBeenCalledTimes(1);
  });
});

describe('SOFT_DISPOSE_FLAG <-> MaterialManager contract', () => {
  // We can't reach into the private subscribedMaterials directly
  // without spinning up the full manager + a real luxar material. The
  // simpler contract test: a listener registered for dispose that
  // mimics the manager's check should skip its cleanup when the flag
  // is present. This regression-tests the wiring used by the manager.
  it('listeners that check SOFT_DISPOSE_FLAG skip cleanup during soft-dispose', () => {
    const material = new THREE.MeshBasicMaterial();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);

    const cleanup = vi.fn();
    material.addEventListener('dispose', () => {
      const tagged = material as unknown as Record<symbol, boolean | undefined>;
      if (tagged[SOFT_DISPOSE_FLAG]) return;
      cleanup();
    });

    invalidateRenderObjectFor(mesh);

    // Listener fired, but its cleanup body was skipped because the
    // flag was set.
    expect(cleanup).not.toHaveBeenCalled();
  });

  it('listeners that check SOFT_DISPOSE_FLAG still run cleanup on a true dispose', () => {
    const material = new THREE.MeshBasicMaterial();

    const cleanup = vi.fn();
    material.addEventListener('dispose', () => {
      const tagged = material as unknown as Record<symbol, boolean | undefined>;
      if (tagged[SOFT_DISPOSE_FLAG]) return;
      cleanup();
    });

    // Real dispose path: no flag set, cleanup runs.
    material.dispose();

    expect(cleanup).toHaveBeenCalledTimes(1);
  });
});
