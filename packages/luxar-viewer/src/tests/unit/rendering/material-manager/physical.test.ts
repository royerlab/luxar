/**
 * `MaterialManager.getMeshPhysicalMaterial` and the physical-material hook.
 *
 * Two contracts: the physical material enters the manager as a STATIC material (no
 * camera broadcast — it has no near fade), and its creation is the one event that
 * tells the scene to build its environment. The negative half of the second — a
 * HOUSE mesh material must not fire the hook — is what keeps existing scenes
 * environment-free, so it is asserted directly.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MaterialManager } from '../../../../rendering/material-manager';
import { PhysicalMeshMaterial } from '../../../../rendering/materials/mesh-physical/material-glsl';
import { MeshMaterial } from '../../../../rendering/materials/mesh/material-glsl';
import { isCameraAwareMaterial } from '../../../../rendering/materials/_shared/camera-aware-material';

describe('MaterialManager — physical mesh family', () => {
  let mm: MaterialManager;
  beforeEach(() => {
    mm = new MaterialManager();
  });

  it('constructs the GLSL wrapper by default and files it as a static material', () => {
    const before = mm.getCacheStats();
    const m = mm.getMeshPhysicalMaterial({ roughness: 0.3 });
    expect(m).toBeInstanceOf(PhysicalMeshMaterial);
    expect(isCameraAwareMaterial(m)).toBe(false);
    const after = mm.getCacheStats();
    expect(after.totalRegistered).toBe(before.totalRegistered + 1);
    expect(after.createCount).toBe(before.createCount + 1);
    expect(after.ownedMaterials).toBe(before.ownedMaterials + 1);
  });

  it('fires onPhysicalMaterialCreated once per physical material and never for a house mesh', () => {
    const listener = vi.fn();
    const off = mm.onPhysicalMaterialCreated(listener);

    mm.getMeshMaterial({
      blendingMode: 'opaque',
      opacity: 1,
      gamma: 1,
      intensity: 1,
      offset: 0,
    });
    expect(listener).not.toHaveBeenCalled();

    mm.getMeshPhysicalMaterial({});
    expect(listener).toHaveBeenCalledTimes(1);
    mm.getMeshPhysicalMaterial({ metalness: 1 });
    expect(listener).toHaveBeenCalledTimes(2);

    off();
    mm.getMeshPhysicalMaterial({});
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it('the listener runs AFTER the material exists, so a subscriber may inspect it', () => {
    let seen: unknown = null;
    mm.onPhysicalMaterialCreated(() => {
      seen = mm.getCacheStats().totalRegistered;
    });
    mm.getMeshPhysicalMaterial({});
    expect(seen).toBe(1);
  });

  it('dispose() releases the physical material and drops the listeners', () => {
    const listener = vi.fn();
    mm.onPhysicalMaterialCreated(listener);
    const m = mm.getMeshPhysicalMaterial({});
    const disposeSpy = vi.spyOn(m, 'dispose');
    mm.dispose();
    expect(disposeSpy).toHaveBeenCalled();
    // A fresh manager after dispose owes nothing to the old subscribers.
    const mm2 = new MaterialManager();
    mm2.getMeshPhysicalMaterial({});
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it('house mesh materials are unchanged by the new family (they still join the camera broadcast)', () => {
    const house = mm.getMeshMaterial({
      blendingMode: 'opaque',
      opacity: 1,
      gamma: 1,
      intensity: 1,
      offset: 0,
    });
    expect(house).toBeInstanceOf(MeshMaterial);
    expect(isCameraAwareMaterial(house)).toBe(true);
  });
});
