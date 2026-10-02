/**
 * `buildLodRegistryDeps` — the one LOD-registry deps builder the app pipeline
 * and the layer share. The routing that matters: every loader-derived dep reads
 * the OWNING loader, and the optional wake/playback deps are omitted (not
 * stubbed) when the render-loop owner has none, so the registry's own
 * fallbacks apply.
 */

import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

import { buildLodRegistryDeps } from '../../../../../core/app/init/lod-registry-deps';
import type { LODGroupRegistryOwner } from '../../../../../data/scene-loader';

function makeOwner(): LODGroupRegistryOwner {
  return {
    currentViewVersion: 7,
    gpuBufferPool: { getResidentBytes: () => 1234 } as never,
    archiveFault: null,
    hasNetworkFailureUnder: vi.fn(() => true),
    requestReprocess: vi.fn(),
    isUpdateInProgress: vi.fn(() => false),
    isLoadPassInProgress: vi.fn(() => true),
    committedViewState: undefined,
  } as unknown as LODGroupRegistryOwner;
}

describe('buildLodRegistryDeps', () => {
  it('reads live display dims and registers fade materials with its owner', () => {
    let displayed = [1, 2, 3];
    const registerMaterial = vi.fn();
    const deps = buildLodRegistryDeps(makeOwner(), {
      getCamera: () => new THREE.PerspectiveCamera(),
      getViewportSize: () => ({ width: 10, height: 20 }),
      getDisplayDims: () => displayed,
      registerMaterial,
      requestRender: vi.fn(),
    });
    expect(deps.getDisplayDims()).toEqual([1, 2, 3]);
    displayed = [0, 2];
    expect(deps.getDisplayDims()).toEqual([0, 2]);

    const material = new THREE.MeshBasicMaterial();
    deps.registerMaterial?.(material);
    expect(registerMaterial).toHaveBeenCalledExactlyOnceWith(material);
  });

  it('routes every loader-derived dep to the owning loader', () => {
    const owner = makeOwner();
    const deps = buildLodRegistryDeps(owner, {
      getCamera: () => new THREE.PerspectiveCamera(),
      getViewportSize: () => ({ width: 10, height: 20 }),
      getDisplayDims: () => [0, 1, 2],
      registerMaterial: vi.fn(),
      requestRender: vi.fn(),
    });

    expect(deps.getViewVersion?.()).toBe(7);
    expect(deps.getResidentBytes?.()).toBe(1234);
    expect(deps.hasArchiveFault?.()).toBe(false);
    expect(deps.hasNetworkFailureUnder?.('/p')).toBe(true);
    // The PASS-level predicate, not the lock-level one.
    expect(deps.isUpdateInProgress?.()).toBe(true);
    expect(owner.isUpdateInProgress).not.toHaveBeenCalled();
    deps.requestReprocess?.(['/part']);
    expect(owner.requestReprocess).toHaveBeenCalledWith(['/part']);
    expect(deps.getViewportSize()).toEqual({ width: 10, height: 20 });
  });

  it('applies the documented LOD flag defaults', () => {
    const deps = buildLodRegistryDeps(makeOwner(), {
      getCamera: () => new THREE.PerspectiveCamera(),
      getViewportSize: () => ({ width: 1, height: 1 }),
      getDisplayDims: () => [0, 1, 2],
      registerMaterial: vi.fn(),
      requestRender: vi.fn(),
    });
    expect(deps.getCrossFadeEnabled?.()).toBe(true);
    expect(deps.getEnergyCompEnabled?.()).toBe(true);
    expect(deps.getForceFinestLOD?.()).toBe(false);
    expect(deps.getLodBias?.()).toBeUndefined();
  });

  it('omits the optional deps an owner does not supply, and forwards the ones it does', () => {
    const bare = buildLodRegistryDeps(makeOwner(), {
      getCamera: () => new THREE.PerspectiveCamera(),
      getViewportSize: () => ({ width: 1, height: 1 }),
      getDisplayDims: () => [0, 1, 2],
      registerMaterial: vi.fn(),
      requestRender: vi.fn(),
    });
    expect('requestTick' in bare).toBe(false);
    expect('getPlaybackPeriodMs' in bare).toBe(false);
    expect('getViewContext' in bare).toBe(false);

    const requestTick = vi.fn();
    const full = buildLodRegistryDeps(makeOwner(), {
      getCamera: () => new THREE.PerspectiveCamera(),
      getViewportSize: () => ({ width: 1, height: 1 }),
      getDisplayDims: () => [0, 1, 2],
      registerMaterial: vi.fn(),
      requestRender: vi.fn(),
      requestTick,
      getPlaybackPeriodMs: () => 40,
      lodFade: false,
      lodBias: 2,
    });
    full.requestTick?.();
    expect(requestTick).toHaveBeenCalledOnce();
    expect(full.getPlaybackPeriodMs?.()).toBe(40);
    expect(full.getCrossFadeEnabled?.()).toBe(false);
    expect(full.getLodBias?.()).toBe(2);
  });
});
