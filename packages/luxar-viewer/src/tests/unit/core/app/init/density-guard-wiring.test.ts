/**
 * `wireDensityGuard` — the pipeline's density-guard closures, run for real
 * against an injected tracker + guard so every accessor, the per-frame
 * callback and the runtime on/off handle execute (the pipeline test mocks
 * every collaborator and never runs a frame).
 */
import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';

import {
  buildDensityProvider,
  summarizeThinning,
  wireDensityGuard,
  type DensityGuardWiringDeps,
} from '../../../../../core/app/init/density-guard-wiring';
import { densityGuardConfig } from '../../../../../config/sections/density-guard/data';
import { getDensityDrop } from '../../../../../rendering/materials/_shared/density-drop';
import { DensityGuard } from '../../../../../scene/density-guard';
import { ProjectedDensityTracker } from '../../../../../scene/projected-density';
import { setCommittedData } from '../../../../../types/committed-data';

interface StubMat {
  uniforms: { uDensityDrop: { value: number }; uOpacity: { value: number } };
  userData: { blendingMode: string };
  updateOpacity(v: number): void;
  getOpacity(): number;
}
function stubMaterial(): StubMat {
  return {
    uniforms: { uDensityDrop: { value: 0 }, uOpacity: { value: 1 } },
    userData: { blendingMode: 'additive' },
    updateOpacity(v) {
      this.uniforms.uOpacity.value = v;
    },
    getOpacity() {
      return this.uniforms.uOpacity.value;
    },
  };
}

/** A dense committed points node (1 M points in a small sphere) plus a camera framing it. */
function denseScene(): { scene: THREE.Scene; camera: THREE.PerspectiveCamera; mesh: THREE.Mesh } {
  const scene = new THREE.Scene();
  const geometry = new THREE.BufferGeometry();
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1);
  const mesh = new THREE.Mesh(geometry, stubMaterial() as unknown as THREE.Material);
  mesh.name = '/dense';
  mesh.userData.nodeType = 'points';
  mesh.userData.visiblePointCount = 1_000_000;
  mesh.userData._layerMaterialCloned = true;
  setCommittedData(mesh, {});
  scene.add(mesh);
  scene.updateMatrixWorld(true);
  const camera = new THREE.PerspectiveCamera(60, 1.6, 0.1, 1000);
  camera.position.set(0, 0, 100);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  camera.updateProjectionMatrix();
  return { scene, camera, mesh };
}

function makeDeps(overrides: Partial<DensityGuardWiringDeps> = {}): DensityGuardWiringDeps & {
  spies: {
    setProvider: ReturnType<typeof vi.fn>;
    notify: ReturnType<typeof vi.fn>;
    render: ReturnType<typeof vi.fn>;
    resume: ReturnType<typeof vi.fn>;
    register: ReturnType<typeof vi.fn>;
  };
} {
  const { scene, camera } = denseScene();
  const spies = {
    setProvider: vi.fn(),
    notify: vi.fn(),
    render: vi.fn(),
    resume: vi.fn().mockReturnValue(0),
    register: vi.fn(),
  };
  return {
    spies,
    configEnabled: true,
    option: undefined,
    config: densityGuardConfig,
    energyComp: false,
    sceneManager: { scene, camera, renderer: { domElement: { width: 1600, height: 1000 } } },
    registerMaterial: spies.register,
    setRefinementDensityProvider: spies.setProvider,
    getDefaultLoader: () => ({ resumeDensityDeferredRefinement: spies.resume }),
    getAdaptiveDpr: () => ({ notifyContentChanged: spies.notify }),
    requestRender: spies.render,
    tracker: new ProjectedDensityTracker(),
    guard: new DensityGuard(),
    ...overrides,
  };
}

const CAPS = {
  blendable: densityGuardConfig.capElementsPerPixel,
  nonBlendable: densityGuardConfig.nonBlendableCapElementsPerPixel,
};

describe('wireDensityGuard', () => {
  it('hands the loader manager a provider over the tracker records, with both caps', () => {
    const deps = makeDeps();
    const wiring = wireDensityGuard(deps);
    expect(wiring.isEnabled()).toBe(true);
    expect(wiring.sessionDisabled).toBe(false);
    expect(deps.spies.setProvider).toHaveBeenCalledWith(wiring.provider, CAPS);
    // Before any frame the tracker has no record → the gate has no opinion.
    expect(wiring.provider('/dense')).toBeUndefined();
    expect(wiring.thinning()).toEqual({ nodes: 0, minKeep: 1 });
  });

  it('per frame: evaluates, thins the over-dense node once, and signals the DPR controller once', () => {
    const deps = makeDeps();
    const wiring = wireDensityGuard(deps);
    const mesh = deps.sceneManager.scene!.children[0] as THREE.Mesh;

    wiring.perFrame();
    // 1 M points in a ~235 px footprint → floor of the ladder.
    expect(mesh.userData.densityKeep).toBe(densityGuardConfig.minKeepFraction);
    expect(deps.spies.notify).toHaveBeenCalledTimes(1);
    expect(deps.spies.render).toHaveBeenCalledTimes(1);
    expect(deps.spies.resume).toHaveBeenCalledTimes(1);
    // The provider now sees the record the same frame produced.
    const sample = wiring.provider('/dense');
    expect(sample).toMatchObject({ onScreen: true, elements: 1_000_000, blendable: true });
    expect(sample!.areaPx).toBeGreaterThan(0);
    expect(wiring.thinning()).toEqual({ nodes: 1, minKeep: densityGuardConfig.minKeepFraction });

    // Steady state: no further content-change signal, resume still polled.
    wiring.perFrame();
    expect(deps.spies.notify).toHaveBeenCalledTimes(1);
    expect(deps.spies.render).toHaveBeenCalledTimes(1);
    expect(deps.spies.resume).toHaveBeenCalledTimes(2);
  });

  it('is inert when disabled: null provider, no evaluation, no thinning', () => {
    const deps = makeDeps({ option: false });
    const wiring = wireDensityGuard(deps);
    expect(wiring.isEnabled()).toBe(false);
    expect(wiring.sessionDisabled).toBe(true);
    expect(deps.spies.setProvider).toHaveBeenCalledWith(null, CAPS);
    wiring.perFrame();
    const mesh = deps.sceneManager.scene!.children[0] as THREE.Mesh;
    expect(mesh.userData.densityKeep).toBeUndefined();
    expect(deps.spies.resume).not.toHaveBeenCalled();
    expect(wiring.thinning()).toEqual({ nodes: 0, minKeep: 1 });
  });

  it('a ?densityCap override reaches the ladder, the rung gate caps and the readout', () => {
    const deps = makeDeps({ capOverride: 64 });
    const wiring = wireDensityGuard(deps);
    expect(wiring.capElementsPerPixel()).toBe(64);
    expect(deps.spies.setProvider).toHaveBeenCalledWith(wiring.provider, {
      blendable: 64,
      nonBlendable: densityGuardConfig.nonBlendableCapElementsPerPixel,
    });
    // The ladder reads the override too: 1 M points in a ~235 px footprint sit
    // at the 1/64 floor under the default cap (4); under a cap of 500 the same
    // node needs only a few halvings, so its keep lands strictly above the floor.
    const wideDeps = makeDeps({ capOverride: 500 });
    const wide = wireDensityGuard(wideDeps);
    wide.perFrame();
    const mesh = wideDeps.sceneManager.scene!.children[0] as THREE.Mesh;
    const keep = mesh.userData.densityKeep as number;
    expect(keep).toBeGreaterThan(densityGuardConfig.minKeepFraction);
    expect(keep).toBeLessThan(1);
  });

  it('an override below the non-blendable cap pulls that cap down too (it must stay the tighter one)', () => {
    const deps = makeDeps({ capOverride: 0.5 });
    const wiring = wireDensityGuard(deps);
    expect(wiring.capElementsPerPixel()).toBe(0.5);
    expect(deps.spies.setProvider).toHaveBeenCalledWith(wiring.provider, {
      blendable: 0.5,
      nonBlendable: 0.5,
    });
  });

  it('ignores an invalid cap override and reports the configured cap', () => {
    for (const bad of [0, -3, Number.NaN, Number.POSITIVE_INFINITY, undefined]) {
      const wiring = wireDensityGuard(makeDeps({ capOverride: bad }));
      expect(wiring.capElementsPerPixel()).toBe(densityGuardConfig.capElementsPerPixel);
    }
  });

  it('config off is not a session disable (the stored setting may still turn it on)', () => {
    const wiring = wireDensityGuard(makeDeps({ configEnabled: false }));
    expect(wiring.isEnabled()).toBe(false);
    expect(wiring.sessionDisabled).toBe(false);
  });

  describe('setEnabled (runtime toggle)', () => {
    it('off: releases every thinned node, clears the provider and records, signals once', () => {
      const deps = makeDeps();
      const wiring = wireDensityGuard(deps);
      const mesh = deps.sceneManager.scene!.children[0] as THREE.Mesh;
      wiring.perFrame();
      expect(getDensityDrop(mesh.material)).toBeGreaterThan(0);
      deps.spies.setProvider.mockClear();
      deps.spies.notify.mockClear();
      deps.spies.render.mockClear();

      wiring.setEnabled(false);

      expect(wiring.isEnabled()).toBe(false);
      expect(mesh.userData.densityKeep).toBe(1);
      expect(getDensityDrop(mesh.material)).toBe(0);
      // Brightness compensation is gone with the thinning.
      expect((mesh.material as unknown as StubMat).getOpacity()).toBe(1);
      expect(deps.spies.setProvider).toHaveBeenCalledWith(null, CAPS);
      expect(wiring.provider('/dense')).toBeUndefined(); // records dropped
      expect(wiring.thinning()).toEqual({ nodes: 0, minKeep: 1 });
      expect(deps.spies.notify).toHaveBeenCalledTimes(1);
      expect(deps.spies.render).toHaveBeenCalledTimes(1);

      // Off means off: a frame no longer evaluates or thins.
      wiring.perFrame();
      expect(mesh.userData.densityKeep).toBe(1);
      expect(getDensityDrop(mesh.material)).toBe(0);
    });

    it('on again: republishes the provider and the next frame thins from a clean ladder', () => {
      const deps = makeDeps();
      const wiring = wireDensityGuard(deps);
      const mesh = deps.sceneManager.scene!.children[0] as THREE.Mesh;
      wiring.perFrame();
      wiring.setEnabled(false);
      deps.spies.setProvider.mockClear();

      wiring.setEnabled(true);
      expect(deps.spies.setProvider).toHaveBeenCalledWith(wiring.provider, CAPS);
      wiring.perFrame();
      expect(mesh.userData.densityKeep).toBe(densityGuardConfig.minKeepFraction);
      expect(getDensityDrop(mesh.material)).toBeCloseTo(1 - densityGuardConfig.minKeepFraction);
    });

    it('is idempotent: setting the current state does nothing', () => {
      const deps = makeDeps();
      const wiring = wireDensityGuard(deps);
      deps.spies.setProvider.mockClear();
      wiring.setEnabled(true);
      expect(deps.spies.setProvider).not.toHaveBeenCalled();
      expect(deps.spies.notify).not.toHaveBeenCalled();
      expect(deps.spies.render).not.toHaveBeenCalled();
    });

    it('tolerates a missing scene/loader/controller while toggling', () => {
      const deps = makeDeps({
        sceneManager: { scene: null, camera: null, renderer: null },
        getDefaultLoader: () => null,
        getAdaptiveDpr: () => undefined,
      });
      const wiring = wireDensityGuard(deps);
      expect(() => wiring.setEnabled(false)).not.toThrow();
      expect(() => wiring.setEnabled(true)).not.toThrow();
    });
  });

  it('tolerates a missing renderer/loader/controller (early boot, dataset switch)', () => {
    const deps = makeDeps({
      sceneManager: { scene: new THREE.Scene(), camera: null, renderer: null },
      getDefaultLoader: () => null,
      getAdaptiveDpr: () => undefined,
    });
    const wiring = wireDensityGuard(deps);
    expect(() => wiring.perFrame()).not.toThrow();
  });
});

describe('buildDensityProvider', () => {
  it('projects a tracker record onto the gate sample shape', () => {
    const tracker = new ProjectedDensityTracker();
    const { scene, camera } = denseScene();
    tracker.configure({
      enabled: () => true,
      getRoot: () => scene,
      getCamera: () => camera,
      getDrawingBufferSize: () => ({ width: 1600, height: 1000 }),
    });
    tracker.evaluate();
    const provider = buildDensityProvider(tracker);
    expect(provider('/missing')).toBeUndefined();
    expect(provider('/dense')).toEqual({
      areaPx: tracker.get('/dense')!.areaPx,
      elements: 1_000_000,
      onScreen: true,
      blendable: true,
    });
  });
});

describe('densityStates (monitor provider)', () => {
  it('projects every live record onto the monitor state shape', () => {
    const deps = makeDeps();
    const wiring = wireDensityGuard(deps);
    expect(wiring.densityStates().size).toBe(0);
    wiring.perFrame();
    const states = wiring.densityStates();
    expect(states.get('/dense')).toMatchObject({
      keep: densityGuardConfig.minKeepFraction,
      blendable: true,
      onScreen: true,
    });
    expect(states.get('/dense')!.elementsPerPixel).toBeGreaterThan(4);
    wiring.setEnabled(false);
    expect(wiring.densityStates().size).toBe(0);
  });
});

describe('summarizeThinning', () => {
  it('counts thinned records and reports the smallest keep', () => {
    const tracker = new ProjectedDensityTracker();
    const { scene, camera } = denseScene();
    // Two more nodes beside /dense: one thinned less, one untouched.
    for (const name of ['/half', '/whole']) {
      const geometry = new THREE.BufferGeometry();
      geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1);
      const mesh = new THREE.Mesh(geometry, stubMaterial() as unknown as THREE.Material);
      mesh.name = name;
      mesh.userData.nodeType = 'points';
      mesh.userData.visiblePointCount = 10;
      setCommittedData(mesh, {});
      scene.add(mesh);
    }
    scene.updateMatrixWorld(true);
    // Stand in for the guard: stamp each keep straight onto its record.
    const keeps: Record<string, number> = { '/dense': 1 / 64, '/half': 0.5, '/whole': 1 };
    tracker.configure({
      enabled: () => true,
      getRoot: () => scene,
      getCamera: () => camera,
      getDrawingBufferSize: () => ({ width: 1600, height: 1000 }),
      onVisit: (m, rec) => {
        rec.keep = keeps[m.name] ?? 1;
      },
    });
    tracker.evaluate();
    expect(summarizeThinning(tracker)).toEqual({ nodes: 2, minKeep: 1 / 64 });
  });
});
