/**
 * `wireDensityGuard` — the pipeline's density-guard closures, run for real
 * against an injected tracker + guard so every accessor and the per-frame
 * callback execute (the pipeline test mocks every collaborator and never runs
 * a frame).
 */
import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';

import {
  buildDensityProvider,
  wireDensityGuard,
  type DensityGuardWiringDeps,
} from '../../../../../core/app/init/density-guard-wiring';
import { densityGuardConfig } from '../../../../../config/sections/density-guard/data';
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

describe('wireDensityGuard', () => {
  it('hands the loader manager a provider over the tracker records, with both caps', () => {
    const deps = makeDeps();
    const wiring = wireDensityGuard(deps);
    expect(wiring.enabled).toBe(true);
    expect(deps.spies.setProvider).toHaveBeenCalledWith(wiring.provider, {
      blendable: densityGuardConfig.capElementsPerPixel,
      nonBlendable: densityGuardConfig.nonBlendableCapElementsPerPixel,
    });
    // Before any frame the tracker has no record → the gate has no opinion.
    expect(wiring.provider?.('/dense')).toBeUndefined();
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
    const sample = wiring.provider?.('/dense');
    expect(sample).toMatchObject({ onScreen: true, elements: 1_000_000, blendable: true });
    expect(sample!.areaPx).toBeGreaterThan(0);

    // Steady state: no further content-change signal, resume still polled.
    wiring.perFrame();
    expect(deps.spies.notify).toHaveBeenCalledTimes(1);
    expect(deps.spies.render).toHaveBeenCalledTimes(1);
    expect(deps.spies.resume).toHaveBeenCalledTimes(2);
  });

  it('is inert when disabled: null provider, no evaluation, no thinning', () => {
    const deps = makeDeps({ option: false });
    const wiring = wireDensityGuard(deps);
    expect(wiring.enabled).toBe(false);
    expect(wiring.provider).toBeNull();
    expect(deps.spies.setProvider).toHaveBeenCalledWith(null, expect.anything());
    wiring.perFrame();
    const mesh = deps.sceneManager.scene!.children[0] as THREE.Mesh;
    expect(mesh.userData.densityKeep).toBeUndefined();
    expect(deps.spies.resume).not.toHaveBeenCalled();
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
