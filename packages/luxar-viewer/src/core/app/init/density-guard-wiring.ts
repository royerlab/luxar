/**
 * Wiring for the projected-density guard: tracker deps, keep-fraction guard,
 * the refinement rung-gate provider, and the per-frame callback.
 *
 * Kept out of `pipeline.ts` so the closures can be exercised by a unit test:
 * the pipeline test mocks every collaborator and never runs a frame, so
 * everything registered here would otherwise be dead to coverage — and a typo
 * in, say, the drawing-buffer size accessor would ship silently.
 *
 * @module core/app/init/density-guard-wiring
 */

import type * as THREE from 'three';

import type { DensityGuardConfig } from '../../../config/sections/density-guard/types';
import type {
  DensityGateCaps,
  ProjectedDensityProvider,
} from '../../../data/scene-loader/progressive/density-gate';
import { DensityGuard, getDensityGuard } from '../../../scene/density-guard';
import {
  ProjectedDensityTracker,
  getProjectedDensityTracker,
  resolveDensityGuardEnabled,
} from '../../../scene/projected-density';

/** What the wiring reads live each frame. Every accessor is called, never captured. */
export interface DensityGuardWiringDeps {
  /** `config.densityGuard.enabled`. */
  configEnabled: boolean;
  /** `LuxarAppOptions.densityGuard` (`?no-density-guard` → false). */
  option: boolean | undefined;
  config: DensityGuardConfig;
  /** The `?no-lod-energy` flag, passed through to `applyLodFade`. */
  energyComp: boolean;
  sceneManager: {
    readonly scene: THREE.Object3D | null;
    readonly camera: THREE.Camera | null;
    readonly renderer: { domElement: { width: number; height: number } } | null;
  };
  registerMaterial(material: THREE.Material): void;
  /** `SceneLoaderManager.setRefinementDensityProvider`. */
  setRefinementDensityProvider(
    provider: ProjectedDensityProvider | null,
    caps: DensityGateCaps
  ): void;
  /** The default loader, read per frame (a dataset switch replaces it). */
  getDefaultLoader(): { resumeDensityDeferredRefinement(): number } | null | undefined;
  /** The adaptive-DPR controller, read per frame (constructed after this wiring). */
  getAdaptiveDpr(): { notifyContentChanged(): void } | null | undefined;
  requestRender(): void;
  /** Injection points for tests; production uses the module singletons. */
  tracker?: ProjectedDensityTracker;
  guard?: DensityGuard;
}

export interface DensityGuardWiring {
  enabled: boolean;
  /** The refinement rung-gate provider handed to the loader manager (null when disabled). */
  provider: ProjectedDensityProvider | null;
  /** Register as the `'projected-density'` per-frame callback. */
  perFrame(): void;
}

/** Build the rung-gate provider over the tracker's records. */
export function buildDensityProvider(tracker: ProjectedDensityTracker): ProjectedDensityProvider {
  return (path) => {
    const rec = tracker.get(path);
    return rec
      ? {
          areaPx: rec.areaPx,
          elements: rec.elements,
          onScreen: rec.onScreen,
          blendable: rec.blendable,
        }
      : undefined;
  };
}

export function wireDensityGuard(deps: DensityGuardWiringDeps): DensityGuardWiring {
  const enabled = resolveDensityGuardEnabled(deps.configEnabled, deps.option);
  const tracker = deps.tracker ?? getProjectedDensityTracker();
  const guard = deps.guard ?? getDensityGuard();

  // The keep-fraction ladder rides the tracker's visit hook (blendable modes
  // only; brightness-compensated through applyLodFade).
  guard.configure({
    config: () => deps.config,
    energyComp: () => deps.energyComp,
    registerMaterial: (material) => deps.registerMaterial(material),
  });
  tracker.configure({
    enabled: () => enabled,
    getRoot: () => deps.sceneManager.scene,
    getCamera: () => deps.sceneManager.camera,
    getDrawingBufferSize: () => {
      const canvas = deps.sceneManager.renderer?.domElement;
      return canvas ? { width: canvas.width, height: canvas.height } : null;
    },
    onVisit: (mesh, record) => guard.observe(mesh, record),
  });

  // Refinement rung gate: the loaders read the same per-node records through
  // this provider (data/ cannot import scene/) and defer a rung that would push
  // a node past its screen cap. Null provider (guard off) = bytes-only.
  const provider = enabled ? buildDensityProvider(tracker) : null;
  deps.setRefinementDensityProvider(provider, {
    blendable: deps.config.capElementsPerPixel,
    nonBlendable: deps.config.nonBlendableCapElementsPerPixel,
  });

  return {
    enabled,
    provider,
    perFrame: () => {
      if (!tracker.evaluate()) return;
      // A keep-step change is a CONTENT change for the DPR controller (its
      // probe baseline no longer describes the scene) and needs a frame.
      if (guard.takeChanged()) {
        deps.getAdaptiveDpr()?.notifyContentChanged();
        deps.requestRender();
      }
      // Deferred rungs resume once the camera has moved in.
      deps.getDefaultLoader()?.resumeDensityDeferredRefinement();
    },
  };
}
