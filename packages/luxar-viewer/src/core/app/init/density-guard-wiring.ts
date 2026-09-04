/**
 * Wiring for the projected-density guard: tracker deps, keep-fraction guard,
 * the refinement rung-gate provider, the per-frame callback, and the runtime
 * on/off handle the Performance popover and settings persistence use.
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
import type { NodeDensityState } from '../../../types/data-monitor-types';
import type { DensityGuardControl } from '../../../ui/rendering-controls/types';

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

export interface DensityGuardWiring extends DensityGuardControl {
  /**
   * The refinement rung-gate provider over the tracker's records. Handed to
   * the loader manager while the guard is on; the manager gets `null`
   * (bytes-only admission) while it is off.
   */
  provider: ProjectedDensityProvider;
  /** Register as the `'projected-density'` per-frame callback. */
  perFrame(): void;
  /** Per-path snapshot for the data monitor's `drawn 1/K` chip (`DensityProvider`). */
  densityStates(): Map<string, NodeDensityState>;
}

/** Project the live records onto the monitor's per-node density state. */
export function collectDensityStates(
  tracker: ProjectedDensityTracker
): Map<string, NodeDensityState> {
  const out = new Map<string, NodeDensityState>();
  for (const rec of tracker.records()) {
    out.set(rec.path, {
      keep: rec.keep,
      elementsPerPixel: rec.elementsPerPixel,
      blendable: rec.blendable,
      onScreen: rec.onScreen,
    });
  }
  return out;
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

/** Thinned-node count and the smallest keep among them, straight off the live records. */
export function summarizeThinning(tracker: ProjectedDensityTracker): {
  nodes: number;
  minKeep: number;
} {
  let nodes = 0;
  let minKeep = 1;
  for (const rec of tracker.records()) {
    if (rec.keep >= 1) continue;
    nodes += 1;
    if (rec.keep < minKeep) minKeep = rec.keep;
  }
  return { nodes, minKeep };
}

export function wireDensityGuard(deps: DensityGuardWiringDeps): DensityGuardWiring {
  const sessionDisabled = deps.option === false;
  let enabled = resolveDensityGuardEnabled(deps.configEnabled, deps.option);
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
  // a node past its screen cap. Null provider (guard off) = bytes-only; the
  // loader kicks refinement for anything the old gate was holding back.
  const provider = buildDensityProvider(tracker);
  const caps: DensityGateCaps = {
    blendable: deps.config.capElementsPerPixel,
    nonBlendable: deps.config.nonBlendableCapElementsPerPixel,
  };
  const publish = (): void => deps.setRefinementDensityProvider(enabled ? provider : null, caps);
  publish();

  // Turning the guard off must undo what it did: every thinned node back to
  // keep 1 (uniform + brightness) and the records dropped, so a later turn-on
  // starts from a clean ladder instead of stale steps.
  const releaseAll = (): void => {
    deps.sceneManager.scene?.traverse((obj) => guard.release(obj));
    tracker.reset();
  };

  return {
    provider,
    sessionDisabled,
    isEnabled: () => enabled,
    setEnabled: (on) => {
      if (on === enabled) return;
      enabled = on;
      publish();
      if (!on) releaseAll();
      // Either direction changes what a frame draws (and which rungs may load):
      // the DPR controller's probe baseline is stale, and a frame is needed —
      // on, so the next walk thins; off, so the released nodes redraw whole.
      deps.getAdaptiveDpr()?.notifyContentChanged();
      deps.requestRender();
    },
    thinning: () => summarizeThinning(tracker),
    densityStates: () => collectDensityStates(tracker),
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
