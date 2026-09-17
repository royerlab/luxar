/**
 * Projected-density guard — the shader keep-fraction ladder.
 *
 * Consumes the per-node density the tracker in `projected-density.ts`
 * measures each frame and, for nodes whose blend mode sums energy
 * (additive / luminous / volumetric — `isBlendableMode`), thins the DRAW to
 * a `keep` fraction of the elements: the material's `uDensityDrop` uniform
 * makes the vertex stage discard every element whose hashed storage index
 * falls below `1 − keep`, and `applyLodFade` multiplies the node's opacity
 * by `1/keep` so the composited brightness stays at the unthinned aggregate.
 * Non-blendable modes (`max`, `normal`, `opaque`) are never thinned — a
 * max-projection or an alpha-over surface has no linear brightness knob.
 * Shaded triangle meshes are excluded: they tile a surface rather than stack
 * emissive energy, and their materials do not carry `uDensityDrop`.
 *
 * Why: on element-dense views the frame cost is elements per pixel, not
 * pixels (2026-09 audit: 1.5 M points framed into ~1 600 px → 42 ms at DPR 1,
 * 83 ms at DPR 0.5). Beyond a few elements per pixel every further element
 * is overdraw the eye cannot resolve. The ladder is quantised
 * (1, 1/2, 1/4, … `minKeepFraction`) with hysteresis so a slow zoom does
 * not flicker between steps, and every step change is a CONTENT change for
 * the adaptive-DPR controller (`notifyContentChanged`), never per-frame
 * churn.
 *
 * Hot-path invariants: no per-frame allocation (the guard mutates the
 * tracker's records and the mesh's own material in place) and idempotent
 * per-frame re-assertion of the uniform, so a material rebuilt or replaced
 * behind the guard's back cannot leave the brightness term without its
 * thinning (or vice versa).
 *
 * @module scene/density-guard
 */

import type * as THREE from 'three';

import {
  getDensityDrop,
  hasDensityDrop,
  setDensityDrop,
} from '../rendering/materials/_shared/density-drop';
import { applyLodFade, isBlendableMode } from './lod-fade';
import type { NodeDensity } from './projected-density';

/** The ladder knobs, a subset of `config.densityGuard`. */
export interface DensityLadderConfig {
  capElementsPerPixel: number;
  minKeepFraction: number;
  enterRatio: number;
  leaveRatio: number;
}

/**
 * The keep fraction (a power of 1/2, floored at `minKeepFraction`) that
 * brings `elementsPerPixel × keep` under the cap.
 */
export function targetKeepFraction(elementsPerPixel: number, cfg: DensityLadderConfig): number {
  const cap = Math.max(cfg.capElementsPerPixel, 1e-9);
  if (!(elementsPerPixel > cap)) return 1;
  const steps = Math.ceil(Math.log2(elementsPerPixel / cap));
  return Math.max(Math.pow(2, -steps), Math.min(1, cfg.minKeepFraction));
}

/**
 * Hysteresis step: from `currentKeep`, move to the target fraction only when
 * the EFFECTIVE density (`elementsPerPixel × currentKeep`) has left the
 * `[cap × leaveRatio, cap × enterRatio]` band in the direction of the
 * target; otherwise hold. `elementsPerPixel` is the UNTHINNED density (the
 * tracker divides the node's full visible count by its footprint).
 */
export function nextKeepFraction(
  currentKeep: number,
  elementsPerPixel: number,
  cfg: DensityLadderConfig
): number {
  const target = targetKeepFraction(elementsPerPixel, cfg);
  if (target === currentKeep) return currentKeep;
  const effective = elementsPerPixel * currentKeep;
  const cap = cfg.capElementsPerPixel;
  if (target < currentKeep) return effective > cap * cfg.enterRatio ? target : currentKeep;
  return effective < cap * cfg.leaveRatio ? target : currentKeep;
}

export interface DensityGuardDeps {
  config(): DensityLadderConfig;
  /** The `?noLodEnergy` flag — passed through to `applyLodFade` unchanged. */
  energyComp(): boolean;
  /** Keeps a clone-on-first-fade material on the camera-uniform loop. */
  registerMaterial?(material: THREE.Material): void;
}

interface GuardUserData {
  densityKeep?: number;
}

/** Whether a single material supports shader-side density thinning. */
function supportsDensityGuard(
  material: THREE.Material | THREE.Material[] | undefined
): material is THREE.Material {
  return Boolean(material && !Array.isArray(material) && hasDensityDrop(material));
}

/** Owns the per-node keep ladder; driven by the tracker's `onVisit` hook. */
export class DensityGuard {
  private deps: DensityGuardDeps | null = null;
  private changed = false;

  configure(deps: DensityGuardDeps): void {
    this.deps = deps;
  }

  /**
   * Tracker visit hook: re-derive this node's keep fraction from its fresh
   * density record, apply a step change to material + brightness, and
   * re-assert the uniform. Off-screen nodes hold their step (no churn on
   * nodes nobody sees; they re-evaluate the frame they return).
   */
  observe(mesh: THREE.Mesh, rec: NodeDensity): void {
    const deps = this.deps;
    const material = mesh.material;
    if (!deps || !supportsDensityGuard(material)) return;
    const ud = mesh.userData as GuardUserData;
    const current = ud.densityKeep ?? 1;
    const blendable = isBlendableMode(material.userData?.blendingMode as string | undefined);
    let next = current;
    if (!blendable) next = 1;
    else if (rec.onScreen) next = nextKeepFraction(current, rec.elementsPerPixel, deps.config());
    if (next !== current) {
      ud.densityKeep = next;
      // Brightness first: applyLodFade may clone-on-first-fade, and the
      // uniform must land on whichever material the mesh ends up drawing.
      applyLodFade(mesh, null, deps.energyComp(), deps.registerMaterial);
      this.changed = true;
    }
    rec.keep = next;
    rec.blendable = blendable;
    // Idempotent re-assertion (returns true only on drift, e.g. a material
    // rebuilt with the default 0 while the node is thinned).
    if (setDensityDrop(mesh.material, 1 - next)) this.changed = true;
  }

  /**
   * Undo this node's thinning (guard turned off at runtime): keep back to 1,
   * brightness recomputed without the density term, uniform cleared. A no-op
   * on anything that is not a thinned data mesh.
   */
  release(obj: THREE.Object3D): void {
    const mesh = obj as THREE.Mesh;
    const deps = this.deps;
    if (!deps || !mesh.isMesh || !supportsDensityGuard(mesh.material)) return;
    const ud = mesh.userData as GuardUserData;
    if ((ud.densityKeep ?? 1) !== 1) {
      ud.densityKeep = 1;
      applyLodFade(mesh, null, deps.energyComp(), deps.registerMaterial);
      this.changed = true;
    }
    if (setDensityDrop(mesh.material, 0)) this.changed = true;
  }

  /** True once since the last call if any node changed step (or drifted). */
  takeChanged(): boolean {
    const c = this.changed;
    this.changed = false;
    return c;
  }

  /** Current keep on a mesh's material (1 when unthinned / no uniform). */
  static keepOf(mesh: THREE.Mesh): number {
    return 1 - getDensityDrop(mesh.material);
  }
}

const guard = new DensityGuard();

/** The app-wide guard; configured once by the init pipeline. */
export function getDensityGuard(): DensityGuard {
  return guard;
}
