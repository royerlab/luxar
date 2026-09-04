/**
 * Projected-density rung gate for progressive refinement.
 *
 * The residency budget (`residency-budget.ts`) bounds a ladder by BYTES. This
 * gate bounds it by what the screen can show: once a node already projects
 * more elements per drawing-buffer pixel than the cap, its next additive rung
 * is pure overdraw — the density guard's shader ladder would thin it straight
 * back out (blendable modes), or a max/normal/opaque node would pay the
 * fragment cost for nothing the eye resolves. So the rung is DEFERRED rather
 * than fetched, decoded, projected and committed.
 *
 * Deferral is camera-dependent, so unlike the byte ceiling it is NOT sticky
 * across runs: the gate remembers, per deferred path, the projected footprint
 * at which the next rung would fit again (`resumeAreaPx`), and
 * `takeResumable()` hands back the paths whose current footprint has grown
 * past it (the user zoomed in). The SceneLoader re-kicks refinement for them
 * (`kickRefinementIfIdle`). Within ONE run the refusal is folded into the
 * residency budget's `declined` set, which is what keeps the documented
 * "declining retires the loader from `anyHasMoreLODs` / `getLoaderProgress`"
 * invariant (see the residency-budget module docstring) — the loop terminates
 * instead of re-offering the loader at frame rate.
 *
 * Density is read through a provider the app pipeline injects
 * (`scene/projected-density.ts` measures it per frame; `data/` cannot import
 * `scene/`). No provider, no record, an off-screen node, or a node that has
 * loaded nothing yet ⇒ no opinion: the gate never refuses on an absent
 * signal, and never refuses a first rung.
 *
 * @module data/scene-loader/progressive/density-gate
 */

import { noteRefinementDensityDeferral } from '../../../profiling/load-timeline';
import { log, Modules } from '../../../utils/log';
import type { LadderResidency } from './residency-budget';

/** One node's projected footprint, as the density tracker last measured it. */
export interface ProjectedDensitySample {
  /** Projected bounding-sphere area in drawing-buffer pixels (0 off-screen). */
  areaPx: number;
  /** Visible (committed) element count. */
  elements: number;
  onScreen: boolean;
  /**
   * True when the node's blend mode sums energy (additive / luminous /
   * volumetric) — the shader ladder can thin it, so its cap is the same few
   * elements per pixel the ladder targets. False for max / normal / opaque,
   * which have no linear brightness knob and get the tighter cap.
   */
  blendable: boolean;
}

/** Injected by the app pipeline; `undefined` = no measurement for this path. */
export type ProjectedDensityProvider = (path: string) => ProjectedDensitySample | undefined;

/** Elements per drawing-buffer pixel above which the next rung is deferred. */
export interface DensityGateCaps {
  blendable: number;
  nonBlendable: number;
}

export interface DensityGateVerdict {
  admitted: boolean;
  /** Density the node would reach with its next rung (mean-rung estimate). */
  predictedElementsPerPixel: number;
  cap: number;
  /** Footprint (px) at which the next rung would fit under the cap. */
  resumeAreaPx: number;
}

/**
 * Estimate the next rung's element count from the mean rung so far — the same
 * estimator the byte budget uses (`estimateNextRungBytes`), for the same
 * reason: a loader may no longer hold its individual rungs after a fold. Zero
 * rungs estimates zero, so a first rung is never refused.
 */
export function estimateNextRungElements(residency: LadderResidency): number {
  if (!(residency.loadedRungs > 0) || !(residency.elementCount > 0)) return 0;
  return residency.elementCount / residency.loadedRungs;
}

/** Pure decision: would the next rung push the node past its density cap? */
export function planDensityAdmission(
  sample: ProjectedDensitySample,
  residency: LadderResidency,
  caps: DensityGateCaps
): DensityGateVerdict {
  const cap = Math.max(sample.blendable ? caps.blendable : caps.nonBlendable, 1e-9);
  const nextElements = estimateNextRungElements(residency);
  const predictedElements = Math.max(0, sample.elements) + nextElements;
  const resumeAreaPx = predictedElements / cap;
  if (!sample.onScreen || nextElements === 0) {
    return { admitted: true, predictedElementsPerPixel: 0, cap, resumeAreaPx };
  }
  const predictedElementsPerPixel = predictedElements / Math.max(sample.areaPx, 1);
  return {
    admitted: predictedElementsPerPixel <= cap,
    predictedElementsPerPixel,
    cap,
    resumeAreaPx,
  };
}

/**
 * Session-scoped gate (one per SceneLoader). `beginRun()` at every refinement
 * run start; `admit()` from the residency budget; `takeResumable()` from the
 * per-frame density walk.
 */
export class RefinementDensityGate {
  /** Deferred path → footprint (px) at which its next rung fits again. */
  private readonly deferred = new Map<string, number>();
  private loggedThisRun = false;

  constructor(
    private readonly provider: ProjectedDensityProvider,
    private readonly caps: DensityGateCaps
  ) {}

  /** Forget the previous run's deferrals: every loader is re-evaluated. */
  beginRun(): void {
    this.deferred.clear();
    this.loggedThisRun = false;
  }

  /** `null` = no opinion (no measurement for this path). */
  admit(path: string, residency: LadderResidency): DensityGateVerdict | null {
    const sample = this.provider(path);
    if (!sample) return null;
    const verdict = planDensityAdmission(sample, residency, this.caps);
    if (!verdict.admitted) {
      this.deferred.set(path, verdict.resumeAreaPx);
      noteRefinementDensityDeferral();
      if (!this.loggedThisRun) {
        this.loggedThisRun = true;
        log.info(
          Modules.SCENE_LOADER,
          `Refinement deferred at the projected-density cap (first: ${path}, ` +
            `${verdict.predictedElementsPerPixel.toFixed(1)} el/px predicted vs cap ` +
            `${verdict.cap}); resumes when the node fills more of the screen.`
        );
      }
    }
    return verdict;
  }

  get deferredCount(): number {
    return this.deferred.size;
  }

  /**
   * Paths whose current footprint now admits their deferred rung. Each is
   * removed from the deferred set; the caller re-kicks refinement.
   */
  takeResumable(): string[] {
    if (this.deferred.size === 0) return [];
    const resumable: string[] = [];
    for (const [path, resumeAreaPx] of this.deferred) {
      const sample = this.provider(path);
      if (sample?.onScreen && sample.areaPx >= resumeAreaPx) resumable.push(path);
    }
    for (const path of resumable) this.deferred.delete(path);
    return resumable;
  }
}
