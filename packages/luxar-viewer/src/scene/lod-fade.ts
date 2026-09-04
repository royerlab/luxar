/**
 * Material-level appliers for the two LOD anti-popping mechanisms.
 *
 * `lod-blend.ts` holds the pure, THREE-free opacity MATH (the coverage
 * cross-fade weight and the streaming `1/e(k)` energy compensation); this
 * module is its material-touching counterpart, extracted from
 * `lod-group-registry.ts`: walk a LOD child's leaf materials
 * (clone-on-first-fade) and write the composed opacity multiplier, or restore
 * the authored opacity. Kept out of `lod-blend.ts` so that module stays
 * dependency-free and unit-testable over plain numbers.
 *
 * @module scene/lod-fade
 */

import * as THREE from 'three';

import { energyCompensation } from './lod-blend';

/**
 * The subset of a leaf material's surface the LOD cross-fade drives: read the
 * authored opacity as a fade base, write `base × α`, and clone-on-first-use
 * (materials are cached by props, so an in-place write would fade every layer
 * sharing the instance). `getOpacity` is the symmetric companion to
 * `updateOpacity` added to the material classes for exactly this.
 */
export interface FadeableMaterial extends THREE.Material {
  updateOpacity(opacity: number): void;
  getOpacity(): number;
  /**
   * Present on every Luxar leaf material. Used after a fade opacity
   * write to re-derive normal mode's opacity-gated depthWrite.
   */
  applyBlendingMode?(mode: string): void;
}

/** True when a material exposes the {@link FadeableMaterial} opacity surface. */
function isFadeable(mat: THREE.Material): mat is FadeableMaterial {
  const m = mat as Partial<FadeableMaterial>;
  return typeof m.updateOpacity === 'function' && typeof m.getOpacity === 'function';
}

/**
 * Blend modes where opacity is a well-behaved linear knob on the composited
 * result, so both LOD anti-popping mechanisms — the coverage cross-fade
 * (mass-conserved levels) and the streaming energy compensation (`1/e(k)`) —
 * are physically sound:
 *
 * - `additive` / `luminous`: order-independent compositing sums energy
 *   linearly in opacity ⇒ both mechanisms are brightness-exact.
 * - `volumetric`: order-dependent emission–absorption, but opacity linearly
 *   scales the optical depth `τ = κ·opacity·intensity`
 *   (VOLUMETRIC_BLENDING_SPEC.md §3.1), which is what makes an opacity fade
 *   well-behaved here — see the two caveats below for what it does and does
 *   NOT guarantee. Both are the documented, accepted tradeoffs of §6.
 *
 *   *Cross-fade.* Because τ adds across fragments and is linear in opacity, a
 *   `w`/`1−w` pair composites to `1 − exp(−(w·τ_fine + (1−w)·τ_coarse))`: the
 *   endpoints are exact, and in between the absorption moves monotonically
 *   between the two levels' own absorptions — a log-space (transmittance-
 *   multiplicative) interpolation, i.e. exactly the ghost-free dissolve an
 *   anti-popping fade wants, and strictly better than the hard swap it
 *   replaces. It collapses to a *constant* `1 − e^(−τ)` only where the two
 *   levels present the same per-ray τ; the build invariant is total mass per
 *   barrier group, NOT per-ray mass, and a coarse level is by construction a
 *   different spatial distribution, so do not build on "absorption is
 *   invariant mid-fade" — it holds only in that mass-matched idealization.
 *
 *   *Streaming `1/e(k)`.* `e(k)` is a GLOBAL energy fraction and a committed
 *   ladder prefix is a SUBSET of splats, so the boost restores τ in
 *   AGGREGATE, not per ray: rays through the committed core are over-boosted
 *   and rays through only-missing splats get nothing. That is the same
 *   structural approximation the additive/luminous path has shipped since the
 *   compensation landed — volumetric is not held to a lower bar. The
 *   volumetric-specific twist: on individually optically-thick splats
 *   (`κ·splat-mass ≳ 1`) the per-splat self-screening `S(τ)` saturates
 *   emission, so a boosted splat deepens occlusion rather than brightening.
 *   Bounded by the shared `ENERGY_FLOOR` cap (≤ 10×), transient (decays as
 *   `e → 1`), and `?no-lod-energy` is the escape hatch; a volumetric-specific
 *   floor is the obvious knob if a thick-splat scene ever shows transient
 *   dark blobs while streaming.
 *
 * A subtree mixing additive/luminous leaves with volumetric leaves is accepted
 * for compatibility because every leaf still has continuous, endpoint-exact
 * opacity control. That is a tolerated authoring edge case, not a shared
 * conservation proof: its mid-band result follows neither one summed-energy
 * model nor one optical-depth model. Authors should keep a LOD subtree within a
 * single compositing family whenever possible.
 *
 * `max` (a max, not a sum), `normal` (nonlinear alpha-over with opacity-gated
 * depthWrite), and `opaque` are excluded from both mechanisms.
 */
const BLENDABLE_MODES: ReadonlySet<string> = new Set(['additive', 'luminous', 'volumetric']);

/** True when `mode` is one of {@link BLENDABLE_MODES} (opacity is a linear knob). */
export function isBlendableMode(mode: string | undefined): boolean {
  return mode != null && BLENDABLE_MODES.has(mode);
}

/**
 * Brightness term for the density guard's keep-fraction thinning: a node
 * drawing only a `keep` fraction of its elements has its opacity multiplied by
 * `1/keep` so the sum-projected (or optical-depth) result stays at the
 * unthinned aggregate. `1` for an unthinned or unstamped node. The guard's
 * ladder already floors `keep` (config `minKeepFraction`), so no cap here.
 */
export function densityCompensation(keep: number | undefined): number {
  return keep != null && Number.isFinite(keep) && keep > 0 && keep < 1 ? 1 / keep : 1;
}

/** Below this the finer level's blend weight is treated as 0/1 (single level). */
export const FADE_EPSILON = 0.01;

/**
 * Floor for the streaming brightness-compensation energy fraction `e(k)`: the
 * `1/e(k)` boost is capped at `1/ENERGY_FLOOR` so a tiny early prefix can't
 * over-brighten its (energy-descending, core-heavy) splats into tone-map
 * clipping. 0.1 ⇒ at most a 10× boost. See `energyCompensation`.
 */
export const ENERGY_FLOOR = 0.1;

/**
 * Whether every fadeable leaf material under ``root`` uses a blend mode that
 * cross-fades correctly ({@link BLENDABLE_MODES} — additive / luminous /
 * volumetric, where opacity is a linear knob on summed energy or on optical
 * depth). A group subtree (overview partition branch) must contain only
 * individually blendable leaves; mixing their compositing families is tolerated
 * for compatibility but has no common mid-fade conservation model. No fadeable
 * material at all ⇒ not blendable (nothing to fade —
 * e.g. a not-yet-loaded placeholder, or a `max`/`normal`/`opaque` layer which
 * keeps the hard swap).
 */
export function isBlendableSubtree(root: THREE.Object3D): boolean {
  let sawFadeable = false;
  let allBlendable = true;
  const visit = (mesh: THREE.Object3D): void => {
    const mat = (mesh as THREE.Mesh).material;
    if (!mat || Array.isArray(mat) || !isFadeable(mat)) return;
    sawFadeable = true;
    const mode = (mat.userData?.blendingMode as string | undefined) ?? '';
    if (!BLENDABLE_MODES.has(mode)) allBlendable = false;
  };
  const obj = root as THREE.Mesh;
  if (obj.material) visit(root);
  else root.traverse(visit);
  return sawFadeable && allBlendable;
}

/**
 * Apply the per-leaf LOD anti-popping opacity to a child's leaf materials, or
 * restore the authored opacity. The effective multiplier is the product of two
 * independent opacity terms:
 *
 * - `coverageWeight` = `weight ?? 1` — the cross-fade blend opacity of this
 *   level (`null` ⇒ 1, no cross-fade in flight). Per-CHILD (the whole level).
 * - `energyFactor` — the streaming brightness compensation `1/e(k)` read
 *   PER-LEAF from `committedEnergyFraction`, applied only when `energyComp` is on
 *   AND the leaf's blend mode sums energy ({@link BLENDABLE_MODES}). Complete /
 *   unstamped / non-blendable leaves ⇒ 1.
 *
 * When the product is ≈ 1 the leaf needs no adjustment: restore the authored
 * opacity if we had faded it (idempotent no-op otherwise) and, crucially, never
 * clone a material we don't have to — so a steady-state / disabled /
 * non-blendable / complete child stays byte-identical. Otherwise snapshot the
 * authored opacity as the fade base and write `base × product`. Leaf materials
 * are per-node since the material-manager rework (all three node factories
 * stamp ``_layerMaterialCloned: true`` at creation), so in practice the write
 * mutates the node's own material in place; the clone-on-first-use branch
 * below is a dormant safety net for any material that ever arrives unstamped
 * (it mirrors the layers panel's marker so the two never double-clone).
 *
 * ``root`` is a leaf mesh or a group subtree (overview/partition branch) →
 * each fadeable leaf is visited individually, so `energyFactor` is genuinely
 * per-leaf across a partition of independently-streaming leaves.
 * ``registerMaterial`` keeps a clone-on-first-fade material receiving
 * per-frame camera-uniform updates (wired to `materialManager.register`;
 * omitted in unit tests, which run no camera loop).
 */
export function applyLodFade(
  root: THREE.Object3D,
  weight: number | null,
  energyComp: boolean,
  registerMaterial?: (material: THREE.Material) => void
): void {
  const coverageWeight = weight ?? 1;
  // Normal mode's depthWrite is opacity-gated (>= 0.99, see
  // normalModeDepthWrite) — but that gate is now the LINE gate only: point
  // materials force depthWrite:false in normal regardless (#1002), so
  // applyBlendingMode('normal') re-derives false for points. Either way,
  // after writing a fade opacity, re-derive the mode state so the gate
  // tracks the live value. Unreachable today (BLENDABLE_MODES =
  // additive/luminous/volumetric, whose depth state is opacity-independent
  // — volumetric's depthWrite is unconditionally false) but preserves the
  // invariant if that set grows.
  const refreshNormalDepthWrite = (m: FadeableMaterial): void => {
    if ((m.userData?.blendingMode as string | undefined) === 'normal') {
      m.applyBlendingMode?.('normal');
    }
  };
  const visit = (mesh: THREE.Object3D): void => {
    const current = (mesh as THREE.Mesh).material;
    if (!current || Array.isArray(current) || !isFadeable(current)) return;
    const ud = mesh.userData as {
      _lodFadeBase?: number;
      _layerMaterialCloned?: boolean;
      committedEnergyFraction?: number;
      densityKeep?: number;
    };
    // The blend mode lives on the MATERIAL's userData; energy compensation
    // only makes physical sense where opacity linearly scales the composited
    // quantity (summed energy for additive/luminous, optical depth τ for
    // volumetric — see BLENDABLE_MODES).
    const blendable = isBlendableMode(current.userData?.blendingMode as string | undefined);
    let energyFactor = 1;
    if (energyComp && blendable) {
      energyFactor = energyCompensation(ud.committedEnergyFraction, ENERGY_FLOOR);
    }
    // Density-guard thinning (scene/density-guard.ts) draws a `keep` fraction
    // of the elements; `1/keep` restores the aggregate brightness. Same
    // blendable-only rule — the guard never thins the other modes.
    const densityFactor = blendable ? densityCompensation(ud.densityKeep) : 1;
    const product = coverageWeight * energyFactor * densityFactor;
    if (Math.abs(product - 1) < FADE_EPSILON) {
      // Nothing to adjust: restore the authored opacity if we faded it, else
      // leave the shared material untouched (no clone).
      if (ud._lodFadeBase != null) {
        current.updateOpacity(ud._lodFadeBase);
        refreshNormalDepthWrite(current);
        ud._lodFadeBase = undefined;
      }
      return;
    }
    let mat = current;
    if (!ud._layerMaterialCloned) {
      const cloned = current.clone() as FadeableMaterial;
      (mesh as THREE.Mesh).material = cloned;
      ud._layerMaterialCloned = true;
      registerMaterial?.(cloned); // keep camera uniforms live
      mat = cloned;
    }
    // Snapshot the composed authored opacity once; hold it steady while the
    // multiplier changes (per-frame as the ladder fills in), clear it on restore.
    if (ud._lodFadeBase == null) ud._lodFadeBase = mat.getOpacity();
    mat.updateOpacity(ud._lodFadeBase * product);
    refreshNormalDepthWrite(mat);
  };
  const obj = root as THREE.Mesh;
  if (obj.material) visit(root);
  else root.traverse(visit);
}
