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
 *   (VOLUMETRIC_BLENDING_SPEC.md §3.1). The cross-fade (weights `w`, `1−w`)
 *   therefore conserves per-ray absorption EXACTLY
 *   (`1 − e^(−wτ)·e^(−(1−w)τ) = 1 − e^(−τ)`) and emission to first order in
 *   τ; the `1/e(k)` boost restores the full per-ray τ of a partially-streamed
 *   ladder. Caveat (spec §6): on individually optically-thick splats
 *   (`κ·splat-mass ≳ 1`) the per-splat self-screening `S(τ)` saturates
 *   emission, so a large boost deepens occlusion more than it brightens — a
 *   bounded, transient artifact accepted under the shared `ENERGY_FLOOR` cap.
 *
 * `max` (a max, not a sum), `normal` (nonlinear alpha-over with opacity-gated
 * depthWrite), and `opaque` are excluded from both mechanisms.
 */
const BLENDABLE_MODES: ReadonlySet<string> = new Set(['additive', 'luminous', 'volumetric']);

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
 * depth). A group subtree (overview partition branch) must be uniformly
 * blendable. No fadeable material at all ⇒ not blendable (nothing to fade —
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
  // normalModeDepthWrite): after writing a fade opacity, re-derive the
  // mode state so the gate tracks the live value. Unreachable today
  // (BLENDABLE_MODES = additive/luminous/volumetric, whose depth state is
  // opacity-independent — volumetric's depthWrite is unconditionally false)
  // but preserves the invariant if that set grows.
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
    };
    // The blend mode lives on the MATERIAL's userData; energy compensation
    // only makes physical sense where opacity linearly scales the composited
    // quantity (summed energy for additive/luminous, optical depth τ for
    // volumetric — see BLENDABLE_MODES).
    let energyFactor = 1;
    if (energyComp && BLENDABLE_MODES.has((current.userData?.blendingMode as string) ?? '')) {
      energyFactor = energyCompensation(ud.committedEnergyFraction, ENERGY_FLOOR);
    }
    const product = coverageWeight * energyFactor;
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
