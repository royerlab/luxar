/**
 * The mesh pick pass's mode-derived state, and the capability the picking system
 * dispatches on.
 *
 * A mesh's pick behaviour depends on the visual material's blending mode in **two**
 * ways (§6.5), and they must never disagree:
 *
 * | Consequence | True for | Why |
 * |---|---|---|
 * | hard alpha cutout | `opaque` | a hole the user sees through must not be pickable, nor depth-occlude picks of what is visible through it |
 * | real projected depth | `opaque`, `normal` | the user sees an occluding surface, so front-most must win rather than brightest |
 *
 * The two overlap but are not the same set, which is exactly why this module exists:
 * a material given "cutout on, brightness-as-depth" would discard holes correctly
 * and then let a dim mesh in front occlude a brighter node behind it. Deriving both
 * from the mode in ONE place makes that combination unreachable.
 *
 * The gsplat pick wrapper's `SurfacePickAwareMaterial` covers only the depth half —
 * gsplats have no cutout — so mesh needs its own capability rather than widening
 * that one. `PickingSystem.renderPickBuffer` prefers this interface when present and
 * falls back to `setSurfacePickDepth` otherwise, reading the mode exactly once
 * either way.
 *
 * @module rendering/picking/mesh/pick-mode
 */

import type * as THREE from 'three';
import { isNormalMode, isOpaqueMode } from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';

/**
 * A pick material that tracks the visual material's whole per-epoch appearance
 * state, not just a single boolean. Implemented by both mesh pick wrappers, and by
 * nothing else — the two members below are the two things a mesh pick pass must copy
 * from its visual twin that the sibling pick materials do not.
 */
export interface MeshPickAwareMaterial {
  /**
   * Apply `mode`'s pick-pass consequences. Both are RUNTIME uniforms, never build
   * flags, so a layers-panel mode switch does not recompile the pick program.
   */
  setPickMode(mode: BlendingMode): void;
  /**
   * Match the visual material's face culling.
   *
   * Grouped with `setPickMode` rather than given its own capability because both are
   * "copy the visual material's current epoch state", both are driven from the same
   * one place (`PickingSystem.renderPickBuffer`), and both are mesh-only — a second
   * guard would buy nothing but a second thing to forget.
   */
  setPickSide(side: THREE.Side): void;
}

/** Type guard for {@link MeshPickAwareMaterial}, mirroring the sibling guards. */
export function isMeshPickAwareMaterial(material: unknown): material is MeshPickAwareMaterial {
  return (
    typeof material === 'object' &&
    material !== null &&
    'setPickMode' in material &&
    typeof (material as Record<string, unknown>).setPickMode === 'function' &&
    typeof (material as Record<string, unknown>).setPickSide === 'function'
  );
}

/** The two uniform values `mode` implies, derived in one place for both backends. */
export interface MeshPickModeState {
  /** Apply the visual shader's identical `a < uAlphaCutoff` discard. */
  readonly cutout: boolean;
  /** Write real projected depth (front-most wins) instead of brightness-as-depth. */
  readonly surfaceDepth: boolean;
}

/**
 * Derive the pick-pass state from a blending mode.
 *
 * `volumetric` needs no special case: it reaches a mesh only by inheritance and the
 * material has already mapped it to `opaque` (`resolveMeshBlendingMode`), so it
 * arrives here as `opaque` — and if it ever arrived raw, both predicates return
 * false, which is the conservative commutative-mode answer rather than a wrong
 * cutout.
 */
export function resolveMeshPickModeState(mode: BlendingMode): MeshPickModeState {
  return {
    cutout: isOpaqueMode(mode),
    surfaceDepth: isNormalMode(mode) || isOpaqueMode(mode),
  };
}
