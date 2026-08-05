/**
 * The mesh pick pass's mode-derived state (spec §6.5), and the two ways it can go
 * wrong that no rendered pixel would reveal.
 *
 * @module tests/unit/rendering/picking/mesh/pick-mode
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  resolveMeshPickModeState,
  isMeshPickAwareMaterial,
} from '../../../../../rendering/picking/mesh/pick-mode';
import { MESH_SUPPORTED_BLENDING_MODES } from '../../../../../rendering/materials/mesh/appearance';
import type { BlendingMode } from '../../../../../types/blending';

describe('resolveMeshPickModeState', () => {
  it('applies the cutout in exactly one mode — `opaque`', () => {
    // Written as an enumeration over the SUPPORTED list rather than a handful of
    // spot checks, so adding a mesh blending mode forces a decision here instead of
    // silently inheriting whichever arm its predicate happened to land on.
    const cutoutModes = MESH_SUPPORTED_BLENDING_MODES.filter(
      (mode) => resolveMeshPickModeState(mode).cutout
    );
    expect(cutoutModes).toEqual(['opaque']);
  });

  it('writes real projected depth for the two depth-ordered surface modes', () => {
    const surfaceModes = MESH_SUPPORTED_BLENDING_MODES.filter(
      (mode) => resolveMeshPickModeState(mode).surfaceDepth
    );
    expect([...surfaceModes].sort()).toEqual(['normal', 'opaque']);
  });

  it('never applies a cutout without also writing surface depth', () => {
    // The invariant the whole module exists for: `cutout && !surfaceDepth` is the
    // combination that discards holes correctly and then lets a DIM mesh in front
    // depth-occlude a brighter node behind it. Asserted across every mode, including
    // the ones a mesh should never see.
    const everyMode: BlendingMode[] = [
      ...MESH_SUPPORTED_BLENDING_MODES,
      'volumetric' as BlendingMode,
    ];
    for (const mode of everyMode) {
      const { cutout, surfaceDepth } = resolveMeshPickModeState(mode);
      expect(cutout && !surfaceDepth, `${mode}: cutout without surface depth`).toBe(false);
    }
  });

  it('treats a raw `volumetric` conservatively rather than guessing', () => {
    // It should never arrive — `resolveMeshBlendingMode` maps it to `opaque` first —
    // but if it did, the commutative answer is the safe one: no cutout invented for
    // a mode with no cutout semantics.
    expect(resolveMeshPickModeState('volumetric' as BlendingMode)).toEqual({
      cutout: false,
      surfaceDepth: false,
    });
  });
});

describe('isMeshPickAwareMaterial', () => {
  it('accepts an object carrying BOTH members', () => {
    expect(isMeshPickAwareMaterial({ setPickMode: () => {}, setPickSide: () => {} })).toBe(true);
  });

  it('rejects an object carrying only one of them', () => {
    // A partial implementation is the realistic failure: someone adds `setPickMode`
    // to a sibling wrapper and the picking system then calls `setPickSide` on it.
    expect(isMeshPickAwareMaterial({ setPickMode: () => {} })).toBe(false);
    expect(isMeshPickAwareMaterial({ setPickSide: () => {} })).toBe(false);
  });

  it('rejects a plain material, null and a non-object', () => {
    expect(isMeshPickAwareMaterial(new THREE.MeshBasicMaterial())).toBe(false);
    expect(isMeshPickAwareMaterial(null)).toBe(false);
    expect(isMeshPickAwareMaterial('setPickMode')).toBe(false);
  });

  it('rejects an object whose members are present but not callable', () => {
    expect(isMeshPickAwareMaterial({ setPickMode: 1, setPickSide: 2 })).toBe(false);
  });
});
