/**
 * Direct unit tests for `scene/lod-fade.ts` — `isBlendableSubtree` (the
 * BLENDABLE_MODES gate for both LOD anti-popping mechanisms) and
 * `applyLodFade`'s opacity composition, previously exercised only
 * indirectly through the LODGroupRegistry suite.
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { applyLodFade, isBlendableSubtree } from '../../../scene/lod-fade';

interface FadeMatStub {
  userData: { blendingMode?: string };
  _op: number;
  updateOpacity(v: number): void;
  getOpacity(): number;
  clone(): FadeMatStub;
}
function fadeMat(blendingMode?: string): FadeMatStub {
  return {
    userData: blendingMode != null ? { blendingMode } : {},
    _op: 1,
    updateOpacity(v: number) {
      this._op = v;
    },
    getOpacity() {
      return this._op;
    },
    clone() {
      const c = fadeMat(blendingMode);
      c._op = this._op;
      return c;
    },
  };
}
function leafMesh(mode?: string): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.material = fadeMat(mode) as unknown as THREE.Material;
  // Per-node material from creation (mirrors the node factories) so
  // applyLodFade mutates in place instead of exercising the dormant clone.
  mesh.userData = { _layerMaterialCloned: true };
  return mesh;
}
const liveOpacity = (mesh: THREE.Mesh): number =>
  (mesh.material as unknown as FadeMatStub).getOpacity();

describe('isBlendableSubtree — BLENDABLE_MODES matrix', () => {
  it.each(['additive', 'luminous', 'volumetric'])('%s ⇒ blendable', (mode) => {
    expect(isBlendableSubtree(leafMesh(mode))).toBe(true);
  });

  it.each(['max', 'normal', 'opaque'])('%s ⇒ not blendable', (mode) => {
    expect(isBlendableSubtree(leafMesh(mode))).toBe(false);
  });

  it('missing mode ⇒ not blendable', () => {
    expect(isBlendableSubtree(leafMesh())).toBe(false);
  });

  it('no fadeable material at all ⇒ not blendable (nothing to fade)', () => {
    expect(isBlendableSubtree(new THREE.Group())).toBe(false);
    const bare = new THREE.Mesh();
    bare.material = new THREE.MeshBasicMaterial(); // no updateOpacity/getOpacity
    expect(isBlendableSubtree(bare)).toBe(false);
  });

  it('group subtree: uniformly blendable across DIFFERENT blendable modes ⇒ blendable', () => {
    const group = new THREE.Group();
    group.add(leafMesh('additive'), leafMesh('volumetric'), leafMesh('luminous'));
    expect(isBlendableSubtree(group)).toBe(true);
  });

  it('group subtree: one non-blendable leaf poisons the subtree', () => {
    const group = new THREE.Group();
    group.add(leafMesh('volumetric'), leafMesh('max'));
    expect(isBlendableSubtree(group)).toBe(false);
  });
});

describe('applyLodFade — opacity composition and fade-base rebase', () => {
  it('writes base × weight and restores the base at weight ≈ 1', () => {
    const mesh = leafMesh('volumetric');
    applyLodFade(mesh, 0.5, false);
    expect(liveOpacity(mesh)).toBeCloseTo(0.5, 6);
    expect(mesh.userData._lodFadeBase).toBe(1);
    applyLodFade(mesh, null, false); // fade over ⇒ restore, snapshot cleared
    expect(liveOpacity(mesh)).toBe(1);
    expect(mesh.userData._lodFadeBase).toBeUndefined();
  });

  it('a mid-fade base rebase (the layers-panel path) renders newBase × product next frame', () => {
    // The panel edits opacity while a fade is in flight: LayerApplyEngine
    // writes the composed value into `_lodFadeBase` instead of the live
    // uniform (which the fade owns), and the next applyLodFade frame
    // composes it. Restoring lands on the NEW base, not the stale snapshot.
    const mesh = leafMesh('volumetric');
    applyLodFade(mesh, 0.5, false);
    expect(liveOpacity(mesh)).toBeCloseTo(0.5, 6);
    mesh.userData._lodFadeBase = 0.6; // panel edit mid-fade
    applyLodFade(mesh, 0.5, false);
    expect(liveOpacity(mesh)).toBeCloseTo(0.3, 6);
    applyLodFade(mesh, null, false);
    expect(liveOpacity(mesh)).toBeCloseTo(0.6, 6);
  });

  it('energy compensation applies per-leaf to volumetric (opacity linearly scales τ)', () => {
    const mesh = leafMesh('volumetric');
    mesh.userData.committedEnergyFraction = 0.5;
    applyLodFade(mesh, null, true);
    expect(liveOpacity(mesh)).toBeCloseTo(2, 6);
  });

  it('energy compensation skips a non-blendable (max) leaf', () => {
    const mesh = leafMesh('max');
    mesh.userData.committedEnergyFraction = 0.5;
    applyLodFade(mesh, null, true);
    expect(liveOpacity(mesh)).toBe(1);
  });

  it('group subtree: energy factors are genuinely per-leaf', () => {
    const group = new THREE.Group();
    const streaming = leafMesh('volumetric');
    streaming.userData.committedEnergyFraction = 0.25;
    const complete = leafMesh('volumetric');
    group.add(streaming, complete);
    applyLodFade(group, 0.5, true);
    expect(liveOpacity(streaming)).toBeCloseTo(0.5 * 4, 6); // weight × 1/e
    expect(liveOpacity(complete)).toBeCloseTo(0.5, 6); // weight only
  });
});
