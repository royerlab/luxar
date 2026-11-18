/**
 * Direct unit tests for `scene/lod-fade.ts` — `isBlendableSubtree` (the
 * BLENDABLE_MODES gate for both LOD anti-popping mechanisms) and
 * `applyLodFade`'s opacity composition, previously exercised only
 * indirectly through the LODGroupRegistry suite.
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import {
  getLadderEnergyStampVersion,
  stampLadderComplete,
} from '../../../data/scene-loader/commit/stamp-view-version';

import {
  applyLodFade,
  applyStandaloneLadderFade,
  densityCompensation,
  invalidateStandaloneLadderFade,
  isBlendableMode,
  isBlendableSubtree,
} from '../../../scene/lod-fade';

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

describe('standalone additive ladder compensation', () => {
  it('skips the tree until a commit changes energy, and reapplies on an opt-out change', () => {
    const scene = new THREE.Group();
    const leaf = leafMesh('additive');
    const loader = { committedEnergyFraction: 0.5 };
    leaf.userData.loader = loader;
    scene.add(leaf);
    stampLadderComplete(leaf.userData);
    const version = getLadderEnergyStampVersion();
    expect(applyStandaloneLadderFade(scene, true, undefined, version)).toBe(true);
    expect(liveOpacity(leaf)).toBe(2);
    loader.committedEnergyFraction = 0.8;
    stampLadderComplete(leaf.userData);
    expect(applyStandaloneLadderFade(scene, true, undefined, version)).toBe(false);
    expect(liveOpacity(leaf)).toBe(2);
    const nextVersion = getLadderEnergyStampVersion();
    expect(applyStandaloneLadderFade(scene, true, undefined, nextVersion)).toBe(true);
    expect(liveOpacity(leaf)).toBeCloseTo(1.25);
    (leaf.material as unknown as FadeMatStub).userData.blendingMode = 'max';
    invalidateStandaloneLadderFade();
    expect(applyStandaloneLadderFade(scene, true, undefined, nextVersion)).toBe(true);
    expect(liveOpacity(leaf)).toBe(1);
    (leaf.material as unknown as FadeMatStub).userData.blendingMode = 'additive';
    invalidateStandaloneLadderFade();
    expect(applyStandaloneLadderFade(scene, true, undefined, nextVersion)).toBe(true);
    expect(liveOpacity(leaf)).toBeCloseTo(1.25);
    expect(applyStandaloneLadderFade(scene, false, undefined, nextVersion)).toBe(true);
    expect(liveOpacity(leaf)).toBe(1);
  });

  it('brightens a committed prefix, follows deeper commits, and restores full brightness', () => {
    const scene = new THREE.Group();
    const leaf = leafMesh('additive');
    leaf.userData.committedEnergyFraction = 0.5;
    scene.add(leaf);
    expect(applyStandaloneLadderFade(scene, true)).toBe(true);
    expect(liveOpacity(leaf)).toBeCloseTo(2);
    leaf.userData.committedEnergyFraction = 0.8;
    expect(applyStandaloneLadderFade(scene, true)).toBe(true);
    expect(liveOpacity(leaf)).toBeCloseTo(1.25);
    leaf.userData.committedEnergyFraction = 1;
    expect(applyStandaloneLadderFade(scene, true)).toBe(true);
    expect(liveOpacity(leaf)).toBe(1);
    expect(applyStandaloneLadderFade(scene, true)).toBe(false);
  });

  it('prepares a hidden leaf so showing it needs no new energy stamp', () => {
    const scene = new THREE.Group();
    const leaf = leafMesh('additive');
    leaf.visible = false;
    leaf.userData.loader = { committedEnergyFraction: 0.5 };
    scene.add(leaf);
    stampLadderComplete(leaf.userData);
    const version = getLadderEnergyStampVersion();
    expect(applyStandaloneLadderFade(scene, true, undefined, version)).toBe(true);
    leaf.visible = true;
    expect(applyStandaloneLadderFade(scene, true, undefined, version)).toBe(false);
    expect(liveOpacity(leaf)).toBe(2);
  });

  it('leaves registry-owned LOD children to their dissolve, and honors the opt-out', () => {
    const scene = new THREE.Group();
    const lod = new THREE.Group();
    lod.userData.kind = 'lod';
    const owned = leafMesh('additive');
    owned.userData.committedEnergyFraction = 0.5;
    lod.add(owned);
    const standalone = leafMesh('additive');
    standalone.userData.committedEnergyFraction = 0.5;
    scene.add(lod, standalone);
    applyStandaloneLadderFade(scene, false);
    expect(liveOpacity(owned)).toBe(1);
    expect(liveOpacity(standalone)).toBe(1);
    applyStandaloneLadderFade(scene, true);
    expect(liveOpacity(owned)).toBe(1);
    expect(liveOpacity(standalone)).toBe(2);
    applyStandaloneLadderFade(scene, false);
    expect(liveOpacity(standalone)).toBe(1);
  });
});

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

  it('group subtree: mixed blendable families are tolerated for compatibility', () => {
    const group = new THREE.Group();
    group.add(leafMesh('additive'), leafMesh('volumetric'), leafMesh('luminous'));
    // Continuous and endpoint-exact, but intentionally not described as sharing
    // one summed-energy or optical-depth conservation model mid-fade.
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

  it('density-guard keep multiplies by 1/keep on blendable leaves only', () => {
    const additive = leafMesh('additive');
    additive.userData.densityKeep = 1 / 8;
    applyLodFade(additive, null, false);
    expect(liveOpacity(additive)).toBeCloseTo(8, 6);
    // Restored once the guard writes keep = 1 back.
    additive.userData.densityKeep = 1;
    applyLodFade(additive, null, false);
    expect(liveOpacity(additive)).toBe(1);
    // Composes multiplicatively with the streaming energy term and the
    // cross-fade weight (independent opacity factors).
    additive.userData.densityKeep = 1 / 4;
    additive.userData.committedEnergyFraction = 0.5;
    applyLodFade(additive, 0.5, true);
    expect(liveOpacity(additive)).toBeCloseTo(0.5 * 2 * 4, 6);
    // A max leaf is never thinned by the guard; a stale stamp must not brighten it.
    const max = leafMesh('max');
    max.userData.densityKeep = 1 / 8;
    applyLodFade(max, null, true);
    expect(liveOpacity(max)).toBe(1);
  });

  it('densityCompensation / isBlendableMode helpers', () => {
    expect(densityCompensation(undefined)).toBe(1);
    expect(densityCompensation(1)).toBe(1);
    expect(densityCompensation(0)).toBe(1);
    expect(densityCompensation(NaN)).toBe(1);
    expect(densityCompensation(0.25)).toBe(4);
    expect(isBlendableMode('volumetric')).toBe(true);
    expect(isBlendableMode('normal')).toBe(false);
    expect(isBlendableMode(undefined)).toBe(false);
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

describe('applyLodFade — drawn-state change report (render-on-change)', () => {
  it('reports true when it writes a new opacity, false on the idempotent rewrite', () => {
    const mesh = leafMesh('additive');
    expect(applyLodFade(mesh, 0.5, false)).toBe(true);
    expect(liveOpacity(mesh)).toBeCloseTo(0.5, 9);
    // The same weight next frame changes nothing drawn.
    expect(applyLodFade(mesh, 0.5, false)).toBe(false);
    expect(applyLodFade(mesh, 0.25, false)).toBe(true);
  });

  it('reports true when it restores a faded opacity, false when there is nothing to restore', () => {
    const mesh = leafMesh('additive');
    applyLodFade(mesh, 0.5, false);
    expect(applyLodFade(mesh, null, false)).toBe(true);
    expect(liveOpacity(mesh)).toBe(1);
    expect(applyLodFade(mesh, null, false)).toBe(false);
  });
});
