/**
 * Density guard — the keep-fraction ladder (pure) and the per-node
 * controller driven by the tracker's visit hook.
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import {
  DensityGuard,
  getDensityGuard,
  nextKeepFraction,
  targetKeepFraction,
  type DensityLadderConfig,
} from '../../../scene/density-guard';
import type { NodeDensity } from '../../../scene/projected-density';

const CFG: DensityLadderConfig = {
  capElementsPerPixel: 4,
  minKeepFraction: 1 / 64,
  enterRatio: 1.5,
  leaveRatio: 0.75,
};

describe('targetKeepFraction', () => {
  it('is 1 at or below the cap', () => {
    expect(targetKeepFraction(0, CFG)).toBe(1);
    expect(targetKeepFraction(4, CFG)).toBe(1);
    expect(targetKeepFraction(NaN, CFG)).toBe(1);
  });

  it('is the power of 1/2 that brings the density under the cap', () => {
    expect(targetKeepFraction(5, CFG)).toBe(1 / 2);
    expect(targetKeepFraction(8, CFG)).toBe(1 / 2);
    expect(targetKeepFraction(8.1, CFG)).toBe(1 / 4);
    expect(targetKeepFraction(100, CFG)).toBe(1 / 32);
  });

  it('floors at minKeepFraction', () => {
    expect(targetKeepFraction(539, CFG)).toBe(1 / 64);
    expect(targetKeepFraction(1e6, CFG)).toBe(1 / 64);
  });
});

describe('nextKeepFraction — hysteresis band', () => {
  it('holds at 1 until the effective density passes cap × enterRatio', () => {
    expect(nextKeepFraction(1, 5, CFG)).toBe(1); // target 1/2 but 5 < 6
    expect(nextKeepFraction(1, 6.5, CFG)).toBe(1 / 2);
  });

  it('jumps several steps at once from a cold start', () => {
    expect(nextKeepFraction(1, 539, CFG)).toBe(1 / 64);
  });

  it('holds a thinned step while the effective density stays inside the band', () => {
    // keep 1/2: effective 2.75 (> 3 = cap·leave? no: 2.75 < 3 → would restore
    // only if the target were coarser than keep, and target(5.5) = 1/2 = keep).
    expect(nextKeepFraction(1 / 2, 5.5, CFG)).toBe(1 / 2);
    // keep 1/2, density 7: effective 3.5, inside [3, 6] → hold.
    expect(nextKeepFraction(1 / 2, 7, CFG)).toBe(1 / 2);
  });

  it('restores only below cap × leaveRatio', () => {
    // keep 1/2, density 4.5: still above the cap unthinned, so the target IS
    // 1/2 → hold (effective 2.25 is under the cap, as intended).
    expect(nextKeepFraction(1 / 2, 4.5, CFG)).toBe(1 / 2);
    // keep 1/2, density 3.9: target 1 (3.9 ≤ 4), effective 1.95 < 3 → restore.
    expect(nextKeepFraction(1 / 2, 3.9, CFG)).toBe(1);
    // keep 1/4, density 7: target 1/2 (3.5 ≤ 4), effective 1.75 < 3 → step up one.
    expect(nextKeepFraction(1 / 4, 7, CFG)).toBe(1 / 2);
    // keep 1/4, density 10: target stays 1/4 (1/2 would give 5 > cap) → hold.
    expect(nextKeepFraction(1 / 4, 10, CFG)).toBe(1 / 4);
  });

  it('off-screen (zero density) restores fully when asked', () => {
    expect(nextKeepFraction(1 / 64, 0, CFG)).toBe(1);
  });
});

interface StubMat {
  uniforms: { uDensityDrop: { value: number }; uOpacity: { value: number } };
  userData: { blendingMode?: string };
  updateOpacity(v: number): void;
  getOpacity(): number;
}
function stubMaterial(mode: string, withUniform = true): StubMat {
  const uniforms = { uOpacity: { value: 1 } } as StubMat['uniforms'];
  if (withUniform) uniforms.uDensityDrop = { value: 0 };
  return {
    uniforms,
    userData: { blendingMode: mode },
    updateOpacity(v) {
      this.uniforms.uOpacity.value = v;
    },
    getOpacity() {
      return this.uniforms.uOpacity.value;
    },
  };
}
function leaf(mode: string, withUniform = true): { mesh: THREE.Mesh; mat: StubMat } {
  const mat = stubMaterial(mode, withUniform);
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), mat as unknown as THREE.Material);
  mesh.name = 'points/dense';
  mesh.userData._layerMaterialCloned = true;
  return { mesh, mat };
}
function record(elementsPerPixel: number, onScreen = true): NodeDensity {
  return {
    path: 'points/dense',
    areaPx: 1000,
    elements: elementsPerPixel * 1000,
    elementsPerPixel,
    onScreen,
    frame: 1,
    keep: 1,
    blendable: true,
  };
}
function guard(energyComp = false): { g: DensityGuard; registered: THREE.Material[] } {
  const registered: THREE.Material[] = [];
  const g = new DensityGuard();
  g.configure({
    config: () => CFG,
    energyComp: () => energyComp,
    registerMaterial: (m) => registered.push(m),
  });
  return { g, registered };
}

describe('DensityGuard.observe', () => {
  it('thins an over-dense additive node: uniform, userData, brightness, changed flag', () => {
    const { g } = guard();
    const { mesh, mat } = leaf('additive');
    const rec = record(539);
    g.observe(mesh, rec);
    expect(mesh.userData.densityKeep).toBe(1 / 64);
    expect(rec.keep).toBe(1 / 64);
    expect(mat.uniforms.uDensityDrop.value).toBeCloseTo(1 - 1 / 64, 12);
    expect(mat.getOpacity()).toBeCloseTo(64, 9);
    expect(g.takeChanged()).toBe(true);
    // Second frame at the same density: nothing changes, flag stays clear.
    g.observe(mesh, rec);
    expect(g.takeChanged()).toBe(false);
  });

  it('restores when the node is zoomed in (density under the leave threshold)', () => {
    const { g } = guard();
    const { mesh, mat } = leaf('luminous');
    g.observe(mesh, record(539));
    g.takeChanged();
    g.observe(mesh, record(1));
    expect(mesh.userData.densityKeep).toBe(1);
    expect(mat.uniforms.uDensityDrop.value).toBe(0);
    expect(mat.getOpacity()).toBe(1);
    expect(g.takeChanged()).toBe(true);
  });

  it('never thins non-blendable modes, and undoes a stale step on a mode switch', () => {
    const { g } = guard();
    const { mesh, mat } = leaf('volumetric');
    g.observe(mesh, record(100));
    expect(mesh.userData.densityKeep).toBe(1 / 32);
    g.takeChanged();
    // Layers panel switches the node to max projection.
    mat.userData.blendingMode = 'max';
    g.observe(mesh, record(100));
    expect(mesh.userData.densityKeep).toBe(1);
    expect(mat.uniforms.uDensityDrop.value).toBe(0);
    expect(g.takeChanged()).toBe(true);
    // Stays at 1 no matter the density.
    g.observe(mesh, record(1e5));
    expect(mesh.userData.densityKeep).toBe(1);
    expect(g.takeChanged()).toBe(false);
  });

  it('holds the current step while off-screen', () => {
    const { g } = guard();
    const { mesh, mat } = leaf('additive');
    g.observe(mesh, record(539));
    g.takeChanged();
    g.observe(mesh, record(0, false));
    expect(mesh.userData.densityKeep).toBe(1 / 64);
    expect(mat.uniforms.uDensityDrop.value).toBeCloseTo(1 - 1 / 64, 12);
    expect(g.takeChanged()).toBe(false);
  });

  it('re-asserts the uniform when a rebuilt material lost it (drift)', () => {
    const { g } = guard();
    const { mesh, mat } = leaf('additive');
    g.observe(mesh, record(539));
    g.takeChanged();
    mat.uniforms.uDensityDrop.value = 0; // e.g. a material rebuilt from defaults
    g.observe(mesh, record(539));
    expect(mat.uniforms.uDensityDrop.value).toBeCloseTo(1 - 1 / 64, 12);
    expect(g.takeChanged()).toBe(true);
  });

  it('composes with the streaming energy term when energyComp is on', () => {
    const { g } = guard(true);
    const { mesh, mat } = leaf('additive');
    mesh.userData.committedEnergyFraction = 0.5;
    g.observe(mesh, record(6.5)); // → keep 1/2
    expect(mesh.userData.densityKeep).toBe(1 / 2);
    expect(mat.getOpacity()).toBeCloseTo(2 * 2, 9);
  });

  it('is a no-op on a material without the uniform, an array material, or before configure', () => {
    const { g } = guard();
    const { mesh, mat } = leaf('additive', false);
    g.observe(mesh, record(539));
    expect(mesh.userData.densityKeep).toBeUndefined();
    expect(mat.getOpacity()).toBe(1);
    expect(g.takeChanged()).toBe(false);

    const arr = new THREE.Mesh(new THREE.BufferGeometry(), [new THREE.MeshBasicMaterial()]);
    g.observe(arr, record(539));
    expect(arr.userData.densityKeep).toBeUndefined();

    const unconfigured = new DensityGuard();
    const { mesh: m2 } = leaf('additive');
    unconfigured.observe(m2, record(539));
    expect(m2.userData.densityKeep).toBeUndefined();
    expect(unconfigured.takeChanged()).toBe(false);
  });

  it('keepOf reads the material uniform; the module singleton is stable', () => {
    const { mesh, mat } = leaf('additive');
    mat.uniforms.uDensityDrop.value = 0.75;
    expect(DensityGuard.keepOf(mesh)).toBeCloseTo(0.25, 12);
    expect(getDensityGuard()).toBe(getDensityGuard());
  });
});
