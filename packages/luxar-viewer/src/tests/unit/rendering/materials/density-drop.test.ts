/**
 * `uDensityDrop` plumbing: the duck-typed uniform helper, the GLSL hash
 * helper's presence in the shared ordering snippet, and the geometry-behaviour
 * matrix row `densityGuard`: every leaf material (visual + picking, GLSL + TSL,
 * every primitive variant) of a type that thins declares the uniform, so the
 * guard and the pick pass can find it, and the real guard thins an over-dense
 * node through it. Mesh declares the row absent: no mesh material carries the
 * uniform and the guard leaves a mesh untouched.
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import {
  DENSITY_DROP_UNIFORM,
  getDensityDrop,
  hasDensityDrop,
  setDensityDrop,
} from '../../../../rendering/materials/_shared/density-drop';
import { GLSL_SORTED_INDEX } from '../../../../rendering/materials/_shared/glsl-lib';
import { DensityGuard } from '../../../../scene/density-guard';
import type { GeometryTypeName } from '../../../../types/format-contract';
import { allMaterialsOf, GEOMETRY_MATERIAL_VARIANTS } from '../../../helpers/geometry-materials';
import { defineBehaviourConformance } from '../../../_conformance/define-behaviour-conformance';

describe('density-drop helper', () => {
  it('reads 0 when the material has no uniform record or a non-numeric value', () => {
    expect(hasDensityDrop(undefined)).toBe(false);
    expect(hasDensityDrop({ uniforms: {} })).toBe(false);
    expect(getDensityDrop(undefined)).toBe(0);
    expect(getDensityDrop({})).toBe(0);
    expect(getDensityDrop({ uniforms: {} })).toBe(0);
    expect(getDensityDrop({ uniforms: { uDensityDrop: { value: 'x' } } })).toBe(0);
    expect(getDensityDrop({ uniforms: { uDensityDrop: { value: NaN } } })).toBe(0);
  });

  it('writes a clamped value and reports whether it changed', () => {
    const mat = { uniforms: { [DENSITY_DROP_UNIFORM]: { value: 0 } } };
    expect(hasDensityDrop(mat)).toBe(true);
    expect(setDensityDrop(mat, 0.5)).toBe(true);
    expect(getDensityDrop(mat)).toBe(0.5);
    expect(setDensityDrop(mat, 0.5)).toBe(false);
    expect(setDensityDrop(mat, -1)).toBe(true);
    expect(getDensityDrop(mat)).toBe(0);
    // Never 1: a drop of 1 would discard EVERY element, including the one
    // whose hash lands exactly on 0.
    expect(setDensityDrop(mat, 1)).toBe(true);
    expect(getDensityDrop(mat)).toBeLessThan(1);
    expect(setDensityDrop({ uniforms: {} }, 0.5)).toBe(false);
  });
});

describe('GLSL_SORTED_INDEX density-drop helper', () => {
  it('declares the uniform and a predicate over the hashed storage index', () => {
    expect(GLSL_SORTED_INDEX).toContain('uniform float uDensityDrop;');
    expect(GLSL_SORTED_INDEX).toContain('bool luxarDensityDropped()');
    // Hashes the ORDERING-resolved index so the visual and pick passes agree
    // per element irrespective of draw slot.
    expect(GLSL_SORTED_INDEX).toMatch(/luxarDensityDropped\(\)[\s\S]*luxarSortedIndex\(\)/);
    // Fast exit when the guard is idle: no hash work at drop 0.
    expect(GLSL_SORTED_INDEX).toContain('if (uDensityDrop <= 0.0) return false;');
  });
});

type Uniforms = { uniforms?: Record<string, { value: unknown }> };

/**
 * Run the real guard over one over-dense frame of `type`'s GLSL visual
 * material in `additive` (every variant); returns the nodes it observed.
 */
function observeOverDense(type: GeometryTypeName): THREE.Mesh[] {
  const guard = new DensityGuard();
  guard.configure({
    config: () => ({
      capElementsPerPixel: 4,
      minKeepFraction: 1 / 64,
      enterRatio: 1.5,
      leaveRatio: 0.75,
    }),
    energyComp: () => false,
  });
  return GEOMETRY_MATERIAL_VARIANTS[type].map(({ visual }) => {
    const material = visual.glsl('additive');
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    mesh.userData._layerMaterialCloned = true;
    guard.observe(mesh, {
      path: '/dense',
      areaPx: 1000,
      elements: 539_000,
      elementsPerPixel: 539,
      onScreen: true,
      frame: 1,
      keep: 1,
      blendable: true,
    });
    return mesh;
  });
}

defineBehaviourConformance('densityGuard', {
  holds(type) {
    for (const [label, mat] of allMaterialsOf(type)) {
      expect((mat as Uniforms).uniforms?.[DENSITY_DROP_UNIFORM], label).toEqual({ value: 0 });
      expect(setDensityDrop(mat, 0.25), label).toBe(true);
      expect(getDensityDrop(mat), label).toBe(0.25);
      // A pick material is cloned per pick render target; the clone keeps the drop.
      if (label.includes('pick glsl')) expect(getDensityDrop(mat.clone()), label).toBe(0.25);
    }
    for (const observed of observeOverDense(type)) {
      expect(getDensityDrop(observed.material)).toBeCloseTo(1 - 1 / 64, 12);
      expect(observed.userData.densityKeep).toBe(1 / 64);
    }
  },
  enforced: {
    'no-op': (type) => {
      for (const [label, mat] of allMaterialsOf(type)) {
        expect(hasDensityDrop(mat), label).toBe(false);
      }
      for (const observed of observeOverDense(type)) {
        expect(getDensityDrop(observed.material)).toBe(0);
        expect(observed.userData.densityKeep).toBeUndefined();
      }
    },
  },
});
