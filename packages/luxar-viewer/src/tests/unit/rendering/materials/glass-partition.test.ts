/**
 * `uGlassPartition` / `uGlassDepth` plumbing (spec MESH_PHYSICAL_MATERIALS §3.4 Phase
 * 3): the duck-typed uniform helper, the one shared depth texture, the GLSL guard's
 * shape, and the geometry-behaviour matrix row `glassPartitionGuard`: every VISUAL
 * data material of every type, variant and backend declares the pair and its fragment
 * shader classifies before its first discard — while no PICK material does (picking
 * must see all the data).
 */
import { afterEach, describe, expect, it } from 'vitest';
import * as THREE from 'three';

import {
  GLASS_DEPTH_UNIFORM,
  GLASS_PARTITION_BEHIND,
  GLASS_PARTITION_FRONT,
  GLASS_PARTITION_OFF,
  GLASS_PARTITION_UNIFORM,
  GLSL_GLASS_PARTITION_GUARD,
  GLSL_GLASS_PARTITION_UNIFORMS,
  getGlassDepthTexture,
  getGlassPartition,
  hasGlassPartition,
  resetGlassDepthTextureForTests,
  setGlassPartition,
} from '../../../../rendering/materials/_shared/glass-partition';
import { POINT_FRAGMENT_SHADER } from '../../../../rendering/materials/point/shader-glsl';
import { LINE_FRAGMENT_SHADER } from '../../../../rendering/materials/line/shader-glsl';
import { CAPSULE_LINE_FRAGMENT_SHADER } from '../../../../rendering/materials/line/shader-glsl-capsule';
import { GSPLAT_FRAGMENT_SHADER } from '../../../../rendering/materials/gsplat/shader-glsl';
import { MESH_FRAGMENT_SHADER } from '../../../../rendering/materials/mesh/shader-glsl';
import type { GeometryTypeName } from '../../../../types/format-contract';
import {
  GEOMETRY_MATERIAL_VARIANTS,
  type PrimitiveVariant,
} from '../../../helpers/geometry-materials';
import { defineBehaviourConformance } from '../../../_conformance/define-behaviour-conformance';
import { POINT_PICK_FRAGMENT_SHADER } from '../../../../rendering/picking/point/shaders';

describe('glass-partition helper', () => {
  afterEach(() => resetGlassDepthTextureForTests());

  it('reads 0 when the material has no uniform record or an unknown value', () => {
    expect(hasGlassPartition(undefined)).toBe(false);
    expect(hasGlassPartition({ uniforms: {} })).toBe(false);
    expect(getGlassPartition(undefined)).toBe(0);
    expect(getGlassPartition({ uniforms: {} })).toBe(0);
    expect(getGlassPartition({ uniforms: { uGlassPartition: { value: 7 } } })).toBe(0);
    expect(getGlassPartition({ uniforms: { uGlassPartition: { value: 'x' } } })).toBe(0);
  });

  it('writes the mode and reports whether it changed; a material without the uniform is left alone', () => {
    const mat = { uniforms: { [GLASS_PARTITION_UNIFORM]: { value: 0 } } };
    expect(hasGlassPartition(mat)).toBe(true);
    expect(setGlassPartition(mat, GLASS_PARTITION_BEHIND)).toBe(true);
    expect(getGlassPartition(mat)).toBe(1);
    expect(setGlassPartition(mat, GLASS_PARTITION_BEHIND)).toBe(false);
    expect(setGlassPartition(mat, GLASS_PARTITION_FRONT)).toBe(true);
    expect(getGlassPartition(mat)).toBe(2);
    expect(setGlassPartition(mat, GLASS_PARTITION_OFF)).toBe(true);
    expect(getGlassPartition(mat)).toBe(0);
    expect(setGlassPartition({ uniforms: {} }, GLASS_PARTITION_FRONT)).toBe(false);
    expect(setGlassPartition(new THREE.MeshBasicMaterial(), GLASS_PARTITION_FRONT)).toBe(false);
  });

  it('hands out ONE depth texture, nearest-filtered, until the test reset', () => {
    const a = getGlassDepthTexture();
    expect(a).toBeInstanceOf(THREE.DepthTexture);
    expect(getGlassDepthTexture()).toBe(a);
    expect(a.minFilter).toBe(THREE.NearestFilter);
    expect(a.magFilter).toBe(THREE.NearestFilter);
    resetGlassDepthTextureForTests();
    expect(getGlassDepthTexture()).not.toBe(a);
  });
});

describe('the GLSL guard', () => {
  it('declares the pair and classifies with the one predicate both passes share', () => {
    expect(GLSL_GLASS_PARTITION_UNIFORMS).toContain('uniform int uGlassPartition;');
    expect(GLSL_GLASS_PARTITION_UNIFORMS).toContain('uniform sampler2D uGlassDepth;');
    // The cleared 1.0 means "no glass here": such a fragment is never "in front", so
    // the behind pass keeps it and the front pass discards it.
    expect(GLSL_GLASS_PARTITION_GUARD).toContain(
      'bool inFrontOfGlass = glassDepth < 1.0 && gl_FragCoord.z < glassDepth;'
    );
    expect(GLSL_GLASS_PARTITION_GUARD).toContain(
      'if (uGlassPartition == 1 && inFrontOfGlass) discard;'
    );
    expect(GLSL_GLASS_PARTITION_GUARD).toContain(
      'if (uGlassPartition == 2 && !inFrontOfGlass) discard;'
    );
    // Exact texel at the fragment's own pixel: no resolution uniform, no filtering.
    expect(GLSL_GLASS_PARTITION_GUARD).toContain(
      'texelFetch(uGlassDepth, ivec2(gl_FragCoord.xy), 0)'
    );
    // Dead at mode 0, so a frame outside the split pays one uniform compare.
    expect(GLSL_GLASS_PARTITION_GUARD.trim().startsWith('if (uGlassPartition != 0) {')).toBe(true);
  });

  it('the pick shaders know nothing of the partition (picking must see all the data)', () => {
    expect(POINT_PICK_FRAGMENT_SHADER).not.toContain('uGlassPartition');
  });
});

/** The GLSL visual fragment shader of each type's primitive variants. */
const VISUAL_FRAGMENT_SHADERS: Record<
  GeometryTypeName,
  Partial<Record<PrimitiveVariant, string>>
> = {
  points: { quad: POINT_FRAGMENT_SHADER },
  lines: { quad: LINE_FRAGMENT_SHADER, capsule: CAPSULE_LINE_FRAGMENT_SHADER },
  gsplats: { quad: GSPLAT_FRAGMENT_SHADER },
  mesh: { triangle: MESH_FRAGMENT_SHADER },
};

type Uniforms = { uniforms: Record<string, { value: unknown }> };

describe('glass partition, per geometry type', () => {
  afterEach(() => resetGlassDepthTextureForTests());

  defineBehaviourConformance('glassPartitionGuard', {
    holds(type) {
      for (const { variant, visual, pick } of GEOMETRY_MATERIAL_VARIANTS[type]) {
        // The fragment shader classifies before it discards, computes or samples.
        const src = VISUAL_FRAGMENT_SHADERS[type][variant];
        expect(src, `${type} ${variant}: no GLSL fragment shader listed`).toBeDefined();
        expect(src).toContain('uniform int uGlassPartition;');
        expect(src).toContain('uniform sampler2D uGlassDepth;');
        const guardAt = src!.indexOf('if (uGlassPartition != 0) {');
        const mainAt = src!.indexOf('void main() {');
        expect(guardAt, `${type} ${variant}: guard inside main`).toBeGreaterThan(mainAt);
        const body = src!.slice(mainAt, guardAt);
        expect(body, `${type} ${variant}: discard before the guard`).not.toContain('discard');
        expect(body, `${type} ${variant}: sample before the guard`).not.toContain('texture(');

        for (const backend of ['glsl', 'tsl'] as const) {
          const label = `${type} ${variant} ${backend}`;
          // Every visual material declares the pair, bound to the one shared texture…
          const mat = visual[backend]() as unknown as Uniforms;
          expect(hasGlassPartition(mat), label).toBe(true);
          expect(getGlassPartition(mat), label).toBe(0);
          expect(mat.uniforms[GLASS_DEPTH_UNIFORM].value, label).toBe(getGlassDepthTexture());
          expect(setGlassPartition(mat, GLASS_PARTITION_FRONT), label).toBe(true);
          expect(mat.uniforms[GLASS_PARTITION_UNIFORM].value, label).toBe(2);
          // …and no pick material does, so a broadcast no-ops on it.
          const pickMat = pick[backend]();
          expect(hasGlassPartition(pickMat), `${label} pick`).toBe(false);
          expect(setGlassPartition(pickMat, GLASS_PARTITION_BEHIND), `${label} pick`).toBe(false);
        }
      }
    },
  });
});
