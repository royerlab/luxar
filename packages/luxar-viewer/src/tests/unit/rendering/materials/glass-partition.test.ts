/**
 * `uGlassPartition` / `uGlassDepth` plumbing (spec MESH_PHYSICAL_MATERIALS §3.4 Phase
 * 3): the duck-typed uniform helper, the one shared depth texture, the GLSL guard's
 * shape, and the invariant that every VISUAL data material on both backends declares
 * the pair — while no PICK material does (picking must see all the data).
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
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { MeshMaterial } from '../../../../rendering/materials/mesh/material-glsl';
import { PointTSLMaterial } from '../../../../rendering/materials/point/material-tsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';
import { MeshTSLMaterial } from '../../../../rendering/materials/mesh/material-tsl';
import { PointPickingMaterial } from '../../../../rendering/picking/point/material';
import { LinePickingMaterial } from '../../../../rendering/picking/line/material';
import { GSplatPickingMaterial } from '../../../../rendering/picking/gsplat/material';
import { MeshPickingMaterial } from '../../../../rendering/picking/mesh/material';
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

  it.each([
    ['point', POINT_FRAGMENT_SHADER],
    ['line', LINE_FRAGMENT_SHADER],
    ['line-capsule', CAPSULE_LINE_FRAGMENT_SHADER],
    ['gsplat', GSPLAT_FRAGMENT_SHADER],
    ['mesh', MESH_FRAGMENT_SHADER],
  ])(
    '%s fragment shader carries the uniforms and runs the guard before its first discard',
    (_n, src) => {
      expect(src).toContain('uniform int uGlassPartition;');
      expect(src).toContain('uniform sampler2D uGlassDepth;');
      const guardAt = src.indexOf('if (uGlassPartition != 0) {');
      const mainAt = src.indexOf('void main() {');
      expect(guardAt).toBeGreaterThan(mainAt);
      // Nothing else discards, computes or samples before the classification.
      const body = src.slice(mainAt, guardAt);
      expect(body).not.toContain('discard');
      expect(body).not.toContain('texture(');
    }
  );

  it('the pick shaders know nothing of the partition (picking must see all the data)', () => {
    expect(POINT_PICK_FRAGMENT_SHADER).not.toContain('uGlassPartition');
  });
});

describe('every visual data material declares the pair, bound to the shared texture', () => {
  afterEach(() => resetGlassDepthTextureForTests());

  it.each([
    ['PointMaterial', () => new PointMaterial({})],
    ['LineMaterial', () => new LineMaterial({})],
    ['GSplatMaterial', () => new GSplatMaterial({})],
    ['MeshMaterial', () => new MeshMaterial({})],
    ['PointTSLMaterial', () => new PointTSLMaterial({})],
    ['LineTSLMaterial', () => new LineTSLMaterial({})],
    ['GSplatTSLMaterial', () => new GSplatTSLMaterial({})],
    ['MeshTSLMaterial', () => new MeshTSLMaterial({})],
  ])('%s', (_n, make) => {
    const mat = make() as unknown as { uniforms: Record<string, { value: unknown }> };
    expect(hasGlassPartition(mat)).toBe(true);
    expect(getGlassPartition(mat)).toBe(0);
    expect(mat.uniforms[GLASS_DEPTH_UNIFORM].value).toBe(getGlassDepthTexture());
    expect(setGlassPartition(mat, GLASS_PARTITION_FRONT)).toBe(true);
    expect(mat.uniforms[GLASS_PARTITION_UNIFORM].value).toBe(2);
  });

  it.each([
    ['PointPickingMaterial', () => new PointPickingMaterial({ nodeId: 1 })],
    ['LinePickingMaterial', () => new LinePickingMaterial({ nodeId: 1 })],
    ['GSplatPickingMaterial', () => new GSplatPickingMaterial({ nodeId: 1 })],
    ['MeshPickingMaterial', () => new MeshPickingMaterial({ nodeId: 1 })],
  ])('%s does NOT (a broadcast no-ops on it)', (_n, make) => {
    const mat = make();
    expect(hasGlassPartition(mat)).toBe(false);
    expect(setGlassPartition(mat, GLASS_PARTITION_BEHIND)).toBe(false);
  });
});
