/**
 * `uDensityDrop` plumbing: the duck-typed uniform helper, the GLSL hash
 * helper's presence in the shared ordering snippet, and the invariant that
 * every leaf material (visual + picking, GLSL + TSL) declares the uniform so
 * the guard and the pick pass can find it.
 */
import { describe, expect, it } from 'vitest';

import {
  DENSITY_DROP_UNIFORM,
  getDensityDrop,
  hasDensityDrop,
  setDensityDrop,
} from '../../../../rendering/materials/_shared/density-drop';
import { GLSL_SORTED_INDEX } from '../../../../rendering/materials/_shared/glsl-lib';
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { PointPickingMaterial } from '../../../../rendering/picking/point/material';
import { LinePickingMaterial } from '../../../../rendering/picking/line/material';
import { GSplatPickingMaterial } from '../../../../rendering/picking/gsplat/material';

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

describe('every GLSL leaf material declares uDensityDrop at 0', () => {
  it.each([
    ['PointMaterial', () => new PointMaterial({})],
    ['LineMaterial', () => new LineMaterial({})],
    ['GSplatMaterial', () => new GSplatMaterial({})],
    ['PointPickingMaterial', () => new PointPickingMaterial({ nodeId: 1 })],
    ['LinePickingMaterial', () => new LinePickingMaterial({ nodeId: 1 })],
    ['GSplatPickingMaterial', () => new GSplatPickingMaterial({ nodeId: 1 })],
  ])('%s', (_name, make) => {
    const mat = make() as unknown as { uniforms: Record<string, { value: unknown }> };
    expect(mat.uniforms[DENSITY_DROP_UNIFORM]).toEqual({ value: 0 });
    expect(setDensityDrop(mat, 0.25)).toBe(true);
    expect(getDensityDrop(mat)).toBe(0.25);
  });

  it('picking clones carry the drop value', () => {
    for (const make of [
      () => new PointPickingMaterial({ nodeId: 1 }),
      () => new LinePickingMaterial({ nodeId: 1 }),
      () => new GSplatPickingMaterial({ nodeId: 1 }),
    ]) {
      const src = make();
      setDensityDrop(src, 0.75);
      expect(getDensityDrop(src.clone())).toBe(0.75);
    }
  });
});
