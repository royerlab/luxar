/**
 * Every visual material's clone carries the runtime uniforms of its type's
 * shared list (`_shared/runtime-uniforms.ts`) — both twins, one list — and
 * `copyRuntimeUniforms` copies vectors into the clone's own instance.
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import {
  copyRuntimeUniforms,
  GSPLAT_RUNTIME_UNIFORMS,
  LINE_RUNTIME_UNIFORMS,
  MESH_RUNTIME_UNIFORMS,
  POINT_RUNTIME_UNIFORMS,
} from '../../../../rendering/materials/_shared/runtime-uniforms';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { MeshMaterial } from '../../../../rendering/materials/mesh/material-glsl';
import { MeshTSLMaterial } from '../../../../rendering/materials/mesh/material-tsl';
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';
import { PointTSLMaterial } from '../../../../rendering/materials/point/material-tsl';

type Uniforms = { uniforms: Record<string, { value: unknown }>; clone(): Uniforms };

describe('copyRuntimeUniforms', () => {
  it('assigns scalars, copies vectors into the target instance, skips missing names', () => {
    const own = new THREE.Vector2(1, 1);
    const from = { uniforms: { a: { value: 3 }, v: { value: new THREE.Vector2(4, 5) } } };
    const to = { uniforms: { a: { value: 0 }, v: { value: own } } };
    copyRuntimeUniforms(from, to, ['a', 'v', 'missing']);
    expect(to.uniforms.a.value).toBe(3);
    expect(to.uniforms.v.value).toBe(own);
    expect(own.toArray()).toEqual([4, 5]);
  });
});

describe.each([
  ['GSplatMaterial', () => new GSplatMaterial({}), GSPLAT_RUNTIME_UNIFORMS],
  ['GSplatTSLMaterial', () => new GSplatTSLMaterial({}), GSPLAT_RUNTIME_UNIFORMS],
  ['PointMaterial', () => new PointMaterial({}), POINT_RUNTIME_UNIFORMS],
  ['PointTSLMaterial', () => new PointTSLMaterial({}), POINT_RUNTIME_UNIFORMS],
  ['LineMaterial', () => new LineMaterial({}), LINE_RUNTIME_UNIFORMS],
  ['LineTSLMaterial', () => new LineTSLMaterial({}), LINE_RUNTIME_UNIFORMS],
  ['MeshMaterial', () => new MeshMaterial({}), MESH_RUNTIME_UNIFORMS],
  ['MeshTSLMaterial', () => new MeshTSLMaterial({}), MESH_RUNTIME_UNIFORMS],
])('%s.clone() carries its runtime uniforms', (_n, make, names) => {
  it('every listed uniform it declares', () => {
    const mat = make() as unknown as Uniforms;
    const expected: Record<string, unknown> = {};
    let n = 0;
    for (const name of names) {
      const u = mat.uniforms[name];
      if (!u) continue;
      n++;
      if (u.value instanceof THREE.Vector2) {
        u.value.set(123 + n, 45 + n);
        expected[name] = [123 + n, 45 + n];
      } else if (name === 'uSortedIndexSlot') {
        u.value = 1; // 0/1 flag
        expected[name] = 1;
      } else {
        u.value = 0.25 + n / 64;
        expected[name] = u.value;
      }
    }
    expect(n).toBeGreaterThan(0);
    const cloned = mat.clone();
    for (const [name, value] of Object.entries(expected)) {
      const got = cloned.uniforms[name].value;
      expect(got instanceof THREE.Vector2 ? got.toArray() : got, name).toEqual(value);
    }
  });
});
