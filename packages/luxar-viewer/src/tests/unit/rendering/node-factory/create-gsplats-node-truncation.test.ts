/**
 * End-to-end contract: a hostile `truncation_radius` attr can never produce a
 * degenerate uniform or cull box.
 *
 * The attr arrives unvalidated from dataset attrs and sets both the kernel
 * support and the `1/(1-C)` normalization, so 0/NaN/negative must not survive
 * to the GPU. TWO layers enforce this and the test deliberately pins the
 * observable outcome rather than either layer:
 *
 *   1. `createGSplatsNode` clamps at the single earliest read of the attr, so
 *      the value is sanitized once at the boundary.
 *   2. the material constructors clamp again (`material-glsl` / `material-tsl`).
 *
 * Layer 2 alone is currently sufficient — every downstream consumer
 * (`gsplat-geometry`, `gsplats-adapter`, `data-processor-gsplats`,
 * `commit-gsplats-geometry`) sizes itself from `material.uniforms.uTruncate`,
 * which is already clamped — so layer 1 is defence in depth, not a live bug
 * fix. It matters if a future consumer reads `attrs.truncation_radius`
 * directly, or a material is constructed without going through its clamp.
 * Pinning the outcome keeps the contract true whichever layer provides it.
 */

import * as THREE from 'three';
import { beforeEach, describe, expect, it } from 'vitest';

import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../../../config/constants';
import { MIN_TRUNCATION_RADIUS } from '../../../../rendering/materials/gsplat/math';
import { createGSplatsNode } from '../../../../rendering/node-factory/create-gsplats-node';
import { resetElementTextureLayoutForTests } from '../../../../rendering/element-texture-layout';
import type { GSplatsMetadata, GSplatsDataLoader } from '../../../../types/gsplats';
import type { InstancedGSplatsMeshConfig } from '../../../../rendering/gsplat-geometry';

/** One splat at the origin with unit Cholesky (maxRowNorm = 1). */
function meshConfig(): InstancedGSplatsMeshConfig {
  return {
    centers: new Float32Array([0, 0, 0]),
    choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
    amplitudes: new Float32Array([1]),
    colors: new Float32Array([1, 1, 1]),
    splatCount: 1,
  };
}

const loader = {} as GSplatsDataLoader;

function build(truncation_radius: unknown): THREE.Mesh {
  return createGSplatsNode(
    '/gs',
    {},
    { truncation_radius } as unknown as GSplatsMetadata,
    meshConfig(),
    loader,
    null
  );
}

function uTruncateOf(mesh: THREE.Mesh): number {
  // `THREE.Mesh.material` is `Material | Material[]`, so the cast has to go
  // through `unknown`; these meshes always carry a single GSplatMaterial.
  const material = mesh.material as unknown as { uniforms: { uTruncate: { value: number } } };
  return material.uniforms.uTruncate.value;
}

describe('createGSplatsNode — truncation_radius sanitization', () => {
  beforeEach(() => {
    resetElementTextureLayoutForTests();
  });

  it('uses the shared default when the attr is absent', () => {
    const mesh = createGSplatsNode(
      '/gs',
      {},
      {} as unknown as GSplatsMetadata,
      meshConfig(),
      loader,
      null
    );
    expect(uTruncateOf(mesh)).toBe(GSPLAT_DEFAULT_TRUNCATION_RADIUS);
  });

  it('passes a legitimate value through untouched', () => {
    expect(uTruncateOf(build(2.75))).toBe(2.75);
    expect(uTruncateOf(build(3.0))).toBe(3.0);
    // Small but float32-valid — the write-side validator accepts these, and
    // the on-disk chunk bounds are computed from them, so the viewer must
    // render at the stored value rather than silently raising it.
    expect(uTruncateOf(build(0.05))).toBe(0.05);
  });

  it.each([
    ['zero', 0],
    ['negative', -1],
    ['sub-float32-minimum', 1e-5],
  ])('clamps a %s radius to the minimum', (_label, value) => {
    expect(uTruncateOf(build(value))).toBe(MIN_TRUNCATION_RADIUS);
  });

  it.each([
    ['NaN', Number.NaN],
    ['Infinity', Number.POSITIVE_INFINITY],
    // Finite in float64, Infinity once narrowed to the float32 GPU uniform —
    // the hostile-attr case the write-side validator rejects with
    // MAX_TRUNCATION_RADIUS_FLOAT32; the read-side clamp mirrors it.
    ['above-float32-max (3.5e38)', 3.5e38],
    ['float64-only-finite (1e308)', 1e308],
  ])('falls back to the default for a %s radius', (_label, value) => {
    expect(uTruncateOf(build(value))).toBe(GSPLAT_DEFAULT_TRUNCATION_RADIUS);
  });

  it.each([
    ['zero', 0],
    ['NaN', Number.NaN],
    ['negative', -1],
  ])('keeps the cull box finite and non-degenerate for a %s radius', (_label, value) => {
    // The bounding box is sized from the resolved radius, so an unsanitized
    // value would collapse it (0) or NaN-poison it — and a NaN box silently
    // breaks frustum culling rather than erroring.
    const box = build(value).geometry.boundingBox!;
    expect(box).not.toBeNull();
    for (const v of [box.min.x, box.min.y, box.min.z, box.max.x, box.max.y, box.max.z]) {
      expect(Number.isFinite(v)).toBe(true);
    }
    expect(box.max.x - box.min.x).toBeGreaterThan(0);
  });
});
