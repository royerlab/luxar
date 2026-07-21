/**
 * Unit tests for material-colormap-helpers.
 *
 * Covers:
 * - supportsScalarColormap predicate per node type
 * - applyColormapTextureToMaterial(null) clears uniforms
 * - applyScalarRangeToMaterial round-trip
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
  supportsScalarColormap,
} from '../../../rendering/material-colormap-helpers';

function makeMockMaterial(): THREE.ShaderMaterial {
  // Lightweight stand-in: a real ShaderMaterial with the
  // ColormapAwareMaterial setters bolted on. The helper delegates
  // uniform / define mutation to these setters; without them the
  // helper would be a no-op (correct for materials like picking
  // that don't implement the interface).
  const m = new THREE.ShaderMaterial({
    uniforms: {},
    defines: {},
  });
  (
    m as unknown as { setColormapTexture: (t: THREE.DataTexture | null) => void }
  ).setColormapTexture = function setColormapTexture(texture: THREE.DataTexture | null): void {
    if (texture) {
      m.defines.USE_COLORMAP = '';
      if (!m.uniforms.uColormapTex) {
        m.uniforms.uColormapTex = { value: texture };
        m.uniforms.uScalarMin = { value: 0.0 };
        m.uniforms.uScalarScale = { value: 1.0 };
      } else {
        m.uniforms.uColormapTex.value = texture;
      }
    } else {
      delete m.defines.USE_COLORMAP;
      if (m.uniforms.uColormapTex) m.uniforms.uColormapTex.value = null;
      if (m.uniforms.uScalarMin) m.uniforms.uScalarMin.value = 0.0;
      if (m.uniforms.uScalarScale) m.uniforms.uScalarScale.value = 1.0;
    }
  };
  (m as unknown as { setScalarRange: (min: number, max: number) => void }).setScalarRange =
    function setScalarRange(min: number, max: number): void {
      if (m.uniforms.uScalarMin) m.uniforms.uScalarMin.value = min;
      if (m.uniforms.uScalarScale) {
        m.uniforms.uScalarScale.value = 1.0 / Math.max(1e-10, max - min);
      }
    };
  return m;
}

describe('supportsScalarColormap', () => {
  it('returns true for gsplats unconditionally (uses always-present aAmplitude)', () => {
    expect(supportsScalarColormap('gsplats')).toBe(true);
    expect(supportsScalarColormap('gsplats', new THREE.BufferGeometry())).toBe(true);
  });

  it('returns false for points without the hasScalars stamp', () => {
    // The fixed 3-texel point layout always has a scalar slot, so scalar
    // presence is the `userData.hasScalars` stamp set by the texel
    // writers — an unstamped (or false-stamped) geometry fails closed.
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(3, 3));
    expect(supportsScalarColormap('points', g)).toBe(false);
    g.userData.hasScalars = false;
    expect(supportsScalarColormap('points', g)).toBe(false);
  });

  it('returns true for points with the `userData.hasScalars` stamp', () => {
    // Stamped by createPointsGeometry / the pool points adapter when
    // `data.scalars !== undefined` (the same signal that used to bind
    // the aScalar attribute).
    const g = new THREE.BufferGeometry();
    g.userData.hasScalars = true;
    expect(supportsScalarColormap('points', g)).toBe(true);
  });

  it('returns false for points when geometry omitted', () => {
    expect(supportsScalarColormap('points')).toBe(false);
  });

  it('returns false for lines with only one of aStartScalar/aEndScalar', () => {
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('aStartScalar', new THREE.InstancedBufferAttribute(new Float32Array(1), 1));
    expect(supportsScalarColormap('lines', g)).toBe(false);
  });

  it('returns true for lines with both scalar attributes', () => {
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('aStartScalar', new THREE.InstancedBufferAttribute(new Float32Array(1), 1));
    g.setAttribute('aEndScalar', new THREE.InstancedBufferAttribute(new Float32Array(1), 1));
    expect(supportsScalarColormap('lines', g)).toBe(true);
  });
});

describe('applyColormapTextureToMaterial', () => {
  it('enables colormap by setting USE_COLORMAP and creating uniforms', () => {
    const mat = makeMockMaterial();
    const tex = new THREE.DataTexture(new Uint8Array(1024), 256, 1, THREE.RGBAFormat);
    const { wasEnabled, nowEnabled } = applyColormapTextureToMaterial(mat, tex);
    expect(wasEnabled).toBe(false);
    expect(nowEnabled).toBe(true);
    expect(mat.defines.USE_COLORMAP).toBe('');
    expect(mat.uniforms.uColormapTex.value).toBe(tex);
    expect(mat.uniforms.uScalarMin.value).toBe(0.0);
    expect(mat.uniforms.uScalarScale.value).toBe(1.0);
  });

  it('disabling clears define + uColormapTex.value + scalar uniforms + userData.scalarRange', () => {
    const mat = makeMockMaterial();
    const tex = new THREE.DataTexture(new Uint8Array(1024), 256, 1, THREE.RGBAFormat);
    applyColormapTextureToMaterial(mat, tex);
    applyScalarRangeToMaterial(mat, 5, 15);
    expect(mat.userData.scalarRange).toEqual([5, 15]);

    const { wasEnabled, nowEnabled } = applyColormapTextureToMaterial(mat, null);
    expect(wasEnabled).toBe(true);
    expect(nowEnabled).toBe(false);
    expect(mat.defines.USE_COLORMAP).toBeUndefined();
    expect(mat.uniforms.uColormapTex.value).toBeNull();
    expect(mat.uniforms.uScalarMin.value).toBe(0.0);
    expect(mat.uniforms.uScalarScale.value).toBe(1.0);
    expect(mat.userData.scalarRange).toBeUndefined();
  });
});

describe('applyScalarRangeToMaterial', () => {
  it('writes uScalarMin and 1/(max-min) to uScalarScale', () => {
    const mat = makeMockMaterial();
    applyColormapTextureToMaterial(
      mat,
      new THREE.DataTexture(new Uint8Array(1024), 256, 1, THREE.RGBAFormat)
    );
    applyScalarRangeToMaterial(mat, 0.5, 2.5);
    expect(mat.uniforms.uScalarMin.value).toBe(0.5);
    expect(mat.uniforms.uScalarScale.value).toBeCloseTo(0.5, 5);
    expect(mat.userData.scalarRange).toEqual([0.5, 2.5]);
  });

  it('guards against zero-width range (no Inf)', () => {
    const mat = makeMockMaterial();
    applyColormapTextureToMaterial(
      mat,
      new THREE.DataTexture(new Uint8Array(1024), 256, 1, THREE.RGBAFormat)
    );
    applyScalarRangeToMaterial(mat, 1, 1);
    expect(Number.isFinite(mat.uniforms.uScalarScale.value)).toBe(true);
  });
});
