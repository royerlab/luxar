/**
 * Tests that materials work correctly with and without colormaps.
 *
 * Scalars live in the element textures (points texel2.x, lines
 * texel5.xy), so there are no scalar attributes to leave unbound; the
 * #ifdef USE_COLORMAP pattern gates the scalar texel fetch + LUT path so
 * it is compiled only when a colormap texture is actually provided.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { PointMaterial } from '../../../rendering/materials/point/material-glsl';
import { GSplatMaterial } from '../../../rendering/materials/gsplat/material-glsl';
import { LineMaterial } from '../../../rendering/materials/line/material-glsl';

describe('Material colormap guards', () => {
  describe('PointMaterial', () => {
    it('creates without colormap (no USE_COLORMAP define)', () => {
      const mat = new PointMaterial();
      expect(mat.defines.USE_COLORMAP).toBeUndefined();
      expect(mat.uniforms.uColormapTex).toBeUndefined();
    });

    it('creates with colormap (USE_COLORMAP define set)', () => {
      const tex = new THREE.DataTexture(new Uint8Array(1024), 256, 1, THREE.RGBAFormat);
      const mat = new PointMaterial({
        colormapTexture: tex,
        scalarRange: [0, 1],
      });
      expect(mat.defines.USE_COLORMAP).toBe('');
      expect(mat.uniforms.uColormapTex.value).toBe(tex);
      expect(mat.uniforms.uScalarMin.value).toBe(0);
      expect(mat.uniforms.uScalarScale.value).toBeCloseTo(1.0, 5);
    });

    it('clone without colormap has no colormap uniforms', () => {
      const mat = new PointMaterial({ opacity: 0.5 });
      const cloned = mat.clone();
      expect(cloned.defines.USE_COLORMAP).toBeUndefined();
      expect(cloned.uniforms.uColormapTex).toBeUndefined();
    });

    it('clone with colormap preserves colormap state', () => {
      const tex = new THREE.DataTexture(new Uint8Array(1024), 256, 1, THREE.RGBAFormat);
      const mat = new PointMaterial({ colormapTexture: tex, scalarRange: [0.5, 2.0] });
      const cloned = mat.clone();
      expect(cloned.defines.USE_COLORMAP).toBe('');
      expect(cloned.uniforms.uColormapTex.value).toBe(tex);
    });

    it('updateColormapTexture enables colormap and sets needsUpdate', () => {
      const mat = new PointMaterial();
      expect(mat.defines.USE_COLORMAP).toBeUndefined();
      // vertexColors is unconditionally false — the shader reads aColor
      // as an explicit attribute and toggles USE_COLORMAP independently.
      expect(mat.vertexColors).toBe(false);

      const tex = new THREE.DataTexture(new Uint8Array(1024), 256, 1, THREE.RGBAFormat);
      mat.updateColormapTexture(tex);
      expect(mat.defines.USE_COLORMAP).toBe('');
      expect(mat.uniforms.uColormapTex.value).toBe(tex);
      expect(mat.vertexColors).toBe(false);
    });

    it('updateColormapTexture(null) disables colormap', () => {
      const tex = new THREE.DataTexture(new Uint8Array(1024), 256, 1, THREE.RGBAFormat);
      const mat = new PointMaterial({ colormapTexture: tex });
      expect(mat.defines.USE_COLORMAP).toBe('');
      expect(mat.vertexColors).toBe(false);

      mat.updateColormapTexture(null);
      expect(mat.defines.USE_COLORMAP).toBeUndefined();
      // Stays false; vertexColors is not toggled by colormap state.
      expect(mat.vertexColors).toBe(false);
    });

    it('updateColormapTexture(null) clears uColormapTex.value and userData.scalarRange', () => {
      const tex = new THREE.DataTexture(new Uint8Array(1024), 256, 1, THREE.RGBAFormat);
      const mat = new PointMaterial({ colormapTexture: tex, scalarRange: [0.5, 2.5] });
      expect(mat.uniforms.uColormapTex.value).toBe(tex);
      expect(mat.userData.scalarRange).toEqual([0.5, 2.5]);

      mat.updateColormapTexture(null);
      expect(mat.defines.USE_COLORMAP).toBeUndefined();
      expect(mat.uniforms.uColormapTex.value).toBeNull();
      expect(mat.uniforms.uScalarMin.value).toBe(0.0);
      expect(mat.uniforms.uScalarScale.value).toBe(1.0);
      expect(mat.userData.scalarRange).toBeUndefined();
    });

    it('disable→clone does not resurrect colormap state', () => {
      const tex = new THREE.DataTexture(new Uint8Array(1024), 256, 1, THREE.RGBAFormat);
      const mat = new PointMaterial({ colormapTexture: tex, scalarRange: [0, 1] });
      mat.updateColormapTexture(null);
      const cloned = mat.clone();
      expect(cloned.defines.USE_COLORMAP).toBeUndefined();
      // Clone may still allocate uColormapTex uniform on the cloned material
      // depending on which path the constructor took, but its value must not
      // resurrect the disabled texture.
      expect(cloned.uniforms.uColormapTex?.value ?? null).toBeNull();
      // vertexColors is unconditionally false.
      expect(cloned.vertexColors).toBe(false);
    });

    it('vertex shader contains #ifdef USE_COLORMAP guard', () => {
      const mat = new PointMaterial();
      expect(mat.vertexShader).toContain('#ifdef USE_COLORMAP');
      expect(mat.vertexShader).toContain('#else');
      expect(mat.vertexShader).toContain('#endif');
    });

    it('vertex shader does NOT have unconditional scalar attribute', () => {
      const mat = new PointMaterial();
      // aScalar attribute should only appear inside #ifdef USE_COLORMAP.
      const lines = mat.vertexShader.split('\n');
      for (const line of lines) {
        if (line.trim().startsWith('in float aScalar') && !line.includes('//')) {
          const idx = lines.indexOf(line);
          const before = lines.slice(Math.max(0, idx - 5), idx).join('\n');
          expect(before).toContain('#ifdef USE_COLORMAP');
        }
      }
    });
  });

  describe('GSplatMaterial', () => {
    it('creates without colormap', () => {
      const mat = new GSplatMaterial();
      expect(mat.defines.USE_COLORMAP).toBeUndefined();
      expect(mat.uniforms.uColormapTex).toBeUndefined();
    });

    it('creates with colormap', () => {
      const tex = new THREE.DataTexture(new Uint8Array(1024), 256, 1, THREE.RGBAFormat);
      const mat = new GSplatMaterial({ colormapTexture: tex, scalarRange: [0, 10] });
      expect(mat.defines.USE_COLORMAP).toBe('');
      expect(mat.uniforms.uColormapTex.value).toBe(tex);
    });

    it('clone without colormap is safe', () => {
      const mat = new GSplatMaterial();
      const cloned = mat.clone();
      expect(cloned.defines.USE_COLORMAP).toBeUndefined();
    });
  });

  describe('LineMaterial', () => {
    it('creates without colormap', () => {
      const mat = new LineMaterial();
      expect(mat.defines.USE_COLORMAP).toBeUndefined();
      expect(mat.uniforms.uColormapTex).toBeUndefined();
    });

    it('creates with colormap', () => {
      const tex = new THREE.DataTexture(new Uint8Array(1024), 256, 1, THREE.RGBAFormat);
      const mat = new LineMaterial({ colormapTexture: tex, scalarRange: [0, 1] });
      expect(mat.defines.USE_COLORMAP).toBe('');
    });

    it('clone without colormap is safe', () => {
      const mat = new LineMaterial();
      const cloned = mat.clone();
      expect(cloned.defines.USE_COLORMAP).toBeUndefined();
    });

    it('vertex shader USES texel5 scalars only under USE_COLORMAP (fetch is unconditional since phase 4)', () => {
      const mat = new LineMaterial();
      const shader = mat.vertexShader;
      // Scalars live in texel5.xy of the fixed 6-texel line texture, so
      // there are no scalar ATTRIBUTES to declare at all (the interleaved
      // era's attribute-set toggle and its GL_MAX_VERTEX_ATTRIBS pressure
      // are gone). Since volumetric phase 4 the texel5 FETCH itself is
      // unconditional — texel5.zw carries the per-endpoint alphas, read
      // in every mode — so the colormap guard moved to the scalar USE:
      // the mix(lineT5.x, lineT5.y, t) LUT feed must stay inside the
      // #ifdef USE_COLORMAP branch.
      expect(shader).not.toContain('in float aStartScalar');
      expect(shader).not.toContain('in float aEndScalar');

      // Exactly ONE fetch site (no colormap-gated duplicate).
      const texel5Fetch = 'texelFetch(uLineTex, ivec2(texel0.x + 5, texel0.y), 0)';
      const fetchIdx = shader.indexOf(texel5Fetch);
      expect(fetchIdx).toBeGreaterThan(-1);
      expect(shader.indexOf(texel5Fetch, fetchIdx + 1)).toBe(-1);

      // The scalar USE sits inside a still-open #ifdef USE_COLORMAP
      // block: the nearest preceding #ifdef USE_COLORMAP comes after any
      // #endif.
      const scalarUse = 'mix(lineT5.x, lineT5.y, t)';
      const useIdx = shader.indexOf(scalarUse);
      expect(useIdx).toBeGreaterThan(-1);
      const preceding = shader.slice(0, useIdx);
      expect(preceding.lastIndexOf('#ifdef USE_COLORMAP')).toBeGreaterThan(
        preceding.lastIndexOf('#endif')
      );
    });
  });
});
