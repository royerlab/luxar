/**
 * GSplat shader family for the TSL ↔ GLSL parity harness: the visual
 * gaussian-splat variants (covariance projection, tiny-sigma fade/
 * reject, gamma fast path, normal premult, opaque peak, thin-covariance
 * dilation, colormap LUT, behind-camera guard) plus the gsplat-pick
 * counterparts including the surface-pick depth pair and the multi-row
 * texture-orientation variant. 18 registry entries.
 *
 * @module tests/e2e/harnesses/tsl-harness/gsplats
 */

import * as THREE from 'three';
import { GSPLAT_SOURCE } from '../../../../rendering/materials/gsplat/shader-glsl';
import {
  gsplatWebGPUFactory,
  buildGSplatTSLNodesFromUniforms,
} from '../../../../rendering/materials/gsplat/shader-tsl';
import { GSPLAT_PICK_SOURCE } from '../../../../rendering/picking/gsplat/shaders';
import {
  gsplatPickWebGPUFactory,
  buildGSplatPickTSLNodesFromUniforms,
} from '../../../../rendering/picking/gsplat/pick.tsl';
import {
  createInstancedGSplatsMesh,
  writeSplatTexels,
} from '../../../../rendering/gsplat-geometry';
import type { RegistryEntry } from './types';
import { buildBehindCamera, buildColormapTexture } from './shared';

/**
 * Build a single-splat gsplat mesh. Isotropic covariance (identity
 * Cholesky) at world origin, fixed amplitude. Test exercises 3D→2D
 * covariance projection + Mahalanobis fragment math.
 */
function buildGSplatInstancedMesh(
  material: THREE.Material,
  center: readonly [number, number, number] = [0, 0, 0],
  sigma: number = 0.1,
  alpha: number = 1.0
): THREE.Object3D {
  // PRODUCTION assembly (createInstancedGSplatsMesh), not a hand-rolled
  // geometry: the previous version decorated the plain BufferGeometry
  // quad TEMPLATE (createGSplatQuadGeometry) with instanced attributes.
  // That accidentally rendered on the WebGLRenderer path but drew ZERO
  // pixels through WebGPURenderer — every gsplat parity variant's TSL
  // side was black and the tests passed vacuously under the tolerance.
  // Isotropic: L = sigma · I, packed [L00, L10, L11, L20, L21, L22].
  // Colors are RGBA (the production colorComponents === 4 layout) so
  // the optional per-splat `alpha` reaches texel3.y; the default 1.0 is
  // the per-element-opacity identity — byte-identical to what the RGB
  // path writes, so every pre-existing entry is unchanged.
  const mesh = createInstancedGSplatsMesh(
    {
      centers: new Float32Array([center[0], center[1], center[2]]),
      cholesky01: new Float32Array([sigma, 0]),
      cholesky23: new Float32Array([sigma, 0]),
      cholesky45: new Float32Array([0, sigma]),
      amplitudes: new Float32Array([1.0]),
      colors: new Float32Array([1.0, 0.5, 0.25, alpha]),
      colorComponents: 4,
      splatCount: 1,
    },
    material
  );
  mesh.frustumCulled = false;
  return mesh;
}

/**
 * Pre-built splat data texture for a gsplat parity variant. Since the
 * texture-storage migration the shaders read splat data via
 * `texelFetch(uSplatTex, ...)`; the TSL texture node is FACTORY-time
 * bound, so the texture must exist in the uniforms record BEFORE the
 * material is built (the same reason the production wrapper rebuilds
 * its graph on a texture identity change). Every gsplat registry
 * entry's `buildUniforms` supplies one of these with the SAME
 * (center, sigma) its `buildMesh` passes to
 * `buildGSplatInstancedMesh`, and the production `writeSplatTexels`
 * writes the layout so the harness can never drift from the real
 * texel packing. (The mesh's own geometry-attached texture holds
 * identical data; the shaders sample the uniform's.)
 */
function buildGSplatSplatDataTexture(
  center: readonly [number, number, number] = [0, 0, 0],
  sigma: number = 0.1,
  alpha: number = 1.0
): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(16), 4, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  // RGBA colors (production colorComponents === 4 branch of
  // writeSplatTexels) so the optional per-splat `alpha` lands in
  // texel3.y; the default 1.0 writes exactly what the RGB branch would
  // (the per-element-opacity identity), preserving every existing entry.
  writeSplatTexels(
    tex,
    {
      centers: new Float32Array([center[0], center[1], center[2]]),
      cholesky01: new Float32Array([sigma, 0]),
      cholesky23: new Float32Array([sigma, 0]),
      cholesky45: new Float32Array([0, sigma]),
      amplitudes: new Float32Array([1.0]),
      colors: new Float32Array([1.0, 0.5, 0.25, alpha]),
      colorComponents: 4,
    },
    1
  );
  return tex;
}

/**
 * MULTI-ROW splat data texture (4×2): element 0 (row 0) is a DECOY
 * splat (green, parked in a corner); element 1 (row 1) holds the
 * standard centered splat. Paired with `aSortedIndex = [1]` (see
 * `buildGSplatMultiRowMesh`), both backends must fetch ROW 1 — a
 * texture-orientation (Y-flip) mismatch between the GLSL `texelFetch`
 * and the TSL `textureLoad` codegen would render the decoy on one
 * backend only and fail pixel parity. 1-row textures (every other
 * entry) are structurally blind to this bug class.
 */
function buildGSplatSplatDataTextureMultiRow(): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(32), 4, 2, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  const sigma = 0.1;
  writeSplatTexels(
    tex,
    {
      centers: new Float32Array([0.8, 0.8, 0, 0, 0, 0]), // decoy corner, real center
      cholesky01: new Float32Array([sigma, 0, sigma, 0]),
      cholesky23: new Float32Array([sigma, 0, sigma, 0]),
      cholesky45: new Float32Array([0, sigma, 0, sigma]),
      amplitudes: new Float32Array([1.0, 1.0]),
      colors: new Float32Array([0, 1, 0, 1.0, 0.5, 0.25]), // decoy green, real standard
    },
    2
  );
  return tex;
}

/**
 * Standard gsplat mesh redirected to STORAGE SLOT 1 (the multi-row
 * texture's real splat). The mesh's own attached texture is ignored —
 * the shaders sample the uniform's — but its `aSortedIndex` drives the
 * fetch index on both backends.
 */
function buildGSplatMultiRowMesh(material: THREE.Material): THREE.Object3D {
  const mesh = buildGSplatInstancedMesh(material) as THREE.Mesh;
  const idx = mesh.geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
  (idx.array as Uint32Array)[0] = 1;
  idx.needsUpdate = true;
  return mesh;
}

/**
 * A highly anisotropic (near-flat) splat: full extent in X and Z, a near-zero
 * minor axis in Y. Under the default ortho camera (which projects the XY block)
 * this yields a near-degenerate 2D covariance — the case the 2D low-pass
 * dilation exists for. Cholesky diagonal = (sx, sy, sz); off-diagonals zero.
 */
function buildThinCovSplatTexture(sx = 0.4, sy = 0.004, sz = 0.4): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(16), 4, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  writeSplatTexels(
    tex,
    {
      centers: new Float32Array([0, 0, 0]),
      cholesky01: new Float32Array([sx, 0]),
      cholesky23: new Float32Array([sy, 0]),
      cholesky45: new Float32Array([0, sz]),
      amplitudes: new Float32Array([1.0]),
      colors: new Float32Array([1.0, 0.5, 0.25]),
    },
    1
  );
  return tex;
}

/**
 * Two overlapping splats along the VIEW axis for the surface-pick depth
 * scenario (front-most-wins for normal-mode gsplats):
 *   - instance 0: NEARER but DIMMER  — amplitude 0.3, world z=-4
 *     (view depth 5 under the default camera at z=1);
 *   - instance 1: FARTHER but BRIGHTER — amplitude 1.0, world z=-8
 *     (view depth 9, inside the default ortho far plane of 10).
 * Both are centred on the optical axis with the SAME isotropic sigma, so
 * under the default ortho camera their footprints coincide exactly at the
 * viewport-centre probe pixel (32,32). Shared by the texture and mesh
 * builders below — they must carry identical data (see the
 * buildGSplatSplatDataTexture doc for why the texture is factory-bound).
 */
const SURFACE_PICK_SPLATS = {
  centers: new Float32Array([0, 0, -4, 0, 0, -8]),
  cholesky01: new Float32Array([0.1, 0, 0.1, 0]),
  cholesky23: new Float32Array([0.1, 0, 0.1, 0]),
  cholesky45: new Float32Array([0, 0.1, 0, 0.1]),
  amplitudes: new Float32Array([0.3, 1.0]),
  colors: new Float32Array([1.0, 0.5, 0.25, 1.0, 0.5, 0.25]),
  splatCount: 2,
} as const;

/** Two-splat data texture for the surface-pick depth variants (4 texels/splat). */
function buildSurfacePickSplatTexture(): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(32), 8, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  writeSplatTexels(tex, SURFACE_PICK_SPLATS, SURFACE_PICK_SPLATS.splatCount);
  return tex;
}

/** Two-splat instanced mesh matching {@link SURFACE_PICK_SPLATS}. */
function buildSurfacePickMesh(material: THREE.Material): THREE.Object3D {
  const mesh = createInstancedGSplatsMesh(SURFACE_PICK_SPLATS, material);
  mesh.frustumCulled = false;
  return mesh;
}

/**
 * Shared uniforms for the surface-pick depth variants — identical to the
 * `gsplat-pick` parity entry plus the `uSurfaceDepth` selector under test.
 */
function buildSurfacePickUniforms(surfaceDepth: 0 | 1): Record<string, THREE.IUniform> {
  return {
    uSplatTex: { value: buildSurfacePickSplatTexture() },
    uResolution: { value: new THREE.Vector2(64, 64) },
    uFx: { value: 32.0 },
    uFy: { value: 32.0 },
    uTruncate: { value: 1.5 },
    uTruncateSq: { value: 2.25 },
    uIsOrtho: { value: 1 },
    uNearCull: { value: 0.01 },
    uMaxExtentFactor: { value: 1.0 },
    uNodeId: { value: 42 },
    uShiftC: { value: Math.exp(-0.5 * 2.25) },
    uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 2.25)) },
    uSurfaceDepth: { value: surfaceDepth },
  };
}

export const GSPLAT_SHADERS: Record<string, RegistryEntry> = {
  // GSplat parity: isotropic Gaussian splat at world origin with
  // identity Cholesky factor. Ortho camera for deterministic projection.
  // Tests the 3D→2D covariance Jacobian, Cholesky factorisation,
  // eigendecomposition, oriented-quad expansion, Mahalanobis fragment.
  gsplat: {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 }, // ortho frustum 2 units → 32 px/unit
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 }, // max projection — no Σ⁻¹ ray-integral path
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) }, // exp(-T²/2) for T=3
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      // blendingMode 'max' matches uProjectionMode=1 above: the TSL
      // factory JS-specializes the graph on the mode (sum emits the
      // Σ⁻¹ ray-integral block), so an inconsistent pair compares a
      // GLSL max-projection against a TSL sum-projection — a real
      // mismatch that was hidden while the TSL side rendered nothing.
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // Multi-row texture-orientation parity: the splat renders from
  // STORAGE SLOT 1 of a 2-row texture (row 0 is a green decoy in a
  // corner). Both backends must resolve the same row — a Y-flip
  // mismatch between the GLSL texelFetch and the TSL textureLoad
  // codegen shows up as the decoy rendering on one backend only.
  // 1-row textures (every other entry) cannot catch this bug class.
  'gsplat-multirow': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTextureMultiRow() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatMultiRowMesh,
  },
  // GSplat at an OFF-CENTER position (world y=0.5 → screen y≈48 of 64).
  // Every other sprite fixture sits at the exact viewport center and is
  // mirror-symmetric about y = H/2, which makes the parity suite BLIND
  // to top-left/bottom-left fragcoord convention bugs (a y-mirror is
  // the identity on them). This variant exists to catch exactly that
  // class — the screenCoordinate-vs-vCenterScreen mismatch made every
  // off-center TSL splat invisible while all centered parity passed.
  'gsplat-offcenter': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture([0, 0.5, 0]) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (material) => buildGSplatInstancedMesh(material, [0, 0.5, 0]),
  },
  // TINY-SIGMA splat (sigma = 0.005, maxLateralVar = 2.5e-5) whose
  // PROJECTED extent lands in the coverage-fade band: uFx = 3200 →
  // projectedExtent = 3200·0.005·3 = 48 px, band (32, 64) → fade 0.5.
  // Guards the fade being computed UNCONDITIONALLY: the former
  // maxLateralVar > 0.01 gate skipped it for sub-0.1-sigma splats, so
  // this splat rendered at FULL amplitude (peak ~255) instead of the
  // faded ~127. Both backends shared the gate, so plain parity is
  // blind — the spec also asserts the ABSOLUTE peak brightness.
  'gsplat-tiny-sigma-fade': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture([0, 0, 0], 0.005) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 3200.0 },
      uFy: { value: 3200.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (material) => buildGSplatInstancedMesh(material, [0, 0, 0], 0.005),
  },
  // TINY-SIGMA splat pushed PAST the fade band (uFx = 12800 →
  // projectedExtent = 192 px > maxExtent = 64) — the coverage cull must
  // reject the vertex entirely. Pre-fix, the gate skipped the fade and
  // the unconditional extent clamp squashed the 192-px quad into a
  // full-intensity hard-edged rectangle (the deep-zoom artifact).
  'gsplat-tiny-sigma-reject': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture([0, 0, 0], 0.005) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 12800.0 },
      uFy: { value: 12800.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (material) => buildGSplatInstancedMesh(material, [0, 0, 0], 0.005),
  },
  // GSplat with the gamma==1 fast path enabled. Same geometry as `gsplat`
  // but uInvGamma=1 + `gammaOne: true`, so the fragment-stage color pow()
  // is replaced with an identity. The GLSL counterpart defines
  // `LUXAR_GAMMA_ONE`. Mirrors `line-gamma-one` (three-geometry symmetry).
  'gsplat-gamma-one': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        gammaOne: true,
        blendingMode: 'max', // matches uProjectionMode=1 (see `gsplat` variant note)
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // GSplat with the no-GOG fast path. Same geometry as `gsplat`,
  // but the TSL factory is built with `noGOG: true` so the
  // `vColor * uIntensity + uOffset` + `max(..., 0)` chain is replaced
  // with `adjusted = vColor`. The GLSL counterpart defines
  // `LUXAR_NO_GOG`. The gain-aware visibility discard keeps reading
  // uIntensity (== 1 in this regime). Mirrors `line-no-gog`
  // (three-geometry symmetry).
  'gsplat-no-gog': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 }, // gamma kept slow path; only no-GOG is exercised
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildDefines: () => ({ LUXAR_NO_GOG: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        noGOG: true,
        blendingMode: 'max', // matches uProjectionMode=1 (see `gsplat` variant note)
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // GSplat 'normal' premultiplied coverage alpha (LUXAR_NORMAL_PREMULT ↔
  // TSL blendingMode:'normal'). The interesting channel is ALPHA: the
  // fragment writes clamp(intensity·uOpacity, 0, 1) instead of 1.0, and
  // GSplat 'volumetric' — emission–absorption (VOLUMETRIC_BLENDING_SPEC.md).
  // SUM projection (uProjectionMode=0 pairs with the TSL sum graph, which
  // emits the Σ⁻¹ ray-integral block) + the LUXAR_VOLUMETRIC fragment
  // branch: RGB = self-screened emission (S(τ) series/quotient), alpha =
  // 1 − e^(−τ) with τ = κ·opacity·intensity. κ=1.5 and opacity=0.7 keep
  // τ mid-range across the footprint so BOTH channels (screened RGB and
  // absorption alpha) vary — a parity mismatch in either the series
  // branch, the exp, or the discard bypass shows up in meanAbsDiff.
  'gsplat-volumetric': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 0 }, // SUM ray-integral (volumetric = emissive)
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 0.7 },
      uAbsorption: { value: 1.5 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildDefines: () => ({ LUXAR_VOLUMETRIC: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'volumetric',
      }) as unknown as THREE.Material;
      // The harness compares raw fragment output — override the
      // factory-applied blend state exactly like the other variants.
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // GSplat 'volumetric' with a PER-SPLAT alpha (RGBA colors +
  // uHasElementAlpha=1). Same scene/uniforms as `gsplat-volumetric`
  // except texel3.y = 0.5 and the gate uniform is ON, so the
  // α → optical-density fold
  //   intensity *= mix(1, −ln(1 − min(a, ALPHA_CLAMP)), uHasElementAlpha)
  // actually executes. Every other gsplat entry leaves uHasElementAlpha
  // at its 0 default (TSL `?? 0`; the GLSL uniform uninitialized), so
  // without this variant the fold ran in ZERO parity cases — a
  // divergence in the mix lanes or the log clamp would ship invisibly.
  // α = 0.5 puts w = −ln(0.5) ≈ 0.693 mid-range, so both the screened
  // RGB and the absorption alpha move measurably relative to α = 1.
  'gsplat-volumetric-rgba': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture([0, 0, 0], 0.1, 0.5) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 0 }, // SUM ray-integral (volumetric = emissive)
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 0.7 },
      uAbsorption: { value: 1.5 },
      uHasElementAlpha: { value: 1 }, // the gate under test
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildDefines: () => ({ LUXAR_VOLUMETRIC: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'volumetric',
      }) as unknown as THREE.Material;
      // The harness compares raw fragment output — override the
      // factory-applied blend state exactly like the other variants.
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildGSplatInstancedMesh(m, [0, 0, 0], 0.1, 0.5),
  },
  // meanAbsDiff compares full RGBA. uOpacity=0.6 keeps the coverage
  // sub-saturated so alpha varies across the splat. uProjectionMode=1
  // (PEAK) matches the TSL factory's normal-mode graph: alpha-over is the
  // surface model, so normal mode uses the 2D-projected peak, not the
  // Σ⁻¹ ray-integral (that path is now exercised only by additive variants).
  'gsplat-normal-premult': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 }, // peak projection (normal = surface)
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 0.6 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildDefines: () => ({ LUXAR_NORMAL_PREMULT: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'normal',
      }) as unknown as THREE.Material;
      // The harness compares raw fragment output — override the
      // factory-applied blend state exactly like the other variants.
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // GSplat 'opaque' — a SURFACE mode like max/normal (usesPeakProjection),
  // so the TSL factory emits the peak-projection graph (no Σ⁻¹
  // ray-integral block) while the fragment keeps the alpha=1.0 contract
  // (no premult branch). Same graph family as the base `gsplat`/max
  // entry; exists to pin the opaque→peak mapping in the codegen
  // snapshot. uProjectionMode=1 keeps the GLSL side on the same branch.
  'gsplat-opaque': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 }, // peak projection (opaque = surface)
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'opaque',
      }) as unknown as THREE.Material;
      // The harness compares raw fragment output — override the
      // factory-applied blend state exactly like the other variants.
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // GSplat 2D-covariance DILATION on a near-degenerate splat. The splat is
  // near-flat (tiny minor axis in Y), so its projected Σ_2D is near-singular —
  // exactly what the low-pass dilation guards. uCov2DDilation=0.3 is set on
  // BOTH backends (GLSL reads the uniform; the TSL adapter reads the same
  // value), so this asserts GLSL and TSL dilate identically. No existing
  // variant exercises a degenerate covariance, so without this a shared
  // dilation regression would be invisible. (max projection isolates the
  // dilation from the ray-integral path.)
  'gsplat-thin-cov': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildThinCovSplatTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 }, // max projection (isolate dilation)
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uCov2DDilation: { value: 0.3 }, // the term under test — same on both backends
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // GSplat colormap parity: USE_COLORMAP LUT path keyed on aAmplitude.
  // uScalarScale=0.5 maps the amplitude (1.0) to t=0.5 so gamma — applied
  // to the value pre-LUT — actually shifts the lookup (pow is identity at
  // t=1). Color GOG bypassed.
  'gsplat-colormap': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
      uColormapTex: { value: buildColormapTexture() },
      uScalarMin: { value: 0.0 },
      uScalarScale: { value: 0.5 },
    }),
    buildDefines: () => ({ USE_COLORMAP: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        useColormap: true,
        blendingMode: 'max', // matches uProjectionMode=1 (see `gsplat` variant note)
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // GSplat-pick parity: same covariance projection as `gsplat`
  // but fragment outputs (nodeId, elementId, brightness, 1.0) and
  // depth = 1 - brightness. No GOG, no ray-integration boost.
  'gsplat-pick': {
    source: GSPLAT_PICK_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 1.5 }, // tighter for picking (vs 3.0 visual)
      uTruncateSq: { value: 2.25 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uNodeId: { value: 42 },
      uShiftC: { value: Math.exp(-0.5 * 2.25) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 2.25)) },
    }),
    buildTSLMaterial: (uniforms) =>
      gsplatPickWebGPUFactory(
        buildGSplatPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: buildGSplatInstancedMesh,
  },
  // B9c: gsplat behind-camera parity (was point-only coverage). The
  // unified perspectiveNearFade subsumes the old standalone reject —
  // a center at view z=+2 must yield an empty frame on both backends.
  'gsplat-behind': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture([0, 0, 3]) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildGSplatInstancedMesh(m, [0, 0, 3]),
    buildCamera: buildBehindCamera,
  },
  'gsplat-pick-behind': {
    source: GSPLAT_PICK_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture([0, 0, 3]) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 1.5 },
      uTruncateSq: { value: 2.25 },
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uNodeId: { value: 42 },
      uShiftC: { value: Math.exp(-0.5 * 2.25) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 2.25)) },
    }),
    buildTSLMaterial: (uniforms) =>
      gsplatPickWebGPUFactory(
        buildGSplatPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: (m) => buildGSplatInstancedMesh(m, [0, 0, 3]),
    buildCamera: buildBehindCamera,
  },
  // Surface-pick depth (S2, front-most-wins for normal-mode gsplats):
  // two overlapping splats along the view axis — instance 0 NEARER but
  // DIMMER (amplitude 0.3), instance 1 FARTHER but BRIGHTER (1.0); see
  // SURFACE_PICK_SPLATS. With uSurfaceDepth=1 the fragment writes REAL
  // projected depth, so the centre probe pixel must carry elementId 0
  // (the near, dim splat — what the user's cursor is on under the
  // depth-sorted occluding surface). `depthCompete` opts the GLSL
  // material into depthTest/depthWrite so the two fragments actually
  // compete (the TSL pick factory already sets them).
  'gsplat-pick-surface': {
    source: GSPLAT_PICK_SOURCE,
    depthCompete: true,
    buildUniforms: () => buildSurfacePickUniforms(1),
    buildTSLMaterial: (uniforms) =>
      gsplatPickWebGPUFactory(
        buildGSplatPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: buildSurfacePickMesh,
  },
  // Control twin: SAME two-splat scene with uSurfaceDepth=0
  // (brightness-as-depth, the commutative-mode convention) — the centre
  // probe pixel must instead carry elementId 1 (the farther, brighter
  // splat). Together the pair proves the uSurfaceDepth selector flips
  // the winner identically on both backends.
  'gsplat-pick-surface-off': {
    source: GSPLAT_PICK_SOURCE,
    depthCompete: true,
    buildUniforms: () => buildSurfacePickUniforms(0),
    buildTSLMaterial: (uniforms) =>
      gsplatPickWebGPUFactory(
        buildGSplatPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: buildSurfacePickMesh,
  },
};
