/**
 * Point shader family for the TSL ↔ GLSL parity harness: the visual
 * point-sprite variants (falloff sweep, gamma fast path, max-mode
 * premultiply, colormap LUT, perspective sizing, subpixel floor, near
 * fade, behind-camera guard) plus the point-pick counterparts.
 * 13 registry entries.
 *
 * @module tests/e2e/harnesses/tsl-harness/points
 */

import * as THREE from 'three';
import { POINT_SOURCE } from '../../../../rendering/materials/point/shader-glsl';
import {
  pointWebGPUFactory,
  buildPointTSLNodesFromUniforms,
} from '../../../../rendering/materials/point/shader-tsl';
import { POINT_PICK_SOURCE } from '../../../../rendering/picking/point/shaders';
import {
  pointPickWebGPUFactory,
  buildPointPickTSLNodesFromUniforms,
} from '../../../../rendering/picking/point/pick.tsl';
import {
  createPointQuadGeometry,
  attachPointStorage,
  writePointTexels,
  type PointTexelSource,
} from '../../../../rendering/point-geometry';
import { writeSortedIndexIdentity } from '../../../../rendering/element-storage';
import type { RegistryEntry } from './types';
import { buildBehindCamera, buildColormapTexture } from './shared';

/**
 * Single-point texel source shared by the texture and mesh builders —
 * they must carry identical data (the shaders sample the UNIFORM's
 * texture; the mesh's geometry-attached texture holds the same values).
 * Realistic attributes: radius 0.5, color (1.0, 0.5, 0.25); `sharpness`
 * is the normalised [0, 1] knob -> super-Gaussian exponent
 * beta = 2^(6s - 2) (default 0.5 -> beta=2, a true Gaussian); `center`
 * is the world-space point position (a behind-camera center exercises
 * the perspective behind-camera guard); `scalar` feeds texel2.x for the
 * colormap-parity case.
 */
function pointTexelSource(
  center: readonly [number, number, number] = [0, 0, 0],
  sharpness: number = 0.5,
  scalar?: number
): PointTexelSource {
  return {
    positions: new Float32Array([center[0], center[1], center[2]]),
    colors: new Float32Array([1.0, 0.5, 0.25]),
    radii: new Float32Array([0.5]),
    sharpness: new Float32Array([sharpness]),
    scalars: scalar !== undefined ? new Float32Array([scalar]) : undefined,
  };
}

/**
 * Pre-built point data texture for a point parity variant. Since the
 * texture-storage migration the shaders read point data via
 * `texelFetch(uPointTex, ...)`; the TSL texture node is FACTORY-time
 * bound, so the texture must exist in the uniforms record BEFORE the
 * material is built (the same reason the production wrapper rebuilds
 * its graph on a texture identity change). Every point registry entry's
 * `buildUniforms` supplies one of these with the SAME (center,
 * sharpness) its `buildMesh` passes to `buildPointInstancedMesh`, and
 * the production `writePointTexels` writes the layout so the harness
 * can never drift from the real texel packing (3 texels/point).
 */
function buildPointDataTexture(
  center: readonly [number, number, number] = [0, 0, 0],
  sharpness: number = 0.5,
  scalar?: number
): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(12), 3, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  writePointTexels(tex, pointTexelSource(center, sharpness, scalar), 1);
  return tex;
}

/**
 * MULTI-ROW point data texture (3×2): element 0 (row 0) is a DECOY
 * point parked in a screen corner with a distinct green color; element
 * 1 (row 1) holds the standard centered point. Paired with
 * `aSortedIndex = [1]` (see `buildPointMultiRowMesh`), both backends
 * must fetch ROW 1 — a texture-orientation (Y-flip) mismatch between
 * the hand-written GLSL `texelFetch` and the TSL `textureLoad` codegen
 * (which wraps fetches in three's `height − y − 1` flip on the WebGL
 * fallback) would sample the decoy on one backend only and fail pixel
 * parity. Every other entry's 1-row texture is structurally blind to
 * this bug class: row 0 maps to row 0 under any flip.
 */
function buildPointDataTextureMultiRow(): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(24), 3, 2, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  writePointTexels(
    tex,
    {
      positions: new Float32Array([0.8, 0.8, 0, 0, 0, 0]), // decoy corner, real center
      colors: new Float32Array([0, 1, 0, 1.0, 0.5, 0.25]), // decoy green, real standard
      radii: new Float32Array([0.5, 0.5]),
      sharpness: new Float32Array([0.5, 0.5]),
    },
    2
  );
  return tex;
}

/**
 * Standard point mesh redirected to STORAGE SLOT 1 (the multi-row
 * texture's real element). The mesh's own attached texture is ignored —
 * the shaders sample the uniform's — but its `aSortedIndex` drives the
 * fetch index on both backends.
 */
function buildPointMultiRowMesh(material: THREE.Material): THREE.Object3D {
  const mesh = buildPointInstancedMesh(material) as THREE.Mesh;
  const idx = mesh.geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
  (idx.array as Uint32Array)[0] = 1;
  idx.needsUpdate = true;
  return mesh;
}

/** Point mesh with the texel2.x scalar written for the colormap-parity case. */
function buildPointColormapMesh(material: THREE.Material): THREE.Object3D {
  // Mid-range scalar so gamma (pow(t, invGamma)) actually moves the
  // lookup off the t=0/1 fixed points where pow is the identity.
  return buildPointInstancedMesh(material, 0.5, [0, 0, 0], 0.5);
}

/**
 * Build a real instanced-points mesh for the point parity test.
 * One point at world origin with realistic attributes; 4-vertex quad
 * base + the production point texture / `aSortedIndex` storage pair
 * (texture-backed geometry — mirrors the gsplat harness assembly).
 */
function buildPointInstancedMesh(
  material: THREE.Material,
  sharpness: number = 0.5,
  center: readonly [number, number, number] = [0, 0, 0],
  scalar?: number
): THREE.Object3D {
  const geom = createPointQuadGeometry();
  const texture = attachPointStorage(geom, 1);
  writePointTexels(texture, pointTexelSource(center, sharpness, scalar), 1);
  writeSortedIndexIdentity(geom, 1);
  // Match the production points-mesh contract (the gpu-buffer-pool points
  // adapter): the WebGLRenderer only issues an instanced draw when `instanceCount`
  // is finite, and the drawRange must cap at the 6 indices that
  // form the unit quad — otherwise r184 falls back to a single
  // non-instanced draw call and produces a degenerate parity image.
  geom.instanceCount = 1;
  geom.setDrawRange(0, 6);
  const mesh = new THREE.Mesh(geom, material);
  mesh.frustumCulled = false;
  return mesh;
}

export const POINT_SHADERS: Record<string, RegistryEntry> = {
  // Point parity: full PointMaterial sprite + GOG + super-Gaussian falloff.
  // Uniforms mirror the production PointMaterial constructor; ortho mode
  // keeps `invDistance = 1` so the test is deterministic across cameras.
  // The default mesh sharpness is 0.5 -> beta=2 (a true Gaussian).
  point: {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture() },
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      // Disable blending for raw-pixel parity against the harness's
      // ShaderMaterial path (which uses transparent: false). Production
      // sets AdditiveBlending; the parity test only checks fragment
      // output, not blending semantics.
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildPointInstancedMesh,
  },
  // Multi-row texture-orientation parity: the point renders from
  // STORAGE SLOT 1 of a 2-row texture (row 0 is a green decoy in a
  // corner). Both backends must resolve the same row — a Y-flip
  // mismatch between the GLSL texelFetch and the TSL textureLoad
  // codegen shows up as the decoy rendering on one backend only.
  // 1-row textures (every other entry) cannot catch this bug class.
  'point-multirow': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTextureMultiRow() },
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildPointMultiRowMesh,
  },
  // Super-Gaussian exponent sweep: the GLSL `pow(rho, beta)` / `exp(...)`
  // falloff must match TSL across the beta range, not just at the default.
  // `point-soft` exercises a peaky cusp (s=0.1 -> beta≈0.36); `point-hard`
  // a near-flat-top hard edge (s=0.9 -> beta≈12.1).
  'point-soft': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture([0, 0, 0], 0.1) },
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildPointInstancedMesh(m, 0.1),
  },
  'point-hard': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture([0, 0, 0], 0.9) },
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildPointInstancedMesh(m, 0.9),
  },
  // Point with the gamma==1 fast path enabled. Same geometry as `point`
  // but invGamma=1 + `gammaOne: true`, so the fragment-stage color pow()
  // is replaced with an identity. The GLSL counterpart defines
  // `LUXAR_GAMMA_ONE`. Mirrors `line-gamma-one` (three-geometry symmetry).
  'point-gamma-one': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture() },
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(buildPointTSLNodesFromUniforms(uniforms, {}), {
        gammaOne: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildPointInstancedMesh,
  },
  // Max-mode premultiply parity: `blendingMode: 'max'` builds the TSL
  // graph with the RGB-contribution output (fragment emits
  // `finalColor * alpha, alpha` so MaxEquation + OneFactor/OneFactor
  // compares contribution-weighted colour); the GLSL twin compiles with
  // the LUXAR_MAX_RGB_CONTRIBUTION define. Framebuffer blending itself
  // is NOT under test — like every variant, both sides read back with
  // NoBlending so raw fragment output is compared.
  'point-max': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture() },
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildDefines: () => ({ LUXAR_MAX_RGB_CONTRIBUTION: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(buildPointTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildPointInstancedMesh,
  },
  // Point colormap parity: USE_COLORMAP LUT path. Gamma is applied to the
  // scalar VALUE before the LUT lookup (vertex stage); intensity/offset
  // apply POST-LUT to the mapped color (matching the gsplat shader).
  // invGamma != 1 with a texel2.x scalar of 0.5 so the gamma warp is
  // observable, and non-default uIntensity/uOffset so the post-LUT
  // gain/offset path is exercised and must match across backends.
  'point-colormap': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture([0, 0, 0], 0.5, 0.5) },
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.5 },
      uOffset: { value: 0.05 },
      uColormapTex: { value: buildColormapTexture() },
      uScalarMin: { value: 0.0 },
      uScalarScale: { value: 1.0 },
    }),
    buildDefines: () => ({ USE_COLORMAP: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, { useColormap: true }),
        { useColormap: true }
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildPointColormapMesh,
  },
  // Point-pick parity: identical sprite layout to `point` but
  // the fragment outputs (nodeId, elementId, brightness, 1.0) and
  // depth = 1 - brightness. Pick footprint is half-radius (×0.5).
  'point-pick': {
    source: POINT_PICK_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture() },
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uNodeId: { value: 42 },
      uResolution: { value: new THREE.Vector2(64, 64) },
    }),
    buildTSLMaterial: (uniforms) =>
      pointPickWebGPUFactory(
        buildPointPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: buildPointInstancedMesh,
  },
  // Behind-camera guard parity: perspective camera (uIsOrtho:0) with the point
  // placed behind it (world z=3 → view z=+2). The visual point shader's
  // `uIsOrtho == 0 && mvPosition.z >= 0` reject must fire IDENTICALLY in GLSL and
  // TSL, so both backends produce an empty (background) frame. Without a behind-
  // camera case the guard ships with no rendered parity coverage (every other
  // point case is ortho, where the guard is a no-op).
  'point-behind': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture([0, 0, 3]) },
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 0 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildPointInstancedMesh(m, 0.5, [0, 0, 3]),
    buildCamera: buildBehindCamera,
  },
  // Same behind-camera guard, but on the point PICK shader (the path that was
  // missing the guard before — see the visual/pick symmetry fix). Both backends
  // must cull the behind-camera point so no spurious pick sprite is emitted.
  'point-pick-behind': {
    source: POINT_PICK_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture([0, 0, 3]) },
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 0 },
      uNodeId: { value: 42 },
      uResolution: { value: new THREE.Vector2(64, 64) },
    }),
    buildTSLMaterial: (uniforms) =>
      pointPickWebGPUFactory(
        buildPointPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: (m) => buildPointInstancedMesh(m, 0.5, [0, 0, 3]),
    buildCamera: buildBehindCamera,
  },
  // ---- B9a/B9b/B9c regression variants ----
  //
  // B9a proof pair: identical points at the SAME view depth, one centered
  // and one off-axis, under PERSPECTIVE. Post-fix (view-z sizing, matching
  // lines/gsplats) their footprints are equal; the old Euclidean-distance
  // sizing shrank the off-axis sprite by cos(theta) (~11% linear at 26.6°
  // here). The spec compares covered-pixel counts across the two renders.
  'point-persp-center': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture([0, 0, 0]) },
      pointSizeFactor: { value: 221.7 }, // 2*64/tan(30°) — fov 60 at 64px
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 0.1 }, // ~11px sprite at view depth 1
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.01 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildPointInstancedMesh(m, 0.5, [0, 0, 0]),
    buildCamera: buildBehindCamera,
  },
  'point-persp-offaxis': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture([0.5, 0, 0]) },
      pointSizeFactor: { value: 221.7 }, // 2*64/tan(30°) — fov 60 at 64px
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 0.1 }, // ~11px sprite at view depth 1
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.01 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    // World x=0.5 at view depth 1 → 26.6° off-axis, NDC x ≈ 0.87 (on-screen
    // at fov 60, aspect 1).
    buildMesh: (m) => buildPointInstancedMesh(m, 0.5, [0.5, 0, 0]),
    buildCamera: buildBehindCamera,
  },
  // B9b proof: raw projected size ≈ 0.96px (radiusScale 0.06 × attr 0.5 ×
  // factor 32). The 1.5px sprite floor guarantees rasterization, and the
  // fragment's sizeScale² compensation scales the ALPHA output by
  // (0.96/1.5)² ≈ 0.41. The point is positioned so the sprite center
  // lands EXACTLY on pixel (32,32)'s center (world 1.5/96 with the
  // [-1,1] ortho frustum on 64px) — falloff there is exactly 1, so the
  // written alpha is deterministically ≈ 0.41·255 ≈ 105 (pre-fix: 255,
  // indistinguishable from the opaque clear).
  'point-subpixel': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture([0.015625, 0.015625, 0]) },
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 0.06 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildPointInstancedMesh(m, 0.5, [0.015625, 0.015625, 0]),
  },
  // B9c: unified near fade, mid-band. Point at view depth 1 with
  // uNearCull 0.7 → smoothstep((1-0.7)/0.7) ≈ 0.39 fade — non-empty,
  // identical across backends (shared perspectiveNearFade helper).
  'point-near-fade': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      uPointTex: { value: buildPointDataTexture() },
      pointSizeFactor: { value: 221.7 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 0.1 },
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.7 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildPointInstancedMesh,
    buildCamera: buildBehindCamera,
  },
};
