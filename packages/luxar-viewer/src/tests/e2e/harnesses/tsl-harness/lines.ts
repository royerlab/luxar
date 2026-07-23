/**
 * Line shader family for the TSL ↔ GLSL parity harness: the visual
 * instanced-line variants (gamma / no-GOG fast paths, max-mode
 * premultiply, colormap LUT, behind-camera + ortho-near culling) plus
 * the line-pick counterparts + the multi-row texture-orientation
 * variant. 10 registry entries.
 *
 * @module tests/e2e/harnesses/tsl-harness/lines
 */

import * as THREE from 'three';
import { LINE_SOURCE } from '../../../../rendering/materials/line/shader-glsl';
import {
  lineWebGPUFactory,
  buildLineTSLNodesFromUniforms,
} from '../../../../rendering/materials/line/shader-tsl';
import { LINE_PICK_SOURCE } from '../../../../rendering/picking/line/shaders';
import {
  linePickWebGPUFactory,
  buildLinePickTSLNodesFromUniforms,
} from '../../../../rendering/picking/line/pick.tsl';
import {
  createInstancedLinesMesh,
  writeLineTexels,
  type LineTexelSource,
} from '../../../../rendering/line-geometry';
import type { RegistryEntry } from './types';
import { buildBehindCamera, buildColormapTexture } from './shared';

/**
 * Single-segment texel source shared by the mesh builder and the
 * standalone `uLineTex` data texture below. Horizontal segment across
 * the viewport in NDC, generous width so it covers many pixels and
 * exposes both the perpendicular falloff and edge AA.
 */
function lineTexelSource(
  start: readonly [number, number, number] = [-0.5, 0, 0],
  end: readonly [number, number, number] = [0.5, 0, 0],
  scalars?: readonly [number, number]
): LineTexelSource {
  return {
    startPositions: new Float32Array([start[0], start[1], start[2]]),
    endPositions: new Float32Array([end[0], end[1], end[2]]),
    startColors: new Float32Array([1.0, 0.5, 0.25]),
    endColors: new Float32Array([1.0, 0.5, 0.25]),
    startWidths: new Float32Array([0.1]),
    endWidths: new Float32Array([0.1]),
    // Sharpness is the normalised [0, 1] knob -> super-Gaussian exponent
    // beta = 2^(6s - 2). 0.5 -> beta=2 (a true Gaussian, the default).
    startSharpness: new Float32Array([0.5]),
    endSharpness: new Float32Array([0.5]),
    segmentLengths: new Float32Array([1.0]),
    startClipped: new Uint8Array([0]),
    endClipped: new Uint8Array([0]),
    startScalars: scalars ? new Float32Array([scalars[0]]) : undefined,
    endScalars: scalars ? new Float32Array([scalars[1]]) : undefined,
  };
}

/**
 * Standalone 6×1 line data texture for `buildUniforms`. The shaders
 * sample the segment from `uLineTex` (`texelFetch(uLineTex, ...)`); the
 * TSL texture node is FACTORY-time bound from the uniforms record, so
 * each registry entry supplies one of these with the SAME (start, end,
 * scalars) its `buildMesh` passes to `buildLineInstancedMesh` — mirrors
 * the point harness's `buildPointDataTexture`.
 */
function buildLineDataTexture(
  start: readonly [number, number, number] = [-0.5, 0, 0],
  end: readonly [number, number, number] = [0.5, 0, 0],
  scalars?: readonly [number, number]
): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(24), 6, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  writeLineTexels(tex, lineTexelSource(start, end, scalars), 1);
  return tex;
}

/**
 * MULTI-ROW line data texture (6×2): element 0 (row 0) is a DECOY
 * segment (short, vertical, green, parked in a corner); element 1
 * (row 1) holds the standard horizontal segment. Paired with
 * `aSortedIndex = [1]` (see `buildLineMultiRowMesh`), both backends
 * must fetch ROW 1 — a texture-orientation (Y-flip) mismatch between
 * the GLSL `texelFetch` and the TSL `textureLoad` codegen would render
 * the decoy on one backend only and fail pixel parity. 1-row textures
 * (every other entry) are structurally blind to this bug class.
 */
function buildLineDataTextureMultiRow(): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(48), 6, 2, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  const real = lineTexelSource();
  writeLineTexels(
    tex,
    {
      startPositions: new Float32Array([0.7, 0.5, 0, ...real.startPositions]),
      endPositions: new Float32Array([0.7, 0.9, 0, ...real.endPositions]),
      startColors: new Float32Array([0, 1, 0, ...real.startColors]),
      endColors: new Float32Array([0, 1, 0, ...real.endColors]),
      startWidths: new Float32Array([0.05, ...real.startWidths]),
      endWidths: new Float32Array([0.05, ...real.endWidths]),
      startSharpness: new Float32Array([0.5, ...real.startSharpness]),
      endSharpness: new Float32Array([0.5, ...real.endSharpness]),
      segmentLengths: new Float32Array([0.4, ...real.segmentLengths]),
      startClipped: new Uint8Array([0, ...real.startClipped]),
      endClipped: new Uint8Array([0, ...real.endClipped]),
    },
    2
  );
  return tex;
}

/**
 * Standard line mesh redirected to STORAGE SLOT 1 (the multi-row
 * texture's real segment). The mesh's own attached texture is ignored —
 * the shaders sample the uniform's — but its `aSortedIndex` drives the
 * fetch index on both backends.
 */
function buildLineMultiRowMesh(material: THREE.Material): THREE.Object3D {
  const mesh = buildLineInstancedMesh(material) as THREE.Mesh;
  const idx = mesh.geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
  (idx.array as Uint32Array)[0] = 1;
  idx.needsUpdate = true;
  return mesh;
}

/**
 * Line mesh + per-endpoint scalars (0.2 → 0.8) for the colormap-parity
 * case. Under `USE_COLORMAP` the shader sources colour from the LUT via
 * the texel5 scalars; the fixed 6-texel layout carries both the (unused)
 * colours and the scalars, so no geometry surgery is needed.
 */
function buildLineColormapMesh(material: THREE.Material): THREE.Object3D {
  return buildLineInstancedMesh(material, [-0.5, 0, 0], [0.5, 0, 0], [0.2, 0.8]);
}

/**
 * Build a single-segment line mesh for parity testing (see
 * `lineTexelSource` for the segment shape).
 */
function buildLineInstancedMesh(
  material: THREE.Material,
  start: readonly [number, number, number] = [-0.5, 0, 0],
  end: readonly [number, number, number] = [0.5, 0, 0],
  scalars?: readonly [number, number]
): THREE.Object3D {
  // PRODUCTION assembly (createInstancedLinesMesh), not a hand-rolled
  // geometry: a plain BufferGeometry quad TEMPLATE decorated by hand is
  // never a real InstancedBufferGeometry, which the WebGPU-path draw
  // dispatch (three.webgpu.js drawParams: `instanceCount =
  // geometry.instanceCount` only when isInstancedBufferGeometry) does
  // not draw as intended. Using the production creator keeps parity
  // testing the real path (texture storage + aSortedIndex) and makes
  // instancing correct by construction.
  const mesh = createInstancedLinesMesh(
    { ...lineTexelSource(start, end, scalars), segmentCount: 1 },
    material
  );
  mesh.frustumCulled = false;
  return mesh;
}

export const LINE_SHADERS: Record<string, RegistryEntry> = {
  // Line parity: instanced quad line with width, sharpness, GOG.
  // Ortho camera so screen-space conversion is deterministic.
  line: {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      // Pre-baked pixel-width scales for this ortho config:
      //   uOrthoLineScale = 2 * 64 / 2 = 64 (2*resY/frustumHeight)
      //   uPerspectiveLineScale is unused (uIsOrtho=1) — benign 1.0.
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineInstancedMesh,
  },
  // Multi-row texture-orientation parity: the segment renders from
  // STORAGE SLOT 1 of a 2-row texture (row 0 is a green decoy). Both
  // backends must resolve the same row — a Y-flip mismatch between the
  // GLSL texelFetch and the TSL textureLoad codegen shows up as the
  // decoy rendering on one backend only. 1-row textures (every other
  // entry) cannot catch this bug class.
  'line-multirow': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTextureMultiRow() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineMultiRowMesh,
  },
  // Line with the gamma==1 fast path enabled. Same geometry +
  // uniforms as `line`, but the TSL factory is built with
  // `gammaOne: true` so the fragment-stage pow() is replaced with an
  // identity. The codegen snapshot for this variant pins the
  // pow-free fast path; the parity test compares against a GLSL
  // shader that has `LUXAR_GAMMA_ONE` defined.
  'line-gamma-one': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        gammaOne: true,
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineInstancedMesh,
  },
  // Line with the no-GOG fast path. Same geometry as `line`,
  // but the TSL factory is built with `noGOG: true` so the
  // `vColor * uIntensity + uOffset` + `max(..., 0)` chain is replaced
  // with `adjusted = vColor`. The GLSL counterpart defines
  // `LUXAR_NO_GOG`.
  'line-no-gog': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 }, // gamma kept slow path; only no-GOG is exercised
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildDefines: () => ({ LUXAR_NO_GOG: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        noGOG: true,
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineInstancedMesh,
  },
  // Max-mode premultiply parity: `blendingMode: 'max'` builds the TSL
  // graph with the RGB-contribution output (fragment emits
  // `gammaColor * a, a` with a = intensity·opacity, so MaxEquation +
  // OneFactor/OneFactor compares contribution-weighted colour); the
  // GLSL twin compiles with LUXAR_MAX_RGB_CONTRIBUTION. Framebuffer
  // blending itself is NOT under test — NoBlending readback like every
  // variant. Mirrors `point-max` (three-geometry symmetry).
  'line-max': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildDefines: () => ({ LUXAR_MAX_RGB_CONTRIBUTION: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'max',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineInstancedMesh,
  },
  // Line colormap parity: USE_COLORMAP LUT path with per-endpoint scalars
  // (0.2 → 0.8). Gamma applied to the value pre-LUT (gammaOne=false here);
  // intensity/offset apply POST-LUT to the mapped color (matching the
  // gsplat shader) — non-default values here so the post-LUT gain/offset
  // path is exercised and must match across backends.
  'line-colormap': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.5 },
      uOffset: { value: 0.05 },
      uColormapTex: { value: buildColormapTexture() },
      uScalarMin: { value: 0.0 },
      uScalarScale: { value: 1.0 },
    }),
    buildDefines: () => ({ USE_COLORMAP: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, { useColormap: true }), {
        useColormap: true,
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineColormapMesh,
  },
  // Line-pick parity: same quad-expansion math as `line` but
  // fragment outputs (nodeId, elementId, brightness, 1.0) and
  // depth = 1 - brightness. No edgeAA, no GOG.
  'line-pick': {
    source: LINE_PICK_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNodeId: { value: 42 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      // Pre-baked pixel-width scales (mirror `line` parity entry).
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
    }),
    buildTSLMaterial: (uniforms) =>
      linePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(uniforms), {
        isOrtho: true,
      }) as unknown as THREE.Material,
    buildMesh: buildLineInstancedMesh,
  },
  // B9c: line behind-camera parity (was point-only coverage). Both
  // endpoints at world z=3 → view z=+2 → the perspective-gated
  // bothBehind cull must produce an empty frame on BOTH backends.
  'line-behind': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture([-0.5, 0, 3], [0.5, 0, 3]) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 64.0 },
      uOrthoLineScale: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        isOrtho: false,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildLineInstancedMesh(m, [-0.5, 0, 3], [0.5, 0, 3]),
    buildCamera: buildBehindCamera,
  },
  'line-pick-behind': {
    source: LINE_PICK_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture([-0.5, 0, 3], [0.5, 0, 3]) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 0 },
      uNodeId: { value: 42 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 64.0 },
      uOrthoLineScale: { value: 1.0 },
    }),
    buildTSLMaterial: (uniforms) =>
      linePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(uniforms), {
        isOrtho: false,
      }) as unknown as THREE.Material,
    buildMesh: (m) => buildLineInstancedMesh(m, [-0.5, 0, 3], [0.5, 0, 3]),
    buildCamera: buildBehindCamera,
  },
  // B9c BUG-A regression: ortho line INSIDE the frustum but within the
  // uNearCull slab (view depth 0.15 < nearCull 0.5, camera near 0.1).
  // Pre-fix the ungated bothBehind cull hid it (while a point/gsplat at
  // the same spot drew); post-fix it renders on both backends — NDC
  // clipping is the sole ortho cull authority.
  'line-ortho-near': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture([-0.5, 0, 0.85], [0.5, 0, 0.85]) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.5 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    // Camera at z=1 (ortho near 0.1): world z=0.85 → view depth 0.15.
    buildMesh: (m) => buildLineInstancedMesh(m, [-0.5, 0, 0.85], [0.5, 0, 0.85]),
  },
};
