/**
 * Line shader family for the TSL ↔ GLSL parity harness: the visual
 * instanced-line variants (gamma / no-GOG fast paths, max-mode
 * premultiply, volumetric emission–absorption, colormap LUT,
 * behind-camera + ortho-near culling, sorted-index permutation) plus
 * the line-pick counterparts + the multi-row texture-orientation
 * variant. 13 registry entries.
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
import { writeSortedIndexOrdering } from '../../../../rendering/element-storage';
import type { RegistryEntry } from './types';
import { buildBehindCamera, buildColormapTexture } from './shared';

/**
 * Single-segment texel source shared by the mesh builder and the
 * standalone `uLineTex` data texture below. Horizontal segment across
 * the viewport in NDC, generous width so it covers many pixels and
 * exposes both the perpendicular falloff and edge AA.
 *
 * `alphas` fills texel5.zw (start/end per-endpoint opacity, from an
 * RGBA color column) through the REAL `writeLineTexels` writer — the
 * volumetric variant passes DISTINCT sub-1.0 values so the vertex
 * stage's sanitize + along-t mix and the fragment's w(a) map are
 * exercised on both backends.
 */
function lineTexelSource(
  start: readonly [number, number, number] = [-0.5, 0, 0],
  end: readonly [number, number, number] = [0.5, 0, 0],
  scalars?: readonly [number, number],
  alphas?: readonly [number, number]
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
    startCapSuppression: new Float32Array([0]),
    endCapSuppression: new Float32Array([0]),
    startScalars: scalars ? new Float32Array([scalars[0]]) : undefined,
    endScalars: scalars ? new Float32Array([scalars[1]]) : undefined,
    startAlphas: alphas ? new Float32Array([alphas[0]]) : undefined,
    endAlphas: alphas ? new Float32Array([alphas[1]]) : undefined,
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
  scalars?: readonly [number, number],
  alphas?: readonly [number, number]
): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(24), 6, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  writeLineTexels(tex, lineTexelSource(start, end, scalars, alphas), 1);
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
      startCapSuppression: new Float32Array([0, ...real.startCapSuppression]),
      endCapSuppression: new Float32Array([0, ...real.endCapSuppression]),
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
  scalars?: readonly [number, number],
  alphas?: readonly [number, number]
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
    { ...lineTexelSource(start, end, scalars, alphas), segmentCount: 1 },
    material
  );
  mesh.frustumCulled = false;
  return mesh;
}

/**
 * Per-endpoint texel5.zw alphas for the volumetric variant: DISTINCT
 * sub-1.0 values so the along-segment mix is a real interpolation
 * (centre pixel sees 0.75, neither endpoint value) and the
 * w(a) = −ln(1−a) map is non-trivial without saturating τ.
 */
const VOLUMETRIC_LINE_ALPHAS: readonly [number, number] = [0.6, 0.9];

/**
 * Line mesh whose 6-texel storage carries the per-endpoint RGBA alphas
 * (texel5.zw = 0.6 → 0.9) — the LINES twin of
 * `points.ts::buildPointVolumetricMesh`.
 */
function buildLineVolumetricMesh(material: THREE.Material): THREE.Object3D {
  return buildLineInstancedMesh(
    material,
    [-0.5, 0, 0],
    [0.5, 0, 0],
    undefined,
    VOLUMETRIC_LINE_ALPHAS
  );
}

/**
 * Combined USE_COLORMAP + LUXAR_VOLUMETRIC mesh: texel5 fully populated
 * (scalars 0.2 → 0.8 in .xy AND per-endpoint alphas 0.6 → 0.9 in .zw).
 * The two branches share the single unconditional texel5 fetch since
 * phase 4, so this is the case that would break if either read
 * displaced the other.
 */
function buildLineVolumetricColormapMesh(material: THREE.Material): THREE.Object3D {
  return buildLineInstancedMesh(
    material,
    [-0.5, 0, 0],
    [0.5, 0, 0],
    [0.2, 0.8],
    VOLUMETRIC_LINE_ALPHAS
  );
}

/**
 * Four-segment texel source for the sorted-permutation variant — one
 * short horizontal segment per screen quadrant (midpoints at world
 * ±0.5 → pixels 16/48 under the default ortho camera), each with a
 * DISTINCT color, so a broken `aSortedIndex` → texel indirection moves
 * or recolors a segment and changes pixels instead of passing
 * vacuously. Shared by the texture and mesh builders below
 * (identical-data convention — the shaders sample the uniform's
 * texture). The LINES twin of `points.ts::SORTED_PERMUTED_POINTS`.
 */
const SORTED_PERMUTED_COUNT = 4;
const SORTED_PERMUTED_LINES: LineTexelSource = {
  // prettier-ignore
  startPositions: new Float32Array([
    -0.75, -0.5, 0,
     0.25, -0.5, 0,
    -0.75,  0.5, 0,
     0.25,  0.5, 0,
  ]),
  // prettier-ignore
  endPositions: new Float32Array([
    -0.25, -0.5, 0,
     0.75, -0.5, 0,
    -0.25,  0.5, 0,
     0.75,  0.5, 0,
  ]),
  // prettier-ignore
  startColors: new Float32Array([
    1.0, 0.1, 0.1,
    0.1, 1.0, 0.1,
    0.1, 0.1, 1.0,
    1.0, 1.0, 0.1,
  ]),
  // prettier-ignore
  endColors: new Float32Array([
    1.0, 0.1, 0.1,
    0.1, 1.0, 0.1,
    0.1, 0.1, 1.0,
    1.0, 1.0, 0.1,
  ]),
  startWidths: new Float32Array([0.1, 0.1, 0.1, 0.1]),
  endWidths: new Float32Array([0.1, 0.1, 0.1, 0.1]),
  startSharpness: new Float32Array([0.5, 0.5, 0.5, 0.5]),
  endSharpness: new Float32Array([0.5, 0.5, 0.5, 0.5]),
  segmentLengths: new Float32Array([0.5, 0.5, 0.5, 0.5]),
  startCapSuppression: new Float32Array([0, 0, 0, 0]),
  endCapSuppression: new Float32Array([0, 0, 0, 0]),
};

/**
 * NON-identity draw-slot → storage-slot permutation under test —
 * applied via the production `writeSortedIndexOrdering` (the
 * SortWorker's write path).
 */
const SORTED_PERMUTED_ORDERING = new Uint32Array([2, 0, 3, 1]);

/**
 * Four-segment line data texture, deliberately SIX TEXELS WIDE (one
 * segment per ROW): storage slot i has texel base 6·i, so with W = 6
 * every slot i > 0 resolves to row y = base / W = i > 0. This
 * exercises the shaders' 2D texel-address reconstruction
 * (x = base % W, y = base / W) on a multi-row wrap — every other line
 * variant uses a single-segment 6×1 texture where y is always 0, so a
 * broken row computation was invisible to the whole line suite.
 */
function buildSortedPermutedLineTexture(): THREE.DataTexture {
  const tex = new THREE.DataTexture(
    new Float32Array(6 * SORTED_PERMUTED_COUNT * 4),
    6,
    SORTED_PERMUTED_COUNT,
    THREE.RGBAFormat,
    THREE.FloatType
  );
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  writeLineTexels(tex, SORTED_PERMUTED_LINES, SORTED_PERMUTED_COUNT);
  return tex;
}

/**
 * Multi-instance lines mesh matching {@link SORTED_PERMUTED_LINES},
 * with the NON-identity {@link SORTED_PERMUTED_ORDERING} written
 * through the production `writeSortedIndexOrdering`. Assembled via the
 * production `createInstancedLinesMesh` like every other line entry
 * (which attaches the storage pair via `attachLineStorage` and writes
 * the texels + identity ordering); the permutation write then replaces
 * the identity — the SortWorker's exact commit sequence.
 */
function buildSortedPermutedLinesMesh(material: THREE.Material): THREE.Object3D {
  const mesh = createInstancedLinesMesh(
    { ...SORTED_PERMUTED_LINES, segmentCount: SORTED_PERMUTED_COUNT },
    material
  );
  writeSortedIndexOrdering(
    mesh.geometry as THREE.InstancedBufferGeometry,
    SORTED_PERMUTED_ORDERING,
    SORTED_PERMUTED_COUNT
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
  // Line volumetric parity: the emission–absorption output branch
  // (LUXAR_VOLUMETRIC; the TSL side builds it from
  // `blendingMode: 'volumetric'`) — τ = κ·alpha·vWidthAtT·chord (the
  // transverse ribbon integral, LINE_CHORD_SCALE = √(π/ln 100)), S(τ)
  // screening, physical absorption alpha, AND the per-endpoint
  // texel5.zw alphas (0.6 → 0.9) → w(a) = −ln(1−a) optical depth via
  // uHasElementAlpha = 1. uAbsorption 2.0 + opacity 0.7 keep τ
  // mid-range on the line body (τ ≈ 0.16 at the centre pixel) so S(τ)
  // and volAlpha are non-trivial — neither saturated at 1 nor
  // vanishing — and a GLSL/TSL divergence is visible. Mirrors
  // `point-volumetric` (three-geometry symmetry).
  'line-volumetric': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: {
        value: buildLineDataTexture([-0.5, 0, 0], [0.5, 0, 0], undefined, VOLUMETRIC_LINE_ALPHAS),
      },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 0.7 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uAbsorption: { value: 2.0 },
      uHasElementAlpha: { value: 1 },
    }),
    buildDefines: () => ({ LUXAR_VOLUMETRIC: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'volumetric',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineVolumetricMesh,
  },
  // COMBINED colormap + volumetric: the LUT sources the colour from
  // texel5.xy while the volumetric branch maps texel5.zw alphas through
  // w(a) into τ — both defines co-compiled, both texel5 reads live off
  // the SAME single fetch. Would catch either branch displacing the
  // other (the untested-combination flag from the phase-4 double-check).
  // (Deep-campaign note: this combination is UNREACHABLE from the Python
  // scene API — all three adders make colors/colormap mutually exclusive
  // and scalars require a colormap, so real data never has both an RGBA
  // alpha column and LUT scalars. The coverage is deliberately defensive:
  // hand-crafted zarr can reach it, and the shader must stay correct.)
  'line-volumetric-colormap': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: {
        value: buildLineDataTexture([-0.5, 0, 0], [0.5, 0, 0], [0.2, 0.8], VOLUMETRIC_LINE_ALPHAS),
      },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 0.7 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uAbsorption: { value: 2.0 },
      uHasElementAlpha: { value: 1 },
      uColormapTex: { value: buildColormapTexture() },
      uScalarMin: { value: 0.0 },
      uScalarScale: { value: 1.0 },
    }),
    buildDefines: () => ({ USE_COLORMAP: '', LUXAR_VOLUMETRIC: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, { useColormap: true }), {
        useColormap: true,
        blendingMode: 'volumetric',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineVolumetricColormapMesh,
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
  // Segment CROSSING the camera plane: start at world z=1.5 (view depth
  // -0.5, BEHIND the camera at z=1), end at the origin (view depth 1.0,
  // in front). Exercises the vertex-stage near-plane SEGMENT clipping:
  // the behind endpoint must be moved onto the nearCull plane before the
  // screen-space expansion. Pre-clip, the w <= 0 endpoint turned the
  // quad into an external (wrapped) primitive whose visible half drooped
  // off the centerline with a razor edge through the profile — the
  // close-zoom "one-sided profile" artifact. The parity test also
  // asserts a CONTENT property (every lit column's centroid stays on the
  // projected centerline), so this entry fails pre-fix, not just on
  // backend divergence.
  'line-crossing': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture([0.15, 0, 1.5], [0.15, 0, 0]) },
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
    buildMesh: (m) => buildLineInstancedMesh(m, [0.15, 0, 1.5], [0.15, 0, 0]),
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
  // aSortedIndex indirection under a NON-identity permutation on a
  // MULTI-ROW line texture — the LINES twin of `point-sorted-permuted`.
  // Every other line variant is a single instance with identity
  // ordering on a 6×1 texture, so neither the draw-slot → storage-slot
  // permutation nor a texel base with base / W > 0 (row y > 0) was
  // exercised by ANY line parity variant — a backend that ignored
  // aSortedIndex or mis-reconstructed the 2D texel address would ship
  // invisibly. Four distinct-color segments, one per screen quadrant,
  // drawn through the production writeSortedIndexOrdering permutation
  // [2, 0, 3, 1].
  'line-sorted-permuted': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildSortedPermutedLineTexture() },
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
    buildMesh: buildSortedPermutedLinesMesh,
  },
};
