/**
 * Line shader family for the TSL ↔ GLSL parity harness: the visual
 * instanced-line variants (gamma / no-GOG fast paths, max-mode
 * premultiply, volumetric emission–absorption, colormap LUT,
 * behind-camera + ortho-near culling, sorted-index permutation) plus
 * the line-pick counterparts + multi-row, cap-suppression, clipping-remap,
 * and exact-near-plane boundary variants, plus the two screen-space-miter join
 * pairs (ortho, and the perspective near-plane joint of #1346 with its control).
 * 30 registry entries.
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
import {
  writeSortedIndexOrdering,
  pumpSortedIndexOrderingApply,
  getActiveSortedIndexAttribute,
} from '../../../../rendering/element-storage';
import type { RegistryEntry } from './types';
import { buildBehindCamera, buildColormapTexture } from './shared';
import {
  NEAR_PLANE_B_FAR,
  NEAR_PLANE_CAMERA,
  NEAR_PLANE_CONTROL_B_FAR,
  NEAR_PLANE_WIDTH,
  PERSPECTIVE_LINE_SCALE,
  nearPlaneSegments,
  worldLength,
  type Vec3,
} from '../../../helpers/line-join-near-plane-scenario';

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
interface LineFixtureStyle {
  readonly startColor?: readonly [number, number, number];
  readonly endColor?: readonly [number, number, number];
  readonly startWidth?: number;
  readonly endWidth?: number;
  readonly startSharpness?: number;
  readonly endSharpness?: number;
  readonly startJointCode?: number;
  readonly endJointCode?: number;
}

function lineTexelSource(
  start: readonly [number, number, number] = [-0.5, 0, 0],
  end: readonly [number, number, number] = [0.5, 0, 0],
  scalars?: readonly [number, number],
  alphas?: readonly [number, number],
  style: LineFixtureStyle = {}
): LineTexelSource {
  const startColor = style.startColor ?? [1.0, 0.5, 0.25];
  const endColor = style.endColor ?? [1.0, 0.5, 0.25];
  return {
    startPositions: new Float32Array([start[0], start[1], start[2]]),
    endPositions: new Float32Array([end[0], end[1], end[2]]),
    startColors: new Float32Array(startColor),
    endColors: new Float32Array(endColor),
    startWidths: new Float32Array([style.startWidth ?? 0.1]),
    endWidths: new Float32Array([style.endWidth ?? 0.1]),
    // Sharpness is the normalised [0, 1] knob -> super-Gaussian exponent
    // beta = 2^(6s - 2). 0.5 -> beta=2 (a true Gaussian, the default).
    startSharpness: new Float32Array([style.startSharpness ?? 0.5]),
    endSharpness: new Float32Array([style.endSharpness ?? 0.5]),
    segmentLengths: new Float32Array([1.0]),
    startJointCode: new Float32Array([style.startJointCode ?? 0]),
    endJointCode: new Float32Array([style.endJointCode ?? 0]),
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
  alphas?: readonly [number, number],
  style: LineFixtureStyle = {}
): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(24), 6, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  writeLineTexels(tex, lineTexelSource(start, end, scalars, alphas, style), 1);
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
      startJointCode: new Float32Array([0, ...real.startJointCode]),
      endJointCode: new Float32Array([0, ...real.endJointCode]),
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
  alphas?: readonly [number, number],
  style: LineFixtureStyle = {}
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
    { ...lineTexelSource(start, end, scalars, alphas, style), segmentCount: 1 },
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
  startJointCode: new Float32Array([0, 0, 0, 0]),
  endJointCode: new Float32Array([0, 0, 0, 0]),
};

/**
 * NON-identity draw-slot → storage-slot permutation under test —
 * applied via the production `writeSortedIndexOrdering` (the
 * SortWorker's write path).
 */
const SORTED_PERMUTED_ORDERING = new Uint32Array([2, 0, 3, 1]);

/**
 * Drain a staged ordering and leave it in the FRONT (slot 0) buffer.
 *
 * The parity harness builds materials by hand, so nothing pushes the
 * active slot into `uSortedIndexSlot` the way the depth-sort coordinator
 * does per frame; both backends therefore sample slot 0. Draining and then
 * folding the swapped-in permutation back onto slot 0 keeps the production
 * writer in the loop while matching what the hand-built shaders read.
 */
function drainOrderingOntoFrontBuffer(geom: THREE.InstancedBufferGeometry): void {
  for (let guard = 0; pumpSortedIndexOrderingApply(geom).more; guard++) {
    if (guard > 64) throw new Error('ordering stream did not converge');
  }
  const active = getActiveSortedIndexAttribute(geom);
  const front = geom.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
  if (active && active !== front) {
    (front.array as Uint32Array).set(active.array as Uint32Array);
    front.needsUpdate = true;
  }
}

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
  const geom = mesh.geometry as THREE.InstancedBufferGeometry;
  // Clear the identity `createInstancedLinesMesh` just wrote, so the
  // ordering below is the ONLY thing that can make the four segments read
  // four distinct storage slots. Without this the test passes whether or
  // not the ordering is ever applied — identity and any permutation both
  // draw all four segments, so the image is the same and the assertion is
  // vacuous. (The points twin gets this for free: its builder starts from
  // a bare `attachPointStorage`, which leaves the buffer zero-filled.)
  (geom.getAttribute('aSortedIndex').array as Uint32Array).fill(0);
  writeSortedIndexOrdering(geom, SORTED_PERMUTED_ORDERING, SORTED_PERMUTED_COUNT);
  // An ordering STAGES into the inactive buffer of the double-buffered pair
  // and swaps in when complete, so rendering straight after staging would
  // draw the un-permuted buffer — and this case exists precisely to prove
  // the permutation reaches the shader. Drain the pump the way the
  // per-frame scheduler does, then fold the result back onto slot 0:
  // production pushes the live slot into `uSortedIndexSlot`, but these
  // harness materials are hand-built (the TSL one has no writable uniform
  // map at all), so both backends read the default slot. The permutation
  // still comes from the production writer — only where it lands is
  // normalised.
  drainOrderingOntoFrontBuffer(geom);
  mesh.frustumCulled = false;
  return mesh;
}

function buildVisualLineUniforms(
  texture: THREE.DataTexture,
  isOrtho: boolean,
  nearCull = 0.01
): Record<string, THREE.IUniform> {
  return {
    uLineTex: { value: texture },
    uResolution: { value: new THREE.Vector2(64, 64) },
    uIsOrtho: { value: isOrtho ? 1 : 0 },
    uNearCull: { value: nearCull },
    uMaxLinePixelWidth: { value: 32.0 },
    uPerspectiveLineScale: { value: isOrtho ? 1.0 : 64.0 },
    uOrthoLineScale: { value: isOrtho ? 64.0 : 1.0 },
    uOpacity: { value: 1.0 },
    uInvGamma: { value: 1.0 },
    uIntensity: { value: 1.0 },
    uOffset: { value: 0.0 },
  };
}

function buildPickLineUniforms(
  texture: THREE.DataTexture,
  isOrtho: boolean,
  nearCull = 0.01
): Record<string, THREE.IUniform> {
  return {
    uLineTex: { value: texture },
    uResolution: { value: new THREE.Vector2(64, 64) },
    uIsOrtho: { value: isOrtho ? 1 : 0 },
    uNodeId: { value: 42 },
    uNearCull: { value: nearCull },
    uMaxLinePixelWidth: { value: 32.0 },
    uPerspectiveLineScale: { value: isOrtho ? 1.0 : 64.0 },
    uOrthoLineScale: { value: isOrtho ? 64.0 : 1.0 },
  };
}

function buildVisualLineTSLMaterial(
  uniforms: Record<string, THREE.IUniform>,
  isOrtho: boolean
): THREE.Material {
  const material = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
    gammaOne: true,
    isOrtho,
  }) as unknown as THREE.Material;
  material.transparent = false;
  material.blending = THREE.NoBlending;
  return material;
}

const REMAP_STYLE: LineFixtureStyle = {
  startColor: [1.0, 0.0, 0.0],
  endColor: [0.0, 0.0, 1.0],
  startWidth: 0.02,
  endWidth: 0.2,
  startSharpness: 0.0,
  endSharpness: 1.0,
};

/**
 * A real degree-2 JOINT: two thick segments meeting at a shallow V, wired with
 * slot-bearing joint codes so the miter block in both backends actually runs.
 *
 * Every other joint fixture carries only a SENTINEL code (0 free end, -1
 * clipped, -2 hub), all of which the join block rejects before it fetches
 * anything — so the whole screen-space miter path was structurally unreachable
 * from the parity suite. This is the fixture that reaches it.
 *
 * Geometry, and why each number matters:
 *   seg 0  (-0.6, -0.3) -> (0, 0)      seg 1  (0, 0) -> (0.6, -0.3)
 *   turn = dot(dirIn, dirOut) = 0.6, so grow = sqrt(2/1.6) = 1.12 <= 2 (inside
 *   the miter limit) and the axial reach is 6.4 * sqrt(0.25) = 3.2 px against a
 *   half-segment of ~10.7 px — comfortably inside the overshoot guard, so the
 *   joint IS mitred rather than falling back.
 *   width 0.1 x uOrthoLineScale 64 = 6.4 px half-width, clear of the 2 px
 *   rendered-width gate; a thinner line would skip the block and the fixture
 *   would silently go vacuous again.
 *
 * Codes follow compute_joint_codes: segment 0's END meets segment 1's START, so
 * seg0.endJointCode = +(1 + 1) = 2 and seg1.startJointCode = -(0 + 3) = -3.
 */
const JOIN_A_START: readonly [number, number, number] = [-0.6, -0.3, 0];
const JOIN_SHARED: readonly [number, number, number] = [0, 0, 0];
const JOIN_B_END: readonly [number, number, number] = [0.6, -0.3, 0];

function buildJoinTexelSource(): LineTexelSource {
  return {
    startPositions: new Float32Array([...JOIN_A_START, ...JOIN_SHARED]),
    endPositions: new Float32Array([...JOIN_SHARED, ...JOIN_B_END]),
    startColors: new Float32Array([1, 0.5, 0.25, 1, 0.5, 0.25]),
    endColors: new Float32Array([1, 0.5, 0.25, 1, 0.5, 0.25]),
    startWidths: new Float32Array([0.1, 0.1]),
    endWidths: new Float32Array([0.1, 0.1]),
    startSharpness: new Float32Array([0.5, 0.5]),
    endSharpness: new Float32Array([0.5, 0.5]),
    segmentLengths: new Float32Array([0.671, 0.671]),
    // seg0 free at its start, joining slot 1's START at its end;
    // seg1 joining slot 0's END at its start, free at its end.
    startJointCode: new Float32Array([0, -3]),
    endJointCode: new Float32Array([2, 0]),
  };
}

/** 6x2 data texture for any two-segment joint fixture. */
function buildJoinDataTexture(texels: LineTexelSource): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(48), 6, 2, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  writeLineTexels(tex, texels, 2);
  return tex;
}

function buildJoinMesh(material: THREE.Material, texels: LineTexelSource): THREE.Object3D {
  const mesh = createInstancedLinesMesh({ ...texels, segmentCount: 2 }, material);
  mesh.frustumCulled = false;
  return mesh;
}

/**
 * The camera + uniform configuration one join fixture PAIR is authored for.
 * Factored out so the ortho V above and the perspective near-plane scenario
 * below share `joinEntry`'s style plumbing instead of duplicating it — the
 * `uLineJoin` / factory-`join` coupling is the fiddly part and must be
 * identical in both pairs.
 */
interface JoinFixture {
  readonly texels: () => LineTexelSource;
  readonly isOrtho: boolean;
  /** Everything but `uLineJoin`, which `joinEntry` supplies. */
  readonly uniforms: (texture: THREE.DataTexture) => Record<string, THREE.IUniform>;
  readonly buildCamera?: () => THREE.Camera;
}

/** The original ortho V (see {@link buildJoinTexelSource}). */
const ORTHO_JOIN_FIXTURE: JoinFixture = {
  texels: buildJoinTexelSource,
  isOrtho: true,
  uniforms: (texture) => buildVisualLineUniforms(texture, true),
};

/**
 * @param join - the `uLineJoin` value. BOTH backends must be driven from this
 * one number: GLSL reads it as a runtime uniform while the TSL factory bakes it
 * into the graph (via `lineJoinStyleFromUniform` in the ShaderSource path), so
 * a fixture that set only one of them would compare a mitred quad against an
 * unmitred one and fail for the wrong reason.
 * @param fixture - which joint geometry/camera to render it on.
 */
function joinEntry(join: number, fixture: JoinFixture = ORTHO_JOIN_FIXTURE): RegistryEntry {
  return {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      ...fixture.uniforms(buildJoinDataTexture(fixture.texels())),
      uLineJoin: { value: join },
    }),
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '', LUXAR_MAX_RGB_CONTRIBUTION: '' }),
    buildTSLMaterial: (uniforms) => {
      const material = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'max',
        gammaOne: true,
        isOrtho: fixture.isOrtho,
        join: join > 0.5 ? 'miter' : 'none',
      }) as unknown as THREE.Material;
      material.transparent = false;
      material.blending = THREE.NoBlending;
      return material;
    },
    buildMesh: (material) => buildJoinMesh(material, fixture.texels()),
    ...(fixture.buildCamera ? { buildCamera: fixture.buildCamera } : {}),
  };
}

/**
 * The PERSPECTIVE near-plane joint of issue #1346, built entirely from
 * `tests/helpers/line-join-near-plane-scenario.ts` — the same module the vitest
 * mirror test consumes, so the two are provably one configuration.
 *
 * Segment A runs from the front to the shared vertex; segment B runs from the
 * shared vertex to a point BEHIND the near-cull plane while its own shared
 * endpoint stays unclipped. Under the old one-sided guard, A saw B's far
 * endpoint behind the plane and fell back to the plain perpendicular while B
 * saw A's in front and mitred ALONE — an asymmetric flap at the joint. With the
 * guard two-sided BOTH sides fall back and keep their code-implied cap, so the
 * `-miter` render must be pixel-identical to `-none`; that identity is what the
 * parity spec asserts, and it is sharp (pre-fix the two differed).
 *
 * The existing ortho `line-join-miter` pair cannot see this: `uIsOrtho == 1`
 * short-circuits the near-plane test to true on both sides, so no ortho fixture
 * can ever exercise the conjunction.
 *
 * Paired with a CONTROL fixture that moves B's far endpoint in front of the
 * plane and changes nothing else. "miter == none" is satisfied by every way the
 * join block can fail to RUN — width gate closed, mis-encoded joint code, wrong
 * storage slot, culled segment, quad off-screen — so on its own it would be a
 * weak claim. The control proves the miter fires under this exact camera and
 * these exact uniforms, which is what gives the identity next to it meaning.
 *
 * See the scenario module for why each number is what it is; the uniforms here
 * are the ones it is authored against (`uNearCull = 0.5` deliberately ABOVE the
 * camera's own 0.1 near plane, and the real
 * `uPerspectiveLineScale = resolutionY / tan(fov/2)`).
 *
 * @param bFar - segment B's far endpoint: `NEAR_PLANE_B_FAR` (behind the
 * near-cull plane, the reproducing case) or `NEAR_PLANE_CONTROL_B_FAR`.
 */
function buildNearPlaneJoinTexelSource(bFar: Vec3): LineTexelSource {
  const segments = nearPlaneSegments(bFar);
  const color = [1, 0.5, 0.25];
  return {
    startPositions: new Float32Array(segments.flatMap((s) => [...s.start])),
    endPositions: new Float32Array(segments.flatMap((s) => [...s.end])),
    startColors: new Float32Array([...color, ...color]),
    endColors: new Float32Array([...color, ...color]),
    startWidths: new Float32Array([NEAR_PLANE_WIDTH, NEAR_PLANE_WIDTH]),
    endWidths: new Float32Array([NEAR_PLANE_WIDTH, NEAR_PLANE_WIDTH]),
    startSharpness: new Float32Array([0.5, 0.5]),
    endSharpness: new Float32Array([0.5, 0.5]),
    segmentLengths: new Float32Array(segments.map((s) => worldLength(s.start, s.end))),
    startJointCode: new Float32Array(segments.map((s) => s.startJointCode)),
    endJointCode: new Float32Array(segments.map((s) => s.endJointCode)),
  };
}

/**
 * This pair's OWN camera, constructed from `NEAR_PLANE_CAMERA` rather than
 * reusing `buildBehindCamera`. The shared one is wired into ~20 unrelated point
 * / gsplat / line fixtures, so retuning it for one of those would silently move
 * this joint out of the configuration the scenario module's depths, half-widths
 * and `turn` are computed for. Building it from the same constants keeps the
 * numbers and the camera from drifting apart.
 */
function buildNearPlaneJoinCamera(): THREE.Camera {
  const { fovDegrees, aspect, near, far, camZ } = NEAR_PLANE_CAMERA;
  const camera = new THREE.PerspectiveCamera(fovDegrees, aspect, near, far);
  camera.position.set(0, 0, camZ);
  camera.lookAt(0, 0, 0);
  return camera;
}

/**
 * The reproducing case: B's far endpoint BEHIND the near-cull plane.
 *
 * Every uniform the scenario's arithmetic depends on is taken FROM the scenario
 * module — `uNearCull`, `uPerspectiveLineScale`, `uMaxLinePixelWidth` — so the
 * numbers and the fixture cannot drift apart. `uResolution` is the deliberate
 * exception: it must equal the harness's fixed render-target edge
 * (`HARNESS_SIZE` in `./render`), which `buildVisualLineUniforms` already
 * supplies, and overriding it from `NEAR_PLANE_CAMERA.resolution` would create a
 * shader/target mismatch on a harness resize rather than a coupling. See that
 * field's doc comment.
 */
const NEAR_PLANE_JOIN_FIXTURE: JoinFixture = {
  texels: () => buildNearPlaneJoinTexelSource(NEAR_PLANE_B_FAR),
  isOrtho: false,
  uniforms: (texture) => ({
    ...buildVisualLineUniforms(texture, false, NEAR_PLANE_CAMERA.nearCull),
    uPerspectiveLineScale: { value: PERSPECTIVE_LINE_SCALE },
    uMaxLinePixelWidth: { value: NEAR_PLANE_CAMERA.maxLinePixelWidth },
  }),
  buildCamera: buildNearPlaneJoinCamera,
};

/**
 * The CONTROL: identical camera, uniforms and joint codes, with B's far endpoint
 * moved in FRONT of the near-cull plane. Both sides mitre here, so this pair's
 * `miter` must differ strongly from its `none` — the proof that the join block
 * runs at all under this camera, without which the pair above could pass by
 * never reaching the block.
 */
const NEAR_PLANE_CONTROL_JOIN_FIXTURE: JoinFixture = {
  ...NEAR_PLANE_JOIN_FIXTURE,
  texels: () => buildNearPlaneJoinTexelSource(NEAR_PLANE_CONTROL_B_FAR),
};

function jointCodeEntry(jointCode: number): RegistryEntry {
  const style: LineFixtureStyle = {
    startJointCode: jointCode,
    endJointCode: jointCode,
  };
  return {
    source: LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(
        buildLineDataTexture([-0.5, 0, 0], [0.5, 0, 0], undefined, undefined, style),
        true
      ),
    // Max-mode premultiplies RGB by the cap/profile intensity, making the
    // suppression value observable in readback even with NoBlending (normal
    // mode carries coverage only in alpha, while the opaque target resolves
    // alpha to 1).
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '', LUXAR_MAX_RGB_CONTRIBUTION: '' }),
    buildTSLMaterial: (uniforms) => {
      const material = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'max',
        gammaOne: true,
        isOrtho: true,
      }) as unknown as THREE.Material;
      material.transparent = false;
      material.blending = THREE.NoBlending;
      return material;
    },
    buildMesh: (material) =>
      buildLineInstancedMesh(material, [-0.5, 0, 0], [0.5, 0, 0], undefined, undefined, style),
  };
}

/**
 * Registry of line shader entries for the TSL↔GLSL parity harness, keyed by
 * test name. Each entry carries the GLSL source and a `buildUniforms` factory;
 * merged into `SHADER_REGISTRY` and driven by the parity/codegen specs.
 */
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
  // Rendered endpoint-cap ladder over the JOINT CODES that decide it
  // (texel4.yz; see GLSL_LINE_JOINT_CODE). A free end and a hub keep the soft
  // cap, a slice-clipped endpoint suppresses it — so the parity spec requires
  // free-end and hub to match each other and clipped to be visibly brighter, on
  // BOTH backends. This replaces a ladder of 0.0 / 0.5 / 1.0 that fed the slot
  // a continuous suppression scalar: 0.5 is not a representable code (it decodes
  // to neither predicate, rendering identically to the fully-suppressed case)
  // and 1.0 decodes to a partner reference to storage slot 0 — itself, in a
  // single-segment fixture, which the kernel can never emit.
  'line-joint-free-end': jointCodeEntry(0.0),
  'line-joint-hub': jointCodeEntry(-2.0),
  'line-joint-clipped': jointCodeEntry(-1.0),
  // The screen-space MITER (#790) on a real two-segment joint — the only
  // fixtures whose joint codes name a partner, so the only ones that reach the
  // join block at all. Pinned as a PAIR: `-miter` must match across backends,
  // and the parity spec additionally requires it to DIFFER from `-none`, which
  // is what proves the join is running rather than being silently skipped.
  'line-join-miter': joinEntry(1.0),
  'line-join-none': joinEntry(0.0),
  // The PERSPECTIVE near-plane joint of #1346 (see NEAR_PLANE_JOIN_FIXTURE):
  // segment B runs off behind the near-cull plane while its shared endpoint
  // stays unclipped. Pinned as a pair whose two members must be pixel-IDENTICAL
  // — with the near-plane guard two-sided both segments fall back and keep
  // their code-implied cap, so styling the joint `miter` changes nothing here.
  // Pre-fix B mitred alone and the two differed.
  'line-join-nearplane-miter': joinEntry(1.0, NEAR_PLANE_JOIN_FIXTURE),
  'line-join-nearplane-none': joinEntry(0.0, NEAR_PLANE_JOIN_FIXTURE),
  // ...and its CONTROL, same camera and uniforms with B's far endpoint moved in
  // FRONT of the plane. This pair must DIFFER, which is what proves the join
  // block runs under this camera at all — otherwise the identity above would be
  // satisfied by every way the block can fail to run.
  'line-join-nearplane-control-miter': joinEntry(1.0, NEAR_PLANE_CONTROL_JOIN_FIXTURE),
  'line-join-nearplane-control-none': joinEntry(0.0, NEAR_PLANE_CONTROL_JOIN_FIXTURE),
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
  // `blendingMode: 'volumetric'`) — τ = κ·alpha (κ times the same ray
  // mass the additive branch emits), S(τ)
  // screening, physical absorption alpha, AND the per-endpoint
  // texel5.zw alphas (0.6 → 0.9) → w(a) = −ln(1−a) optical depth via
  // uHasElementAlpha = 1. uAbsorption 2.0 + opacity 0.7 keep τ
  // mid-range on the line body so S(τ)
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
  // fragment outputs (nodeId, elementIdLow16, brightness, elementIdHigh16) and
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
  // Endpoint exactly ON nearCull: strict `<` means neither crossing branch
  // fires. The visual and picking twins below pin that boundary across GLSL
  // and TSL (w == nearCull remains finite and the segment still renders).
  'line-on-near-plane': {
    source: LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(buildLineDataTexture([-0.2, 0, 0.5], [0.2, 0, 0]), false, 0.5),
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '' }),
    buildTSLMaterial: (uniforms) => buildVisualLineTSLMaterial(uniforms, false),
    buildMesh: (material) => buildLineInstancedMesh(material, [-0.2, 0, 0.5], [0.2, 0, 0]),
    buildCamera: buildBehindCamera,
  },
  'line-pick-on-near-plane': {
    source: LINE_PICK_SOURCE,
    buildUniforms: () =>
      buildPickLineUniforms(buildLineDataTexture([-0.2, 0, 0.5], [0.2, 0, 0]), false, 0.5),
    buildTSLMaterial: (uniforms) =>
      linePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(uniforms), {
        isOrtho: false,
      }) as unknown as THREE.Material,
    buildMesh: (material) => buildLineInstancedMesh(material, [-0.2, 0, 0.5], [0.2, 0, 0]),
    buildCamera: buildBehindCamera,
  },
  // Distinct endpoint colour/width/sharpness on a crossing segment. With
  // nearCull=0.5 the clipped start is tA=2/3, so its visible side must already
  // be blue-dominant and substantially wider than the raw red/thin start.
  // This makes a `t`/swapped-endpoint remap bug observable in pixels instead
  // of passing vacuously with identical endpoint attributes.
  'line-crossing-remap': {
    source: LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(
        buildLineDataTexture([0.15, 0, 1.5], [0.15, 0, 0], undefined, undefined, REMAP_STYLE),
        false,
        0.5
      ),
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '' }),
    buildTSLMaterial: (uniforms) => buildVisualLineTSLMaterial(uniforms, false),
    buildMesh: (material) =>
      buildLineInstancedMesh(
        material,
        [0.15, 0, 1.5],
        [0.15, 0, 0],
        undefined,
        undefined,
        REMAP_STYLE
      ),
    buildCamera: buildBehindCamera,
  },
  'line-pick-crossing-remap': {
    source: LINE_PICK_SOURCE,
    buildUniforms: () =>
      buildPickLineUniforms(
        buildLineDataTexture([0.15, 0, 1.5], [0.15, 0, 0], undefined, undefined, REMAP_STYLE),
        false,
        0.5
      ),
    buildTSLMaterial: (uniforms) =>
      linePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(uniforms), {
        isOrtho: false,
      }) as unknown as THREE.Material,
    buildMesh: (material) =>
      buildLineInstancedMesh(
        material,
        [0.15, 0, 1.5],
        [0.15, 0, 0],
        undefined,
        undefined,
        REMAP_STYLE
      ),
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
  // REVERSED crossing: identical segment to `line-crossing` but with the
  // endpoints SWAPPED — start at the origin (view depth 1.0, in front),
  // end at world z=1.5 (view depth -0.5, BEHIND the camera). This drives
  // the DISTINCT tB clip branch (end behind, start in front), whereas
  // `line-crossing` only exercises tA (start behind). The physical
  // clipped segment — and therefore the rendered footprint — is the same,
  // so the parity test reuses `line-crossing`'s content assertions. This
  // entry uses LINE_SOURCE, so it covers only the VISUAL tB branch (both
  // visual backends); its picking twin `line-pick-crossing-reversed`
  // covers the picking tB branch — together the four backends. A
  // sign/ordering slip in the visual tB branch droops the band off the
  // centerline and fails the test.
  'line-crossing-reversed': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture([0.15, 0, 0], [0.15, 0, 1.5]) },
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
    buildMesh: (m) => buildLineInstancedMesh(m, [0.15, 0, 0], [0.15, 0, 1.5]),
    buildCamera: buildBehindCamera,
  },
  // PICKING twin of `line-crossing`: the same camera-plane-crossing
  // segment through the LINE_PICK path (perspective, nodeId/elementId/
  // brightness output). This is the ONLY fixture that reaches the
  // picking-shader near-plane segment clip: `line-pick` is ortho (clip
  // never built) and `line-pick-behind` is both-endpoints-behind (the
  // bothBehind cull fires first). Deleting the picking clip on EITHER
  // backend droops the wrapped-quad wedge off the centerline, which the
  // parity test's per-backend centroid assertion catches.
  'line-pick-crossing': {
    source: LINE_PICK_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture([0.15, 0, 1.5], [0.15, 0, 0]) },
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
    buildMesh: (m) => buildLineInstancedMesh(m, [0.15, 0, 1.5], [0.15, 0, 0]),
    buildCamera: buildBehindCamera,
  },
  // PICKING twin of `line-crossing-reversed`: identical to
  // `line-pick-crossing` but with the endpoints SWAPPED — start at the
  // origin (view depth 1.0, in front), end at world z=1.5 (view depth
  // -0.5, BEHIND the camera). This drives the picking shaders' DISTINCT
  // tB clip branch (end behind, start in front); `line-pick-crossing`
  // only exercises tA. The physical clipped segment is the same, so the
  // parity test reuses `line-pick-crossing`'s per-backend centroid
  // assertion — a broken picking tB branch on either backend is caught:
  // dropping the clip leaves the behind endpoint unclipped and droops the
  // wrapped-quad wedge off the centerline (centroid tooth), while a
  // sign/ordering slip collapses the band to a few columns (qualifying-
  // column-count tooth).
  'line-pick-crossing-reversed': {
    source: LINE_PICK_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture([0.15, 0, 0], [0.15, 0, 1.5]) },
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
    buildMesh: (m) => buildLineInstancedMesh(m, [0.15, 0, 0], [0.15, 0, 1.5]),
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
