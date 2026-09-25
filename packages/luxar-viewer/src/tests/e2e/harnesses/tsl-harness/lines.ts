/**
 * Line shader family for the TSL ↔ GLSL parity harness: the visual
 * instanced-line variants (gamma / no-GOG fast paths, max-mode
 * premultiply, volumetric emission–absorption, colormap LUT,
 * behind-camera + ortho-near culling, sorted-index permutation) plus
 * the line-pick counterparts + multi-row, cap-suppression, clipping-remap,
 * and exact-near-plane boundary variants, plus the capsule-primitive
 * (`line-capsule-*`) family, visual and pick alike (hardcoded entry
 * counts drift — count `Object.keys(LINE_SHADERS)` when it matters).
 *
 * @module tests/e2e/harnesses/tsl-harness/lines
 */

import * as THREE from 'three';
import { LINE_SOURCE } from '../../../../rendering/materials/line/shader-glsl';
import { CAPSULE_LINE_SOURCE } from '../../../../rendering/materials/line/shader-glsl-capsule';
import { capsuleLineWebGPUFactory } from '../../../../rendering/materials/line/shader-tsl-capsule';
import { CAPSULE_LINE_PICK_SOURCE } from '../../../../rendering/picking/line/shaders-capsule';
import { capsuleLinePickWebGPUFactory } from '../../../../rendering/picking/line/pick-capsule.tsl';
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
import {
  buildBehindCamera,
  buildColormapTexture,
  buildCubeFaceCamera,
  buildCubeFaceEquivalentCamera,
  buildOrthoCamera,
} from './shared';

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
  /** World length of the segment (texel slot); defaults to 1.0. */
  readonly segmentLength?: number;
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
    segmentLengths: new Float32Array([style.segmentLength ?? 1.0]),
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

/**
 * FAT geometry for the pick/visible footprint-agreement fixtures
 * (#1352 PR-3): drawn half-width ≈ 19 px at the standard ortho scale 64,
 * so a σ-level footprint divergence moves the coverage boundary by
 * multiple pixels instead of hiding inside the rasterisation ribbon.
 */
const FOOTPRINT_STYLE: LineFixtureStyle = {
  startWidth: 0.3,
  endWidth: 0.3,
};

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
 *   width 0.1 x line scale 64 = 6.4 px half-width, clear of the 2 px
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

/**
 * SAME-PARITY twin of {@link buildJoinTexelSource}: the identical V, but with
 * segment 1 AUTHORED IN REVERSE so the two segments meet END to END.
 *
 * The end->start fixture above cannot see the join's orientation rule at all:
 * there `atEnd` and `partnerSharesItsStart` agree, so orienting the partner leg
 * on either one gives the same `turn`. At an END-END joint they disagree, and
 * the pre-fix orientation handed back the partner's own traversal direction —
 * the NEGATION of what the joint needs. This fixture is the one that renders
 * that difference.
 *
 * Geometry, re-derived the way the fixture above documents it:
 *   seg 0  (-0.6, -0.3) -> (0, 0)      seg 1  (0.6, -0.3) -> (0, 0)
 *   From seg 0's END: lineDir = (0.894, 0.447), the partner's far endpoint is
 *   its START at pixel (19.2, -9.6), so partnerDir = (0.894, -0.447) and
 *   turn = 0.6 — the SAME joint angle as the end->start fixture, as it must be
 *   (turn describes the geometry, not the authoring). grow = sqrt(2/1.6) = 1.12
 *   <= 2 and the axial reach is 6.4 * sqrt(0.25) = 3.2 px against a half-leg of
 *   ~10.7 px, so the joint IS mitred and the cap is 1.0.
 *   Pre-fix the same call read turn = -0.6: grow = 2.24 > 2 rejected the miter
 *   AND clamp(-0.6, 0, 1) = 0 kept the soft endpoint cap, so the joint rendered
 *   DIMMER than with the join disabled. That inversion is what the parity spec
 *   asserts against.
 *   Widths are unchanged at 0.1 x line scale 64 = 6.4 px half-width, clear
 *   of the 2 px rendered-HALF-width gate.
 *
 * Codes follow compute_joint_codes: seg 0's END meets seg 1's END, so
 * seg0.endJointCode = -(1 + 3) = -4 and seg1.endJointCode = -(0 + 3) = -3; both
 * outer endpoints are free ends (0).
 */
function buildSameParityJoinTexelSource(): LineTexelSource {
  const src = buildJoinTexelSource();
  return {
    ...src,
    startPositions: new Float32Array([...JOIN_A_START, ...JOIN_B_END]),
    endPositions: new Float32Array([...JOIN_SHARED, ...JOIN_SHARED]),
    startJointCode: new Float32Array([0, 0]),
    endJointCode: new Float32Array([-4, -3]),
  };
}

/**
 * World position whose projection under {@link buildBehindCamera} (a 60 deg
 * perspective camera at world z = 1) lands `px`/`py` pixels from the centre of
 * the 64x64 viewport at view depth `depth`:
 *   ndc = px / (0.5 * resolution) = px / 32,  world = ndc * tan(fov/2) * depth.
 * Lets the taper fixture below be specified in the pixel space the join math
 * actually works in.
 */
function perspectivePixelPos(px: number, py: number, depth: number): [number, number, number] {
  const scale = (Math.tan(Math.PI / 6) * depth) / 32;
  return [px * scale, py * scale, 1 - depth];
}

/**
 * TAPERED, PERSPECTIVE joint: the first line fixture to reach the join block
 * off the ortho fast path, and the one that pins the DECLINED-miter cap
 * derivation. Neither was reachable from this suite before.
 *
 * It does NOT gate the 2 px WIDTH straddle, despite the taper. Segment 0's
 * rendered half-width really does run from 1.79 px at t = 0 to 6.4 px at t = 1,
 * but this source spreads {@link buildJoinTexelSource} and overrides only the
 * positions, widths and lengths — so the t = 0 end keeps that source's FREE-END
 * sentinel (startJointCode 0) and the block short-circuits on `namesAPartner`
 * there whatever its width is. The two partner-naming ends are both at 6.4 px,
 * 3.2x the gate; raising the 0.028 start width would not move a pixel.
 *
 * Nor does it gate the near-plane conjunction. `buildVisualLineUniforms` leaves
 * nearCull at its 0.01 default and every endpoint here is at depth 1.0 or 1.5,
 * so the predicate is emitted rather than constant-folded but is never false —
 * deleting either conjunct changes nothing on this fixture.
 * {@link NEAR_PLANE_JOIN} is the pair that gates it.
 *
 * It does NOT witness the provoking-vertex divergence a per-VERTEX join width
 * would introduce, either. The TSL side renders through
 * `WebGPURenderer({ forceWebGL: true })` (see `./render.ts`), so both backends
 * compile to GLSL and share one provoking-vertex rule, and a divergent `flat`
 * cap would resolve the same way on each. The per-END width is pinned instead
 * by the source assertion `#790 both vertex stages hand luxarLineJoin each END
 * its own segment-constant width` in
 * `tests/unit/rendering/materials/line/material-glsl.test.ts`, the codegen
 * snapshot `tests/__codegen__/line.vertex.glsl.txt` (generated from the TSL
 * graph, so it pins the TSL twin in local full-suite runs), the
 * always-running TSL-source lock
 * `tests/unit/rendering/materials/line/join-width-tsl.test.ts`, and the CPU
 * mirror in `tests/unit/rendering/line-join-math.test.ts`.
 *
 * The joint is deliberately one the block DECLINES rather than mitres: a mitred
 * joint's cap is 1.0, which is also the code-implied default of a slot-bearing
 * joint code, so the derivation would leave no trace in the image. Declining
 * derives clamp(turn, 0, 1) instead, which here is 0 — maximally far from the
 * default, and the only thing this pair renders that no other join fixture does.
 *
 * Geometry, in the pixel space the join works in (offsets from the viewport
 * centre; see {@link perspectivePixelPos}):
 *   seg 0  (-8, -6) -> (0, 0) at depth 1      seg 1  (0, 0) -> (6, -8) at depth 1.5
 *   Both legs project 10 px long. dirIn = (0.8, 0.6), dirOut = (0.6, -0.8), so
 *   turn = 0 (a right angle): grow = sqrt(2) is inside the miter limit but the
 *   axial reach 6.4 * sqrt(2 - 1) = 6.4 px exceeds 0.5 * min(10, 10) = 5 px, so
 *   both sides decline and derive cap = clamp(0, 0, 1) = 0.
 *   Widths follow the perspective branch, width * lineScale / depth
 *   with the harness's scale of 64: seg 0 is 0.028 -> 1.79 px at t = 0 and
 *   0.1 -> 6.4 px at t = 1; seg 1 is 0.1 -> 6.4 px and 0.15 -> 6.4 px, i.e.
 *   constant on screen. Only the two 6.4 px ends name a partner (see above).
 *   The two legs sit at DIFFERENT depths (1 and 1.5) so the near-plane
 *   conjunction reads two distinct values rather than one number twice — both
 *   of them far in front of nearCull, so it stays unconditionally true.
 */
const TAPER_A_START = perspectivePixelPos(-8, -6, 1);
const TAPER_SHARED = perspectivePixelPos(0, 0, 1);
const TAPER_B_END = perspectivePixelPos(6, -8, 1.5);

function buildTaperJoinTexelSource(): LineTexelSource {
  const src = buildJoinTexelSource();
  return {
    ...src,
    startPositions: new Float32Array([...TAPER_A_START, ...TAPER_SHARED]),
    endPositions: new Float32Array([...TAPER_SHARED, ...TAPER_B_END]),
    startWidths: new Float32Array([0.028, 0.1]),
    endWidths: new Float32Array([0.1, 0.15]),
    // World-space leg lengths (the cap ramp's axial scale).
    segmentLengths: new Float32Array([0.1804, 0.5685]),
  };
}

/**
 * NEAR-PLANE joint: the fixture that makes the two-sided near-plane guard a
 * RENDERED gate rather than a shader-text assertion.
 *
 * The guard is a conjunction over BOTH far endpoints of the joint, not just the
 * fetched partner's. Nothing in this suite could see that: every other join
 * fixture sits far in front of nearCull on both legs, so the predicate is
 * unconditionally true. Here segment 0's far endpoint is BEHIND the cull plane
 * and segment 1's is in front, which is exactly the asymmetry the conjunction
 * exists for.
 *
 *   nearCull 0.6 (threaded through `joinEntry`), perspective camera.
 *   seg 0  depth 0.3 -> shared at depth 1.0     seg 1  shared -> depth 1.5
 *
 * One-sided (pre-fix), each side tested only the endpoint it FETCHED:
 *   - seg 0's END fetches seg 1's far endpoint at depth 1.5, which clears 0.6,
 *     so it accepted and mitred;
 *   - seg 1's START fetches seg 0's far endpoint at depth 0.3, which does not,
 *     so it declined.
 * Segment 0 then mitred alone and its rotated end edge had nothing to tile
 * against — the flap. With the conjunction seg 0 also tests its OWN far
 * endpoint (0.3 < 0.6), so both sides take the `noJoin` path and the mitred
 * render IS the unmitred one, pixel for pixel. That equality is the assertion.
 *
 * Every OTHER reason to decline is ruled out, or the gate would be vacuous
 * (offsets from the viewport centre; see {@link perspectivePixelPos}):
 *   Camera near is 0.1, clear of the 0.3 endpoint. Segment 0 is near-plane
 *   SEGMENT-clipped at tA = 3/7 onto the 0.6 plane, which lands its rendered
 *   start 10 px left of the shared vertex (the unclipped start is 35 px out at
 *   depth 0.3); the shared vertex itself projects to the centre of the 64x64
 *   viewport. Segment 1 leaves along (0.6, 0.8) for 15 px. So dirIn = (1, 0),
 *   dirOut = (0.6, 0.8) and turn = 0.6: grow = sqrt(2/1.6) = 1.12 <= 2 and the
 *   axial reach 6.4 * 0.5 = 3.2 px clears both sides' overshoot guards
 *   (0.5 * min(10, 15) = 5 px from seg 0, 0.5 * min(15, 17.5) = 7.5 px from
 *   seg 1 — its partner leg is projected under the nearCull w-guard, so seg 0
 *   reads 17.5 px there rather than 35). Both partner-naming ends render at
 *   0.1 * 64 / 1.0 = 6.4 px of half-width, 3.2x the 2 px gate.
 *
 * Codes are {@link buildJoinTexelSource}'s: seg 0's END meets seg 1's START,
 * outer ends free.
 */
const NEAR_PLANE_CULL = 0.6;
/**
 * The lowered plane of {@link NEAR_PLANE_CONTROL_JOIN} — under segment 0's 0.3
 * far endpoint, so the same joint mitres instead of declining.
 */
const NEAR_PLANE_CONTROL_CULL = 0.2;
const NEAR_PLANE_A_START = perspectivePixelPos(-35, 0, 0.3);
const NEAR_PLANE_SHARED = perspectivePixelPos(0, 0, 1);
const NEAR_PLANE_B_END = perspectivePixelPos(9, 12, 1.5);

function buildNearPlaneJoinTexelSource(): LineTexelSource {
  const src = buildJoinTexelSource();
  return {
    ...src,
    startPositions: new Float32Array([...NEAR_PLANE_A_START, ...NEAR_PLANE_SHARED]),
    endPositions: new Float32Array([...NEAR_PLANE_SHARED, ...NEAR_PLANE_B_END]),
    // World-space leg lengths (the cap ramp's axial scale).
    segmentLengths: new Float32Array([0.7252, 0.644]),
  };
}

/**
 * SHARP-FOLD joint (#1352): two segments sharing a vertex with a 122.7°
 * direction change. Every interior end renders its half of the joint
 * DISC (round cap partitioned at the bisector), so the wide variant
 * (width 0.3 ⇒ ~12 px capsule radius at ortho scale 64) fills its fold
 * tip; the `thin` variant (width 0.02 ⇒ clamped to the 1.5 px floor)
 * exercises the same path at a hairline-sized joint disc.
 */
function buildFoldJoinTexelSource(width: number): LineTexelSource {
  return {
    startPositions: new Float32Array([-0.8, 0.35, 0, 0.0, 0.0, 0]),
    endPositions: new Float32Array([0.0, 0.0, 0, -0.75, -0.5, 0]),
    startColors: new Float32Array([1, 1, 1, 1, 1, 1]),
    endColors: new Float32Array([1, 1, 1, 1, 1, 1]),
    startWidths: new Float32Array([width, width]),
    endWidths: new Float32Array([width, width]),
    startSharpness: new Float32Array([0.5, 0.5]),
    endSharpness: new Float32Array([0.5, 0.5]),
    segmentLengths: new Float32Array([0.87, 0.9]),
    // seg0's END joins seg1 (slot 1, shares its START ⇒ code slot+1 = 2);
    // seg1's START joins seg0 (slot 0, shares its END ⇒ code −(slot+3) = −3).
    startJointCode: new Float32Array([0, -3]),
    endJointCode: new Float32Array([2, 0]),
  };
}

/**
 * DEFICIT-PACKET joint (#1488): {@link buildJoinTexelSource}'s V with
 * ALTERNATING vertex widths, so the packet gate opens on the TAPER clause and
 * the deficit term has real work to do.
 *
 * The gate is not otherwise unreached here — {@link buildFoldJoinTexelSource}
 * at width 0.3 opens it via the ANGLE clause (its fold turns the direction by
 * 122.7°, giving dot(q_hat, u) = -+0.540, just past the -+0.5 threshold, at
 * radius 12.65 px ⇒ rMax 13.15, well over the 4 px width gate). Its legs are
 * CONGRUENT, though, so the deficit term it computes is ~0 and the branch is
 * exercised without being stressed: reverting the TSL reach to the half disc
 * moves 244 of that fixture's pixels at up to 255 per channel, yet dilutes to
 * a shared-loop mean of 0.1956 and fails nothing (measured before this branch
 * merged main's #1495/#1494 work and not re-measured after — every radius here
 * clears the AA floor, so the widthScale factor is 1 and the comparison should
 * carry, but read the counts as indicative). `line-capsule-fold-mid` (width
 * 0.05, 2.11 px) reaches the branch too, through #1495's floored sharp-turn
 * escape rather than the width gate, and its legs are congruent as well.
 * (`line-capsule-fold-thin` is width-gated off at rMax 2.0 — its 122.7° turn
 * does clear the sharp-turn test, but #1495's escape also demands both RAW
 * radii reach the 1.5 px AA floor and its width 0.02 gives 0.84 px, so the
 * gate holds;
 * `line-capsule-joint`, uniform 0.1 widths, genuinely closes all four clauses
 * at both ends and never reaches the branch at all.)
 * {@link buildShortPartnerJoinTexelSource} (#1490) reaches the branch with a
 * genuinely non-congruent pair, so it is not a fourth congruent case — but it
 * is tuned to the far-cap term rather than the reach: max-mode blending, a
 * 2.56 px partner against a 16 px leg, and its long end's gate opens on the
 * LENGTH clause as much as the taper one.
 *
 * This fixture instead opens the gate the way the deficit rule was designed
 * for — a fat vertex between thin neighbours, where the fat disc keeps the
 * half a thin partner cannot render — so the chop a short reach produces is
 * large, local and worth asserting on. See the dedicated divergent-pixel test
 * in `tsl-shader-parity.spec.ts`, which the shared mean cannot replace.
 *
 * Widths, and why each number matters (ortho, line scale 64, radius =
 * width x 64 x CAPSULE_RADIUS_PER_QUAD_HALFWIDTH = width x 42.18 px):
 *   Shared vertex 0.20 -> 8.44 px; both OUTER ends 0.06 -> 2.53 px. A fat
 *   vertex between two thin neighbours is precisely the case the deficit rule
 *   exists for (its disc keeps the half a thin partner cannot render).
 *   - Both outer radii clear the 1.5 px AA floor (2.53 px), so the taper is
 *     AUTHORED rather than manufactured by the clamp — a floored partner would
 *     open the ratio clause for the wrong reason.
 *   - rMax = max(8.44, 2.53) + 0.5 apron = 8.94 px at BOTH segments, 2.2x
 *     CAPSULE_JOINT_PACKET_MIN_RADIUS_PX = 4 px. Below that width gate the
 *     vertex stage skips the packet math outright — #1495's sharp-turn escape
 *     cannot lift it here, the turn being the wrong sign at both ends (below) —
 *     and the fixture would be vacuous however steep the taper.
 *   - The gate opens on the TAPER-RATIO clause alone, at both ends:
 *     |1 - 2.53/8.44| = 0.700, 35x the 0.02 tolerance. The other three stay
 *     closed and are worth knowing as such — own-widening (2.53 > 8.44 x 1.02
 *     is false), partner length (21.47 px projected vs the 2 x 8.44 = 16.87 px
 *     threshold, clear by 27%), and the turn, whose clause is SIGNED: end A
 *     needs dot(q_hat, u) > +0.5 and measures -0.6, end B needs < -0.5 and
 *     measures +0.6, so both are the wrong sign and the clause is shut.
 *
 * Positions, lengths and joint codes are {@link buildJoinTexelSource}'s
 * untouched: same V, same slot wiring, so the only difference from
 * `line-capsule-joint` is which vertex-stage branch runs.
 *
 * VISUAL ONLY — there is deliberately no pick twin. The two pick surfaces
 * carry the same packet branch and the same reach rule, so one would be
 * worth having, but this harness cannot host it: `line-capsule-pick-joint`
 * ALREADY spends 1.8462 of its 2.0 per-covered-pixel budget on 7 pixels in
 * column x = 31, the shared-vertex column, where the two backends attribute
 * the seam to different legs (GLSL writes [255, 255, b, 0] with b in 0..3;
 * TSL writes [255, 0, B, 0] with B in 22..254 — a different element id AND a
 * different brightness). Widening the joint reproduces that signature on 18
 * pixels and a pick twin measures 3.1265, over budget. The failure is the
 * pick pass's seam attribution, unrelated to the deficit region or the
 * stencil reach, so a twin would only land a red test that blames the wrong
 * thing. Nor would bolting the visual twin's divergent-pixel test onto it
 * help: with the twin built and the pick TSL reach reverted, the >24/255
 * population stays at 18 and the mean moves only 3.1265 -> 3.2840. The reach
 * IS observable on this surface — the population differing AT ALL moves
 * 18 -> 272 of 492 covered, worst channel 255 — just not by either metric
 * this suite scores by. A future pick twin would therefore have to assert an
 * ANY-DIFFERENCE population bound, and to do it outside the shared
 * mean-based loop, which fails the fixture at 3.1265 on the pre-existing seam
 * before the reach is even in question. Until then the pick surfaces'
 * full-disc reach is pinned by the unit source pin in
 * `tests/unit/rendering/materials/line-capsule.test.ts` alone.
 */
function buildDeficitPacketJoinTexelSource(): LineTexelSource {
  const src = buildJoinTexelSource();
  return {
    ...src,
    // seg0 runs thin -> fat into the shared vertex; seg1 runs fat -> thin out.
    startWidths: new Float32Array([0.06, 0.2]),
    endWidths: new Float32Array([0.2, 0.06]),
  };
}

/**
 * SHORT-PARTNER joint (#1490): the first fixture in this suite that publishes
 * a deficit packet whose partner ENDS INSIDE the joint disc — the regime where
 * the fragment's far-cap term (`clamp(xp, 0, cut.w)` on the radius,
 * `xp - cut.w` on the overshoot) is the only thing standing between the
 * reconstructed partner rod and an unbounded one.
 *
 * Nothing else here reaches it. `line-capsule-joint` / `-fold` / `-fold-thin`
 * / `-fold-mid` / `-pick-joint` all carry UNIFORM widths, so their packets
 * (where they open at all) have gradient 0 and a partner far longer than the
 * disc: `-fold`'s ql is ~28.8 px against a ~12 px radius, so `xp` never
 * reaches `cut.w` and the far-cap term is inert.
 * {@link buildDeficitPacketJoinTexelSource} (`-joint-taper`, #1488) DOES taper,
 * so its gradient is nonzero, but its partner is long for the same reason: ql
 * 21.47 px against an 8.44 px joint radius, so `xp` stays well inside `cut.w`
 * over the leg's whole support and the far cap never binds there either.
 * Deleting the far cap from all
 * four capsule shader sources left the whole viewer unit suite green — the
 * companion source lock
 * `tests/unit/rendering/materials/line/capsule-joint-packet-source-lock.test.ts`
 * closes that hole by text; this pair closes it by pixels. Mutation-proved
 * both ways: unbinding the rod in the VISUAL GLSL fragment
 * alone fails `line-capsule-joint-short-partner` at 4.433 per-covered-pixel
 * (bound 2.0) with every other capsule test still green, and the same edit to
 * the PICK GLSL fragment alone fails `line-capsule-pick-joint-short-partner`
 * at 2.715, again with every other one green. (Counts of the selected fleet are
 * deliberately not quoted — they move whenever a fixture lands; the ASYMMETRY
 * is the claim.)
 *
 * Geometry, every number derived at the harness's ortho mapping — the default
 * `OrthographicCamera(-1, 1, 1, -1)` over a 64x64 target is 32 px per world
 * unit, and the capsule pixel radius is
 * `width x lineScale(64) x CAPSULE_RADIUS_PER_QUAD_HALFWIDTH(0.6590102)`
 * (offsets below are px from the viewport centre, where the joint sits):
 *
 *   seg 0 (the SHORT partner)  (0, 0) -> 0.08 x (cos60, sin60)   2.56 px long,
 *                    12.653 px at the joint tapering to 8.435 px at its far end
 *   seg 1 (the LONG leg)       (-0.5, 0) -> (0, 0)  16.00 px long, 12.653 px radius
 *
 * THE SEGMENT ORDER IS LOAD-BEARING — do not "tidy" it back to the natural
 * short-after-long reading. The registry entries render with
 * `THREE.NoBlending`, and for the VISUAL pair nothing ever populates the depth
 * buffer, so the LAST-drawn instance simply overwrites and instance order is
 * storage order. (The two backends get there differently, which is worth
 * knowing: `depthCompete` gates only the GLSL `ShaderMaterial`, which is
 * depth-test-OFF without it, while the TSL NodeMaterial takes max mode's
 * blending state — depthTest true but depthWrite FALSE. Nothing writes depth
 * either way, every fragment tests against the cleared 1.0 and passes, so both
 * sides resolve last-drawn-wins. The pick pair is the opposite case: both
 * backends write brightness-as-depth and the brighter fragment wins, which is
 * why those two entries DO set `depthCompete`.) The only
 * leg whose output the far-cap term changes is the LONG one (it is the leg
 * that reconstructs a SHORT partner; seg 0's own packet has ql = 16 px, past
 * its 12.653 px support, so its far cap is inert). Worse, the region where the
 * two reconstructions differ lies inside the SHORT partner's own support by
 * construction — at 6.0 px along the partner axis and 2 px off it the bounded
 * long leg's deficit is 0 (it discards) while the unbounded one draws 0.33,
 * and the short partner draws 0.605 there either way. So with the long leg
 * authored FIRST the short partner overwrites the whole difference and the
 * fixture is exactly vacuous — measured: the GLSL-visual-only unbounded
 * mutation produced 0 differing pixels in that order. Authored LAST, the long
 * leg's differing fragment is the one left in the framebuffer.
 *
 * Both packet gates open, and for different reasons — checked clause by
 * clause against the vertex stages (`CAPSULE_JOINT_DEFICIT_GATE` = 0.02, and
 * both ends clear the 4 px `CAPSULE_JOINT_PACKET_MIN_RADIUS_PX` width gate at
 * rMax = 13.153):
 *   - seg 1's END (rB = 12.653), the long leg — THE ONE THAT MATTERS: taper
 *     |1 - 8.435/12.653| = 0.333 > 0.02 AND short partner 2.56 < 2 x 12.653 =
 *     25.31. (Own-widening false: uniform, 12.653 does not exceed 12.906.
 *     Sharp turn false: dot(q, u) = +0.500, and this end's clause is < -0.5.)
 *   - seg 0's START (rA = 12.653), the short partner: short partner 16.00 <
 *     25.31. (Taper 0: the long leg's far radius equals the joint radius.
 *     Own-widening false: 8.435 does not exceed 12.906. Sharp turn false:
 *     dot(q, u) = -0.500, and this end's clause is > 0.5.)
 * Neither end is the degenerate near-hairpin fallback: nl = 1.7321 at both,
 * far above the 1e-3 floor, and the two cut normals are exact negations OF
 * EACH OTHER IN WORLD SPACE — +/-(0.8660, 0.5000) there. Each leg stores its
 * own in ITS OWN (u, v) frame, where the two read nLoc = (+0.8660, +0.5) at
 * seg 1's end and (-0.8660, +0.5) at seg 0's start; those two are NOT
 * negations as printed, because the frames differ by the 60 deg turn.
 *
 * Why THIS partner and not a fatter one: the packed gradient at seg 1's end is
 * (8.435 - 12.653) / 2.56 = -1.6475 px/px, so an UNBOUNDED rod keeps shrinking
 * and hits the 1e-4 radius floor 7.68 px along the partner axis — still inside
 * seg 1's 12.653 px joint disc. The partner term collapses to ~0 there and the
 * deficit `max(mine - partner, 0)` returns the leg's FULL profile: sampled at
 * the 4096 pixel centres through the CPU mirror, the unbounded composition
 * exceeds max(mine, partner) by up to +0.529 of peak (74 of those centres
 * differ by more than 0.02), while the bounded one tracks it to 1e-15. A
 * partner WIDER at its far end (gradient > 0) renders identically either way —
 * the unbounded rod only grows, the deficit floors at 0, and the fixture would
 * be silently vacuous (measured: max difference exactly 0 at far width 0.45).
 *
 * Everything stays in frame. The profile has compact support at exactly the
 * pixel radius, so the analytic reach from the joint is the long leg's far cap
 * at 16.00 + 12.653 = 28.65 px (the short partner's is 2.56 + 8.435 = 11.00
 * px), and the stencil quads extend to |x| = 29.15 / |y| = 13.15 px (the long
 * leg) and |x| = 19.25 / |y| = 20.18 px (the short partner) — all inside the
 * +/-32 px half-viewport.
 *
 * Codes follow compute_joint_codes, re-derived for THIS authoring order (the
 * sign says which of the PARTNER's endpoints is the shared one, which is what
 * `luxarPartnerFar` decodes back into `far = code > 0 ? pEnd : pStart`):
 *   - seg 0's START meets seg 1's END, so it names slot 1 at that segment's
 *     END: -(1 + 3) = -4. Decode: slot = int(4 + 0.5) - 3 = 1, code < 0 so
 *     far = slot 1's pStart = (-0.5, 0) with width 0.3 — the long leg's free
 *     end, 12.653 px. Correct.
 *   - seg 1's END meets seg 0's START, so it names slot 0 at that segment's
 *     START: +(0 + 1) = 1. Decode: slot = int(1 + 0.5) - 1 = 0, code > 0 so
 *     far = slot 0's pEnd = 0.08 x (cos60, sin60) with width 0.2 — the short
 *     partner's free end, 8.435 px. Correct.
 * Both outer ends are free (0). Getting either code wrong is a quiet second
 * kind of vacuity: `luxarPartnerFar` returns w = -1 for a non-interior code
 * and no packet is built at all.
 */
function buildShortPartnerJoinTexelSource(): LineTexelSource {
  // 60 deg from the LONG leg's direction: 0.08 x (cos 60, sin 60).
  const partnerFar: readonly [number, number, number] = [0.04, 0.069282, 0];
  return {
    // Storage order IS draw order — see the ordering note above.
    startPositions: new Float32Array([0, 0, 0, -0.5, 0, 0]),
    endPositions: new Float32Array([...partnerFar, 0, 0, 0]),
    startColors: new Float32Array([1, 1, 1, 1, 1, 1]),
    endColors: new Float32Array([1, 1, 1, 1, 1, 1]),
    // The shared vertex carries ONE width (0.3) on both segments, as a real
    // polyline would; only the short partner's FAR end tapers, which is what
    // makes the packed radius gradient non-zero.
    startWidths: new Float32Array([0.3, 0.3]),
    endWidths: new Float32Array([0.2, 0.3]),
    startSharpness: new Float32Array([0.5, 0.5]),
    endSharpness: new Float32Array([0.5, 0.5]),
    segmentLengths: new Float32Array([0.08, 0.5]),
    startJointCode: new Float32Array([-4, 0]),
    endJointCode: new Float32Array([0, 1]),
  };
}

function buildJoinDataTexture(src: LineTexelSource): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(48), 6, 2, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  writeLineTexels(tex, src, 2);
  return tex;
}

function buildJoinMesh(src: LineTexelSource, material: THREE.Material): THREE.Object3D {
  const mesh = createInstancedLinesMesh({ ...src, segmentCount: 2 }, material);
  mesh.frustumCulled = false;
  return mesh;
}

/** The two-segment joint geometry a {@link joinEntry} pair renders. */
interface JoinFixture {
  readonly texels: () => LineTexelSource;
  readonly isOrtho: boolean;
  readonly buildCamera?: () => THREE.Camera;
  /** `uNearCull`; defaults to `buildVisualLineUniforms`'s 0.01. */
  readonly nearCull?: number;
}

const END_TO_START_JOIN: JoinFixture = { texels: buildJoinTexelSource, isOrtho: true };
const SAME_PARITY_JOIN: JoinFixture = { texels: buildSameParityJoinTexelSource, isOrtho: true };
const TAPER_JOIN: JoinFixture = {
  texels: buildTaperJoinTexelSource,
  isOrtho: false,
  buildCamera: buildBehindCamera,
};
const NEAR_PLANE_JOIN: JoinFixture = {
  texels: buildNearPlaneJoinTexelSource,
  isOrtho: false,
  buildCamera: buildBehindCamera,
  nearCull: NEAR_PLANE_CULL,
};

/**
 * CONTROL for {@link NEAR_PLANE_JOIN}: the SAME joint with the cull plane
 * lowered under segment 0's far endpoint, so the guard is TRUE and both sides
 * mitre. This pair must therefore DIFFER, and that difference is what makes the
 * pair above a gate rather than a tautology.
 *
 * `-miter` == `-none` is satisfied by every way the join block can fail to RUN
 * at all: the rendered-width gate closed, a joint code that decodes to a
 * sentinel or the wrong storage slot, a segment culled or a quad projected off
 * screen. The equality assertion cannot tell any of those from the near-plane
 * guard doing its job, and an analytic argument about the turn and the widths
 * only shows the block SHOULD be reached — nothing rendered showed that it is,
 * under this fixture's exact camera and uniforms. Here it visibly is.
 *
 * `nearCull` is the right thing to move, and the only thing moved. It IS the
 * guard's threshold, so lowering it flips exactly the predicate under test; and
 * it lives in the uniforms rather than the geometry, so the control reuses
 * {@link buildNearPlaneJoinTexelSource} verbatim and needs no second set of
 * position constants that could drift out of step with the first. Hence the
 * spread: the two fixtures are the same object but for this one number.
 *
 * The lower plane changes more than the guard, but every one of those changes is
 * driven by `nearCull`, which `uLineJoin` does not touch — so the whole class is
 * shared by BOTH members of the control pair and cancels out of the
 * miter-vs-none comparison. The ones a reader would otherwise be surprised by:
 *   Segment 0's start sits at view depth 0.3 >= 0.2, so tA stays 0 and the
 *   near-plane SEGMENT clip does not fire at all; both partner-naming ends (the
 *   shared vertex, depth 1.0) keep `reachesVertex` either way.
 *   `luxarLinePixelPos` divides by max(clip.w, nearCull), so segment 0's far
 *   endpoint now projects at its full 35 px from the shared vertex instead of
 *   the 17.5 px the 0.6 plane clamped it to, and segment 0's RENDERED leg is
 *   35 px rather than the 10 px the clip left.
 *   The fragment stage's near fade is smoothstep(nearCull, 2*nearCull, depth) on
 *   the interpolated view depth, so the joint's own fade rises from 0.74 to 1.0.
 *   `luxarLineEndPixelWidth` divides by max(-viewZ, nearCull) as well, so
 *   segment 0's FREE-END corner half-width goes 10.67 px -> 21.33 px (its 0.3
 *   depth is no longer clamped to the plane), with `vPixelWidth` following.
 *
 * What the guard then sees, and why the joint is mitred rather than merely
 * accepted: both operands (segment 0's far endpoint at 0.3 and segment 1's at
 * 1.5) clear 0.2, so `bothFarInFront` is true on both sides. The turn is still
 * 0.6 — the clip only ever moved segment 0's start ALONG its own line, so
 * dirIn = (1, 0) either way — giving grow = sqrt(2/1.6) = 1.12 <= 2 and an axial
 * reach of 6.4 * 0.5 = 3.2 px, now against overshoot allowances of
 * 0.5 * min(35, 15) = 7.5 px from segment 0's side and 0.5 * min(15, 35) = 7.5 px
 * from segment 1's (the 0.6 plane's were 5 px and 7.5 px). Both partner-naming
 * ends still render at 0.1 * 64 / 1.0 = 6.4 px of half-width, 3.2x the 2 px gate.
 */
const NEAR_PLANE_CONTROL_JOIN: JoinFixture = {
  ...NEAR_PLANE_JOIN,
  nearCull: NEAR_PLANE_CONTROL_CULL,
};

/**
 * @param fixture - the joint geometry (see {@link JoinFixture}).
 * @param join - the `uLineJoin` value. BOTH backends must be driven from this
 * one number: GLSL reads it as a runtime uniform while the TSL factory bakes it
 * into the graph (`buildTSLMaterial` below passes `join` to the factory, which
 * simply omits the join block for `none`), so a fixture that set only one of
 * them would compare a mitred quad against an unmitred one and fail for the
 * wrong reason.
 */
function joinEntry(fixture: JoinFixture, join: number): RegistryEntry {
  return {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      ...buildVisualLineUniforms(
        buildJoinDataTexture(fixture.texels()),
        fixture.isOrtho,
        fixture.nearCull
      ),
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
    buildMesh: (material) => buildJoinMesh(fixture.texels(), material),
    ...(fixture.buildCamera ? { buildCamera: fixture.buildCamera } : {}),
  };
}

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

/** A slanted segment 3 units down +X, for the cube-capture face cases. */
const CUBE_LINE_START: readonly [number, number, number] = [3, -0.3, 0.2];
const CUBE_LINE_END: readonly [number, number, number] = [3, 0.3, -0.25];
/** Wide enough (~5 px at 3 units) that a clamped-to-floor width would be obvious. */
const CUBE_LINE_STYLE: LineFixtureStyle = { startWidth: 0.5, endWidth: 0.5 };

/**
 * A perspective line seen through a CubeCamera face (fov −90, a flipped
 * projection) or its 180°-rolled +90° equivalent: the same image, because the
 * pixel-width scale is resY·|P11|, never negative.
 */
function buildCubeCaptureLineEntry(buildCamera: () => THREE.Camera): RegistryEntry {
  return {
    source: LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(
        buildLineDataTexture(CUBE_LINE_START, CUBE_LINE_END, undefined, undefined, CUBE_LINE_STYLE),
        false
      ),
    buildTSLMaterial: (uniforms) => buildVisualLineTSLMaterial(uniforms, false),
    buildMesh: (material) =>
      buildLineInstancedMesh(
        material,
        CUBE_LINE_START,
        CUBE_LINE_END,
        undefined,
        undefined,
        CUBE_LINE_STYLE
      ),
    buildCamera,
  };
}

/** Scale of the nanometre case: every length, width included, times 1e-5. */
const TINY = 1e-5;

/**
 * The `line` fixture shrunk by {@link TINY} in every length (segment, width,
 * segment length) and seen through an ortho frustum shrunk by the same factor
 * (height 2e-5): under an orthographic projection that is the SAME image. The
 * historical CPU scale clamped the frustum height to 1e-4 before dividing, so
 * this line rendered 5x narrower than its unit-scale twin.
 */
const TINY_ORTHO_LINE: RegistryEntry = {
  source: LINE_SOURCE,
  buildUniforms: () =>
    buildVisualLineUniforms(
      buildLineDataTexture([-0.5 * TINY, 0, 0], [0.5 * TINY, 0, 0], undefined, undefined, {
        startWidth: 0.1 * TINY,
        endWidth: 0.1 * TINY,
        segmentLength: TINY,
      }),
      true
    ),
  buildTSLMaterial: (uniforms) => buildVisualLineTSLMaterial(uniforms, true),
  buildMesh: (material) =>
    buildLineInstancedMesh(
      material,
      [-0.5 * TINY, 0, 0],
      [0.5 * TINY, 0, 0],
      undefined,
      undefined,
      {
        startWidth: 0.1 * TINY,
        endWidth: 0.1 * TINY,
        segmentLength: TINY,
      }
    ),
  buildCamera: () => buildOrthoCamera(2 * TINY),
};

/** The unit-scale twin of {@link TINY_ORTHO_LINE}, through the same builders. */
const UNIT_ORTHO_LINE: RegistryEntry = {
  source: LINE_SOURCE,
  buildUniforms: () => buildVisualLineUniforms(buildLineDataTexture(), true),
  buildTSLMaterial: (uniforms) => buildVisualLineTSLMaterial(uniforms, true),
  buildMesh: (material) => buildLineInstancedMesh(material),
  buildCamera: () => buildOrthoCamera(2),
};

export const LINE_SHADERS: Record<string, RegistryEntry> = {
  'line-cube-face': buildCubeCaptureLineEntry(buildCubeFaceCamera),
  'line-cube-face-equivalent': buildCubeCaptureLineEntry(buildCubeFaceEquivalentCamera),
  'line-tiny-ortho': TINY_ORTHO_LINE,
  'line-unit-ortho': UNIT_ORTHO_LINE,
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
      // The pixel-width scale is read from the camera's projection:
      //   resY * |P11| = 64 * 2 / 2 = 64 (2*resY/frustumHeight)
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
  'line-join-miter': joinEntry(END_TO_START_JOIN, 1.0),
  'line-join-none': joinEntry(END_TO_START_JOIN, 0.0),
  // The SAME-PARITY (END-END) twin of the pair above. An end->start joint is
  // structurally blind to the partner-leg orientation — `atEnd` and
  // `partnerSharesItsStart` agree there — so this is the pair that pins it.
  // The parity spec additionally requires the mitred render to be AT LEAST AS
  // BRIGHT as the unmitred one around the shared vertex: the pre-fix
  // orientation read turn = -0.6 here, which both declined the miter and kept
  // the soft endpoint cap, i.e. it rendered DIMMER than `none`.
  'line-join-same-parity-miter': joinEntry(SAME_PARITY_JOIN, 1.0),
  'line-join-same-parity-none': joinEntry(SAME_PARITY_JOIN, 0.0),
  // TAPERED joint under a PERSPECTIVE camera. The first line fixtures to reach
  // the join block off the ortho fast path, and the ones that pin the
  // DECLINED-miter cap derivation — this joint is a right angle the block
  // declines rather than mitres. They do NOT gate the 2 px width straddle or
  // the near-plane conjunction; see {@link TAPER_JOIN} for why.
  'line-join-taper-miter': joinEntry(TAPER_JOIN, 1.0),
  'line-join-taper-none': joinEntry(TAPER_JOIN, 0.0),
  // NEAR-PLANE joint: nearCull 0.6 with segment 0's far endpoint at depth 0.3,
  // so the guard's two operands genuinely disagree. Both sides must decline,
  // which makes `-miter` and `-none` pixel-identical; the one-sided guard let
  // segment 0 miter alone and flap. See {@link NEAR_PLANE_JOIN}.
  'line-join-near-plane-miter': joinEntry(NEAR_PLANE_JOIN, 1.0),
  'line-join-near-plane-none': joinEntry(NEAR_PLANE_JOIN, 0.0),
  // CONTROL for the pair above: the same joint with nearCull lowered to 0.2,
  // under segment 0's 0.3 far endpoint, so the guard is true and both sides
  // MITRE. The parity spec requires this pair to DIFFER — which is what proves
  // the block is reached under that fixture's camera and uniforms, so its
  // pixel-identity is the guard declining and not the join being skipped for
  // some unrelated reason. See {@link NEAR_PLANE_CONTROL_JOIN}.
  'line-join-near-plane-control-miter': joinEntry(NEAR_PLANE_CONTROL_JOIN, 1.0),
  'line-join-near-plane-control-none': joinEntry(NEAR_PLANE_CONTROL_JOIN, 0.0),
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
  // Opaque-mode contribution cutout parity: the GLSL twin compiles with
  // LUXAR_OPAQUE_RGB_CONTRIBUTION and the TSL graph is built from the
  // opaque blending mode. Low opacity moves the discard boundary inside
  // the ribbon so a predicate mismatch changes the rendered footprint.
  'line-opaque': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uLineTex: { value: buildLineDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uOpacity: { value: 0.02 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildDefines: () => ({ LUXAR_OPAQUE_RGB_CONTRIBUTION: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'opaque',
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

  // ============================================================
  // CAPSULE LINE PRIMITIVE (#1352, ?linePrimitive=capsule).
  // Gaussian-like quartic profile of the 2D point-to-segment distance —
  // see _shared/line-capsule.ts. These fixtures pin value-level parity
  // of the two backends over the capsule's behaviour surface: side-on
  // ribbon, end-on disc, half-disc bisector joints (gentle and sharp
  // folds), taper, colormap, max + volumetric mode tails, near-plane
  // straddling, and the pick twins.
  // NOTE for snapshot hygiene: append new entries at the END.
  // ============================================================
  'line-capsule-sideon': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () => buildVisualLineUniforms(buildLineDataTexture(), true),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'additive',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineInstancedMesh,
  },
  // End-on ortho: the segment projects to a POINT — the capsule's whole
  // reason to exist. Must render a finite radial disc on both backends.
  'line-capsule-endon-ortho': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(buildLineDataTexture([0, 0, 0.3], [0, 0, -0.5]), true),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'additive',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildLineInstancedMesh(m, [0, 0, 0.3], [0, 0, -0.5]),
  },
  'line-capsule-endon-persp': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(buildLineDataTexture([0, 0, 0.5], [0, 0, -0.3]), false),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'additive',
        isOrtho: false,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildLineInstancedMesh(m, [0, 0, 0.5], [0, 0, -0.3]),
    buildCamera: buildBehindCamera,
  },
  // The V joint: partner fetch + 2D bisector cuts on both backends. The
  // two capsules must tile the bend seamlessly under additive.
  'line-capsule-joint': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(buildJoinDataTexture(buildJoinTexelSource()), true),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'additive',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (material) => buildJoinMesh(buildJoinTexelSource(), material),
  },
  // 122.7° fold, WIDE: the end's half of the joint disc fills the fold
  // tip (no chopped notch, no double-bright overlap).
  'line-capsule-fold': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(buildJoinDataTexture(buildFoldJoinTexelSource(0.3)), true),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'additive',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (material) => buildJoinMesh(buildFoldJoinTexelSource(0.3), material),
  },
  // The same fold, HAIRLINE-thin: the joint disc is hairline-sized (the
  // 1.5 px AA floor), pinning that a thin fold stays compact.
  'line-capsule-fold-thin': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(buildJoinDataTexture(buildFoldJoinTexelSource(0.02)), true),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'additive',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (material) => buildJoinMesh(buildFoldJoinTexelSource(0.02), material),
  },
  // The same fold at the width that lands INSIDE the band #1495 opens:
  // 0.05 x line scale 64 x 0.659 = 2.11 px raw, so the joint is under
  // the 4 px packet width gate (rMax 2.61) yet at or above the 1.5 px AA
  // floor, and the fold's own axis dot is 0.54 > 0.5. That is exactly the
  // combination the floored sharp-turn exception opens — the wide fixture
  // above clears the gate outright and the hairline one is held back by the
  // floor conjunct, so without this variant neither backend ever executes
  // the new branch.
  'line-capsule-fold-mid': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(buildJoinDataTexture(buildFoldJoinTexelSource(0.05)), true),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'additive',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (material) => buildJoinMesh(buildFoldJoinTexelSource(0.05), material),
  },
  // Width taper + endpoint colour/sharpness remap in one (REMAP_STYLE).
  'line-capsule-taper': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(
        buildLineDataTexture([-0.5, 0, 0], [0.5, 0, 0], undefined, undefined, REMAP_STYLE),
        true
      ),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'additive',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) =>
      buildLineInstancedMesh(m, [-0.5, 0, 0], [0.5, 0, 0], undefined, undefined, REMAP_STYLE),
  },
  // Colormap path: interpolated scalar varying + fragment LUT sample.
  'line-capsule-colormap': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () => ({
      ...buildVisualLineUniforms(buildLineDataTexture([-0.5, 0, 0], [0.5, 0, 0], [0.2, 0.8]), true),
      uColormapTex: { value: buildColormapTexture() },
      uScalarMin: { value: 0.0 },
      uScalarScale: { value: 1.0 },
    }),
    buildDefines: () => ({ USE_COLORMAP: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(
        buildLineTSLNodesFromUniforms(uniforms, { useColormap: true }),
        {
          useColormap: true,
          blendingMode: 'additive',
          isOrtho: true,
        }
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildLineInstancedMesh(m, [-0.5, 0, 0], [0.5, 0, 0], [0.2, 0.8]),
  },
  // MAX mode: the premultiplied-RGB output tail.
  'line-capsule-max': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () => buildVisualLineUniforms(buildLineDataTexture(), true),
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '', LUXAR_MAX_RGB_CONTRIBUTION: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'max',
        gammaOne: true,
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineInstancedMesh,
  },
  // Opaque-mode contribution cutout: the capsule has its own fragment
  // predicate, so pin it independently from the quad line primitive.
  'line-capsule-opaque': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () => ({
      ...buildVisualLineUniforms(buildLineDataTexture(), true),
      uOpacity: { value: 0.02 },
    }),
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '', LUXAR_OPAQUE_RGB_CONTRIBUTION: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'opaque',
        gammaOne: true,
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineInstancedMesh,
  },
  // VOLUMETRIC blending: the τ tail with per-element alpha optical-depth
  // mapping (uHasElementAlpha) — mirrors the quad's 'line-volumetric'.
  'line-capsule-volumetric': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () => ({
      ...buildVisualLineUniforms(
        buildLineDataTexture([-0.5, 0, 0], [0.5, 0, 0], undefined, VOLUMETRIC_LINE_ALPHAS),
        true
      ),
      uOpacity: { value: 0.7 },
      uAbsorption: { value: 2.0 },
      uHasElementAlpha: { value: 1 },
    }),
    buildDefines: () => ({ LUXAR_VOLUMETRIC: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'volumetric',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) =>
      buildLineInstancedMesh(m, [-0.5, 0, 0], [0.5, 0, 0], undefined, VOLUMETRIC_LINE_ALPHAS),
  },
  // Near-plane straddling in perspective: the endpoint clip reshapes the
  // profile domain itself (no behind-eye notion in 2D).
  'line-capsule-nearclip': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(buildLineDataTexture([0, 0, 1.5], [0, 0, -0.5]), false, 0.35),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'additive',
        isOrtho: false,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildLineInstancedMesh(m, [0, 0, 1.5], [0, 0, -0.5]),
    buildCamera: buildBehindCamera,
  },
  // Near-plane straddling with TAPER + colour gradient: the clipped end
  // must carry the attribute values interpolated AT the clip parameter,
  // not the behind-camera endpoint's (width/colour would jump — the
  // review finding the plain nearclip fixture could not see).
  'line-capsule-nearclip-taper': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(
        buildLineDataTexture([0, 0, 1.5], [0, 0, -0.5], undefined, undefined, {
          startColor: [1.0, 0.1, 0.1],
          endColor: [0.1, 0.1, 1.0],
          startWidth: 0.06,
          endWidth: 0.5,
          startSharpness: 0.1,
          endSharpness: 0.9,
        }),
        false,
        0.35
      ),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'additive',
        isOrtho: false,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) =>
      buildLineInstancedMesh(m, [0, 0, 1.5], [0, 0, -0.5], undefined, undefined, {
        startColor: [1.0, 0.1, 0.1],
        endColor: [0.1, 0.1, 1.0],
        startWidth: 0.06,
        endWidth: 0.5,
        startSharpness: 0.1,
        endSharpness: 0.9,
      }),
    buildCamera: buildBehindCamera,
  },
  // FAT side-on: the footprint-sensitivity twin of the pick fixture below.
  'line-capsule-fat': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(
        buildLineDataTexture(undefined, undefined, undefined, undefined, FOOTPRINT_STYLE),
        true
      ),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'additive',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) =>
      buildLineInstancedMesh(m, undefined, undefined, undefined, undefined, FOOTPRINT_STYLE),
  },
  // === Capsule PICK twins: same stencil + profile, pick output contract ===
  'line-capsule-pick-sideon': {
    source: CAPSULE_LINE_PICK_SOURCE,
    buildUniforms: () =>
      buildPickLineUniforms(
        buildLineDataTexture(undefined, undefined, undefined, undefined, FOOTPRINT_STYLE),
        true
      ),
    buildTSLMaterial: (uniforms) =>
      capsuleLinePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(uniforms), {
        isOrtho: true,
      }) as unknown as THREE.Material,
    buildMesh: (m) =>
      buildLineInstancedMesh(m, undefined, undefined, undefined, undefined, FOOTPRINT_STYLE),
  },
  'line-capsule-pick-endon': {
    source: CAPSULE_LINE_PICK_SOURCE,
    buildUniforms: () =>
      buildPickLineUniforms(
        buildLineDataTexture([0, 0, 0.3], [0, 0, -0.5], undefined, undefined, FOOTPRINT_STYLE),
        true
      ),
    buildTSLMaterial: (uniforms) =>
      capsuleLinePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(uniforms), {
        isOrtho: true,
      }) as unknown as THREE.Material,
    buildMesh: (m) =>
      buildLineInstancedMesh(m, [0, 0, 0.3], [0, 0, -0.5], undefined, undefined, FOOTPRINT_STYLE),
  },
  // TWO segments, so the two legs' fragments overlap in the joint region and
  // COMPETE through the depth test: the pick fragment writes brightness-as-
  // depth (`gl_FragDepth = 1 - brightness`) and its element id, so whichever
  // fragment wins decides the READBACK VALUE, not just a tie-break. The TSL
  // pick factory already sets depthTest/depthWrite on its NodeMaterial
  // (see types.ts), so the GLSL3 ShaderMaterial must opt in too or the two
  // backends resolve different winners — measured 1.846 per-covered-pixel
  // across 7 pixels without it, i.e. passing only by sitting under the 2.0
  // bound. The single-segment pick entries above cannot compete and stay
  // depth-off.
  'line-capsule-pick-joint': {
    source: CAPSULE_LINE_PICK_SOURCE,
    depthCompete: true,
    buildUniforms: () => buildPickLineUniforms(buildJoinDataTexture(buildJoinTexelSource()), true),
    buildTSLMaterial: (uniforms) =>
      capsuleLinePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(uniforms), {
        isOrtho: true,
      }) as unknown as THREE.Material,
    buildMesh: (material) => buildJoinMesh(buildJoinTexelSource(), material),
  },
  // The V again with alternating vertex widths, so the joint packet gate
  // opens and the DEFICIT rule (plus the full-disc stencil reach it needs,
  // #1488) runs on both backends — see buildDeficitPacketJoinTexelSource.
  'line-capsule-joint-taper': {
    source: CAPSULE_LINE_SOURCE,
    buildUniforms: () =>
      buildVisualLineUniforms(buildJoinDataTexture(buildDeficitPacketJoinTexelSource()), true),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'additive',
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (material) => buildJoinMesh(buildDeficitPacketJoinTexelSource(), material),
  },
  // SHORT-partner joint: the deficit packet's far-cap term, live on both
  // backends for the first time (#1490). See
  // {@link buildShortPartnerJoinTexelSource} for every derived number.
  'line-capsule-joint-short-partner': {
    source: CAPSULE_LINE_SOURCE,
    // MAX mode, for the reason `joinEntry` and `jointCodeEntry` above give:
    // max premultiplies RGB by the profile, so the deficit lands in all four
    // channels. In additive mode it rides ALPHA alone, and the parity metric
    // averages over four channels — measured, the GLSL-only unbounded mutation
    // moved 87 pixels but scored 1.108, quietly under the 2.0 bound. The same
    // 87 pixels in max mode score 4.433.
    //
    // How that 87 was counted, since a CPU model of the same scene gives 80-86
    // depending on how the 1 px AA ramp and the `profile <= 0 => no write` rule
    // are sampled: it is the HARNESS's own readback — the 64x64 render target,
    // GLSL-with-the-mutation against TSL-bounded, counting RGBA quadruplets
    // that differ in any channel, out of 916 covered. The pick twin measures
    // 70 of 908 under its own shader's mutation.
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '', LUXAR_MAX_RGB_CONTRIBUTION: '' }),
    buildUniforms: () =>
      buildVisualLineUniforms(buildJoinDataTexture(buildShortPartnerJoinTexelSource()), true),
    buildTSLMaterial: (uniforms) => {
      const m = capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        blendingMode: 'max',
        gammaOne: true,
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (material) => buildJoinMesh(buildShortPartnerJoinTexelSource(), material),
  },
  // The pick twin of the same geometry: the pick stencil rebuilds the partner
  // rod from its own copy of the packet, so it regresses independently — and
  // it does witness the far-cap term on its own (unbinding the rod in the PICK
  // GLSL fragment alone fails this entry at 2.715, 70 of 908 covered pixels,
  // with every other capsule test still green). It needs NO max-mode define:
  // the pick fragment already carries brightness in a colour channel.
  // `depthCompete` for the same reason as the entry above, and here it is not
  // marginal: the fat 12.653 px legs overlap deeply, and depth-off measures
  // 6.311 per-covered-pixel — flipped pick WINNERS (GLSL element id 1 against
  // TSL element id 0), not near-ties.
  'line-capsule-pick-joint-short-partner': {
    source: CAPSULE_LINE_PICK_SOURCE,
    depthCompete: true,
    buildUniforms: () =>
      buildPickLineUniforms(buildJoinDataTexture(buildShortPartnerJoinTexelSource()), true),
    buildTSLMaterial: (uniforms) =>
      capsuleLinePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(uniforms), {
        isOrtho: true,
      }) as unknown as THREE.Material,
    buildMesh: (material) => buildJoinMesh(buildShortPartnerJoinTexelSource(), material),
  },
};
