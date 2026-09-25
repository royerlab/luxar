/**
 * Mesh shader family for the TSL ↔ GLSL parity harness: the six variants of
 * `docs/specs/MESH_NODE_SPEC.md` §6.4 — the `opaque` default (alpha cutout), the
 * alpha-weighted emission shared by `additive`/`luminous`/`normal`, the `max`
 * premultiply, the derivative flat-normal build, the colormap LUT build, and
 * `mesh-pick`.
 *
 * ## What the pick entries can and cannot show
 *
 * The harness renders to an `UnsignedByteType` target, so the pick pass's id
 * channels — `nodeId` in R and the two 16-bit halves in G/A — clamp and quantize to
 * 8 bits. The pick entries therefore test what survives that: whether the two
 * backends **discard the same fragments** under the cutout, and whether they agree on
 * the `brightness` channel. The id SPLIT is pinned elsewhere, by construction rather
 * than by pixels — both backends route through one shared helper
 * (`luxarElementIdSplit`, single-sourced in `glsl-lib.ts`), the codegen snapshot
 * shows the TSL side's `/ 65536` and `- hi * 65536`, and a unit test round-trips the
 * split against `voteWinner`'s recombination.
 *
 * ## Five entries render under PERSPECTIVE
 *
 * Every other entry uses the shared ORTHOGRAPHIC default camera, under which
 * `perspectiveNearFade` is the identity — so the near fade would ship with no
 * rendered parity coverage at all. `mesh-near-fade`, `mesh-additive-near-fade` and
 * `mesh-pick-near-fade` override `buildCamera` with the perspective
 * `buildBehindCamera` and pick a `uNearCull` that puts the whole quad at a partial
 * fade; the arithmetic is on `NEAR_FADE_UNIFORMS` below.
 *
 * The two `*-near-fade-reference` entries render under the SAME perspective camera
 * with the fade made the identity (a `uNearCull` far inside the quad's depth), and
 * exist because the anti-vacuity
 * half of the parity test needs an un-faded frame of the *same surface points*. The
 * ortho default camera cannot supply one: its frame is 2.0 wide at z = 0 against the
 * perspective frame's 2·tan(30°) = 1.155, so pixel (i, j) is a different point on the
 * quad in the two framings and the "exactly 0.15625 ×" comparison would be measuring
 * that mismatch as well as the fade.
 *
 * ## The fixture is a tilted-normal quad, on purpose
 *
 * Two triangles in the z = 0 plane, but with per-corner normals fanned outward
 * (`normalize(x·0.6, y·0.6, 1)`). That is what makes the flat-vs-smooth pair a real
 * test rather than a tautology: the *geometric* normal of this quad is uniformly
 * +z, so a build that ignored `shading` and always shaded from derivatives would
 * render the flat and smooth variants IDENTICALLY. With fanned stored normals the
 * smooth variant carries a visible radial gradient and the flat one is uniform.
 *
 * The colour attribute is `float32` RGBA at four components, so `vAlpha` is a real
 * authored value (0.25 at one corner) rather than the constant 1.0 an RGB fixture
 * would supply — which is what gives the cutout and the max premultiply something
 * to act on.
 *
 * @module tests/e2e/harnesses/tsl-harness/mesh
 */

import * as THREE from 'three';
import { MESH_SOURCE } from '../../../../rendering/materials/mesh/shader-glsl';
import {
  meshWebGPUFactory,
  buildMeshTSLNodesFromUniforms,
  type MeshTSLConfig,
} from '../../../../rendering/materials/mesh/shader-tsl';
import { MESH_DEFAULTS } from '../../../../rendering/materials/mesh/appearance';
import { MESH_PICK_SOURCE } from '../../../../rendering/picking/mesh/shaders';
import {
  meshPickWebGPUFactory,
  buildMeshPickTSLNodesFromUniforms,
} from '../../../../rendering/picking/mesh/pick.tsl';
import type { RegistryEntry } from './types';
import { buildBehindCamera, buildColormapTexture } from './shared';

/** Quad corners in the z = 0 plane, filling the ortho camera's [-1, 1] frame. */
const QUAD_POSITIONS = new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]);
const QUAD_INDICES = new Uint32Array([0, 1, 2, 0, 2, 3]);

/**
 * Per-corner normals fanned outward from +z. Deliberately NOT the quad's geometric
 * normal — see the module doc for why an axis-aligned fixture would make the
 * flat-vs-smooth comparison vacuous.
 */
const QUAD_NORMALS = ((): Float32Array => {
  const out = new Float32Array(12);
  for (let v = 0; v < 4; v++) {
    const x = QUAD_POSITIONS[v * 3] * 0.6;
    const y = QUAD_POSITIONS[v * 3 + 1] * 0.6;
    const len = Math.hypot(x, y, 1);
    out[v * 3] = x / len;
    out[v * 3 + 1] = y / len;
    out[v * 3 + 2] = 1 / len;
  }
  return out;
})();

/** RGBA, with a partially-transparent corner so `vAlpha` is not the constant 1.0. */
const QUAD_COLORS = new Float32Array([
  1.0, 0.5, 0.25, 1.0, 0.25, 1.0, 0.5, 1.0, 0.5, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25,
]);

/** Per-vertex scalars spanning [0, 1] for the colormap variant. */
const QUAD_SCALARS = new Float32Array([0.0, 0.33, 0.66, 1.0]);

/**
 * The same colours as RGB — three components, no alpha column.
 *
 * This is the production no-alpha layout (`float32x3`, which is what the
 * absent-colours white fill also binds), and the one case where the opaque
 * `vAlpha = 1.0` comes from the **attribute default** for the missing `w` rather
 * than from a CPU-side pad. Worth its own fixture because getting it wrong is
 * invisible in the common path: a `w` that defaulted to 0 instead of 1 would make
 * every fragment fail the `opaque` cutout and the surface would simply vanish.
 */
const QUAD_COLORS_RGB = new Float32Array([
  1.0, 0.5, 0.25, 0.25, 1.0, 0.5, 0.5, 0.25, 1.0, 1.0, 1.0, 1.0,
]);

/**
 * Per-corner UVs covering the full [0, 1] square.
 *
 * Chosen so the 2x2 texture below lands one texel per corner under NEAREST — which
 * makes the parity comparison read the sampler itself rather than an interpolation
 * both backends happen to agree on. A degenerate mapping (all corners at the same
 * UV) would render a flat colour and pass even if the varying were never assigned.
 */
const QUAD_UVS = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);

/**
 * A 2x2 RGBA texture with a transparent texel, NEAREST-filtered.
 *
 * The transparent texel is the load-bearing part: texture alpha multiplies
 * coverage, so under the `opaque` cutout one quadrant must be DISCARDED. A fully
 * opaque texture would render the same on a backend that dropped the alpha term
 * entirely, so the fixture would not distinguish them.
 *
 * NEAREST plus no mipmaps so the sampled value at each corner is exactly one
 * authored texel — a linear filter would blend across the texels and make an
 * off-by-one in the UV mapping invisible.
 */
function buildBaseColorTexture(): THREE.DataTexture {
  const data = new Uint8Array([
    255,
    0,
    0,
    255, // red, opaque
    0,
    255,
    0,
    255, // green, opaque
    0,
    0,
    255,
    255, // blue, opaque
    255,
    255,
    255,
    0, // white, FULLY TRANSPARENT — the cutout quadrant
  ]);
  const tex = new THREE.DataTexture(data, 2, 2, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function buildMeshGeometry(
  withScalars = false,
  rgbOnly = false,
  withUVs = false
): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(QUAD_POSITIONS, 3, false));
  geometry.setAttribute('normal', new THREE.BufferAttribute(QUAD_NORMALS, 3, false));
  geometry.setAttribute(
    'color',
    rgbOnly
      ? new THREE.BufferAttribute(QUAD_COLORS_RGB, 3, false)
      : new THREE.BufferAttribute(QUAD_COLORS, 4, false)
  );
  if (withScalars) {
    geometry.setAttribute('aScalar', new THREE.BufferAttribute(QUAD_SCALARS, 1, false));
  }
  if (withUVs) {
    geometry.setAttribute('uv', new THREE.BufferAttribute(QUAD_UVS, 2, false));
  }
  geometry.setIndex(new THREE.BufferAttribute(QUAD_INDICES, 1, false));
  return geometry;
}

const buildMeshObject =
  (withScalars = false, rgbOnly = false, withUVs = false) =>
  (material: THREE.Material): THREE.Object3D => {
    const mesh = new THREE.Mesh(buildMeshGeometry(withScalars, rgbOnly, withUVs), material);
    // Both faces, matching the `double_sided` default — and required for the
    // `gl_FrontFacing` flip to be exercised at all by a camera on either side.
    material.side = THREE.DoubleSide;
    return mesh;
  };

/**
 * Near-fade inputs for the ORTHO default camera (`buildDefaultCamera`).
 *
 * `uIsOrtho: 1` is the honest value for that camera, and it is also what keeps
 * every pre-existing entry's pixels unchanged: `perspectiveNearFade` returns 1.0
 * under ortho, so `uNearCull` is inert here.
 */
const ORTHO_FADE_UNIFORMS = { uIsOrtho: 1, uNearCull: 0.1 } as const;

/**
 * Near-fade inputs for the PERSPECTIVE `buildBehindCamera` entries below.
 *
 * That camera sits at z = 1 looking down −Z and the quad is in the z = 0 plane, so
 * every fragment has `viewZ = -1` exactly. With `uNearCull = 0.8` the fade is
 * `smoothstep(0.8, 1.6, 1.0) = 3t² − 2t³` at `t = 0.25` — **0.15625**: partial (so
 * a backend that dropped the fade renders visibly brighter), well clear of the 0.01
 * reject (so it is not silently testing the discard instead), and constant across
 * the quad (so the two backends must agree to the last bit).
 */
const NEAR_FADE_UNIFORMS = { uIsOrtho: 0, uNearCull: 0.8 } as const;

/**
 * The un-faded reference for the perspective entries: the SAME camera, with a
 * `uNearCull` so small that the quad's depth (1.0) is far past the fade band, so
 * `perspectiveNearFade` = smoothstep(1e-6, 2e-6, 1.0) = 1.0 exactly and every other
 * term of the fragment is untouched. Dividing one frame by the other therefore
 * isolates the fade and nothing else. (The ortho test is read from the camera the
 * frame is drawn with, so a mismatched `uIsOrtho` can no longer switch the fade off;
 * the ortho branch has its own entry, `mesh-ortho-near-cull`.)
 */
const UNFADED_REFERENCE_UNIFORMS = { uIsOrtho: 0, uNearCull: 1e-6 } as const;

/**
 * The ortho branch, non-vacuously: the `mesh` quad under an ORTHOGRAPHIC camera with
 * the perspective entries' `uNearCull` of 0.8. A perspective fade would be 0.15625
 * here; the ortho branch must return 1.0, so this frame equals `mesh`'s exactly.
 */
const ORTHO_NEAR_CULL_UNIFORMS = { uIsOrtho: 1, uNearCull: 0.8 } as const;

/** Uniforms mirroring the production `MeshMaterial` constructor. */
function meshUniforms(
  withColormap = false,
  fade: { uIsOrtho: number; uNearCull: number } = ORTHO_FADE_UNIFORMS,
  withTexture = false
): Record<string, THREE.IUniform> {
  return {
    uOpacity: { value: 1.0 },
    uInvGamma: { value: 1.0 / 2.2 },
    uIntensity: { value: 1.0 },
    uOffset: { value: 0.0 },
    uAmbient: { value: MESH_DEFAULTS.ambient },
    uShadeExponent: { value: MESH_DEFAULTS.shadeExponent },
    uSpecular: { value: MESH_DEFAULTS.specular },
    uShininess: { value: MESH_DEFAULTS.shininess },
    uAlphaCutoff: { value: MESH_DEFAULTS.alphaCutoff },
    uIsOrtho: { value: fade.uIsOrtho },
    uNearCull: { value: fade.uNearCull },
    ...(withColormap
      ? {
          uColormapTex: { value: buildColormapTexture() },
          uScalarMin: { value: 0.0 },
          uScalarScale: { value: 1.0 },
        }
      : {}),
    ...(withTexture ? { uBaseColorTex: { value: buildBaseColorTexture() } } : {}),
  };
}

/**
 * Uniforms mirroring the production `MeshPickingMaterial` constructor.
 *
 * `nodeId` is 1 rather than a realistic id: the harness target is 8-bit, so anything
 * above 1.0 saturates to the same 255 on both backends and the channel stops
 * distinguishing anything. 1 keeps R at full scale while staying an honest value.
 *
 * @param surfaceMode `true` = the `opaque` default (cutout on, real depth);
 *   `false` = a commutative mode (no cutout, brightness-as-depth).
 */
function meshPickUniforms(
  surfaceMode: boolean,
  fade: { uIsOrtho: number; uNearCull: number } = ORTHO_FADE_UNIFORMS,
  withTexture = false
): Record<string, THREE.IUniform> {
  return {
    uNodeId: { value: 1 },
    uOpacity: { value: 1.0 },
    uAlphaCutoff: { value: MESH_DEFAULTS.alphaCutoff },
    uAlphaCutout: { value: surfaceMode ? 1 : 0 },
    uSurfaceDepth: { value: surfaceMode ? 1 : 0 },
    uIsOrtho: { value: fade.uIsOrtho },
    uNearCull: { value: fade.uNearCull },
    ...(withTexture ? { uBaseColorTex: { value: buildBaseColorTexture() } } : {}),
  };
}

/**
 * Build the TSL material for a variant, with blending neutralized for raw-pixel
 * parity against the harness's `ShaderMaterial` path — the parity spec compares
 * fragment output, not composite semantics (same treatment as the point entries).
 */
function buildMeshTSL(config: MeshTSLConfig) {
  return (uniforms: Record<string, THREE.IUniform>): THREE.Material => {
    const m = meshWebGPUFactory(
      buildMeshTSLNodesFromUniforms(uniforms, config),
      config
    ) as unknown as THREE.Material;
    m.transparent = false;
    m.blending = THREE.NoBlending;
    return m;
  };
}

/**
 * Registry of mesh shader entries for the TSL↔GLSL parity harness, keyed by
 * test name. Each entry carries the GLSL source and a `buildUniforms` factory;
 * merged into `SHADER_REGISTRY` and driven by the parity/codegen specs.
 */
export const MESH_SHADERS: Record<string, RegistryEntry> = {
  // The DEFAULT build: `opaque`, i.e. the hard alpha cutout, shading from stored
  // normals. Note the mesh default differs from the siblings' `additive`.
  mesh: {
    source: MESH_SOURCE,
    buildUniforms: () => meshUniforms(),
    buildDefines: () => ({ LUXAR_MESH_ALPHA_CUTOUT: '' }),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'opaque' }),
    buildMesh: buildMeshObject(),
  },
  // `mesh` with a uNearCull (0.8) that WOULD fade the quad under perspective: under
  // the ortho default camera the fade is the identity, so the frame equals `mesh`.
  'mesh-ortho-near-cull': {
    source: MESH_SOURCE,
    buildUniforms: () => meshUniforms(false, ORTHO_NEAR_CULL_UNIFORMS),
    buildDefines: () => ({ LUXAR_MESH_ALPHA_CUTOUT: '' }),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'opaque' }),
    buildMesh: buildMeshObject(),
  },
  // Alpha-weighted emission — the shape `additive` / `luminous` / `normal` share.
  // Distinct generated code from `mesh`: no cutout discard, and alpha reaches the
  // output instead of the constant 1.0.
  'mesh-additive': {
    source: MESH_SOURCE,
    buildUniforms: () => meshUniforms(),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'additive' }),
    buildMesh: buildMeshObject(),
  },
  // `max`: RGB premultiplied by coverage, because MaxEquation + OneFactor/OneFactor
  // does not weight source RGB by alpha at composite.
  'mesh-max': {
    source: MESH_SOURCE,
    buildUniforms: () => meshUniforms(),
    buildDefines: () => ({ LUXAR_MAX_RGB_CONTRIBUTION: '' }),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'max' }),
    buildMesh: buildMeshObject(),
  },
  // The derivative flat-normal variant: neither the `normal` attribute nor its
  // varying appears in the generated code at all. On this fixture it shades
  // uniformly where `mesh` shades with a radial gradient.
  'mesh-flat-normal': {
    source: MESH_SOURCE,
    buildUniforms: () => meshUniforms(),
    buildDefines: () => ({
      LUXAR_MESH_ALPHA_CUTOUT: '',
      LUXAR_MESH_FLAT_NORMAL: '',
    }),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'opaque', shading: 'flat' }),
    buildMesh: buildMeshObject(),
  },
  // A size-3 `float32` colour attribute — the RGB layout, where the opaque
  // `vAlpha = 1.0` comes from the attribute default for the missing `w` rather than
  // from a CPU-side pad. Under the `opaque` cutout that default is load-bearing: a
  // `w` of 0 would fail the cutout on every fragment and the surface would vanish.
  // Deliberately NOT in the codegen snapshot list — the shader is byte-identical to
  // `mesh`'s (the attribute width is invisible to it), so a snapshot would only
  // duplicate one. What this entry tests is the BINDING, which only rendering shows.
  'mesh-rgb': {
    source: MESH_SOURCE,
    buildUniforms: () => meshUniforms(),
    buildDefines: () => ({ LUXAR_MESH_ALPHA_CUTOUT: '' }),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'opaque' }),
    buildMesh: buildMeshObject(false, true),
  },
  // Colormap LUT: the vertex stage reads `aScalar` and samples the LUT instead of
  // the `color` attribute's RGB — while still reading its alpha, which is the one
  // part of the colour attribute the colormap does NOT replace.
  'mesh-colormap': {
    source: MESH_SOURCE,
    buildUniforms: () => meshUniforms(true),
    buildDefines: () => ({
      LUXAR_MESH_ALPHA_CUTOUT: '',
      USE_COLORMAP: '',
    }),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'opaque', useColormap: true }),
    buildMesh: buildMeshObject(true),
  },
  // A base-colour TEXTURE: the fragment stage samples `uBaseColorTex` through the
  // interpolated `vUv` instead of reading the `color` attribute's RGB. Distinct
  // generated code from every entry above — it is the only one with a sampler in the
  // FRAGMENT stage (the colormap's is in the vertex stage) and the only one that
  // declares a `uv` attribute.
  //
  // The fixture's transparent texel is what makes this non-vacuous under `opaque`:
  // texture alpha multiplies coverage, so one quadrant must be discarded. A backend
  // that dropped the alpha term would fill it.
  'mesh-texture': {
    source: MESH_SOURCE,
    buildUniforms: () => meshUniforms(false, ORTHO_FADE_UNIFORMS, true),
    buildDefines: () => ({
      LUXAR_MESH_ALPHA_CUTOUT: '',
      LUXAR_MESH_BASE_COLOR_TEX: '',
    }),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'opaque', useBaseColorTexture: true }),
    buildMesh: buildMeshObject(false, false, true),
  },
  // `shading: 'none'` — the unlit arm. No normal is computed at all, so neither the
  // `normal` attribute, its varying, nor the derivative pair appears in the
  // generated code, and the shade/specular terms are absent from the fragment tail.
  // On this fixture it renders the base colour flat where `mesh` shades it with a
  // radial gradient.
  'mesh-none-shading': {
    source: MESH_SOURCE,
    buildUniforms: () => meshUniforms(),
    buildDefines: () => ({
      LUXAR_MESH_ALPHA_CUTOUT: '',
      LUXAR_MESH_NO_SHADING: '',
    }),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'opaque', shading: 'none' }),
    buildMesh: buildMeshObject(),
  },
  // The pick pass in its DEFAULT state: `opaque`, so the cutout is on and the real
  // projected depth is written. The RGBA fixture's 0.25-alpha corner is below the
  // 0.5 cutoff, so part of the quad is discarded — which is the whole point of
  // asserting parity here, since a backend that dropped the discard would fill it.
  'mesh-pick': {
    source: MESH_PICK_SOURCE,
    buildUniforms: () => meshPickUniforms(true),
    buildTSLMaterial: (uniforms) =>
      meshPickWebGPUFactory(
        buildMeshPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: buildMeshObject(),
  },
  // The pick twin of `mesh-texture`. Its own entry rather than a variant of
  // `mesh-pick` because the sampler is DEFINE-gated there too (unlike the cutout,
  // which is a runtime uniform), so this is genuinely distinct generated code.
  //
  // What it pins is the correctness point the whole pick-side texture work exists
  // for: the transparent texel must be discarded HERE as well as in the visual
  // pass, or the hole the user can see stays pickable and depth-occluding.
  'mesh-pick-texture': {
    source: MESH_PICK_SOURCE,
    buildUniforms: () => meshPickUniforms(true, ORTHO_FADE_UNIFORMS, true),
    buildDefines: () => ({ LUXAR_MESH_PICK_BASE_COLOR_TEX: '' }),
    buildTSLMaterial: (uniforms) =>
      meshPickWebGPUFactory(
        buildMeshPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: buildMeshObject(false, false, true),
  },
  // The OTHER arm of both runtime uniforms: no cutout, brightness-as-depth. Not in
  // the codegen snapshot list, and deliberately so — the branch is a runtime uniform
  // (§6.5), so this generates the byte-identical shader `mesh-pick` snapshots and a
  // second snapshot would only duplicate one. What it adds is the rendered proof
  // that with the cutout OFF the low-alpha corner survives on BOTH backends, which
  // is the assertion that would catch a build flag creeping back in.
  'mesh-pick-commutative': {
    source: MESH_PICK_SOURCE,
    buildUniforms: () => meshPickUniforms(false),
    buildTSLMaterial: (uniforms) =>
      meshPickWebGPUFactory(
        buildMeshPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: buildMeshObject(),
  },
  // The near fade under PERSPECTIVE, in the `opaque` default — the one mode where
  // the fade ramps the shaded RGB rather than the coverage, because its output
  // alpha is the constant 1.0 and there is no alpha left to fade. Every fragment of
  // the quad sits at viewZ = -1, so the fade is a uniform 0.15625 (see
  // NEAR_FADE_UNIFORMS for the arithmetic) and the two backends must agree exactly.
  //
  // Deliberately NOT in the codegen snapshot list: `uIsOrtho` is a runtime uniform,
  // so this generates the shader `mesh` already snapshots and a second copy would
  // only duplicate one. What it adds is the rendered proof — same reasoning as
  // `mesh-pick-commutative`.
  'mesh-near-fade': {
    source: MESH_SOURCE,
    // Disable the additive highlight for this ratio measurement: the un-faded
    // reference otherwise clips above 1.0 while the faded frame does not, so the
    // 8-bit readback no longer preserves the exact 0.15625 ratio being tested.
    buildUniforms: () => ({
      ...meshUniforms(false, NEAR_FADE_UNIFORMS),
      uSpecular: { value: 0.0 },
    }),
    buildDefines: () => ({ LUXAR_MESH_ALPHA_CUTOUT: '' }),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'opaque' }),
    buildMesh: buildMeshObject(),
    buildCamera: buildBehindCamera,
  },
  // The un-faded twin of the entry above: same camera, same uNearCull, `uIsOrtho`
  // flipped to 1 so the fade is the identity. This is what the parity spec divides
  // against — the ortho-camera `mesh` entry would be a DIFFERENT crop of the quad,
  // so its pixel (32, 32) is not the same surface point. Out of the codegen list for
  // the same runtime-uniform reason as `mesh-near-fade`.
  'mesh-near-fade-reference': {
    source: MESH_SOURCE,
    buildUniforms: () => ({
      ...meshUniforms(false, UNFADED_REFERENCE_UNIFORMS),
      uSpecular: { value: 0.0 },
    }),
    buildDefines: () => ({ LUXAR_MESH_ALPHA_CUTOUT: '' }),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'opaque' }),
    buildMesh: buildMeshObject(),
    buildCamera: buildBehindCamera,
  },
  // The OTHER fade fold, and the one `mesh-near-fade` structurally cannot reach: with
  // no cutout the fade multiplies into COVERAGE (`a *= nearFade`) and leaves the
  // shaded RGB alone — the arm `additive` / `luminous` / `normal` all take, and the
  // one `max` inherits through its premultiply. Rendering it is the only proof that
  // fold is wired: the codegen snapshot shows the line, but a snapshot cannot tell a
  // multiply into alpha from a multiply into nothing.
  //
  // Out of the codegen list, same reasoning as `mesh-pick-commutative`: `uIsOrtho` is
  // a runtime uniform, so this generates the shader `mesh-additive` already snapshots.
  'mesh-additive-near-fade': {
    source: MESH_SOURCE,
    buildUniforms: () => meshUniforms(false, NEAR_FADE_UNIFORMS),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'additive' }),
    buildMesh: buildMeshObject(),
    buildCamera: buildBehindCamera,
  },
  // Its un-faded reference, on the same terms as `mesh-near-fade-reference`.
  'mesh-additive-near-fade-reference': {
    source: MESH_SOURCE,
    buildUniforms: () => meshUniforms(false, UNFADED_REFERENCE_UNIFORMS),
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'additive' }),
    buildMesh: buildMeshObject(),
    buildCamera: buildBehindCamera,
  },
  // The same fade on the PICK pass, which folds it into `brightness` instead. Pick
  // coverage has to track visible coverage as the camera closes in, or a surface
  // the user can barely see stays fully pickable. Same runtime-uniform argument for
  // staying out of the codegen list.
  //
  // This one renders a UNIFORM frame, and legitimately: the quad overfills the
  // perspective frame; the node id is a per-node constant and the element id, though
  // a per-VERTEX ordinal (`gl_VertexID`, flat), lands on the same byte for both
  // triangles of this 4-vertex quad — the harness target is RGBA8, so any non-zero
  // ordinal clamps to 255 and the high half is 0 throughout; and the cutout arm's
  // brightness is the constant 1.0 before the fade scales it — 4096 pixels of the
  // same RGBA. So the parity spec cannot use its `assertBothRendered` helper here
  // (whose "did anything render?" proxy is "some pixel differs from pixel 0"); it
  // pins the centre pixel against the un-faded `mesh-pick` instead, and then that
  // every other pixel equals the centre.
  'mesh-pick-near-fade': {
    source: MESH_PICK_SOURCE,
    buildUniforms: () => meshPickUniforms(true, NEAR_FADE_UNIFORMS),
    buildTSLMaterial: (uniforms) =>
      meshPickWebGPUFactory(
        buildMeshPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: buildMeshObject(),
    buildCamera: buildBehindCamera,
  },
};
