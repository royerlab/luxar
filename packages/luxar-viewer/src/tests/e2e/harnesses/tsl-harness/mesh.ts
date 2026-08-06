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
import { buildColormapTexture } from './shared';

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

function buildMeshGeometry(withScalars = false, rgbOnly = false): THREE.BufferGeometry {
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
  geometry.setIndex(new THREE.BufferAttribute(QUAD_INDICES, 1, false));
  return geometry;
}

const buildMeshObject =
  (withScalars = false, rgbOnly = false) =>
  (material: THREE.Material): THREE.Object3D => {
    const mesh = new THREE.Mesh(buildMeshGeometry(withScalars, rgbOnly), material);
    // Both faces, matching the `double_sided` default — and required for the
    // `gl_FrontFacing` flip to be exercised at all by a camera on either side.
    material.side = THREE.DoubleSide;
    return mesh;
  };

/** Uniforms mirroring the production `MeshMaterial` constructor. */
function meshUniforms(withColormap = false): Record<string, THREE.IUniform> {
  return {
    uOpacity: { value: 1.0 },
    uInvGamma: { value: 1.0 / 2.2 },
    uIntensity: { value: 1.0 },
    uOffset: { value: 0.0 },
    uAmbient: { value: MESH_DEFAULTS.ambient },
    uShadeExponent: { value: MESH_DEFAULTS.shadeExponent },
    uAlphaCutoff: { value: MESH_DEFAULTS.alphaCutoff },
    ...(withColormap
      ? {
          uColormapTex: { value: buildColormapTexture() },
          uScalarMin: { value: 0.0 },
          uScalarScale: { value: 1.0 },
        }
      : {}),
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
function meshPickUniforms(surfaceMode: boolean): Record<string, THREE.IUniform> {
  return {
    uNodeId: { value: 1 },
    uOpacity: { value: 1.0 },
    uAlphaCutoff: { value: MESH_DEFAULTS.alphaCutoff },
    uAlphaCutout: { value: surfaceMode ? 1 : 0 },
    uSurfaceDepth: { value: surfaceMode ? 1 : 0 },
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
    buildTSLMaterial: buildMeshTSL({ blendingMode: 'opaque', flatNormal: true }),
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
};
