/**
 * Mesh appearance constants and the blending-mode → emission-shape mapping.
 *
 * Small on purpose, and shared by four consumers that must not disagree: the GLSL
 * material wrapper, the TSL material wrapper, the TSL factory's build-time branch,
 * and `createMeshNode`. Each of them has to answer the same two questions — "what
 * does a mesh look like by default" and "which fragment emission does this blending
 * mode want" — and a per-file copy of either answer is a divergence waiting to
 * happen between the two backends.
 *
 * @module rendering/materials/mesh/appearance
 */

import { getCompleteBlendingState, type CompleteBlendingState } from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';

/**
 * Mesh material defaults.
 *
 * `ambient` / `shadeExponent` parameterize the §6.2 headlight
 * `shade = mix(ambient, 1, pow(saturate(dot(N, V) * 0.5 + 0.5), shadeExponent))`:
 *
 * - `ambient` is the shade floor. It is what a surface facing *away* from the
 *   camera keeps, so it is what makes a silhouette readable rather than black.
 *   `1.0` collapses the whole term to 1 and reproduces the emissive look of the
 *   other three geometry types.
 * - `shadeExponent` shapes the falloff between the head-on and edge-on extremes.
 *   `1.0` is the plain linear wrap.
 *
 * `alphaCutoff` is the `opaque`-mode cutout threshold (§6.2). `0.5` is the
 * conventional alpha-test midpoint, and — because node opacity folds INTO the
 * coverage that is compared against it — it also fixes where a plain RGB mesh
 * (whose `vAlpha` is identically 1) dissolves as `opacity` sweeps down: at 0.5.
 */
export const MESH_DEFAULTS = {
  ambient: 0.25,
  shadeExponent: 1.5,
  alphaCutoff: 0.5,
} as const;

/**
 * The blending modes a mesh supports (§6.3).
 *
 * `volumetric` is absent, and that is the point: it integrates emission and
 * absorption along the view ray through a participating medium, and a triangle is a
 * zero-thickness surface — its path length through the medium is zero, so there is
 * nothing for `absorption` to attenuate. The Python adder refuses it outright on a
 * mesh node; the viewer still has to cope with it arriving by INHERITANCE from an
 * ancestor group the mesh knows nothing about, which is what
 * {@link resolveMeshBlendingMode} is for.
 */
export const MESH_SUPPORTED_BLENDING_MODES = [
  'opaque',
  'normal',
  'additive',
  'luminous',
  'max',
] as const;

/**
 * Map an incoming blending mode onto one a mesh can actually draw.
 *
 * Driven by {@link MESH_SUPPORTED_BLENDING_MODES} rather than by an
 * `=== 'volumetric'` test, so the list above is load-bearing: anything not on it
 * falls back to `opaque`. Today that is only `volumetric`, but the fragment shader
 * implements exactly three emissions, and a future mode reaching it unmapped would
 * silently take whichever branch its `shaderOutputMode` happened to land on.
 *
 * Callers decide whether to warn: `createMeshNode` warns once per node (naming it),
 * while the material wrappers apply the same mapping silently, because by then the
 * warning has been issued and a runtime mode switch from the layers panel would
 * otherwise re-warn on every click.
 *
 * A warning rather than a load failure, deliberately — the mode may be inherited
 * from an ancestor, so refusing would make an unrelated group setting break an
 * otherwise valid mesh.
 */
export function resolveMeshBlendingMode(mode: BlendingMode): BlendingMode {
  // The widening cast is the same one `load-scene.ts` needs for its version
  // allowlists: a literal-typed readonly tuple's `.includes` rejects any argument
  // outside the tuple's own union, which is exactly the argument this has to test.
  return (MESH_SUPPORTED_BLENDING_MODES as readonly BlendingMode[]).includes(mode)
    ? mode
    : 'opaque';
}

/**
 * The fragment emission shape for a mesh in `mode`, after the volumetric mapping.
 *
 * Three shapes, one per §6.2 bullet:
 * - `'opaque'` → hard alpha cutout, fully-opaque survivors, depth written.
 * - `'rgb-contribution'` → RGB premultiplied by coverage, for `max`'s
 *   `MaxEquation + OneFactor/OneFactor` state which does not weight source RGB by
 *   alpha at composite.
 * - `'alpha-weighted'` → plain `vec4(rgb, a)`; the framebuffer applies the coverage.
 *
 * Read from `getCompleteBlendingState` rather than re-derived, so the shader branch
 * and the framebuffer state can never disagree about which one is in force.
 */
export function resolveMeshOutput(mode: BlendingMode): CompleteBlendingState['shaderOutputMode'] {
  return getCompleteBlendingState(resolveMeshBlendingMode(mode)).shaderOutputMode;
}
