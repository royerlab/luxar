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
 * Clamp the headlight wrap exponent to a safe range, supplying the default when
 * undefined.
 *
 * The same class of hazard `clampGamma` exists for, and the same `0.001` bound. The
 * shade term computes `pow(wrap, exponent)` where `wrap = saturate(N·V · 0.5 + 0.5)`,
 * and `wrap` is **exactly 0** for any fragment whose normal faces directly away from
 * the camera (`N·V == -1`, which the two-sided flip leaves reachable on the
 * derivative-fallback path and on any unflipped geometry). GLSL leaves `pow(0, y)`
 * undefined for `y <= 0`, so an author setting `shadeExponent = 0` — a perfectly
 * plausible "I want no gradient" value — produces driver-dependent output (1, 0 or
 * NaN) at precisely the silhouette.
 *
 * Clamping keeps that fragment DEFINED and sensible: at `exponent = 0.001`,
 * `pow(0, 0.001)` is 0, so a face-away fragment shades at `ambient` — which is what
 * the ambient floor means. The honest way to ask for no gradient at all is
 * `ambient = 1`, which collapses the `mix` and is documented as such.
 *
 * Lives here rather than in `_shared/uniform-helpers.ts` because that module states
 * its own scope as helpers "that already appear in every material constructor", and
 * mesh is the only shaded type.
 */
export function clampShadeExponent(exponent: number | undefined): number {
  return Math.max(0.001, exponent ?? MESH_DEFAULTS.shadeExponent);
}

/**
 * Clamp a `[0, 1]` appearance fraction, supplying `fallback` when absent or unusable.
 *
 * `ambient` and `alphaCutoff` are both genuine fractions, not gains: `ambient` is the
 * shade floor a face-away fragment keeps, and `alphaCutoff` is compared against a
 * coverage that is itself in `[0, 1]`. Values outside the range are not merely odd,
 * they are meaningless — `ambient = 1e9` multiplies the surface to white and
 * `alphaCutoff = 1e9` discards every fragment, i.e. the mesh disappears with no
 * diagnostic. Both are reachable: `add_mesh(**attrs)` writes arbitrary attrs to zarr,
 * so these arrive from authored (or hand-crafted) metadata rather than from code.
 *
 * NaN/Inf route to `fallback` rather than clamping, matching the sibling shaders'
 * sanitizer policy — corruption resolves to the documented default, loudly, instead
 * of to a range boundary that looks deliberate.
 */
export function clampAppearanceFraction(value: number | undefined, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  return Math.min(1, Math.max(0, value));
}

/**
 * Squared-length floor below which an interpolated stored normal is not trusted.
 *
 * Compared against `dot(N, N)`, so this is `(1e-6)²` — a normal shorter than one
 * part per million of unit length. Two sources produce them: a legitimately
 * degenerate triangle (the writer *warns* rather than rejecting zero-length normals,
 * spec §3.5), and interpolation across a triangle whose corner normals oppose, which
 * cancels to ~0 somewhere in between.
 *
 * Lives HERE rather than once per shader file, and that placement is the point: the
 * GLSL side needs it as a source string and the TSL side as a JS number, so the
 * obvious arrangement is two constants — which is a value that can silently drift in
 * two directions, with the two backends then switching to the derivative fallback on
 * *different* fragments. One number, interpolated into the GLSL and passed to TSL
 * directly, makes that unrepresentable rather than merely tested for.
 *
 * `String(1e-12)` is `"1e-12"`, a valid GLSL ES 3.0 floating constant (the exponent
 * form needs no decimal point), so the interpolation needs no formatting helper.
 */
export const MESH_NORMAL_EPS_SQ = 1e-12;

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

/**
 * The two emission defines, keyed to the shape the mode selected. Exactly one is set,
 * or neither.
 */
export const MESH_EMISSION_DEFINES = {
  opaque: 'LUXAR_MESH_ALPHA_CUTOUT',
  'rgb-contribution': 'LUXAR_MAX_RGB_CONTRIBUTION',
} as const satisfies Partial<Record<CompleteBlendingState['shaderOutputMode'], string>>;

/**
 * Bring a material's emission defines in line with `output`, and report whether
 * anything changed.
 *
 * Shared by both wrappers because the invariant it maintains — **at most one
 * emission define is ever set** — is not local to either. The mesh fragment has three
 * emissions where its siblings have two, so leaving a stale flag behind is a live
 * failure mode rather than a theoretical one: an `opaque → additive` switch that
 * stranded `LUXAR_MESH_ALPHA_CUTOUT` would keep discarding fragments in a mode with
 * no cutout, and a mesh would appear to lose parts of itself at random. Implemented
 * twice, the unit test asserting "never both set" would only prove that both copies
 * happen to be right today.
 *
 * @returns `true` when a define was added or removed — the caller's signal to
 *   recompile (GLSL) or rebuild the graph (TSL).
 */
export function syncMeshEmissionDefines(
  defines: Record<string, unknown>,
  output: CompleteBlendingState['shaderOutputMode']
): boolean {
  let changed = false;
  for (const [shape, flag] of Object.entries(MESH_EMISSION_DEFINES)) {
    const wanted = output === shape;
    const had = flag in defines;
    if (wanted && !had) {
      defines[flag] = '';
      changed = true;
    } else if (!wanted && had) {
      delete defines[flag];
      changed = true;
    }
  }
  return changed;
}
