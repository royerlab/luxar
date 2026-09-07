/**
 * Mesh visual shader pair (GLSL3) — the hand-written reference twin of
 * `shader-tsl.ts`.
 *
 * Mesh is the first geometry type in Luxar that **shades**. The other three are
 * purely emissive per-element sprites: their fragment stage computes a soft falloff
 * and emits colour, with no notion of a surface orientation. A triangle has one, and
 * a mesh drawn flat is an unreadable silhouette — so this pair adds a light-free
 * shade term, and with it the two hazards that come from reading a *normal*:
 * degenerate/corrupt stored normals, and back faces.
 *
 * It is also the first pair whose per-vertex data arrives in **vertex attributes**
 * rather than an RGBA32F element texture, so there is no `texelFetch` prologue, no
 * `aSortedIndex` indirection, and no quad expansion — `position` and `normal` are
 * three's own auto-declared attributes and the draw is an ordinary indexed
 * `drawElements`.
 *
 * ## Shader variants (spec §6.2 / §6.4)
 *
 * | Define | Effect |
 * |---|---|
 * | `USE_COLORMAP` | read `aScalar` + LUT instead of the `color` attribute |
 * | `LUXAR_GAMMA_ONE` | skip the gamma `pow()` (`pow(x, 1) == x`) |
 * | `LUXAR_NO_GOG` | skip the `intensity`/`offset` mul-add-clamp chain |
 * | `LUXAR_MESH_FLAT_NORMAL` | derivative-only normal; the `normal` attribute and its varying are not read at all |
 * | `LUXAR_MESH_ALPHA_CUTOUT` | `opaque` mode: hard, order-independent alpha cutout |
 * | `LUXAR_MAX_RGB_CONTRIBUTION` | `max` mode: premultiply RGB by coverage |
 *
 * The stored-normal-vs-flat choice is a **compile-time variant**, not a runtime
 * branch, because a declared-but-unbound `normal` attribute reads `(0, 0, 0, 1)`
 * rather than "absent" — there is no runtime value that means "no normals". It is
 * decided once per node in `createMeshNode` and handed to both backends, so the two
 * can never derive it differently (spec §6.2).
 *
 * @module rendering/materials/mesh/shader-glsl
 */

import { GLSL_SANITIZE_FUNCTIONS, GLSL_NEAR_FADE_FUNCTIONS } from '../_shared/glsl-lib';
import {
  GLSL_GLASS_PARTITION_GUARD,
  GLSL_GLASS_PARTITION_UNIFORMS,
} from '../_shared/glass-partition';
import { MESH_LIGHT_DIRECTION, MESH_NORMAL_EPS_SQ } from './appearance';
import type { ShaderSource } from '../_shared/shader-source';
import { requireTslMaterials } from '../../tsl/slot';

/**
 * Vertex stage.
 *
 * `position`, `normal` and `normalMatrix` are **not declared here** — three's
 * `ShaderMaterial` prefix auto-injects all three (`attribute vec3 position;`,
 * `attribute vec3 normal;`, `uniform mat3 normalMatrix;`, with
 * `#define attribute in` under GLSL3). Re-declaring any of them is a compile error.
 * `color` and `aScalar` are ours and are declared below.
 *
 * `color` is read as a **`vec4`** regardless of how many components the attribute
 * carries: GL fills the missing components of a size-3 attribute with `(0, 0, 0, 1)`,
 * so RGB data supplies the opaque `w = 1.0` for free. The 8/16-bit family is padded
 * to 4 components CPU-side instead (`mesh-geometry.ts`), because three r184's WebGPU
 * backend has no valid 3-component `unorm8`/`unorm16` vertex format — see §6.1.1.
 * Either way there is ONE `color` attribute and ONE shader that reads it.
 */
export const MESH_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}

    // Per-vertex colour, always bound (never left to the GL default black —
    // see createMeshDefaultColorAttribute). RGB input arrives as w = 1.0.
    in vec4 color;

    #ifdef USE_COLORMAP
    in float aScalar;                  // per-vertex scalar for the LUT lookup
    uniform sampler2D uColormapTex;    // 256x1 LUT texture
    uniform float uScalarMin;          // display-range minimum
    uniform float uScalarScale;        // 1.0 / (max - min)
    uniform mediump float uInvGamma;   // gamma on the VALUE, pre-LUT
    #endif

    #ifdef LUXAR_MESH_BASE_COLOR_TEX
    // \`uv\` needs no declaration: three.js's vertex prefix declares
    // position/normal/uv unconditionally for every program, which is also why
    // \`position\` and \`normal\` are used below without appearing here. Declaring it
    // would be a redefinition error.
    out mediump vec2 vUv;
    #endif

    out mediump vec3 vColor;
    // SMOOTHLY interpolated, unlike the gsplat shader's \`flat out … vAlpha\`:
    // a splat's alpha is a per-INSTANCE constant, so interpolating it is a no-op
    // and \`flat\` documents that. A mesh vertex is not an instance constant — its
    // opacity must vary across the face exactly as the line shader's vAlpha
    // varies along a segment.
    out mediump float vAlpha;
    // VIEW-space position. highp because the fragment stage differentiates it:
    // \`cross(dFdx(vViewPos), dFdy(vViewPos))\` is the flat-normal fallback, and
    // mediump would quantize the inter-fragment delta into a noisy normal.
    out highp vec3 vViewPos;

    #if !defined(LUXAR_MESH_FLAT_NORMAL) && !defined(LUXAR_MESH_NO_SHADING)
    // VIEW-space normal. The attribute is in the node's local display frame and
    // every other input to the shade term is view-space, so it is carried across
    // by \`normalMatrix\` (the inverse-transpose of the model-view matrix) HERE,
    // before interpolation. The inverse-transpose is load-bearing rather than
    // pedantic: anisotropic voxel spacing (z != xy) is routine in this domain,
    // and under it the plain model-view linear map skews normals off
    // perpendicular. No vertex-stage normalize — the fragment renormalizes
    // anyway, which absorbs the length change normalMatrix introduces.
    out highp vec3 vNormal;
    #endif

    void main() {
      #ifdef LUXAR_MESH_BASE_COLOR_TEX
      vUv = uv;
      #endif

      #ifdef USE_COLORMAP
      // The display-range window and gamma shape the scalar VALUE before the LUT
      // lookup, not the resulting colour — same rule as the point/line shaders.
      // Intensity/offset apply POST-LUT in the fragment stage.
      float t = clamp((aScalar - uScalarMin) * uScalarScale, 0.0, 1.0);
      #ifndef LUXAR_GAMMA_ONE
      t = pow(t, uInvGamma);
      #endif
      vColor = texture(uColormapTex, vec2(t, 0.5)).rgb;
      #elif defined(LUXAR_MESH_BASE_COLOR_TEX)
      // Left at white: the base colour comes from a PER-FRAGMENT texture fetch, so
      // there is nothing per-vertex to carry. Not skipped altogether because the
      // fragment stage multiplies by \`vColor\` unconditionally — that keeps the
      // GOG/gamma tail identical across all three colour sources rather than
      // forking it three ways, and white is the multiplicative identity.
      vColor = vec3(1.0);
      #else
      vColor = color.rgb;
      #endif

      // Read unconditionally, including under USE_COLORMAP: the scalar replaces
      // the colour, not the opacity. Sanitized for the same reason the sibling
      // shaders sanitize theirs — alpha is load-bearing (it is the whole coverage
      // term for a mesh, which has no per-element intensity), and a NaN would
      // survive into the cutout comparison as a fragment that never discards.
      vAlpha = sanitizeAlpha(color.a);

      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      vViewPos = mvPosition.xyz;
      #if !defined(LUXAR_MESH_FLAT_NORMAL) && !defined(LUXAR_MESH_NO_SHADING)
      vNormal = normalMatrix * normal;
      #endif
      gl_Position = projectionMatrix * mvPosition;
    }
  `;

/**
 * Fragment stage.
 *
 * ## The order of the three normal rules is load-bearing
 *
 * 1. The **derivative normal is computed unconditionally**, before any
 *    guard-dependent branch. The epsilon guard below reads an interpolated varying,
 *    so branching on it is non-uniform control flow — where GLSL leaves
 *    `dFdx`/`dFdy` *undefined*, because normal validity can differ between fragments
 *    of the same 2×2 quad. Only the cheap `gl_FrontFacing` flip may sit behind the
 *    guard; never the derivative evaluation.
 * 2. The **epsilon guard is written negated** (`!(dot(N, N) >= eps)`), which is not a
 *    style choice: `NaN` fails *every* comparison, so a corrupt store's NaN normal
 *    fails `>=` and takes the fallback. The positive form `dot(N, N) < eps` would let
 *    it slip through and normalize into NaN shading.
 * 3. The **two-sided flip applies to the stored normal only**. `gl_FrontFacing ? N :
 *    -N` exists because `double_sided` defaults true and §5's whole-triangle cull
 *    exposes the interior back faces of a sliced closed isosurface: without the flip,
 *    the stored back normal shades on the wrong side of its gradient, producing an
 *    inverted result that can collapse toward `uAmbient`.
 *    The derivative normal needs no flip — `cross(dFdx, dFdy)` is defined by the
 *    rasterized fragment, not by the winding, so it always faces the viewer — and
 *    flipping it would *reintroduce* exactly that inverted shade, worst precisely at
 *    the degenerate vertices the guard is there to rescue. So the exemption is per
 *    FRAGMENT, not per variant.
 *
 * ## The near fade is evaluated here, not in the vertex stage
 *
 * The three sibling types evaluate `perspectiveNearFade` per VERTEX (points/gsplats)
 * or partly so (lines clip the segment in the vertex stage), because an instanced
 * quad has one center depth and a per-vertex value is exact for the whole sprite. A
 * triangle is not a sprite: it spans depth, so a per-vertex fade would interpolate
 * the RAMP across the face and a large triangle straddling the fade band would
 * render a linear smear instead of the smoothstep. Hence the fade is computed here,
 * off the existing `vViewPos` varying.
 */
export const MESH_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;
    ${GLSL_GLASS_PARTITION_UNIFORMS}

    ${GLSL_NEAR_FADE_FUNCTIONS}

    uniform mediump float uOpacity;
    uniform mediump float uInvGamma;   // Pre-computed 1/gamma
    uniform mediump float uIntensity;  // Per-node linear colour multiplier (gain)
    uniform mediump float uOffset;     // Per-node additive brightness shift
    // Shade term (§6.2): view-anchored lighting, no scene light.
    uniform mediump float uAmbient;        // shade floor at the silhouette
    uniform mediump float uShadeExponent;  // wrap-term contrast
    uniform mediump float uSpecular;       // additive highlight strength
    uniform mediump float uShininess;      // highlight exponent
    // Cutout threshold, read only under LUXAR_MESH_ALPHA_CUTOUT.
    uniform mediump float uAlphaCutoff;
    // Near-fade inputs, the same pair the three siblings carry: 0 = perspective,
    // 1 = orthographic (where the fade is the identity), and the scene-relative
    // fade start in world units.
    uniform int uIsOrtho;
    uniform float uNearCull;

    #ifdef LUXAR_MESH_BASE_COLOR_TEX
    uniform sampler2D uBaseColorTex;
    in mediump vec2 vUv;
    #endif

    in mediump vec3 vColor;
    in mediump float vAlpha;
    in highp vec3 vViewPos;
    #if !defined(LUXAR_MESH_FLAT_NORMAL) && !defined(LUXAR_MESH_NO_SHADING)
    in highp vec3 vNormal;
    #endif

    out vec4 fragColor;

    void main() {
      ${GLSL_GLASS_PARTITION_GUARD}
      #ifndef LUXAR_MESH_NO_SHADING
      // (1) Derivative normal, UNCONDITIONALLY — see the module doc.
      highp vec3 derivativeNormal = normalize(cross(dFdx(vViewPos), dFdy(vViewPos)));
      // Forced viewer-facing rather than assumed so, which makes the fallback
      // convention-INDEPENDENT. \`cross(dFdx, dFdy)\` carries the sign of the
      // fragment-space y axis, and the two shading-language specs differ on it: GLSL's
      // \`dFdy\` is with respect to a BOTTOM-UP window coordinate while WGSL's \`dpdy\`
      // is TOP-DOWN. Under opposite conventions the identical surface yields +z on one
      // backend and -z on the other, and an unforced flat variant would then shade
      // correctly on one and collapse toward \`uAmbient\` on the other. Since the
      // fixed view axis is (0, 0, 1), "faces the viewer" is exactly
      // \`z >= 0\` and one sign flip settles it for either convention.
      //
      // MEASURED, so the comment does not overstate: on Chrome + Apple Silicon the two
      // backends agree with the flip REMOVED — a face-on flat quad renders identically
      // (37,151 lit pixels either way, byte-identical means), so the conventions
      // coincide there and the flip is currently INERT on that platform. It is kept
      // because it costs one instruction, is correct under either convention, and the
      // GLSL/WGSL specs do not promise they agree; it is insurance, not a fix for an
      // observed bug. It is also winding-independent, which is what §6.2 asserts.
      if (derivativeNormal.z < 0.0) derivativeNormal = -derivativeNormal;

      #ifdef LUXAR_MESH_FLAT_NORMAL
      highp vec3 N = derivativeNormal;
      #else
      highp float nn = dot(vNormal, vNormal);
      // (2) AFFIRMATIVE on purpose — the stored normal is used only when the length
      // test is positively TRUE, never when its negation is false. NaN fails every
      // comparison, so a corrupt store lands in the fallback here, whereas a
      // \`nn < eps\` test would pass it through to normalize into NaN shading.
      //
      // TWO-SIDED, which the spec's \`!(dot(N, N) >= eps)\` phrasing is not: an
      // INFINITE normal component gives \`nn == inf\`, which satisfies \`>= eps\`, and
      // then \`vNormal * inversesqrt(inf)\` is \`inf * 0.0\` = NaN — the same NaN
      // shading the guard exists to prevent, arriving from the other end. The writer
      // rejects non-finite normals, so this is the hand-crafted-zarr case every
      // sibling shader sanitizes for. \`1e30\` is the finite-range convention already
      // used by the TSL sanitizers.
      bool storedUsable = nn >= ${MESH_NORMAL_EPS_SQ} && nn < 1e30;
      highp vec3 storedNormal = vNormal * inversesqrt(max(nn, ${MESH_NORMAL_EPS_SQ}));
      // (3) Flip the STORED normal only, and only when it survived the guard.
      storedNormal = gl_FrontFacing ? storedNormal : -storedNormal;
      highp vec3 N = storedUsable ? storedNormal : derivativeNormal;
      #endif

      // View-anchored lighting: the offset key follows the camera without adding
      // scene light state. V stays the fixed view-space axis (0, 0, 1), so the
      // Blinn-Phong half-vector is constant too.
      mediump vec3 L = normalize(vec3(${MESH_LIGHT_DIRECTION.join(', ')}));
      mediump vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
      mediump float wrap = clamp(dot(N, L) * 0.5 + 0.5, 0.0, 1.0);
      mediump float shade = mix(uAmbient, 1.0, pow(wrap, uShadeExponent));
      mediump float spec = uSpecular * pow(max(dot(N, H), 0.0), uShininess);
      #endif

      // Per-node GOG. Identical chain to the sibling shaders (§6.2 notes it is
      // copied rather than shared — _shared/ carries the sanitizers and the
      // define helpers, but each of the eight shader files writes this tail out).
      // Base colour. The texture is sampled PER FRAGMENT, unlike the colormap LUT,
      // which is a vertex-stage lookup: a LUT maps one scalar per vertex and
      // interpolating the resulting colour is a close-enough model of
      // interpolating the scalar, whereas an image has structure BETWEEN vertices
      // and a per-vertex fetch would resolve exactly one texel per vertex —
      // reproducing the point-cloud limitation this feature exists to remove.
      #ifdef LUXAR_MESH_BASE_COLOR_TEX
      mediump vec4 texel = texture(uBaseColorTex, vUv);
      #ifdef LUXAR_MESH_TEX_LUMINANCE
      // A single-channel texture uploads as RedFormat, which samples as
      // (r, 0, 0, 1) — so without this swizzle a greyscale basemap renders pure
      // red rather than grey. Its own define because the alternative, expanding
      // 1 channel to RGBA on the CPU, would quadruple the upload for data that
      // is one \`.rrr\` away from correct.
      mediump vec3 baseColor = vColor * texel.rrr;
      #else
      mediump vec3 baseColor = vColor * texel.rgb;
      #endif
      // Texture alpha MULTIPLIES coverage, so an RGBA basemap gets real cutout
      // holes under \`opaque\` rather than an all-or-nothing silhouette. A
      // 3-channel texture is expanded to RGBA with alpha 1 at upload and a
      // 1-channel one samples alpha 1, so this term is a free no-op for both.
      mediump float texAlpha = texel.a;
      #else
      mediump vec3 baseColor = vColor;
      mediump float texAlpha = 1.0;
      #endif

      #ifdef LUXAR_NO_GOG
      mediump vec3 adjusted = baseColor;
      #else
      mediump vec3 adjusted = max(baseColor * uIntensity + uOffset, vec3(0.0));
      #endif

      // Colormap mode already applied gamma to the scalar VALUE pre-LUT, and
      // LUXAR_GAMMA_ONE means pow(x, 1) == x. Either way, skip the pow().
      #if defined(USE_COLORMAP) || defined(LUXAR_GAMMA_ONE)
      mediump vec3 finalColor = adjusted;
      #else
      mediump vec3 finalColor = pow(adjusted, vec3(uInvGamma));
      #endif

      // The shade factor is a LIGHTING term: it multiplies RGB and must never
      // enter the coverage below, or a silhouette fragment would also turn
      // transparent (and, under the cutout, dissolve).
      #ifdef LUXAR_MESH_NO_SHADING
      // Unlit: the base colour reaches the screen unmodulated. This is what every
      // other Luxar geometry type does — the other three are purely emissive — and
      // what a data basemap needs, since a view-anchored key would make a
      // colour-coded surface read differently as the camera moved.
      mediump vec3 shadedColor = finalColor;
      #else
      mediump vec3 shadedColor = finalColor * shade + vec3(spec);
      #endif

      // Mesh has no per-element intensity/amplitude/falloff scalar (§2.2) — it is
      // a solid surface — so coverage is just the per-vertex alpha times node
      // opacity. NOT \`intensity * uOpacity\` like the emissive types.
      mediump float a = vAlpha * uOpacity * texAlpha;

      // Perspective near fade, PER FRAGMENT (see the module doc for why not per
      // vertex). uNearCull is scene-bounds-scaled (diagonal * 0.001); the 1e-20
      // floor only guards the degenerate smoothstep (edge0 == edge1) when
      // uNearCull is exactly 0 — an ABSOLUTE floor here would override the
      // scene-relative value and fade out a whole tiny-unit scene, as it did on
      // the sibling shaders. Kept at the file's highp default rather than
      // mediump like the appearance uniforms: only the RESULT is in [0, 1], and
      // the depths being compared are not (same as the line shader's twin).
      float nearFade = perspectiveNearFade(uIsOrtho, vViewPos.z, max(uNearCull, 1e-20));
      // Rejected in EVERY mode, at the siblings' 0.01 threshold. Not optional in
      // the depth-writing ones: 'opaque' always writes depth and 'normal' does at
      // opacity >= 0.99 (blending-state.ts::normalModeDepthWrite, which mesh feeds
      // its real opacity), so a fully-faded-but-still-rasterized fragment would
      // occlude everything behind it while contributing nothing visible. The
      // unconditional form is the cheap one anyway: the alternative is a runtime
      // uniform for a branch that only ever saves a discard. The cost is early-z: the four
      // non-cutout variants had no discard at all before this, and any discard
      // forfeits it. Paid knowingly — the same trade the mesh pick fragment makes
      // for gl_FragDepth — because a wrong depth buffer is not recoverable and a
      // lost early-z is only slower.
      if (nearFade < 0.01) discard;

      #ifndef LUXAR_MESH_ALPHA_CUTOUT
      // Every non-cutout mode folds the fade into COVERAGE, exactly as the
      // point/line shaders do — which is also how 'max' picks it up in its RGB
      // premultiply below for free.
      a *= nearFade;
      #endif

      #ifdef LUXAR_MESH_ALPHA_CUTOUT
      // 'opaque' (the mesh default): a hard, ORDER-INDEPENDENT cutout. Smooth
      // partial transparency would be self-contradictory in a depth-writing mode
      // drawn without per-triangle sorting (§6.3), so authored alpha means masks
      // and holes here. Survivors are fully opaque and write depth normally.
      //
      // The cutout compares the UNFADED coverage: the fade is a distance effect,
      // not an authored mask, and letting it move the comparison would dissolve
      // the cutout's holes open as the camera approached. This mode emits alpha
      // 1.0, so there is no alpha left to fade — the fade ramps the SHADED RGB
      // instead. Note what that means: the surface DARKENS toward black over the
      // band rather than dissolving, and only the reject above removes it. Over a
      // black background the two look the same; over a lit one they do not. The
      // trade, and why the alternative is worse, is written up in README.md.
      if (a < uAlphaCutoff) discard;
      fragColor = vec4(shadedColor * nearFade, 1.0);
      #elif defined(LUXAR_MAX_RGB_CONTRIBUTION)
      // 'max': MaxEquation + OneFactor/OneFactor does NOT weight source RGB by
      // alpha at composite, so premultiply by coverage here — otherwise a
      // barely-covering fragment would win the max with its full bright RGB.
      fragColor = vec4(shadedColor * a, a);
      #else
      // 'additive' / 'luminous' / 'normal': the framebuffer applies \`a\` at
      // composite (SrcAlpha/One or SrcAlpha/OneMinusSrcAlpha).
      fragColor = vec4(shadedColor, a);
      #endif
    }
  `;

export const MESH_SOURCE: ShaderSource = {
  name: 'mesh',
  webgl: { vertex: MESH_VERTEX_SHADER, fragment: MESH_FRAGMENT_SHADER },
  // Default config — the stored-normal, opaque-cutout build. Consumers needing
  // USE_COLORMAP / the flat-normal variant / a different blend mode call
  // `meshWebGPUFactory(nodes, { ...flags })` directly.
  webgpu: (uniforms: Record<string, unknown>) => {
    const { meshWebGPUFactory, buildMeshTSLNodesFromUniforms } =
      requireTslMaterials().factories.mesh;
    return meshWebGPUFactory(
      buildMeshTSLNodesFromUniforms(uniforms as Record<string, import('three').IUniform>)
    );
  },
};
