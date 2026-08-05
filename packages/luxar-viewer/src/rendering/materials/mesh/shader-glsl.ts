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

import { GLSL_SANITIZE_FUNCTIONS } from '../_shared/glsl-lib';
import { MESH_NORMAL_EPS_SQ } from './appearance';
import type { ShaderSource } from '../_shared/shader-source';
import { meshWebGPUFactory, buildMeshTSLNodesFromUniforms } from './shader-tsl';

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

    #ifndef LUXAR_MESH_FLAT_NORMAL
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
      #ifdef USE_COLORMAP
      // The display-range window and gamma shape the scalar VALUE before the LUT
      // lookup, not the resulting colour — same rule as the point/line shaders.
      // Intensity/offset apply POST-LUT in the fragment stage.
      float t = clamp((aScalar - uScalarMin) * uScalarScale, 0.0, 1.0);
      #ifndef LUXAR_GAMMA_ONE
      t = pow(t, uInvGamma);
      #endif
      vColor = texture(uColormapTex, vec2(t, 0.5)).rgb;
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
      #ifndef LUXAR_MESH_FLAT_NORMAL
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
 *    exposes the interior back faces of a sliced closed isosurface: without the flip
 *    a back fragment has `dot(N, V) < 0`, the wrap term lands in `[0, 0.5)`, and the
 *    back side shades with a dimmed inverted gradient collapsing toward `uAmbient`.
 *    The derivative normal needs no flip — `cross(dFdx, dFdy)` is defined by the
 *    rasterized fragment, not by the winding, so it always faces the viewer — and
 *    flipping it would *reintroduce* exactly that inverted shade, worst precisely at
 *    the degenerate vertices the guard is there to rescue. So the exemption is per
 *    FRAGMENT, not per variant.
 */
export const MESH_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    uniform mediump float uOpacity;
    uniform mediump float uInvGamma;   // Pre-computed 1/gamma
    uniform mediump float uIntensity;  // Per-node linear colour multiplier (gain)
    uniform mediump float uOffset;     // Per-node additive brightness shift
    // Shade term (§6.2): a view-anchored headlight, no scene light.
    uniform mediump float uAmbient;        // shade floor at the silhouette
    uniform mediump float uShadeExponent;  // wrap-term contrast
    // Cutout threshold, read only under LUXAR_MESH_ALPHA_CUTOUT.
    uniform mediump float uAlphaCutoff;

    in mediump vec3 vColor;
    in mediump float vAlpha;
    in highp vec3 vViewPos;
    #ifndef LUXAR_MESH_FLAT_NORMAL
    in highp vec3 vNormal;
    #endif

    out vec4 fragColor;

    void main() {
      // (1) Derivative normal, UNCONDITIONALLY — see the module doc.
      highp vec3 derivativeNormal = normalize(cross(dFdx(vViewPos), dFdy(vViewPos)));
      // Forced viewer-facing rather than assumed so. \`cross(dFdx, dFdy)\` carries the
      // sign of the fragment-space y axis, and the two backends disagree about it:
      // GLSL's \`dFdy\` is with respect to a BOTTOM-UP window coordinate while WGSL's
      // \`dpdy\` is TOP-DOWN, so the identical surface yields +z on one and -z on the
      // other. Left alone, the flat variant would shade correctly on WebGL and
      // collapse to \`uAmbient\` everywhere on WebGPU — a §6.4 matching-output
      // violation that no amount of GLSL-side testing can see. Since the headlight's
      // V is the fixed view axis (0, 0, 1), "faces the viewer" is exactly
      // \`z >= 0\`, so one sign flip makes the fallback convention-independent (and
      // winding-independent, which is what §6.2 asserts about it).
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

      // View-anchored headlight: V is the fixed view-space axis (0, 0, 1) — a
      // camera headlight — so dot(N, V) reduces to N.z. The wrap term
      // (N.V * 0.5 + 0.5) keeps the silhouette readable instead of black.
      mediump float wrap = clamp(N.z * 0.5 + 0.5, 0.0, 1.0);
      mediump float shade = mix(uAmbient, 1.0, pow(wrap, uShadeExponent));

      // Per-node GOG. Identical chain to the sibling shaders (§6.2 notes it is
      // copied rather than shared — _shared/ carries the sanitizers and the
      // define helpers, but each of the six shader files writes this tail out).
      #ifdef LUXAR_NO_GOG
      mediump vec3 adjusted = vColor;
      #else
      mediump vec3 adjusted = max(vColor * uIntensity + uOffset, vec3(0.0));
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
      mediump vec3 shadedColor = finalColor * shade;

      // Mesh has no per-element intensity/amplitude/falloff scalar (§2.2) — it is
      // a solid surface — so coverage is just the per-vertex alpha times node
      // opacity. NOT \`intensity * uOpacity\` like the emissive types.
      mediump float a = vAlpha * uOpacity;

      #ifdef LUXAR_MESH_ALPHA_CUTOUT
      // 'opaque' (the mesh default): a hard, ORDER-INDEPENDENT cutout. Smooth
      // partial transparency would be self-contradictory in a depth-writing mode
      // drawn without per-triangle sorting (§6.3), so authored alpha means masks
      // and holes here. Survivors are fully opaque and write depth normally.
      if (a < uAlphaCutoff) discard;
      fragColor = vec4(shadedColor, 1.0);
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
  webgpu: (uniforms: Record<string, unknown>) =>
    meshWebGPUFactory(
      buildMeshTSLNodesFromUniforms(uniforms as Record<string, import('three').IUniform>)
    ),
};
