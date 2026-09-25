/**
 * GLSL3 picking shader source for meshes + the `ShaderSource` record.
 *
 * The simplest of the four pick shaders, and the only one whose element id is a
 * **built-in** rather than an attribute. Three structural differences from its
 * siblings, all from spec §6.5:
 *
 * 1. **No `aSortedIndex` indirection.** Mesh has no per-triangle depth sort (§9),
 *    so there is no draw-slot → storage-slot mapping and no `luxarElementIdParts()`
 *    call. The element id is `gl_VertexID`, split by the same shared
 *    `luxarElementIdSplit()` the sorted-index helper uses, so both paths agree
 *    with `voteWinner`'s recombination by construction.
 * 2. **Vertex granularity, not element granularity.** The draw is indexed
 *    (`faces` is the index buffer), so under `drawElements` `gl_VertexID` is the
 *    ordinal of the vertex in the `vertices` array — a stable per-vertex id.
 *    `gl_VertexID / 3` would be meaningless (shared vertices break it, and WebGL2
 *    has no `gl_PrimitiveID`), and a FACE ordinal would be renumbered on every
 *    slice change, since §5.4 rewrites only the index buffer. A vertex ordinal is
 *    invariant across slices, which is why it is the chosen granularity — and it
 *    indexes the per-vertex label CSR (§3.2) directly.
 * 3. **No quad expansion, and only half the camera uniforms.** A mesh has no
 *    screen-space footprint to size, so this pair binds no `uResolution` and no
 *    focal length. It does bind `uIsOrtho` + `uNearCull` and the wrapper IS a
 *    `CameraAwareMaterial` — matching the visual mesh material, whose near fade
 *    the pick coverage has to reproduce or a fading surface would stay fully
 *    pickable (#1431).
 *
 * Both mode-dependent behaviours are **runtime uniforms, not defines**, so
 * `mesh-pick` stays a single codegen variant and a blending-mode switch from the
 * layers panel never recompiles the pick program:
 *
 * | Uniform | Set from | Effect |
 * |---|---|---|
 * | `uSurfaceDepth` | `isNormalMode(mode) \|\| isOpaqueMode(mode)` | real projected depth (front-most wins) vs brightness-as-depth (brightest wins) |
 * | `uAlphaCutout` | `isOpaqueMode(mode)` | apply the visual shader's identical `a < uAlphaCutoff` discard |
 * | `LUXAR_MESH_PICK_BASE_COLOR_TEX` | the node has a texture | sample the texture's ALPHA into coverage, so cutout holes are unpickable |
 *
 * ## The surface-depth VALUE matches across backends — despite how the snapshot reads
 *
 * Under `uSurfaceDepth == 1` this shader writes `gl_FragCoord.z`. Its TSL twin
 * writes three's `depth` node, whose `DEPTH` scope is camera-aware at build time:
 * a perspective camera expands to `viewZToPerspectiveDepth(positionView.z, near,
 * far)` — exactly the hyperbolic window-space depth `gl_FragCoord.z` is — and an
 * orthographic camera to `viewZToOrthographicDepth`, which is exactly the
 * orthographic `gl_FragCoord.z`. So the two backends write the SAME value per
 * fragment, and cross-node depth comparisons — including against the commutative
 * modes' `1 - brightness` fragments sharing this buffer — resolve identically.
 *
 * Recording this because `mesh-pick.fragment.glsl.txt` is easy to misread as a
 * divergence: the snapshot expands `depth` to the LINEAR
 * `(positionView.z + cameraNear) / (cameraNear - cameraFar)` form only because the
 * codegen harness renders with an ORTHOGRAPHIC camera (see
 * `tsl-harness/mesh.ts`), for which that linear form IS `gl_FragCoord.z` — it is
 * not the general perspective expansion. The same GLSL-`gl_FragCoord.z`-vs-TSL-
 * `depth` pairing already ships in the gsplat pick shaders, where it is
 * undocumented.
 *
 * Source-of-truth for GLSL3; the WebGPU counterpart lives in `./pick.tsl` and is
 * referenced through the `ShaderSource.webgpu` factory below.
 *
 * @module rendering/picking/mesh/shaders
 */

import type { ShaderSource } from '../../materials/_shared/shader-source';
import {
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_ELEMENT_ID_SPLIT,
  GLSL_NEAR_FADE_FUNCTIONS,
} from '../../materials/_shared/glsl-lib';
import { requireTslMaterials } from '../../tsl/slot';

/**
 * Picking vertex stage for meshes.
 *
 * `position` is three's own auto-declared attribute; `color` is ours. It is the
 * ONLY attribute this stage needs beyond position — the element id comes from the
 * `gl_VertexID` built-in, and `normal`/`aScalar` are shading inputs with no
 * bearing on which vertex was clicked.
 *
 * `color` is read as a `vec4` for the same reason the visual shader does: GL fills
 * a size-3 attribute's missing components with `(0, 0, 0, 1)`, so RGB data supplies
 * the opaque `w = 1.0` for free.
 *
 * The two id varyings are `flat` — see the provoking-vertex note on the fragment
 * stage. `vAlpha` is smoothly interpolated, exactly as in the visual shader, so the
 * cutout hole in the pick pass has the same shape as the one on screen.
 *
 * `vViewZ` carries the view-space depth the fragment stage's near fade needs. Just
 * the z, not the whole `vec3` the visual pair carries — the visual stage
 * differentiates its `vViewPos` for the flat-normal fallback, and the pick pass has
 * no shading to do.
 */
export const MESH_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_ELEMENT_ID_SPLIT}

    // Per-vertex colour; only .a is read here (the pick pass has no colour output).
    // Always bound — see createMeshDefaultColorAttribute. RGB input gives w = 1.0.
    in vec4 color;

    uniform float uNodeId;

    flat out highp float vNodeId;
    flat out highp vec2 vElementId;
    out mediump float vAlpha;
    #ifdef LUXAR_MESH_PICK_BASE_COLOR_TEX
    // \`uv\` is three's own auto-declared attribute, like \`position\` above.
    out mediump vec2 vUv;
    #endif
    // VIEW-space depth for the fragment stage's near fade. highp: it is compared
    // against a scene-relative uNearCull that can be ~1e-3 of the scene diagonal,
    // which mediump cannot resolve on a large scene.
    out highp float vViewZ;

    void main() {
      vNodeId = uNodeId;
      // Vertex ordinal, NOT a triangle ordinal and NOT a storage slot: mesh has
      // no ordering attribute to indirect through, and gl_VertexID under an
      // indexed draw is already the stable per-vertex id (§6.5).
      vElementId = luxarElementIdSplit(uint(gl_VertexID));
      // Sanitized identically to the visual shader: alpha is the whole coverage
      // term for a mesh, and a NaN would survive into the cutout comparison as a
      // fragment that never discards — pickable where the visual has a hole.
      vAlpha = sanitizeAlpha(color.a);
      #ifdef LUXAR_MESH_PICK_BASE_COLOR_TEX
      vUv = uv;
      #endif
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      vViewZ = mvPosition.z;
      gl_Position = projectionMatrix * mvPosition;
    }
  `;

/**
 * Picking fragment stage for meshes.
 *
 * Writes the shared RGBA32F pick encoding
 * `vec4(nodeId, elementId-low16, brightness, elementId-high16)`.
 *
 * ## Why `vElementId` is `flat`, and which corner it reports
 *
 * A `flat` varying is sourced from ONE corner of the triangle, and the two
 * backends do not default to the same one: OpenGL ES 3.0 fixes the provoking
 * vertex to the **last** vertex of the primitive, while the WGSL
 * `@interpolate(flat)` three emits defaults to **first**-vertex sampling. The
 * pick contract at vertex granularity is therefore "*a* corner vertex of the
 * front-most triangle under the cursor" — the cursor is over the face, not a
 * vertex, so every corner is an equally valid answer and no consumer may assume a
 * specific one. The GLSL path still aligns with WebGPU where the platform allows
 * (`WEBGL_provoking_vertex`, see `./material.ts`); where it does not, the
 * divergence stands as a documented §6.4 exception and the parity tests assert
 * *membership* in the expected face rather than one exact corner.
 *
 * Non-`flat` would be far worse than a corner ambiguity: `gl_VertexID` differs at
 * every corner of a shared-vertex indexed draw, so a linearly interpolated
 * `vElementId` arrives FRACTIONAL and the readback's `Math.round` resolves to
 * arbitrary unrelated vertices. (The point TSL pick escapes without `flat` only
 * because a single-instance quad's four corners carry the same id, making
 * interpolation the identity — an accident that does not survive to mesh.)
 *
 * ## The cutout must match the visual shader exactly
 *
 * Without the identical `a < uAlphaCutoff` discard, a hole the user can see
 * through would still rasterize here at true surface depth — becoming pickable
 * AND depth-occluding picks of the nodes visible through it.
 *
 * The near fade is here for exactly the same reason, and is evaluated PER FRAGMENT
 * to match the visual pair (a triangle spans depth). It folds into `brightness`,
 * the way the point and gsplat pick shaders fold theirs, so pick salience tracks
 * visible salience; and a fragment below the 0.01 reject is discarded, so it writes
 * neither the id nor depth.
 */
export const MESH_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_NEAR_FADE_FUNCTIONS}

    flat in highp float vNodeId;
    flat in highp vec2 vElementId;
    in mediump float vAlpha;
    in highp float vViewZ;
    #ifdef LUXAR_MESH_PICK_BASE_COLOR_TEX
    // The pick pass samples the texture for its ALPHA ONLY — it has no colour
    // output. Not an optimisation that could be skipped: texture alpha multiplies
    // coverage in the visual shader, so an RGBA basemap's cutout holes exist on
    // screen. A pick pass that ignored the texture would leave those holes
    // pickable AND depth-occluding, which is exactly the visual/pick divergence
    // \`syncMeshPickAppearance\` exists to prevent.
    //
    // A define rather than the runtime uniform \`uAlphaCutout\` uses, because a
    // sampler has to be DECLARED and \`has_texture\` is a per-node constant: unlike
    // the blending mode, no layers-panel action can turn a texture on for a node
    // whose store had none.
    uniform sampler2D uBaseColorTex;
    in mediump vec2 vUv;
    #endif

    uniform mediump float uOpacity;
    uniform mediump float uAlphaCutoff;
    // Near-fade inputs, mirroring the visual material's pair: 0 = perspective,
    // 1 = orthographic (fade is the identity there), and the scene-relative fade
    // start in world units.
    uniform int uIsOrtho;
    uniform float uNearCull;
    // 1 = 'opaque': apply the visual shader's hard cutout. Runtime uniform, not a
    // define — a layers-panel mode switch must not recompile the pick program.
    uniform int uAlphaCutout;
    // 1 = the depth-ordered surface modes ('opaque'/'normal'): write real
    // projected depth (front-most wins). 0 = the commutative modes
    // (additive/luminous/max): brightness-as-depth (brightest wins), because real
    // surface depth there would let a dim mesh in front depth-occlude a brighter
    // node behind it, contradicting the brightest-wins vote.
    uniform int uSurfaceDepth;

    out vec4 fragColor;

    void main() {
      // Identical coverage to the visual shader (§6.2): a mesh has no per-element
      // intensity/amplitude, so coverage is the per-vertex alpha times node
      // opacity — NOT \`intensity * uOpacity\` like the emissive types.
      mediump float a = vAlpha * uOpacity;
      #ifdef LUXAR_MESH_PICK_BASE_COLOR_TEX
      // Alpha only, and NO luminance swizzle: a 1-channel texture samples alpha
      // 1.0, so the swizzle the visual shader needs for \`.rgb\` has no analogue
      // here. Multiplied in BEFORE the cutout comparison below, matching the
      // visual shader's ordering exactly — comparing a different quantity is how
      // the two passes would disagree about where the holes are.
      a *= texture(uBaseColorTex, vUv).a;
      #endif

      // Same fade, same 1e-20 degenerate-smoothstep floor and same 0.01 reject as
      // the visual shader — pick coverage must keep matching visible coverage as
      // the camera flies into the surface. Rejected before anything is written, so
      // a faded-out fragment contributes neither an id nor depth.
      // Ortho test from three's per-draw 'isOrthographic' (the camera being
      // drawn with), not a CPU-pushed flag; the fragment stage has no
      // projectionMatrix to read it from.
      float nearFade = perspectiveNearFade(isOrthographic ? 1 : 0, vViewZ, max(uNearCull, 1e-20));
      if (nearFade < 0.01) discard;

      if (uAlphaCutout == 1) {
        if (a < uAlphaCutoff) discard;
        // Survivors of the cutout are FULLY OPAQUE on screen — the visual shader
        // emits \`vec4(rgb, 1.0)\` for them — so the pick brightness must be 1.0
        // too. Carrying the pre-cutout \`a\` through instead would under-weight a
        // solid mesh in the cross-node brightness vote purely because its author
        // wrote 0.6 into a channel the visual output ignores.
        a = 1.0;
      }

      // The fade folds in HERE rather than into \`a\` above, so it reaches both
      // arms with one multiply: the cutout arm has already replaced \`a\` with the
      // constant 1.0 (its visual twin ramps the shaded RGB by the same factor),
      // and the commutative arm carries the coverage the visual twin faded.
      // Folding earlier would instead move the cutout comparison and dissolve the
      // holes open as the camera approached.
      mediump float brightness = clamp(a * nearFade, 0.0, 1.0);

      fragColor = vec4(vNodeId, vElementId.x, brightness, vElementId.y);
      // Pick depth convention, synced from the MAIN material's blending mode by
      // PickingSystem.renderPickBuffer(). Writing gl_FragDepth at all forfeits
      // early-z; the pick pass is half-resolution and the sibling gsplat pick
      // makes the same trade.
      gl_FragDepth = (uSurfaceDepth == 1) ? gl_FragCoord.z : 1.0 - brightness;
    }
  `;

export const MESH_PICK_SOURCE: ShaderSource = {
  name: 'mesh-pick',
  webgl: { vertex: MESH_PICK_VERTEX_SHADER, fragment: MESH_PICK_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    const { meshPickWebGPUFactory, buildMeshPickTSLNodesFromUniforms } =
      requireTslMaterials().factories.pickMesh;
    return meshPickWebGPUFactory(buildMeshPickTSLNodesFromUniforms(u));
  },
};
