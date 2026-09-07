/**
 * `uGlassPartition` / `uGlassDepth` — the per-fragment depth partition of the emissive
 * data around `refract_data` glass (spec MESH_PHYSICAL_MATERIALS §3.4, Phase 3).
 *
 * A glass mesh that refracts the data must draw AFTER it, and the emissive layers write
 * no depth (additive mode has the depth test off altogether), so nothing in the
 * hardware can tell a point in front of the glass from one behind it: the glass would
 * paint over both. The refraction split therefore renders the glass's front-face depth
 * first (`post-processing/post-processing-manager/refraction-split.ts`) and every data
 * fragment classifies ITSELF against that texture:
 *
 * ```
 * inFront = glassDepth < 1.0 && fragmentDepth < glassDepth
 * ```
 *
 * where `1.0` is the cleared depth, i.e. "no refracting glass covers this pixel". Pass A
 * keeps the fragments that are NOT in front (behind the glass, or under no glass at all)
 * and the glass refracts exactly that image; pass C, after the glass, keeps only the
 * fragments in front, so they land crisp on top. Both predicates come from the one
 * `inFront` test, which is what makes the two passes a partition: every data fragment is
 * drawn exactly once. The mode is a runtime uniform, never a define — it flips several
 * times per frame and must not recompile anything.
 *
 * Every Luxar data material (points / quad lines / capsule lines / gsplats / house mesh,
 * GLSL and TSL) declares the pair. The pick materials deliberately do NOT: picking must
 * see all the data, and it runs while the mode is 0 anyway. Duck-typed on
 * `material.uniforms` like `density-drop.ts`, so one helper serves `ShaderMaterial`
 * uniform records and the TSL materials' `proxyIUniform` records alike.
 *
 * The depth texture is ONE module-level object: every material binds it once at
 * construction, and the split's glass-depth render target is created WITH it, so no
 * material ever needs a rebind (a TSL graph rebuild would otherwise be required) and a
 * frame that never runs the split samples a texture that is simply never read (the
 * uniform branch is dead at mode 0).
 *
 * @module rendering/materials/_shared/glass-partition
 */

import * as THREE from 'three';

/** Uniform name of the partition mode (`int` in GLSL, `float` in TSL). */
export const GLASS_PARTITION_UNIFORM = 'uGlassPartition';
/** Uniform name of the refracting-glass front-face depth sampler. */
export const GLASS_DEPTH_UNIFORM = 'uGlassDepth';

/**
 * The partition mode: 0 draws everything (every frame outside the split), 1 keeps the
 * fragments behind the glass or under no glass (pass A), 2 keeps the fragments in front
 * of the glass (pass C).
 */
export type GlassPartition = 0 | 1 | 2;

/** Mode 0: no partition (the state every frame outside the split sees). */
export const GLASS_PARTITION_OFF: GlassPartition = 0;
/** Mode 1: keep fragments behind the glass or under no glass at all (pass A). */
export const GLASS_PARTITION_BEHIND: GlassPartition = 1;
/** Mode 2: keep fragments in front of the glass (pass C). */
export const GLASS_PARTITION_FRONT: GlassPartition = 2;

interface GlassPartitionUniforms {
  uniforms?: Record<string, { value: unknown } | undefined>;
}

let glassDepthTexture: THREE.DepthTexture | null = null;

/**
 * The one depth texture every data material samples and the refraction split renders
 * the glass front faces into. Created lazily so importing a shader module allocates
 * nothing; a 1×1 placeholder until the split's render target adopts and resizes it.
 */
export function getGlassDepthTexture(): THREE.DepthTexture {
  if (!glassDepthTexture) {
    glassDepthTexture = new THREE.DepthTexture(1, 1);
    glassDepthTexture.name = 'Luxar.glassDepth';
    // Exact texel reads: the data shaders compare their own window depth against the
    // glass depth at the same pixel, so any filtering would blur the front/behind edge.
    glassDepthTexture.minFilter = THREE.NearestFilter;
    glassDepthTexture.magFilter = THREE.NearestFilter;
  }
  return glassDepthTexture;
}

/** Test-only: forget the shared depth texture so a suite starts from a fresh one. */
export function resetGlassDepthTextureForTests(): void {
  glassDepthTexture = null;
}

/** Whether `material` carries the partition uniform. */
export function hasGlassPartition(material: unknown): boolean {
  return Boolean(
    (material as GlassPartitionUniforms | null | undefined)?.uniforms?.[GLASS_PARTITION_UNIFORM]
  );
}

/** The partition mode on `material`, or 0 when it has no such uniform. */
export function getGlassPartition(material: unknown): GlassPartition {
  const u = (material as GlassPartitionUniforms | null | undefined)?.uniforms?.[
    GLASS_PARTITION_UNIFORM
  ];
  const v = u?.value;
  return v === 1 || v === 2 ? v : 0;
}

/**
 * Set the partition mode on `material`. Returns true when the material carries the
 * uniform and the value changed.
 */
export function setGlassPartition(material: unknown, mode: GlassPartition): boolean {
  const u = (material as GlassPartitionUniforms | null | undefined)?.uniforms?.[
    GLASS_PARTITION_UNIFORM
  ];
  if (!u) return false;
  if (u.value === mode) return false;
  u.value = mode;
  return true;
}

/**
 * The GLSL uniform pair, for the fragment stage of every WebGL data shader. Emitted as a
 * string so the shader sources stay plain template literals.
 */
export const GLSL_GLASS_PARTITION_UNIFORMS = /* glsl */ `
    // Refraction split (glass-partition.ts): 0 off, 1 keep behind-or-none, 2 keep front.
    uniform int uGlassPartition;
    // Refracting-glass front-face window depth; the cleared 1.0 means "no glass here".
    uniform sampler2D uGlassDepth;
`;

/**
 * The GLSL classification, to be the FIRST statement of `main()` in every WebGL data
 * fragment shader. `gl_FragCoord.z` is the fragment's own window depth (no visual shader
 * writes `gl_FragDepth`) and the depth target is the HDR target's size, so the integer
 * `texelFetch` at the fragment's own pixel is exact and needs no resolution uniform.
 */
export const GLSL_GLASS_PARTITION_GUARD = /* glsl */ `
      if (uGlassPartition != 0) {
        float glassDepth = texelFetch(uGlassDepth, ivec2(gl_FragCoord.xy), 0).r;
        bool inFrontOfGlass = glassDepth < 1.0 && gl_FragCoord.z < glassDepth;
        if (uGlassPartition == 1 && inFrontOfGlass) discard;
        if (uGlassPartition == 2 && !inFrontOfGlass) discard;
      }
`;
