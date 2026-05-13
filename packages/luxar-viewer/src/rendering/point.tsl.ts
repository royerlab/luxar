/**
 * Point material TSL factory — NodeMaterial counterpart to the
 * GLSL3 pair in `shaders/point-shaders.ts`.
 *
 * Renders one soft Gaussian-falloff sprite per point with:
 *   - world-space sizing (perspective + orthographic)
 *   - sharpness compensation (visibility-threshold normalisation)
 *   - per-node Gain/Offset/Gamma colour adjustment
 *   - zero-radius nD-slicing discard
 *
 * The mesh layout matches the post-migration container: 4-vertex
 * unit-quad base + InstancedBufferAttribute per-instance data
 * (aCenter, aRadius, aSharpness, aColor, aScalar). The vertex stage
 * projects aCenter to clip space and expands the unit quad by the
 * per-instance pointSize; the fragment stage discards outside the
 * inscribed circle and computes Gaussian falloff.
 *
 * Feature toggles map to {@link PointTSLConfig}, mirroring the GLSL3
 * `#define` semantics where flipping a flag triggers a recompile:
 *   - `useColormap` → reads `aScalar` + samples `uColormapTex`
 *     instead of `aColor`.
 *   - `useMaxRGBContribution` → premultiplies output RGB by alpha
 *     for the CustomBlending + MaxEquation rendering mode.
 *
 * @module rendering/point.tsl
 */

import * as THREE from 'three';
import {
  Fn,
  uniform,
  attribute,
  varying,
  vec2,
  vec3,
  vec4,
  float,
  int,
  max,
  clamp,
  length,
  dot,
  pow,
  Discard,
  texture,
  modelViewMatrix,
  cameraProjectionMatrix,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';

/**
 * Loosely-typed TSL node alias. TSL's typed overloads return many
 * mutually-incompatible inner constructor types — relaxing at helper
 * boundaries lets the runtime TSL builder do the real type checking
 * when it compiles to GLSL/WGSL.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TSLNode = any;

export interface PointTSLConfig {
  readonly useColormap?: boolean;
  readonly useMaxRGBContribution?: boolean;
}

/** Sanitise a positive scalar. Mirrors GLSL `sanitizePositive`. */
function sanitizePositive(value: TSLNode, fallback: TSLNode): TSLNode {
  const isFinite = value.lessThan(1e30).and(value.greaterThan(-1e30));
  const isPositive = value.greaterThan(0.0);
  return isFinite.and(isPositive).select(value, fallback);
}

/** Sanitise a non-negative scalar. Mirrors GLSL `sanitizeNonNegative`. */
function sanitizeNonNegative(value: TSLNode, fallback: TSLNode): TSLNode {
  const isFinite = value.lessThan(1e30).and(value.greaterThan(-1e30));
  const isNonNeg = value.greaterThanEqual(0.0);
  return isFinite.and(isNonNeg).select(value, fallback);
}

/**
 * Point-material TSL factory.
 *
 * The `uniforms` table must include the full set the GLSL3 shader
 * reads. Missing fields are tolerated only on optional branches
 * (colormap uniforms when `useColormap === false`).
 */
export function pointWebGPUFactory(
  uniforms: Record<string, THREE.IUniform>,
  config: PointTSLConfig = {}
): NodeMaterial {
  // Per-vertex (4 corners, ±1).
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  // Per-instance.
  const aCenter: TSLNode = attribute<'vec3'>('aCenter', 'vec3');
  const aRadius: TSLNode = attribute<'float'>('aRadius', 'float');
  const aSharpness: TSLNode = attribute<'float'>('aSharpness', 'float');
  const aColor: TSLNode = attribute<'vec3'>('aColor', 'vec3');
  const aScalar: TSLNode = config.useColormap
    ? attribute<'float'>('aScalar', 'float')
    : null;

  // Uniforms — vertex stage.
  const uPointSizeFactor = uniform((uniforms.pointSizeFactor.value as number) ?? 1.0);
  const uMaxPointSize = uniform((uniforms.maxPointSize.value as number) ?? 1.0);
  const uRadiusScale = uniform((uniforms.radiusScale.value as number) ?? 1.0);
  const uSharpnessScale = uniform((uniforms.sharpnessScale.value as number) ?? 1.0);
  const uIsOrtho = uniform((uniforms.uIsOrtho.value as number) ?? 0);
  const uResolution = uniform(
    (uniforms.uResolution.value as THREE.Vector2) ?? new THREE.Vector2(1, 1)
  );

  // Colormap (optional).
  const uColormapTex =
    config.useColormap && uniforms.uColormapTex
      ? texture((uniforms.uColormapTex.value as THREE.Texture | null) ?? new THREE.Texture())
      : null;
  const uScalarMin =
    config.useColormap && uniforms.uScalarMin
      ? uniform((uniforms.uScalarMin.value as number) ?? 0.0)
      : null;
  const uScalarScale =
    config.useColormap && uniforms.uScalarScale
      ? uniform((uniforms.uScalarScale.value as number) ?? 1.0)
      : null;

  // Uniforms — fragment stage.
  const uOpacity = uniform((uniforms.opacity.value as number) ?? 1.0);
  const uInvGamma = uniform((uniforms.invGamma.value as number) ?? 1.0);
  const uIntensity = uniform((uniforms.uIntensity.value as number) ?? 1.0);
  const uOffset = uniform((uniforms.uOffset.value as number) ?? 0.0);

  // ---- Vertex computation ----

  // Sanitise per-instance attributes (NaN/Inf-safe).
  const normalizedSharpness: TSLNode = sanitizePositive(
    aSharpness.mul(uSharpnessScale),
    float(2.0)
  );
  const normalizedRadius: TSLNode = sanitizeNonNegative(
    aRadius.mul(uRadiusScale),
    float(0.0)
  );

  // Per-instance colour from LUT or attribute.
  let perPointColor: TSLNode;
  if (config.useColormap && aScalar && uColormapTex && uScalarMin && uScalarScale) {
    const t = clamp(aScalar.sub(uScalarMin).mul(uScalarScale), 0.0, 1.0);
    perPointColor = uColormapTex.sample(vec2(t, 0.5)).rgb;
  } else {
    perPointColor = aColor;
  }

  // Project per-instance centre to view + clip space.
  const mvPos: TSLNode = modelViewMatrix.mul(vec4(aCenter, 1.0));
  const projCenter: TSLNode = cameraProjectionMatrix.mul(mvPos);

  // World-space size: perspective gets 1/length(view-z) attenuation;
  // ortho stays constant.
  const invDistance: TSLNode = int(uIsOrtho)
    .equal(int(1))
    .select(float(1.0), length(vec3(mvPos)).reciprocal());
  const basePointSize: TSLNode = normalizedRadius.mul(uPointSizeFactor).mul(invDistance);

  // Sharpness compensation: keep visible disk extent constant.
  const sharpnessCompRaw: TSLNode = float(1.0).div(
    float(1.0).sub(pow(float(0.01), float(1.0).div(max(normalizedSharpness, float(0.01)))))
  );
  const sharpnessCompFinite = sharpnessCompRaw
    .lessThan(1e30)
    .and(sharpnessCompRaw.greaterThan(-1e30));
  const sharpnessComp: TSLNode = sharpnessCompFinite.select(sharpnessCompRaw, float(1.0));
  const computedPointSize: TSLNode = basePointSize.mul(sharpnessComp);
  const pointSize: TSLNode = max(
    float(1.0),
    clamp(computedPointSize, float(1.0), uMaxPointSize)
  );

  // Expand the unit quad to a sprite in clip space.
  const offsetClip: TSLNode = aQuadCorner
    .mul(pointSize.div(uResolution))
    .mul(projCenter.w);
  const clipPos: TSLNode = projCenter.add(vec4(offsetClip, 0.0, 0.0));

  // Sprite UV (replaces gl_PointCoord). Computed per-vertex,
  // interpolated to the fragment via the `varying()` wrapper —
  // matches `vSpriteCoord = (aQuadCorner + 1.0) * 0.5` from GLSL.
  const vSpriteCoord: TSLNode = varying(aQuadCorner.add(1.0).mul(0.5));
  // Per-instance vRadius and vSharpness are constant within a quad
  // (4 verts share the same instance) so `varying()` interpolation
  // is a no-op but the wrapper is what gets TSL to pass them to the
  // fragment stage.
  const vRadius: TSLNode = varying(normalizedRadius);
  const vSharpness: TSLNode = varying(normalizedSharpness);
  const vColor: TSLNode = varying(perPointColor);

  // ---- Fragment computation ----

  const colorNode = Fn(() => {
    // Zero-radius nD-slicing discard.
    Discard(vRadius.lessThan(0.0001));

    const centered: TSLNode = vec2(vSpriteCoord.sub(0.5));
    const r2: TSLNode = dot(centered, centered);
    Discard(r2.greaterThan(0.25));

    const normalizedR: TSLNode = r2.mul(4.0).sqrt();
    const falloff: TSLNode = float(1.0).sub(normalizedR).max(float(0.0)).pow(vSharpness);

    // GOG: colour × intensity + offset, clamped, then gamma.
    const adjusted: TSLNode = max(vColor.mul(uIntensity).add(uOffset), vec3(0.0));
    Discard(max(adjusted.r, max(adjusted.g, adjusted.b)).lessThan(1e-4));
    const finalColor: TSLNode = adjusted.pow(vec3(uInvGamma));

    const alpha: TSLNode = falloff.mul(uOpacity);

    if (config.useMaxRGBContribution) {
      // RGB premultiplied by alpha — CustomBlending + MaxEquation.
      return vec4(finalColor.mul(alpha), alpha);
    }
    return vec4(finalColor, alpha);
  });

  const material = new NodeMaterial();
  // Override the vertex output entirely — we project + expand the
  // sprite ourselves. NodeMaterial.vertexNode replaces the default
  // modelViewProjection chain.
  material.vertexNode = clipPos;
  material.colorNode = colorNode();
  material.transparent = true;
  material.toneMapped = false;
  material.depthTest = true;
  material.depthWrite = false;
  return material;
}
