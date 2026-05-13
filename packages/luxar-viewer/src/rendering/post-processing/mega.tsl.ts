/**
 * Mega-shader TSL factory — NodeMaterial counterpart to the GLSL3
 * fragment in `mega-shader.glsl.ts`.
 *
 * The mega-shader fuses post-processing into a single pass. The TSL
 * port mirrors the GLSL operation order so the two backends produce
 * pixel-identical output for any given configuration:
 *
 *   1. Sample HDR scene (+ optional bloom)
 *   2. Lens distortion (chromatic + radial), per-channel sampling
 *   3. Detector noise   ← NOT YET PORTED — see "Deferred" below.
 *   4. EOG (Exposure → Offset → Gamma)
 *   5. Tone mapping (mode-switched on `config.toneMappingMode`)
 *   6. Vignette
 *   7. sRGB encoding (final write)
 *
 * Feature toggles live in {@link MegaTSLConfig} as JS-side flags
 * rather than uniform-bool nodes. This matches the existing GLSL3
 * semantics where flipping `USE_BLOOM` (etc.) triggers a recompile
 * via the `#define`-driven shader cache. The consumer
 * (`MegaShaderMaterial`) treats both backends symmetrically:
 * recompile via `defines` for WebGL, re-call the factory for WebGPU.
 *
 * Tone mapping uses the dedicated functions from `three/tsl`
 * (`linearToneMapping`, `reinhardToneMapping`, etc.) rather than the
 * GLSL `<tonemapping_pars_fragment>` include. The mapping from
 * Luxar's internal IDs (1..6) to TSL functions is in
 * {@link applyToneMapping}.
 *
 * ## Deferred — detector noise
 *
 * The GLSL detector-noise path uses a Bob Jenkins hash + Anscombe
 * variance-stabilising transform to approximate Poisson shot noise.
 * Porting this faithfully needs `floatBitsToUint` + bitwise ops in
 * TSL — those primitives exist (`bitcast`, `shiftLeft`, `bitXor`)
 * but the port is non-mechanical because the hash is a long chain of
 * uint-only ops. Until that lands the TSL factory throws if
 * `config.useDetectorNoise` is true, so callers see a loud failure
 * rather than a silent missing effect. Tracked in
 * `MIGRATION_PROGRESS.md` (M9-bis).
 *
 * @module rendering/post-processing/mega.tsl
 */

import * as THREE from 'three';
import {
  Fn,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
  texture,
  mat3,
  float,
  dot,
  max,
  exp2,
  smoothstep,
  mix,
  step,
  linearToneMapping,
  reinhardToneMapping,
  cineonToneMapping,
  acesFilmicToneMapping,
  agxToneMapping,
  neutralToneMapping,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';

/**
 * Loosely-typed TSL node, used for internal helper signatures.
 *
 * TSL's `vec2()` / `vec3()` overloads return many mutually-incompatible
 * inner constructor types — `JoinNode`, `ConvertNode`, `ConstNode`,
 * `VarNode` — that can't be unified in a parameter annotation without
 * pinning one overload and rejecting the others. The runtime TSL
 * builder still checks types when it compiles to GLSL/WGSL, so the
 * lost JS-side strictness doesn't cost real safety.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TSLNode = any;

/**
 * Luxar-internal tone-mapping IDs. Match the
 * `LUXAR_TONE_MAPPING_MODE` define values in `mega-shader.glsl.ts`.
 * `NoToneMapping` aliases to `Linear` upstream so callers never need
 * to set 0.
 */
export type LuxarToneMappingMode = 1 | 2 | 3 | 4 | 5 | 6;

/**
 * Compile-time configuration for {@link megaWebGPUFactory}. Mirrors
 * the GLSL `#define` set; toggling any field triggers a re-call of
 * the factory (which builds a new NodeMaterial).
 */
export interface MegaTSLConfig {
  readonly useLensDistortion?: boolean;
  readonly useBloom?: boolean;
  /**
   * Detector noise is not yet ported (see module header). Setting
   * this to `true` throws.
   */
  readonly useDetectorNoise?: boolean;
  readonly useVignette?: boolean;
  readonly toneMappingMode?: LuxarToneMappingMode;
  /** Output the linear-HDR sample (post-bloom) and skip everything else. */
  readonly captureRawHDR?: boolean;
  /** Skip ONLY the final sRGB encoding step. */
  readonly captureLinearLDR?: boolean;
}

/**
 * Tone-mapping dispatch. Picks the matching TSL function for the
 * Luxar-internal mode ID. Exposure is pinned to `1.0` because the
 * EOG step has already pre-multiplied — matches the GLSL behaviour
 * where `toneMappingExposure` is also pinned to 1.0.
 *
 * The TSL toneMapping functions return an untyped `Node`; we coerce
 * back to `Node<'vec3'>` via `vec3(...)` at the call site so the
 * downstream `.mul`/`.add` chain keeps its type info.
 */
function applyToneMapping(color: TSLNode, mode: LuxarToneMappingMode): TSLNode {
  const exposureOne = float(1.0);
  switch (mode) {
    case 1:
      return linearToneMapping(color, exposureOne);
    case 2:
      return reinhardToneMapping(color, exposureOne);
    case 3:
      return cineonToneMapping(color, exposureOne);
    case 4:
      return acesFilmicToneMapping(color, exposureOne);
    case 5:
      return agxToneMapping(color, exposureOne);
    case 6:
      return neutralToneMapping(color, exposureOne);
  }
}

/**
 * sRGB OETF (linear → sRGB). Inlined here because TSL's
 * `output.toColorSpace(SRGBColorSpace)` is gated on renderer-output
 * paths we explicitly bypass with `material.toneMapped = false`.
 *
 * Matches the GLSL `linearToSRGB(vec3)` helper bit-for-bit:
 *  - safe = max(c, 0)
 *  - mix(1.055 · safe^(1/2.4) - 0.055,  safe · 12.92,  step(safe, 0.0031308))
 */
function linearToSRGB(c: TSLNode): TSLNode {
  // Pre-cast every vec3 const to TSLNode at the call site. TSL's
  // `step` / `pow` exports only declare scalar overloads in the
  // current d.ts, even though the runtime accepts vec3 args.
  const ZERO_VEC3: TSLNode = vec3(0.0);
  const THRESHOLD_VEC3: TSLNode = vec3(0.0031308);
  const INV_GAMMA_VEC3: TSLNode = vec3(1.0 / 2.4);

  const safe: TSLNode = max(c, ZERO_VEC3);
  // GLSL: lessThanEqual(safe, 0.0031308) returns bvec3 → as vec3 in mix.
  // Equivalent: step(safe, threshold) → 1 if safe<=threshold else 0.
  const isLinear: TSLNode = step(safe, THRESHOLD_VEC3);
  const gammaPart: TSLNode = safe.pow(INV_GAMMA_VEC3).mul(1.055).sub(0.055);
  const linearPart: TSLNode = safe.mul(12.92);
  return mix(gammaPart, linearPart, isLinear);
}

/**
 * Mega-shader TSL factory. Returns a `NodeMaterial` whose
 * `fragmentNode` evaluates the configured pipeline.
 *
 * The `uniforms` table must include the full set the GLSL3 shader
 * reads: `uHdrScene`, `uResolution`, `uExposure`, `uGlobalOffset`,
 * `uGlobalGamma`, plus the lens-distortion / bloom / vignette
 * uniforms when their respective `use*` flags are set in `config`.
 */
export function megaWebGPUFactory(
  uniforms: Record<string, THREE.IUniform>,
  config: MegaTSLConfig = {}
): NodeMaterial {
  if (config.useDetectorNoise) {
    throw new Error(
      'megaWebGPUFactory: useDetectorNoise=true is not yet supported (M9-bis). ' +
        'Use the GLSL3 path for detector-noise rendering until the Poisson + ' +
        'Bob Jenkins hash port lands.'
    );
  }

  const toneMappingMode: LuxarToneMappingMode = config.toneMappingMode ?? 6;

  // Common inputs
  const uHdrScene = texture(
    (uniforms.uHdrScene.value as THREE.Texture | null) ?? new THREE.Texture()
  );
  const uExposure = uniform((uniforms.uExposure.value as number) ?? 0.0);
  const uGlobalOffset = uniform((uniforms.uGlobalOffset.value as number) ?? 0.0);
  const uGlobalGamma = uniform((uniforms.uGlobalGamma.value as number) ?? 1.0);

  // Optional inputs — declared at factory time only when the feature
  // is enabled, so unused uniforms don't end up in the compiled
  // shader.
  const uBloomTexture =
    config.useBloom && uniforms.uBloomTexture
      ? texture((uniforms.uBloomTexture.value as THREE.Texture | null) ?? new THREE.Texture())
      : null;
  const uBloomIntensity =
    config.useBloom && uniforms.uBloomIntensity
      ? uniform((uniforms.uBloomIntensity.value as number) ?? 0.0)
      : null;

  const uDistortion =
    config.useLensDistortion && uniforms.uDistortion
      ? uniform((uniforms.uDistortion.value as THREE.Vector2) ?? new THREE.Vector2(0, 0))
      : null;
  const uPrincipalPoint =
    config.useLensDistortion && uniforms.uPrincipalPoint
      ? uniform((uniforms.uPrincipalPoint.value as THREE.Vector2) ?? new THREE.Vector2(0, 0))
      : null;
  const uFocalLength =
    config.useLensDistortion && uniforms.uFocalLength
      ? uniform((uniforms.uFocalLength.value as THREE.Vector2) ?? new THREE.Vector2(1, 1))
      : null;
  const uSkew =
    config.useLensDistortion && uniforms.uSkew
      ? uniform((uniforms.uSkew.value as number) ?? 0.0)
      : null;
  const uDispersion =
    config.useLensDistortion && uniforms.uDispersion
      ? uniform((uniforms.uDispersion.value as number) ?? 0.0)
      : null;

  const uVignetteDarkness =
    config.useVignette && uniforms.uVignetteDarkness
      ? uniform((uniforms.uVignetteDarkness.value as number) ?? 0.5)
      : null;
  const uVignetteOffset =
    config.useVignette && uniforms.uVignetteOffset
      ? uniform((uniforms.uVignetteOffset.value as number) ?? 0.5)
      : null;

  const fragmentNode = Fn(() => {
    // Use the geometry's `uv` attribute. screenUV is Y-flipped under
    // forceWebGL, which would sample the HDR texture upside-down
    // relative to the GLSL3 path (vUv = position.xy * 0.5 + 0.5).
    const coord = uv();

    // Sample (scene + bloom) at one UV — closure factored as a JS
    // helper rather than a TSL `Fn` so we can branch on
    // `config.useBloom` at factory time.
    //
    const sampleSceneAndBloom = (sampleUV: TSLNode) => {
      const base = uHdrScene.sample(sampleUV).rgb;
      if (uBloomTexture && uBloomIntensity) {
        return vec3(base.add(uBloomTexture.sample(sampleUV).rgb.mul(uBloomIntensity)));
      }
      return vec3(base);
    };

    // (1+2) Sample with optional chromatic lens distortion. The GLSL
    // path applies per-channel distortion + a border mask to suppress
    // out-of-bounds reads (clamped texels bleed garbage otherwise).
    //
    // We materialise `color` as a TSL `.toVar()` so subsequent stages
    // can `.assign()` new values without TS choking on the union of
    // overload return types each stage produces.
    const color = vec3(0.0, 0.0, 0.0).toVar();
    if (
      config.useLensDistortion &&
      uDistortion &&
      uPrincipalPoint &&
      uFocalLength &&
      uSkew &&
      uDispersion
    ) {
      const applyDistortion = (sampleUV: TSLNode, distortionCoeff: TSLNode) => {
        const xn: TSLNode = vec2(sampleUV.sub(0.5).mul(2.0));
        const r2 = dot(xn, xn);
        const xDistorted: TSLNode = vec3(xn.mul(distortionCoeff.mul(r2).add(1.0)), 1.0);
        // Camera intrinsic matrix. Three columns: focal x, skew + focal
        // y, principal point. mat3 in TSL is column-major like GLSL.
        const kk: TSLNode = mat3(
          vec3(uFocalLength.x, 0.0, 0.0),
          vec3(uSkew.mul(uFocalLength.x), uFocalLength.y, 0.0),
          vec3(uPrincipalPoint.x, uPrincipalPoint.y, 1.0)
        );
        // mat3 × vec3 returns vec3 in TSL but the typed `.mul()`
        // overload only sees the mat-typed result. Casting `kk` to
        // TSLNode lets `.xy` resolve.
        return vec2(kk.mul(xDistorted).xy.mul(0.5).add(0.5));
      };
      const distortionBorder = (sampleUV: TSLNode) => {
        // Bool mask: 1 inside [0,1]², 0 outside. Implemented as
        // step(0, uv) - step(1, uv) componentwise and reduced.
        const inX = step(0.0, sampleUV.x).sub(step(1.0, sampleUV.x));
        const inY = step(0.0, sampleUV.y).sub(step(1.0, sampleUV.y));
        return float(inX.mul(inY));
      };

      const dispersion = uDispersion;
      const distR = vec2(uDistortion.mul(float(1.0).sub(dispersion)));
      const distG = uDistortion;
      const distB = vec2(uDistortion.mul(float(1.0).add(dispersion)));
      const uvR = applyDistortion(coord, distR);
      const uvG = applyDistortion(coord, distG);
      const uvB = applyDistortion(coord, distB);
      const r = sampleSceneAndBloom(uvR).r.mul(distortionBorder(uvR));
      const g = sampleSceneAndBloom(uvG).g.mul(distortionBorder(uvG));
      const b = sampleSceneAndBloom(uvB).b.mul(distortionBorder(uvB));
      color.assign(vec3(r, g, b));
    } else {
      color.assign(sampleSceneAndBloom(coord));
    }

    // RAW_HDR capture mode: skip everything past sample+bloom. Used
    // by `captureHDRPixels('hdr-effects-pre-tone')` for EXR export.
    if (config.captureRawHDR) {
      return vec4(color, 1.0);
    }

    // (3) Detector noise — deferred; the factory throws upstream if
    //     config.useDetectorNoise is true.

    // (4) EOG: Exposure → Offset → Gamma. Each step routes through
    // TSLNode at the cast boundary because `color.toVar()` is typed
    // by TSL with a specific inner-constructor that `pow`'s vec
    // overload can't unify against.
    const invGamma: TSLNode = vec3(float(1.0).div(uGlobalGamma));
    color.assign(color.mul(exp2(uExposure)));
    color.assign(max(color.add(vec3(uGlobalOffset)), vec3(0.0)));
    color.assign((color as TSLNode).pow(invGamma));

    // (5) Tone mapping — see {@link applyToneMapping}.
    color.assign(applyToneMapping(color, toneMappingMode));

    // (6) Vignette: multiplicative darkening, preserves color ratios.
    if (config.useVignette && uVignetteDarkness && uVignetteOffset) {
      const vc = vec2(coord.sub(0.5).div(uVignetteOffset));
      const d2 = dot(vc, vc);
      const vf = float(1.0).sub(smoothstep(0.0, 1.5, d2).mul(uVignetteDarkness));
      color.assign(color.mul(vf));
    }

    // (7) sRGB encoding — skipped in the linear-LDR capture mode.
    if (!config.captureLinearLDR) {
      color.assign(linearToSRGB(color));
    }

    return vec4(color, 1.0);
  });

  const material = new NodeMaterial();
  material.fragmentNode = fragmentNode();
  // We bypass renderer-injected tone mapping; our shader does it
  // explicitly per `toneMappingMode`.
  material.toneMapped = false;
  material.depthTest = false;
  material.depthWrite = false;
  material.transparent = false;
  return material;
}
