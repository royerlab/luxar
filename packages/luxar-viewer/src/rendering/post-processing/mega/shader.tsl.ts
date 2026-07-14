/**
 * Mega-shader TSL factory — NodeMaterial counterpart to the GLSL3
 * fragment in `shader.glsl.ts`.
 *
 * The mega-shader fuses post-processing into a single pass. The TSL
 * port mirrors the GLSL operation order so the two backends produce
 * pixel-identical output for any given configuration:
 *
 *   1. Sample HDR scene (+ optional bloom)
 *   2. Lens distortion (chromatic + radial), per-channel sampling
 *   3. Detector noise (Bob Jenkins hash + Anscombe Poisson + Gaussian)
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
 * @module rendering/post-processing/mega/shader.tsl
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
  uint,
  floatBitsToUint,
  dot,
  max,
  exp2,
  log,
  sqrt,
  clamp,
  smoothstep,
  mix,
  step,
  mod,
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
 * `LUXAR_TONE_MAPPING_MODE` define values in `shader.glsl.ts`.
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
/**
 * Bob Jenkins integer hash. Mirrors the GLSL3 implementation
 * bit-for-bit; the shift / xor / add / hex-constant sequence is what
 * makes the cascade avalanche-stable.
 *
 * TSL doesn't expose direct `uint(...)` literal hex syntax inline, so
 * we wrap the constants via `uint(0x...)` calls. Operator method
 * dispatch on uint nodes goes through `IntegerExtensions<'uint'>`
 * which exposes `.add()`, `.shiftLeft()`, `.bitXor()`, `.shiftRight()`.
 */
function bobJenkinsHash(a: TSLNode): TSLNode {
  let h = a;
  // a = (a + 0x7ed55d16u) + (a << 12u)
  h = h.add(uint(0x7ed55d16)).add(h.shiftLeft(uint(12)));
  // a = (a ^ 0xc761c23cu) ^ (a >> 19u)
  h = h.bitXor(uint(0xc761c23c)).bitXor(h.shiftRight(uint(19)));
  // a = (a + 0x165667b1u) + (a << 5u)
  h = h.add(uint(0x165667b1)).add(h.shiftLeft(uint(5)));
  // a = (a + 0xd3a2646cu) ^ (a << 9u)
  h = h.add(uint(0xd3a2646c)).bitXor(h.shiftLeft(uint(9)));
  // a = (a + 0xfd7046c5u) + (a << 3u)
  h = h.add(uint(0xfd7046c5)).add(h.shiftLeft(uint(3)));
  // a = (a ^ 0xb55a4f09u) ^ (a >> 16u)
  h = h.bitXor(uint(0xb55a4f09)).bitXor(h.shiftRight(uint(16)));
  return h;
}

/** Mix two float seeds through bobJenkinsHash → single uint. */
function rngUint2(x: TSLNode, y: TSLNode): TSLNode {
  const a = bobJenkinsHash(floatBitsToUint(x));
  const b = bobJenkinsHash(floatBitsToUint(y));
  return bobJenkinsHash(a.bitXor(b));
}

/** Three-input variant of {@link rngUint2}. */
function rngUint3(x: TSLNode, y: TSLNode, z: TSLNode): TSLNode {
  const ab = rngUint2(x, y);
  const c = bobJenkinsHash(floatBitsToUint(z));
  return bobJenkinsHash(ab.bitXor(c));
}

/** Hash → uniform float in [0, 1). 2^32 = 4294967296. */
function rngFloat2(x: TSLNode, y: TSLNode): TSLNode {
  // Convert uint to float by reinterpreting + normalising. The `vec3`
  // overload through float() handles the uint→float conversion.
  return float(rngUint2(x, y)).div(4294967296.0);
}
function rngFloat3(x: TSLNode, y: TSLNode, z: TSLNode): TSLNode {
  return float(rngUint3(x, y, z)).div(4294967296.0);
}

/**
 * Clamped logistic ≈ Gaussian. The 0.5513 coefficient normalises to
 * roughly unit variance after the [-4, 4] clamp on the logit. Cheap
 * Gaussian approximation that avoids Box-Muller's two-RNG cost.
 */
function clampedLogistic(u: TSLNode): TSLNode {
  const f = clamp(u, 0.0001, 0.9999);
  const logit = log(f.div(float(1.0).sub(f)));
  return clamp(logit, -4.0, 4.0).mul(0.5513);
}

/** vec3 of clampedLogistic-driven RNG, seeded from a temporal vec3. */
function normal3Temporal(sx: TSLNode, sy: TSLNode, sz: TSLNode): TSLNode {
  return vec3(
    clampedLogistic(rngFloat3(sx, sy, sz)),
    clampedLogistic(rngFloat3(sx.add(13.37), sy.add(7.31), sz.add(19.93))),
    clampedLogistic(rngFloat3(sx.add(31.17), sy.add(41.23), sz.add(53.59)))
  );
}

/** vec3 of clampedLogistic-driven RNG, seeded from a fixed (no-time) vec2. */
function normal3Fixed(sx: TSLNode, sy: TSLNode): TSLNode {
  return vec3(
    clampedLogistic(rngFloat2(sx, sy)),
    clampedLogistic(rngFloat2(sx.add(13.37), sy.add(7.31))),
    clampedLogistic(rngFloat2(sx.add(31.17), sy.add(41.23)))
  );
}

/**
 * Anscombe variance-stabilising transform: maps Poisson(λ) to
 * approximately N(2√λ + ..., 1). Used to approximate Poisson shot
 * noise via a Gaussian step in transformed space.
 */
function anscombeForward(x: TSLNode): TSLNode {
  return sqrt(max(x.add(0.375), float(0.0))).mul(2.0);
}
function anscombeInverse(y: TSLNode): TSLNode {
  const yHalf: TSLNode = y.mul(0.5);
  return max(yHalf.mul(yHalf).sub(0.375), float(0.0));
}

/** Apply Poisson-distributed noise to a per-channel intensity vec3. */
function poissonNoise(sx: TSLNode, sy: TSLNode, sz: TSLNode, lambda: TSLNode): TSLNode {
  const yChan: TSLNode = vec3(
    anscombeForward(lambda.r),
    anscombeForward(lambda.g),
    anscombeForward(lambda.b)
  );
  const noisy: TSLNode = yChan.add(normal3Temporal(sx, sy, sz));
  return vec3(anscombeInverse(noisy.r), anscombeInverse(noisy.g), anscombeInverse(noisy.b));
}

/**
 * Apply the full detector-noise stack to a vec3 intensity. Mirrors
 * the GLSL `applyDetectorNoise` operation order:
 *
 *   shot noise → fade-zero-pixels → readout noise → fixed-pattern noise
 */
function applyDetectorNoise(
  intensity: TSLNode,
  coord: TSLNode,
  uTimeNode: TSLNode,
  uReadoutSigmaNode: TSLNode,
  uPhotonGainNode: TSLNode,
  uFpnSigmaNode: TSLNode
): TSLNode {
  // Wrap time to prevent precision collapse over long sessions.
  const wrappedTime = mod(uTimeNode, float(1000.0));
  const seedX: TSLNode = coord.x.mul(1000.0);
  const seedY: TSLNode = coord.y.mul(1000.0);

  // 1. Shot noise.
  const photonCount: TSLNode = intensity.div(max(uPhotonGainNode, float(0.0001)));
  const noisyPhotons: TSLNode = poissonNoise(seedX, seedY, wrappedTime, photonCount);
  const afterShotRaw: TSLNode = noisyPhotons.mul(uPhotonGainNode);

  // Anscombe forward-then-inverse adds a 3/8 bias that lifts pure-zero
  // pixels. Smoothstep gate fades shot-noise contribution out for
  // pixels near zero — visually identical to the GLSL helper.
  // The TSL `smoothstep` overload requires scalar edge args; we
  // compute the gate per-channel via three scalar calls and rejoin.
  const shotW: TSLNode = vec3(
    smoothstep(float(0.0), float(0.01), intensity.r),
    smoothstep(float(0.0), float(0.01), intensity.g),
    smoothstep(float(0.0), float(0.01), intensity.b)
  );
  const afterShot: TSLNode = mix(intensity, afterShotRaw, shotW);

  // 2. Readout noise — temporal Gaussian, intensity-independent.
  const readout: TSLNode = normal3Temporal(
    seedX.add(100.0),
    seedY.add(100.0),
    wrappedTime.add(100.0)
  ).mul(uReadoutSigmaNode);

  // 3. Fixed-pattern noise — Gaussian, static per-pixel.
  const fpn: TSLNode = normal3Fixed(seedX, seedY).mul(uFpnSigmaNode);

  return max(afterShot.add(readout).add(fpn), vec3(0.0));
}

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
  config: MegaTSLConfig = {},
  outMaterial?: NodeMaterial
): NodeMaterial {
  const toneMappingMode: LuxarToneMappingMode = config.toneMappingMode ?? 6;

  // Common inputs. Three categories with different live-update contracts:
  //
  //   - **Primitive numeric uniforms** (uExposure, uGlobalOffset, …):
  //     bind via `.onUpdate(() => iuniform.value, 'render')` so
  //     wrapper-class writes to `this.uniforms.X.value` reach the GPU
  //     each render. The raw `uniform(number)` overload captures the
  //     JS value at factory-build time and silently drops subsequent
  //     writes — same trap that bit `bloom.tsl.ts` (now fixed there).
  //   - **Vector2 uniforms** (uResolution, uDistortion, …): the TSL
  //     UniformNode holds the Vector2 by reference; the wrapper
  //     mutates with `.set(x, y)` instead of replacing the object,
  //     so reads stay live without `.onUpdate`.
  //   - **Texture inputs** (`uHdrScene`, `uBloomTexture`): the TSL
  //     `texture(...)` factory captures the THREE.Texture at
  //     factory-call time, so a texture *identity swap* would NOT be
  //     seen by the bound TextureNode. We do not wire `.onUpdate`
  //     here because the wrapper class (`MegaShaderTSLMaterial`) is
  //     designed around graph rebuilds: `setHdrSceneTexture` and
  //     `setBloom(intensity, texture)` call `rebuildGraph()` whenever
  //     the texture identity changes, which re-runs this factory
  //     against the new texture. See
  //     `material-tsl.ts::setHdrSceneTexture` /
  //     `setBloom`. (Contrast with the bloom pyramid's
  //     `bloom/bloom.tsl.ts`, where the input texture changes per pass and
  //     the factory can't be re-run — that path needs `.onUpdate`.)
  const uHdrScene = texture(
    (uniforms.uHdrScene.value as THREE.Texture | null) ?? new THREE.Texture()
  );
  const uExposure = uniform((uniforms.uExposure.value as number) ?? 0.0).onUpdate(
    () => (uniforms.uExposure.value as number) ?? 0.0,
    'render'
  );
  const uGlobalOffset = uniform((uniforms.uGlobalOffset.value as number) ?? 0.0).onUpdate(
    () => (uniforms.uGlobalOffset.value as number) ?? 0.0,
    'render'
  );
  const uGlobalGamma = uniform((uniforms.uGlobalGamma.value as number) ?? 1.0).onUpdate(
    () => (uniforms.uGlobalGamma.value as number) ?? 1.0,
    'render'
  );

  // Optional inputs — declared at factory time only when the feature
  // is enabled, so unused uniforms don't end up in the compiled
  // shader.
  const uBloomTexture =
    config.useBloom && uniforms.uBloomTexture
      ? texture((uniforms.uBloomTexture.value as THREE.Texture | null) ?? new THREE.Texture())
      : null;
  const uBloomIntensity =
    config.useBloom && uniforms.uBloomIntensity
      ? uniform((uniforms.uBloomIntensity.value as number) ?? 0.0).onUpdate(
          () => (uniforms.uBloomIntensity?.value as number) ?? 0.0,
          'render'
        )
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
      ? uniform((uniforms.uSkew.value as number) ?? 0.0).onUpdate(
          () => (uniforms.uSkew?.value as number) ?? 0.0,
          'render'
        )
      : null;
  const uDispersion =
    config.useLensDistortion && uniforms.uDispersion
      ? uniform((uniforms.uDispersion.value as number) ?? 0.0).onUpdate(
          () => (uniforms.uDispersion?.value as number) ?? 0.0,
          'render'
        )
      : null;

  const uVignetteDarkness =
    config.useVignette && uniforms.uVignetteDarkness
      ? uniform((uniforms.uVignetteDarkness.value as number) ?? 0.5).onUpdate(
          () => (uniforms.uVignetteDarkness?.value as number) ?? 0.5,
          'render'
        )
      : null;
  const uVignetteOffset =
    config.useVignette && uniforms.uVignetteOffset
      ? uniform((uniforms.uVignetteOffset.value as number) ?? 0.5).onUpdate(
          () => (uniforms.uVignetteOffset?.value as number) ?? 0.5,
          'render'
        )
      : null;

  // Detector-noise uniforms. Pulled at factory build time so an
  // unused branch doesn't end up in the compiled shader.
  const uTime =
    config.useDetectorNoise && uniforms.uTime
      ? uniform((uniforms.uTime.value as number) ?? 0.0).onUpdate(
          () => (uniforms.uTime?.value as number) ?? 0.0,
          'render'
        )
      : null;
  const uReadoutSigma =
    config.useDetectorNoise && uniforms.uReadoutSigma
      ? uniform((uniforms.uReadoutSigma.value as number) ?? 0.0).onUpdate(
          () => (uniforms.uReadoutSigma?.value as number) ?? 0.0,
          'render'
        )
      : null;
  const uPhotonGain =
    config.useDetectorNoise && uniforms.uPhotonGain
      ? uniform((uniforms.uPhotonGain.value as number) ?? 0.0001).onUpdate(
          () => (uniforms.uPhotonGain?.value as number) ?? 0.0001,
          'render'
        )
      : null;
  const uFpnSigma =
    config.useDetectorNoise && uniforms.uFpnSigma
      ? uniform((uniforms.uFpnSigma.value as number) ?? 0.0).onUpdate(
          () => (uniforms.uFpnSigma?.value as number) ?? 0.0,
          'render'
        )
      : null;

  const fragmentNode = Fn(() => {
    // Read the geometry's caps-aware `uv` attribute. The fullscreen-
    // triangle factory encodes WebGL2/WebGPU framebuffer-Y correction
    // there so we never have to branch on the renderer backend here.
    // screenUV would sample the wrong row of the HDR target under
    // either backend (it tracks gl_FragCoord, which inherits the
    // backend's framebuffer Y orientation).
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
      // This is the CANONICAL top-down distortion map (uv() under
      // WebGPURenderer is top-down; matches the TS picking mirror in
      // picking-system/lens-distortion.ts verbatim). The GLSL twin runs
      // under WebGLRenderer's bottom-up uv and conjugates by negating
      // principalPoint.y and skew — do not copy its matrix here.
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

    // (3) Detector noise — physics-based detector simulation.
    // Three independent components composed additively:
    //   - Poisson shot noise (intensity-dependent) via Anscombe forward/inverse
    //   - Gaussian readout noise (temporal, intensity-independent)
    //   - Gaussian fixed-pattern noise (per-pixel, time-invariant)
    // Hash-driven RNG (Bob Jenkins integer-bit-mix) gives deterministic
    // noise per pixel/frame at zero state cost.
    if (config.useDetectorNoise && uTime && uReadoutSigma && uPhotonGain && uFpnSigma) {
      color.assign(applyDetectorNoise(color, coord, uTime, uReadoutSigma, uPhotonGain, uFpnSigma));
    }

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

  const material = outMaterial ?? new NodeMaterial();
  material.fragmentNode = fragmentNode();
  // We bypass renderer-injected tone mapping; our shader does it
  // explicitly per `toneMappingMode`.
  material.toneMapped = false;
  material.depthTest = false;
  material.depthWrite = false;
  material.transparent = false;
  return material;
}
