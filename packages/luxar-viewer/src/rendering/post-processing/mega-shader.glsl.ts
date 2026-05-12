/**
 * Mega-shader: single-pass post-processing fragment shader.
 *
 * Collapses what used to be a chain of pmndrs/postprocessing Effects
 * (chromatic lens distortion, detector noise, EOG, tone mapping,
 * vignette) into one fullscreen pass. Each effect is gated behind an
 * `#ifdef USE_*` define so a recompile is needed only when an effect
 * is toggled on/off — value tweaks just update uniforms.
 *
 * Pre-stages owned by the host:
 *   - Bloom pyramid renders into `uBloomTexture` (see {@link BloomChain}).
 *     The mega-shader samples bloom at the SAME (chromatic-aberrated)
 *     UVs as the scene; that matches the pmndrs chain semantics in
 *     which bloom was additively blended into the HDR buffer BEFORE
 *     chromatic lens distortion sampled from it.
 *   - FXAA is a separate post-pass after the mega-shader, since edge
 *     detection needs neighbor reads of the *final* LDR pixels.
 *
 * Operation order matches the canonical pmndrs chain order
 * (`effect-orchestrator.ts:buildOrderedEffects`):
 *   1. Bloom additively mixed into the HDR scene (per-channel at
 *      distorted UVs when lens distortion is on)
 *   2. Lens distortion (sampling lookup)
 *   3. Detector noise
 *   4. EOG (Exposure-Offset-Gamma)
 *   5. Tone mapping (mode-switched via `LUXAR_TONE_MAPPING_MODE`
 *      shader-define; values are Luxar-internal IDs 1..6, NOT THREE's
 *      enum values — see `toneMappingModeDefine` in the material file
 *      for the mapping. `NoToneMapping` aliases to Linear so it
 *      clamps to [0,1] like the old pipeline did.)
 *   6. Vignette
 *   7. sRGB encoding (final write; matches outputColorSpace = sRGB)
 *
 * @module rendering/post-processing/mega-shader.glsl
 */

/**
 * Fullscreen-triangle vertex shader. The host supplies a single
 * triangle with positions at NDC {-1,-1}, {3,-1}, {-1,3} — one
 * primitive covers the screen with no clipping waste vs. a quad.
 *
 * Notes for THREE.ShaderMaterial in GLSL3 mode:
 *   - `in vec3 position;` is AUTO-INJECTED by the renderer prefix.
 *     Declaring it explicitly causes a "'position' : redefinition"
 *     compile error.
 *   - THREE also injects built-in uniforms (modelMatrix,
 *     viewMatrix, projectionMatrix, etc.); we use `position`
 *     directly without them since we already hand NDC coords.
 */
export const MEGA_VERTEX_SHADER = /* glsl */ `
  out vec2 vUv;

  void main() {
    // Position is in NDC already; derive UV from it.
    vUv = position.xy * 0.5 + 0.5;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

/**
 * Mega-shader fragment shader. Defines that the host toggles:
 *
 *   - `USE_LENS_DISTORTION`   — chromatic + radial distortion sampling
 *   - `USE_BLOOM`             — additive bloom mix
 *   - `USE_DETECTOR_NOISE`    — physics-based detector noise
 *   - `USE_VIGNETTE`          — multiplicative vignette darkening
 *   - `LUXAR_TONE_MAPPING_MODE`   — internal mode ID (1..6); see
 *                                   mega-shader-material.ts
 *   - `LUXAR_CAPTURE_RAW_HDR`     — early-exit after sample+bloom;
 *                                   bypasses EOG, tone mapping,
 *                                   vignette, sRGB encoding. Used by
 *                                   captureHDRPixels('hdr-effects-pre-tone')
 *                                   to recover the old pmndrs
 *                                   linear-HDR-with-bloom semantics.
 *   - `LUXAR_CAPTURE_LINEAR_LDR`  — skips ONLY the final sRGB
 *                                   encoding. Used by captureHDRPixels('visible-ldr')
 *                                   to match the old linear-LDR
 *                                   capture (post-tone-mapping but
 *                                   pre-sRGB).
 *
 * Note: `toneMappingExposure` is provided by THREE's
 * `<tonemapping_pars_fragment>` chunk; we deliberately do not
 * redeclare it. The host pins it to 1.0 because uExposure already
 * pre-multiplies before the tone-mapping call.
 */
export const MEGA_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;

  in vec2 vUv;
  out vec4 fragColor;

  // ============================================================
  // Common uniforms
  // ============================================================

  uniform sampler2D uHdrScene;
  uniform vec2 uResolution;

  // EOG: applied to every frame regardless of effect toggles
  uniform float uExposure;       // log2 stops; 0 = neutral
  uniform float uGlobalOffset;   // additive lift
  uniform float uGlobalGamma;    // gamma power (1.0 = linear)

  // ============================================================
  // Lens distortion (chromatic + radial)
  // ============================================================
  #ifdef USE_LENS_DISTORTION
  uniform vec2 uDistortion;        // radial coefficient [x, y]
  uniform vec2 uPrincipalPoint;    // optical center offset
  uniform vec2 uFocalLength;       // focal length scale [fx, fy]
  uniform float uSkew;             // skew correction (radians)
  uniform float uDispersion;       // chromatic dispersion strength

  // Brown-Conrady radial distortion + camera intrinsic matrix.
  vec2 applyDistortion(vec2 uv, vec2 distortionCoeff) {
    vec2 xn = 2.0 * (uv - 0.5);
    float r2 = dot(xn, xn);
    vec3 xDistorted = vec3((1.0 + distortionCoeff * r2) * xn, 1.0);
    mat3 kk = mat3(
      vec3(uFocalLength.x, 0.0, 0.0),
      vec3(uSkew * uFocalLength.x, uFocalLength.y, 0.0),
      vec3(uPrincipalPoint.x, uPrincipalPoint.y, 1.0)
    );
    return (kk * xDistorted).xy * 0.5 + 0.5;
  }

  // Mask out-of-bounds samples so distortion doesn't bleed garbage
  // texels in from edge clamping.
  float distortionBorder(vec2 uv) {
    return float(uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0);
  }
  #endif

  // ============================================================
  // Bloom
  // ============================================================
  #ifdef USE_BLOOM
  uniform sampler2D uBloomTexture;
  uniform float uBloomIntensity;
  #endif

  // Sample (scene + bloom) at a single UV. Matches the original
  // pmndrs pipeline where Bloom was additively mixed into the HDR
  // buffer BEFORE chromatic lens distortion sampled it; so a
  // distorted sample picks up bloom at the distorted UV too.
  vec3 sampleHdrPlusBloom(vec2 uv) {
    vec3 result = texture(uHdrScene, uv).rgb;
    #ifdef USE_BLOOM
    result += texture(uBloomTexture, uv).rgb * uBloomIntensity;
    #endif
    return result;
  }

  // ============================================================
  // Detector noise (Bob Jenkins hash + Anscombe Poisson + Gaussian)
  // Lifted verbatim from detector-noise-effect.ts.
  // ============================================================
  #ifdef USE_DETECTOR_NOISE
  uniform float uTime;
  uniform float uReadoutSigma;
  uniform float uPhotonGain;
  uniform float uFpnSigma;

  uint bobJenkinsHash(uint a) {
    a = (a + 0x7ed55d16u) + (a << 12u);
    a = (a ^ 0xc761c23cu) ^ (a >> 19u);
    a = (a + 0x165667b1u) + (a << 5u);
    a = (a + 0xd3a2646cu) ^ (a << 9u);
    a = (a + 0xfd7046c5u) + (a << 3u);
    a = (a ^ 0xb55a4f09u) ^ (a >> 16u);
    return a;
  }

  uint rnguint2(vec2 x) {
    uint a = bobJenkinsHash(floatBitsToUint(x.x));
    uint b = bobJenkinsHash(floatBitsToUint(x.y));
    return bobJenkinsHash(a ^ b);
  }

  uint rnguint3(vec3 x) {
    uint a = rnguint2(x.xy);
    uint b = bobJenkinsHash(floatBitsToUint(x.z));
    return bobJenkinsHash(a ^ b);
  }

  float rngfloat2(vec2 x) { return float(rnguint2(x)) / 4294967296.0; }
  float rngfloat3(vec3 x) { return float(rnguint3(x)) / 4294967296.0; }

  // Clamped logistic ≈ Gaussian (faster than Box-Muller).
  // 0.5513 normalizes to ~unit variance after the [-4, 4] clamp.
  float clampedLogistic(float u) {
    float f = clamp(u, 0.0001, 0.9999);
    float logit = log(f / (1.0 - f));
    return clamp(logit, -4.0, 4.0) * 0.5513;
  }

  vec3 normal3_temporal(vec3 seed) {
    return vec3(
      clampedLogistic(rngfloat3(seed)),
      clampedLogistic(rngfloat3(seed + vec3(13.37, 7.31, 19.93))),
      clampedLogistic(rngfloat3(seed + vec3(31.17, 41.23, 53.59)))
    );
  }

  vec3 normal3_fixed(vec2 seed) {
    return vec3(
      clampedLogistic(rngfloat2(seed)),
      clampedLogistic(rngfloat2(seed + vec2(13.37, 7.31))),
      clampedLogistic(rngfloat2(seed + vec2(31.17, 41.23)))
    );
  }

  // Anscombe variance-stabilizing transform for Poisson approximation.
  float anscombeForward(float x) { return 2.0 * sqrt(max(x + 0.375, 0.0)); }
  float anscombeInverse(float y) {
    float x = (y * 0.5) * (y * 0.5) - 0.375;
    return max(x, 0.0);
  }

  vec3 poissonNoise(vec3 seed, vec3 lambda) {
    vec3 y = vec3(
      anscombeForward(lambda.r),
      anscombeForward(lambda.g),
      anscombeForward(lambda.b)
    );
    y += normal3_temporal(seed);
    return vec3(
      anscombeInverse(y.r),
      anscombeInverse(y.g),
      anscombeInverse(y.b)
    );
  }

  vec3 applyDetectorNoise(vec3 intensity, vec2 uv) {
    float wrappedTime = mod(uTime, 1000.0);
    vec3 temporalSeed = vec3(uv * 1000.0, wrappedTime);
    vec2 fixedSeed = uv * 1000.0;

    // 1. Shot noise (Poisson)
    vec3 photonCount = intensity / max(uPhotonGain, 0.0001);
    vec3 noisyPhotons = poissonNoise(temporalSeed, photonCount);
    vec3 afterShot = noisyPhotons * uPhotonGain;

    // Anscombe forward-then-inverse has a 3/8 bias that brightens
    // pure-zero pixels (visible as glow under vignette). Fade the
    // shot-noise effect to zero in the darkest pixels.
    vec3 shotW = smoothstep(vec3(0.0), vec3(0.01), intensity);
    afterShot = mix(intensity, afterShot, shotW);

    // 2. Readout noise (Gaussian, temporal)
    vec3 readoutNoise = normal3_temporal(temporalSeed + vec3(100.0)) * uReadoutSigma;

    // 3. Fixed-pattern noise (Gaussian, static per-pixel)
    vec3 fpn = normal3_fixed(fixedSeed) * uFpnSigma;

    return max(afterShot + readoutNoise + fpn, vec3(0.0));
  }
  #endif

  // ============================================================
  // Tone mapping — uses THREE's built-in chunk.
  // The chunk declares the toneMappingExposure uniform and the
  // mode functions (Linear/Reinhard/Cineon/ACESFilmic/AgX/Neutral).
  // ============================================================
  #include <tonemapping_pars_fragment>

  // ============================================================
  // Vignette
  // ============================================================
  #ifdef USE_VIGNETTE
  uniform float uVignetteDarkness;
  uniform float uVignetteOffset;
  #endif

  // Linear → sRGB encoding (matches THREE.SRGBColorSpace output).
  // Equivalent to what THREE injects via the colorspace_fragment
  // chunk when outputColorSpace is sRGB. Inlined here because the
  // chunk is keyed to gl_FragColor and we use a GLSL3 out variable.
  vec3 linearToSRGB(vec3 c) {
    vec3 safe = max(c, vec3(0.0));
    return mix(
      1.055 * pow(safe, vec3(1.0 / 2.4)) - 0.055,
      safe * 12.92,
      vec3(lessThanEqual(safe, vec3(0.0031308)))
    );
  }

  // ============================================================
  // Main
  // ============================================================
  void main() {
    vec2 uv = vUv;
    vec3 color;

    // (1+2) Sample (scene + bloom). When lens distortion is on, the
    //       sample happens at chromatically-distorted UVs per channel.
    //       Matches the original pmndrs ordering (bloom additively
    //       mixed BEFORE chromatic distortion sampled the buffer).
    #ifdef USE_LENS_DISTORTION
    {
      vec2 distR = uDistortion * (1.0 - uDispersion);
      vec2 distG = uDistortion;
      vec2 distB = uDistortion * (1.0 + uDispersion);
      vec2 uvR = applyDistortion(uv, distR);
      vec2 uvG = applyDistortion(uv, distG);
      vec2 uvB = applyDistortion(uv, distB);
      float r = sampleHdrPlusBloom(uvR).r * distortionBorder(uvR);
      float g = sampleHdrPlusBloom(uvG).g * distortionBorder(uvG);
      float b = sampleHdrPlusBloom(uvB).b * distortionBorder(uvB);
      color = vec3(r, g, b);
    }
    #else
    color = sampleHdrPlusBloom(uv);
    #endif

    // Capture-mode early exit: when LUXAR_CAPTURE_RAW_HDR is set,
    // the host disables USE_DETECTOR_NOISE / USE_VIGNETTE /
    // USE_LENS_DISTORTION and wants the pre-EOG linear HDR pixels for
    // EXR export. Skip every downstream step (EOG, tone mapping,
    // vignette, sRGB encoding). Output is the scene+bloom sample
    // exactly as the old pmndrs hdr-effects-pre-tone chain wrote into
    // its HalfFloat ping-pong target.
    #ifdef LUXAR_CAPTURE_RAW_HDR
    fragColor = vec4(color, 1.0);
    return;
    #endif

    // (3) Detector noise
    #ifdef USE_DETECTOR_NOISE
    color = applyDetectorNoise(color, uv);
    #endif

    // (4) EOG: Exposure → Offset → Gamma
    color *= exp2(uExposure);
    color = max(color + vec3(uGlobalOffset), vec3(0.0));
    color = pow(color, vec3(1.0 / uGlobalGamma));

    // (5) Tone mapping — LUXAR_TONE_MAPPING_MODE is a Luxar-internal
    //     compressed ID (1..6) set by toneMappingModeDefine() in the
    //     material file. THREE.NoToneMapping is aliased to mode 1
    //     (Linear) at the host so it clamps like the old pipeline.
    //     Functions come from <tonemapping_pars_fragment>.
    #if LUXAR_TONE_MAPPING_MODE == 1
      color = LinearToneMapping(color);
    #elif LUXAR_TONE_MAPPING_MODE == 2
      color = ReinhardToneMapping(color);
    #elif LUXAR_TONE_MAPPING_MODE == 3
      color = CineonToneMapping(color);
    #elif LUXAR_TONE_MAPPING_MODE == 4
      color = ACESFilmicToneMapping(color);
    #elif LUXAR_TONE_MAPPING_MODE == 5
      color = AgXToneMapping(color);
    #elif LUXAR_TONE_MAPPING_MODE == 6
      color = NeutralToneMapping(color);
    #endif

    // (6) Vignette: multiplicative darkening (preserves color ratios).
    #ifdef USE_VIGNETTE
    {
      vec2 vc = (uv - 0.5) / uVignetteOffset;
      float d2 = dot(vc, vc);
      float vf = 1.0 - smoothstep(0.0, 1.5, d2) * uVignetteDarkness;
      color *= vf;
    }
    #endif

    // (7) sRGB encoding. Required when the output reaches a display
    //     surface (backbuffer or sRGB-encoded ldrTarget read by FXAA
    //     and then written to backbuffer).
    //     LUXAR_CAPTURE_LINEAR_LDR disables this step for the
    //     visible-ldr EXR capture mode — the old pmndrs pipeline read
    //     from a HalfFloat ping-pong buffer that held linear LDR
    //     (post-tone-mapping, pre-sRGB) values.
    #ifndef LUXAR_CAPTURE_LINEAR_LDR
    color = linearToSRGB(color);
    #endif

    // Alpha forced to 1.0 — input alpha may be NaN/Inf from heavy
    // additive blending of points/lines (RobustVignetteEffect's
    // original rationale; still applies here).
    fragColor = vec4(color, 1.0);
  }
`;
