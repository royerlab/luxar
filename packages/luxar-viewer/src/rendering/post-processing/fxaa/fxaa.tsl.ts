/**
 * FXAA TSL factory — NodeMaterial counterpart to the GLSL3 shader
 * in `shaders.ts`.
 *
 * Implements the same FXAA Quality preset as the GLSL version:
 * Rec.601 luma → local-contrast early-out → diagonal-tap sub-pixel
 * blend → edge-direction step → 0.5-pixel mix.
 *
 * Reads `uInput` (the LDR texture) and `uResolution`. Uniforms are
 * threaded in via the factory parameter so the WebGL fallback and
 * the WebGPU path share one source of state.
 *
 * @module rendering/post-processing/fxaa/fxaa.tsl
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
  float,
  If,
  max,
  min,
  abs,
  dot,
  clamp,
  smoothstep,
  mix,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { bindLiveTexture } from '../../materials/_shared/live-texture-tsl';

/**
 * FXAA TSL factory. Receives the uniforms table that the
 * `ShaderMaterial`-side path also receives, so callers don't need
 * to branch on backend when wiring uniform values.
 */
export function fxaaWebGPUFactory(uniforms: Record<string, THREE.IUniform>): NodeMaterial {
  // Live-bound texture node — see `bindLiveTexture` for why the swap
  // lives in `updateBefore` rather than `.onUpdate(…, 'render')`. The
  // host (FxaaPass) builds the material with `uInput.value === null`
  // and assigns the real `ldrTarget.texture` per render, so this pass
  // has exactly the many-tap render-target swap that #2584 was about:
  // on the first frame after each material build the already-updated
  // taps would derive their Y-flip from the null-placeholder. (The
  // parity harness hands this factory a real texture at build time, so
  // the swap never happens under test — the fix here is by
  // construction, not by coverage.)
  const fallback = new THREE.Texture();
  const uInput = bindLiveTexture(
    texture((uniforms.uInput.value as THREE.Texture | null) ?? fallback),
    uniforms.uInput,
    fallback
  );
  const uResolution = uniform(
    (uniforms.uResolution.value as THREE.Vector2) ?? new THREE.Vector2(1, 1)
  );

  // Rec.601 luma weights — match the GLSL `luma()` helper.
  const lumaWeights = vec3(0.299, 0.587, 0.114);

  // FXAA-3.11 quality constants (match GLSL `#define`s).
  const FXAA_EDGE_THRESHOLD = float(0.125);
  const FXAA_EDGE_THRESHOLD_MIN = float(0.0312);
  const FXAA_SUBPIX_CAP = float(0.75);

  const fragmentNode = Fn(() => {
    const inv = vec2(1.0).div(uResolution);
    // Read the geometry's caps-aware `uv` attribute. The fullscreen-
    // triangle factory encodes WebGL2/WebGPU framebuffer-Y correction
    // there so we never have to branch on the renderer backend here.
    const coord = uv();

    const cM = uInput.sample(coord).rgb;
    const cN = uInput.sample(coord.add(vec2(float(0), inv.y.negate()))).rgb;
    const cS = uInput.sample(coord.add(vec2(float(0), inv.y))).rgb;
    const cE = uInput.sample(coord.add(vec2(inv.x, float(0)))).rgb;
    const cW = uInput.sample(coord.add(vec2(inv.x.negate(), float(0)))).rgb;

    // Inline luma computation. Factored as a TSL `Fn` it required
    // a `VarNode` parameter that doesn't match the type returned
    // by `.rgb` swizzles; inlining `dot(c, lumaWeights)` is simpler
    // and produces identical output.
    const lM = dot(cM, lumaWeights);
    const lN = dot(cN, lumaWeights);
    const lS = dot(cS, lumaWeights);
    const lE = dot(cE, lumaWeights);
    const lW = dot(cW, lumaWeights);

    const lMin = min(lM, min(min(lN, lS), min(lE, lW)));
    const lMax = max(lM, max(max(lN, lS), max(lE, lW)));
    const range = lMax.sub(lMin);

    const result = vec3(cM).toVar();

    // Local-contrast gate: skip blending if contrast is below the
    // adaptive threshold. Returning early in TSL requires structured
    // `If`; we instead branch the assignment.
    If(range.greaterThanEqual(max(FXAA_EDGE_THRESHOLD_MIN, lMax.mul(FXAA_EDGE_THRESHOLD))), () => {
      const cNW = uInput.sample(coord.add(vec2(inv.x.negate(), inv.y.negate()))).rgb;
      const cNE = uInput.sample(coord.add(vec2(inv.x, inv.y.negate()))).rgb;
      const cSW = uInput.sample(coord.add(vec2(inv.x.negate(), inv.y))).rgb;
      const cSE = uInput.sample(coord.add(vec2(inv.x, inv.y))).rgb;

      const lNW = dot(cNW, lumaWeights);
      const lNE = dot(cNE, lumaWeights);
      const lSW = dot(cSW, lumaWeights);
      const lSE = dot(cSE, lumaWeights);

      const lLowpass = lN.add(lE).add(lW).add(lS).mul(0.25);
      const subRange = abs(lLowpass.sub(lM));
      const subPixelRaw = clamp(subRange.div(range), 0.0, 1.0);
      const subPixelSmoothed = smoothstep(0.0, 1.0, subPixelRaw);
      const subPixel = subPixelSmoothed.mul(subPixelSmoothed).mul(FXAA_SUBPIX_CAP);

      const edgeH = abs(lNW.add(lNE).sub(lSW.add(lSE))).add(abs(lN.sub(lS)).mul(2.0));
      const edgeV = abs(lNW.add(lSW).sub(lNE.add(lSE))).add(abs(lE.sub(lW)).mul(2.0));
      const horizontal = edgeH.greaterThanEqual(edgeV);

      // Determine step direction along the edge gradient.
      const lOpp1 = horizontal.select(lN, lW);
      const lOpp2 = horizontal.select(lS, lE);
      const gradN = abs(lOpp1.sub(lM));
      const gradP = abs(lOpp2.sub(lM));
      const stepUpLeft = gradN.greaterThanEqual(gradP);

      const stepH = vec2(float(0), inv.y);
      const stepV = vec2(inv.x, float(0));
      const stepRaw = horizontal.select(stepH, stepV);
      const stepSigned = stepUpLeft.select(stepRaw.negate(), stepRaw);

      const cBlend = uInput.sample(coord.add(stepSigned.mul(0.5))).rgb;
      result.assign(mix(cM, cBlend, subPixel));
    });

    return vec4(result, 1.0);
  });

  const material = new NodeMaterial();
  material.fragmentNode = fragmentNode();
  // FXAA reads already-tone-mapped LDR values and writes them
  // through. Bypass renderer-level tone mapping injection.
  material.toneMapped = false;
  material.depthTest = false;
  material.depthWrite = false;
  material.transparent = false;
  return material;
}
