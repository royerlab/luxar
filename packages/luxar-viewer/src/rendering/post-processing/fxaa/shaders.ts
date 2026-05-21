/**
 * FXAA post-pass shader sources.
 *
 * Standard "FXAA Quality" preset, GLSL3. Single-pass edge anti-aliasing
 * on tone-mapped LDR pixels — runs after the mega-shader.
 *
 * Lives in its own module so the eventual WebGPU/TSL port has a
 * single place to add a NodeMaterial factory without touching
 * `pass.ts`.
 *
 * @module rendering/post-processing/fxaa/shaders
 */

import type { ShaderSource } from '../../materials/_shared/shader-source';
import { fxaaWebGPUFactory } from './fxaa.tsl';

// THREE's ShaderMaterial auto-declares `in vec3 position;` in the
// GLSL3 prefix — do not redeclare it here.
export const FXAA_VERTEX_SHADER = /* glsl */ `
  out vec2 vUv;
  void main() {
    // Caps-aware uv attribute encodes WebGL2/WebGPU Y-orientation
    // correction. See createFullscreenTriangleGeometry.
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

/**
 * FXAA "Quality" preset. Adapted from Three.js's FXAAShader; rewritten
 * to GLSL ES 3.0 (in/out + `texture()`) and inlined here to avoid an
 * extra example-import.
 */
export const FXAA_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uInput;
  uniform vec2 uResolution;

  // Standard FXAA-3.11 quality constants.
  #define FXAA_EDGE_THRESHOLD       0.125
  #define FXAA_EDGE_THRESHOLD_MIN   0.0312
  #define FXAA_SUBPIX_TRIM          0.25
  #define FXAA_SUBPIX_CAP           0.75

  float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

  void main() {
    vec2 inv = 1.0 / uResolution;

    vec3 cM  = texture(uInput, vUv).rgb;
    vec3 cN  = texture(uInput, vUv + vec2( 0.0, -inv.y)).rgb;
    vec3 cS  = texture(uInput, vUv + vec2( 0.0,  inv.y)).rgb;
    vec3 cE  = texture(uInput, vUv + vec2( inv.x, 0.0)).rgb;
    vec3 cW  = texture(uInput, vUv + vec2(-inv.x, 0.0)).rgb;

    float lM = luma(cM);
    float lN = luma(cN);
    float lS = luma(cS);
    float lE = luma(cE);
    float lW = luma(cW);

    float lMin = min(lM, min(min(lN, lS), min(lE, lW)));
    float lMax = max(lM, max(max(lN, lS), max(lE, lW)));
    float range = lMax - lMin;

    // Local-contrast early-out.
    if (range < max(FXAA_EDGE_THRESHOLD_MIN, lMax * FXAA_EDGE_THRESHOLD)) {
      fragColor = vec4(cM, 1.0);
      return;
    }

    // Sample diagonals for the sub-pixel blend factor.
    vec3 cNW = texture(uInput, vUv + vec2(-inv.x, -inv.y)).rgb;
    vec3 cNE = texture(uInput, vUv + vec2( inv.x, -inv.y)).rgb;
    vec3 cSW = texture(uInput, vUv + vec2(-inv.x,  inv.y)).rgb;
    vec3 cSE = texture(uInput, vUv + vec2( inv.x,  inv.y)).rgb;

    float lNW = luma(cNW);
    float lNE = luma(cNE);
    float lSW = luma(cSW);
    float lSE = luma(cSE);

    // Sub-pixel blend factor.
    float lLowpass = (lN + lE + lW + lS) * 0.25;
    float subRange = abs(lLowpass - lM);
    float subPixel = clamp(subRange / range, 0.0, 1.0);
    subPixel = smoothstep(0.0, 1.0, subPixel);
    subPixel = subPixel * subPixel * FXAA_SUBPIX_CAP;

    // Edge direction (horizontal vs vertical).
    float edgeH = abs((lNW + lNE) - (lSW + lSE)) + 2.0 * abs(lN - lS);
    float edgeV = abs((lNW + lSW) - (lNE + lSE)) + 2.0 * abs(lE - lW);
    bool horizontal = edgeH >= edgeV;

    // Step direction along the edge gradient.
    float lOpp1 = horizontal ? lN : lW;
    float lOpp2 = horizontal ? lS : lE;
    float grad1 = lOpp1 - lM;
    float grad2 = lOpp2 - lM;
    float gradN = abs(grad1);
    float gradP = abs(grad2);
    bool stepUpLeft = gradN >= gradP;
    float gradient = max(gradN, gradP) * 0.25;

    vec2 step = horizontal ? vec2(0.0, inv.y) : vec2(inv.x, 0.0);
    if (stepUpLeft) step = -step;

    // Single-sample blend along the step.
    vec3 cBlend = texture(uInput, vUv + step * 0.5).rgb;

    // Mix in the sub-pixel blend.
    vec3 result = mix(cM, cBlend, subPixel);
    fragColor = vec4(result, 1.0);
  }
`;

export const FXAA_SOURCE: ShaderSource = {
  name: 'fxaa',
  webgl: { vertex: FXAA_VERTEX_SHADER, fragment: FXAA_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) =>
    fxaaWebGPUFactory(uniforms as Record<string, import('three').IUniform>),
};
