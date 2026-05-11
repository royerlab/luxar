/**
 * Line Picking Material for GPU object picking.
 *
 * Renders line segments to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (gl_InstanceID), B = brightness, A = 1.0
 *
 * Vertex shader is based on LINE_VERTEX_SHADER (rendering/shaders/line-shaders.ts)
 * with additions for nodeId/elementId output. Keep in sync with that shader.
 *
 * Mirrors the visual line shader's fragment-side cap factor,
 * near-plane safety, max-pixel-width clamp, and width/sharpness
 * sanitization. Without picking parity the pick footprint diverges
 * from the visible footprint and hover labels can be wrong for thick
 * lines near the camera.
 *
 * Fragment uses full width (lines are already narrow) and brightness-as-depth.
 */

import * as THREE from 'three';
import type { CameraAwareMaterial } from '../camera-aware-material';
import { materialManager } from '../material-manager';

export interface LinePickingMaterialConfig {
  nodeId: number;
}

/**
 * Picking vertex shader for lines.
 * Based on LINE_VERTEX_SHADER — adds uNodeId uniform and vNodeId/vElementId outputs.
 * Strips colormap and color varyings (not needed for picking).
 */
const LINE_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    in vec2 aQuadCorner;

    // Instanced attributes (per segment)
    in vec3 aStartPos;
    in vec3 aEndPos;
    in float aStartWidth;
    in float aEndWidth;
    in float aStartSharpness;
    in float aEndSharpness;
    in float aSegmentLength;
    in float aStartClipped;
    in float aEndClipped;

    uniform float uFOV;
    uniform vec2 uResolution;
    uniform int uIsOrtho;
    uniform float uNodeId;
    uniform float uNearCull;          // visual-shader parity
    uniform float uMaxLinePixelWidth; // visual-shader parity

    out float vSharpness;
    out float vPerpNorm;
    out float vT;             // fragment-side cap math (parity with visual)
    out float vSegmentLength;
    out float vWidthAtT;
    out float vPixelWidth;
    out float vWidthFade;     // visual-shader parity
    flat out float vClippedStart;
    flat out float vClippedEnd;
    flat out highp float vNodeId;
    flat out highp float vElementId;

    void main() {
      float t = aQuadCorner.x > 0.0 ? 1.0 : 0.0;
      vT = t;
      vSegmentLength = aSegmentLength;
      vClippedStart = aStartClipped;
      vClippedEnd = aEndClipped;

      // sanitize width/sharpness against negative/NaN/Inf.
      float startW = (isnan(aStartWidth) || isinf(aStartWidth) || aStartWidth < 0.0) ? 0.0 : aStartWidth;
      float endW = (isnan(aEndWidth) || isinf(aEndWidth) || aEndWidth < 0.0) ? 0.0 : aEndWidth;
      float startS = (isnan(aStartSharpness) || isinf(aStartSharpness) || aStartSharpness <= 0.0) ? 2.0 : aStartSharpness;
      float endS = (isnan(aEndSharpness) || isinf(aEndSharpness) || aEndSharpness <= 0.0) ? 2.0 : aEndSharpness;

      float width = mix(startW, endW, t);
      vSharpness = mix(startS, endS, t);
      vWidthAtT = width;

      vec3 worldPos = mix(aStartPos, aEndPos, t);

      vec4 mvStart = modelViewMatrix * vec4(aStartPos, 1.0);
      vec4 mvEnd = modelViewMatrix * vec4(aEndPos, 1.0);
      vec4 mvPos = mix(mvStart, mvEnd, t);

      // Visual-shader parity: near-plane safety (degenerate quad if both endpoints behind).
      float nearCull = max(uNearCull, 1e-4);
      float startDepth = -mvStart.z;
      float endDepth = -mvEnd.z;
      bool bothBehind = (startDepth < nearCull) && (endDepth < nearCull);
      if (bothBehind) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vPerpNorm = 0.0;
        vPixelWidth = 0.0;
        vWidthFade = 0.0;
        vNodeId = uNodeId;
        vElementId = float(gl_InstanceID);
        return;
      }

      vec4 clipStart = projectionMatrix * mvStart;
      vec4 clipEnd = projectionMatrix * mvEnd;
      vec4 clipPos = projectionMatrix * mvPos;

      float wStart = max(clipStart.w, 1e-4);
      float wEnd = max(clipEnd.w, 1e-4);
      vec2 ndcStart = clipStart.xy / wStart;
      vec2 ndcEnd = clipEnd.xy / wEnd;
      vec2 pixelStart = (ndcStart * 0.5 + 0.5) * uResolution;
      vec2 pixelEnd = (ndcEnd * 0.5 + 0.5) * uResolution;

      vec2 pixelDir = pixelEnd - pixelStart;
      float pixelLen = length(pixelDir);
      vec2 lineDir = pixelLen > 0.0001 ? pixelDir / pixelLen : vec2(1.0, 0.0);
      vec2 perpendicular = vec2(-lineDir.y, lineDir.x);

      float rawPixelWidth;
      if (uIsOrtho == 1) {
        rawPixelWidth = width * 2.0 * uResolution.y / uFOV;
      } else {
        float dist = max(length(mvPos.xyz), nearCull);
        float tanHalfFov = tan(uFOV * 0.5);
        rawPixelWidth = width * uResolution.y / (dist * tanHalfFov);
      }

      float minPixelWidth = 1.5;
      float maxPW = max(uMaxLinePixelWidth, minPixelWidth + 1.0);
      float clampedPixelWidth = clamp(rawPixelWidth, minPixelWidth, maxPW);
      vWidthFade = (rawPixelWidth <= maxPW) ? 1.0 : (maxPW / max(rawPixelWidth, 1e-4));
      vPixelWidth = rawPixelWidth;
      vPerpNorm = aQuadCorner.y;

      vec2 pixelOffset = perpendicular * aQuadCorner.y * clampedPixelWidth;
      vec2 ndcOffset = pixelOffset / uResolution * 2.0;
      clipPos.xy += ndcOffset * clipPos.w;

      gl_Position = clipPos;

      vNodeId = uNodeId;
      vElementId = float(gl_InstanceID);
    }
`;

/**
 * Picking fragment shader for lines.
 * Outputs vec4(nodeId, elementId, brightness, 1.0) with brightness-as-depth.
 *
 * Lines use FULL width for picking (same as visual) — unlike points/splats,
 * lines are already narrow with a sharp parabolic profile. Tighter truncation
 * would make thin lines nearly impossible to pick. The brightness-weighted
 * voting handles overlap correctly (centerline brightness always wins).
 *
 * Cap factor is computed in fragment to match the visual shader.
 */
const LINE_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    in float vSharpness;
    in float vPerpNorm;
    in float vT;
    in float vSegmentLength;
    in float vWidthAtT;
    in float vPixelWidth;
    in float vWidthFade;
    flat in float vClippedStart;
    flat in float vClippedEnd;
    flat in highp float vNodeId;
    flat in highp float vElementId;

    out vec4 fragColor;

    void main() {
      float p = abs(vPerpNorm);

      // Full width — lines are already narrow, no need for tighter truncation
      if (p >= 1.0) discard;

      float perpFalloff = pow(max(1.0 - p * p, 0.0), max(vSharpness, 0.0001));

      float minPixelWidth = 1.5;
      float widthScale = min(vPixelWidth / minPixelWidth, 1.0);

      // cap factor in fragment (matches visual shader).
      float distFromStart = vT * vSegmentLength;
      float distFromEnd = (1.0 - vT) * vSegmentLength;
      float distToNearest = min(distFromStart, distFromEnd);
      float capRamp = vWidthAtT > 1e-4
        ? clamp(distToNearest / vWidthAtT, 0.0, 1.0)
        : 1.0;
      float baseCap = 0.5 + 0.5 * capRamp;
      // step(distFromStart, distFromEnd) is 1 when start is closer (distFromEnd >= distFromStart).
      float nearestIsStart = step(distFromStart, distFromEnd);
      float nearestClipped = mix(vClippedEnd, vClippedStart, nearestIsStart);
      float capFactor = mix(baseCap, 1.0, nearestClipped);

      float brightness = capFactor * perpFalloff * widthScale * vWidthFade;
      if (brightness < 1e-4) discard;

      fragColor = vec4(vNodeId, vElementId, brightness, 1.0);
      gl_FragDepth = 1.0 - clamp(brightness, 0.0, 1.0);
    }
`;

export class LinePickingMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  constructor(config: LinePickingMaterialConfig) {
    super({
      uniforms: {
        uFOV: { value: (60 * Math.PI) / 180 },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uIsOrtho: { value: 0 },
        uNearCull: { value: 0.05 },
        uMaxLinePixelWidth: { value: 540 },
        uNodeId: { value: config.nodeId },
      },
      vertexShader: LINE_PICK_VERTEX_SHADER,
      fragmentShader: LINE_PICK_FRAGMENT_SHADER,
      glslVersion: THREE.GLSL3,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      blending: THREE.NoBlending,
      toneMapped: false,
      side: THREE.DoubleSide,
    });
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    this.uniforms.uFOV.value = fov;
    this.uniforms.uResolution.value.copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    if (nearCull !== undefined && nearCull > 0) {
      this.uniforms.uNearCull.value = nearCull;
    }
    this.uniforms.uMaxLinePixelWidth.value = Math.max(2, resolution.y * 0.5);
  }

  dispose(): void {
    materialManager.unregister(this);
    super.dispose();
  }
}
