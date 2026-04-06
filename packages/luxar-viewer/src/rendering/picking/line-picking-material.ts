/**
 * Line Picking Material for GPU object picking.
 *
 * Renders line segments to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (gl_InstanceID), B = brightness, A = 1.0
 *
 * Vertex shader is based on LINE_VERTEX_SHADER (rendering/shaders/line-shaders.ts)
 * with additions for nodeId/elementId output. Keep in sync with that shader.
 *
 * Fragment uses tighter truncation (60% width) and brightness-as-depth.
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

    out float vSharpness;
    out float vPerpNorm;
    out float vCapFactor;
    out float vPixelWidth;
    flat out highp float vNodeId;
    flat out highp float vElementId;

    void main() {
      float t = aQuadCorner.x > 0.0 ? 1.0 : 0.0;

      float width = mix(aStartWidth, aEndWidth, t);
      vSharpness = mix(aStartSharpness, aEndSharpness, t);

      vec3 worldPos = mix(aStartPos, aEndPos, t);

      vec4 mvStart = modelViewMatrix * vec4(aStartPos, 1.0);
      vec4 mvEnd = modelViewMatrix * vec4(aEndPos, 1.0);
      vec4 mvPos = mix(mvStart, mvEnd, t);
      vec4 clipStart = projectionMatrix * mvStart;
      vec4 clipEnd = projectionMatrix * mvEnd;
      vec4 clipPos = projectionMatrix * mvPos;

      vec2 ndcStart = clipStart.xy / clipStart.w;
      vec2 ndcEnd = clipEnd.xy / clipEnd.w;
      vec2 pixelStart = (ndcStart * 0.5 + 0.5) * uResolution;
      vec2 pixelEnd = (ndcEnd * 0.5 + 0.5) * uResolution;

      vec2 pixelDir = pixelEnd - pixelStart;
      float pixelLen = length(pixelDir);
      vec2 lineDir = pixelLen > 0.0001 ? pixelDir / pixelLen : vec2(1.0, 0.0);
      vec2 perpendicular = vec2(-lineDir.y, lineDir.x);

      float rawPixelWidth;
      if (uIsOrtho == 1) {
        rawPixelWidth = width * uResolution.y / uFOV;
      } else {
        float dist = length(mvPos.xyz);
        float tanHalfFov = tan(uFOV * 0.5);
        rawPixelWidth = width * uResolution.y / (dist * tanHalfFov);
      }

      float minPixelWidth = 1.5;
      float pixelWidth = max(rawPixelWidth, minPixelWidth);
      vPixelWidth = rawPixelWidth;
      vPerpNorm = aQuadCorner.y;

      vec2 pixelOffset = perpendicular * aQuadCorner.y * pixelWidth;
      vec2 ndcOffset = pixelOffset / uResolution * 2.0;
      clipPos.xy += ndcOffset * clipPos.w;

      // Cap factor calculation
      float distFromStart = t * aSegmentLength;
      float distFromEnd = (1.0 - t) * aSegmentLength;
      float distToNearest = min(distFromStart, distFromEnd);
      float baseCap = (distToNearest >= width) ? 1.0 : 0.5 + 0.5 * (distToNearest / width);
      float nearestIsStart = step(distFromEnd, distFromStart);
      float nearestClipped = mix(aEndClipped, aStartClipped, nearestIsStart);
      vCapFactor = mix(baseCap, 1.0, nearestClipped);

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
 */
const LINE_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    in float vSharpness;
    in float vPerpNorm;
    in float vCapFactor;
    in float vPixelWidth;
    flat in highp float vNodeId;
    flat in highp float vElementId;

    out vec4 fragColor;

    void main() {
      float p = abs(vPerpNorm);

      // Full width — lines are already narrow, no need for tighter truncation
      if (p >= 1.0) discard;

      float perpFalloff = pow(1.0 - p * p, vSharpness);

      float minPixelWidth = 1.5;
      float widthScale = min(vPixelWidth / minPixelWidth, 1.0);

      float brightness = vCapFactor * perpFalloff * widthScale;
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
    _nearCull?: number
  ): void {
    this.uniforms.uFOV.value = fov;
    this.uniforms.uResolution.value.copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
  }

  dispose(): void {
    materialManager.unregister(this);
    super.dispose();
  }
}
