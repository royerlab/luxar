/**
 * Point Picking Material for GPU object picking.
 *
 * Renders points to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (gl_VertexID), B = brightness, A = 1.0
 *
 * Vertex shader is based on POINT_VERTEX_SHADER (rendering/shaders/point-shaders.ts)
 * with additions for nodeId/elementId output. Keep in sync with that shader.
 *
 * Fragment uses tighter truncation (50% radius) and brightness-as-depth
 * so the brightest element at each pixel wins the depth test.
 */

import * as THREE from 'three';
import type { CameraAwareMaterial } from '../camera-aware-material';
import { materialManager } from '../material-manager';

export interface PointPickingMaterialConfig {
  nodeId: number;
  radiusScale?: number;
  sharpnessScale?: number;
}

/**
 * Picking vertex shader for points.
 * Based on POINT_VERTEX_SHADER — adds uNodeId uniform and vNodeId/vElementId outputs.
 */
const POINT_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    in float radius;
    in float sharpness;

    uniform float pointSizeFactor;
    uniform float maxPointSize;
    uniform float radiusScale;
    uniform float sharpnessScale;
    uniform int uIsOrtho;
    uniform float uNodeId;

    out highp float vRadius;
    out mediump float vSharpness;
    flat out highp float vNodeId;
    flat out highp float vElementId;

    void main() {
      float normalizedSharpness = sharpness * sharpnessScale;
      vSharpness = normalizedSharpness > 0.0 ? normalizedSharpness : 2.0;

      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;

      float normalizedRadius = radius * radiusScale;
      vRadius = normalizedRadius;

      float invDistance = (uIsOrtho == 1) ? 1.0 : inversesqrt(dot(mvPosition.xyz, mvPosition.xyz));
      float basePointSize = normalizedRadius * pointSizeFactor * invDistance;

      // Tighter truncation for picking: 50% of visual radius
      // Use sharpness compensation * 0.5 so we only pick the bright core
      float sharpnessCompensation = 1.0 / (1.0 - pow(0.01, 1.0 / max(vSharpness, 0.01)));
      float pointSize = basePointSize * sharpnessCompensation * 0.5;

      gl_PointSize = max(1.0, min(pointSize, maxPointSize));

      vNodeId = uNodeId;
      vElementId = float(gl_VertexID);
    }
`;

/**
 * Picking fragment shader for points.
 * Outputs vec4(nodeId, elementId, brightness, 1.0) with brightness-as-depth.
 */
const POINT_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    in highp float vRadius;
    in mediump float vSharpness;
    flat in highp float vNodeId;
    flat in highp float vElementId;

    out vec4 fragColor;

    void main() {
      if (vRadius < 0.0001) discard;

      vec2 centered = gl_PointCoord - 0.5;
      float r2 = dot(centered, centered);

      // Full circle discard (gl_PointSize is already halved in vertex shader for tighter picking)
      if (r2 > 0.25) discard;

      float normalizedR = sqrt(4.0 * r2);
      float falloff = pow(max(1.0 - normalizedR, 0.0), vSharpness);

      float brightness = falloff;
      if (brightness < 1e-4) discard;

      fragColor = vec4(vNodeId, vElementId, brightness, 1.0);
      gl_FragDepth = 1.0 - clamp(brightness, 0.0, 1.0);
    }
`;

export class PointPickingMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  constructor(config: PointPickingMaterialConfig) {
    const defaultFov = (60 * Math.PI) / 180;
    const defaultResolutionY = 1080;
    const defaultTanHalfFov = Math.tan(defaultFov / 2);

    super({
      uniforms: {
        pointSizeFactor: { value: (2.0 * defaultResolutionY) / defaultTanHalfFov },
        maxPointSize: { value: defaultResolutionY * 0.5 },
        radiusScale: { value: config.radiusScale ?? 1.0 },
        sharpnessScale: { value: config.sharpnessScale ?? 1.0 },
        uIsOrtho: { value: 0 },
        uNodeId: { value: config.nodeId },
      },
      vertexShader: POINT_PICK_VERTEX_SHADER,
      fragmentShader: POINT_PICK_FRAGMENT_SHADER,
      glslVersion: THREE.GLSL3,
      // Picking settings: opaque, depth test, no blending
      transparent: false,
      depthTest: true,
      depthWrite: true,
      blending: THREE.NoBlending,
      toneMapped: false,
    });
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    _nearCull?: number
  ): void {
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    if (isOrtho) {
      const halfFrustum = fov * 0.5;
      this.uniforms.pointSizeFactor.value = (2.0 * resolution.y) / halfFrustum;
    } else {
      const tanHalfFov = Math.tan(fov / 2);
      this.uniforms.pointSizeFactor.value = (2.0 * resolution.y) / tanHalfFov;
    }
    this.uniforms.maxPointSize.value = resolution.y * 0.5;
  }

  dispose(): void {
    materialManager.unregister(this);
    super.dispose();
  }
}
