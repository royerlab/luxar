/**
 * GSplat Material for Luxar
 *
 * Specialized THREE.ShaderMaterial for rendering Gaussian splats using instanced quads.
 * Implements volumetric ray integration with oriented, anisotropic Gaussian density functions.
 *
 * Key features:
 * - Instanced oriented quad geometry (4 vertices per splat)
 * - Full 3D covariance via Cholesky factors
 * - Perspective-correct projection of covariance to 2D
 * - Ray-integrated amplitude boost (volumetric rendering)
 * - Generalized Gaussian falloff: exp(-½ · r^sharpness)
 * - Per-splat attributes (center, cholesky, amplitude, sharpness, color)
 *
 * Mathematical basis:
 * - GSplat density: G(x) = a · exp(-½ · ‖L⁻¹(x - μ)‖^s)
 * - Ray integration boosts amplitude by σ_ray · c(s)
 * - 2D covariance: Σ_2D = J · Σ_cam · Jᵀ (perspective Jacobian projection)
 *
 * @module rendering/gsplat-material
 */

import * as THREE from 'three';
import { config } from '../config';
import { materialManager } from './material-manager';

/**
 * Configuration for gsplat material creation
 */
export interface GSplatMaterialConfig {
  /** Opacity multiplier (0.0 to 1.0) */
  opacity?: number;
  /** HDR intensity multiplier */
  hdrMultiplier?: number;
  /** Truncation radius in sigmas (default 3.0) */
  truncationRadius?: number;
  /** Blending mode ('additive' | 'normal' | 'max') */
  blendingMode?: 'additive' | 'normal' | 'max';
}

/**
 * GSplat material uniforms interface
 */
export interface GSplatMaterialUniforms {
  /** Viewport resolution [width, height] */
  uResolution: { value: THREE.Vector2 };
  /** Focal length X in pixels */
  uFx: { value: number };
  /** Focal length Y in pixels */
  uFy: { value: number };
  /** Truncation radius in sigmas */
  uTruncate: { value: number };
  /** HDR intensity multiplier */
  uHDRMultiplier: { value: number };
  /** Opacity multiplier */
  uOpacity: { value: number };
}

/**
 * GSplat material for volumetric Gaussian splatting.
 *
 * Each splat is rendered as an oriented quad expanded based on the 2D
 * projected covariance eigenvalues. The fragment shader evaluates the
 * Gaussian density using Mahalanobis distance from the projected center.
 */
export class GSplatMaterial extends THREE.ShaderMaterial {
  /**
   * Vertex shader with 3D → 2D covariance projection and oriented quad expansion.
   *
   * Per-splat attributes:
   * - aCenter: 3D splat center (after nD slicing)
   * - aCholesky01, aCholesky23, aCholesky45: Packed 3D Cholesky factors
   * - aAmplitude: Already attenuated by hidden dimensions
   * - aSharpness: Generalized Gaussian falloff exponent
   * - aColor: RGB color
   *
   * The shader:
   * 1. Transforms center and Cholesky to camera space
   * 2. Projects 3D covariance to 2D using perspective Jacobian
   * 3. Computes ray variance for amplitude boost
   * 4. Expands oriented quad based on 2D covariance eigenvalues
   */
  private static readonly VERTEX_SHADER = /* glsl */ `
    precision highp float;

    // Quad corner attribute (static geometry)
    attribute vec2 aQuadCorner;  // (-1,-1), (1,-1), (-1,1), (1,1)

    // Per-instance attributes
    attribute vec3 aCenter;           // 3D center (after nD slicing)
    attribute vec2 aCholesky01;       // [L00, L10]
    attribute vec2 aCholesky23;       // [L11, L20]
    attribute vec2 aCholesky45;       // [L21, L22]
    attribute float aAmplitude;       // Already attenuated by hidden dims
    attribute float aSharpness;
    attribute vec3 aColor;

    // Uniforms (modelViewMatrix and projectionMatrix are built-in THREE.js uniforms)
    uniform vec2 uResolution;
    uniform float uFx, uFy;           // Focal lengths in pixels
    uniform float uTruncate;          // Truncation radius (in sigmas)
    uniform int uProjectionMode;      // 0 = sum projection (additive), 1 = max projection (max blending)

    // Varyings to fragment
    varying vec3 vColor;
    varying float vAmplitude2D;
    varying float vSharpness;
    varying vec3 vL2D;                // 2D Cholesky [L00, L10, L11]
    varying vec2 vCenterScreen;       // Splat center in screen pixels
    varying vec2 vQuadCoord;          // Quad coordinate for fragment

    // Unpack 3D Cholesky to matrix (column-major order for GLSL mat3)
    // Packed order: [L00, L10, L11, L20, L21, L22]
    // aCholesky01 = [L00, L10], aCholesky23 = [L11, L20], aCholesky45 = [L21, L22]
    mat3 unpackCholesky3D() {
        return mat3(
            aCholesky01.x, aCholesky01.y, aCholesky23.y,  // Column 0: [L00, L10, L20]
            0.0,           aCholesky23.x, aCholesky45.x,  // Column 1: [0, L11, L21]
            0.0,           0.0,           aCholesky45.y   // Column 2: [0, 0, L22]
        );
    }

    // Compute 2D Cholesky from 2D covariance (symmetric positive definite)
    vec3 cholesky2x2(mat2 S) {
        float L00 = sqrt(max(S[0][0], 1e-8));
        float L10 = S[1][0] / L00;
        float L11 = sqrt(max(S[1][1] - L10 * L10, 1e-8));
        return vec3(L00, L10, L11);
    }

    // Sharpness integral factor c(s) - simple approximation
    // With 3σ truncation, all values in [2.0, 3.6], so simple formula works well
    // Optimized to be nearly exact at s=2 (the common case)
    // Max error: 3.7%, error at s=2: 0.27%
    float sharpnessIntegralFactor(float s) {
        return 1.97 + 1.95 * exp(-0.64 * s);
    }

    void main() {
        // Transform center to camera space
        vec4 centerCam4 = modelViewMatrix * vec4(aCenter, 1.0);
        vec3 centerCam = centerCam4.xyz;

        // Near-plane guard: reject splats too close to camera
        if (-centerCam.z < 0.1) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);  // Behind camera
            return;
        }

        // Transform Cholesky to camera space (rotation only)
        mat3 R = mat3(modelViewMatrix);
        mat3 L3D = unpackCholesky3D();
        mat3 L_cam = R * L3D;
        mat3 Sigma_cam = L_cam * transpose(L_cam);

        // Perspective projection Jacobian at splat center
        float z = -centerCam.z;  // Positive depth (camera looks down -Z)
        float z2 = z * z;

        // Jacobian J = d(screen)/d(camera) at splat center
        // J[0] = [fx/z, 0, fx*x/z²]
        // J[1] = [0, fy/z, fy*y/z²]
        mat3x2 J;
        J[0] = vec2(uFx / z, 0.0);
        J[1] = vec2(0.0, uFy / z);
        J[2] = vec2(uFx * centerCam.x / z2, uFy * centerCam.y / z2);

        // Project covariance to 2D: Σ_2D = J · Σ_cam · Jᵀ
        // Compute J * Sigma_cam first
        vec2 JS0 = J[0] * Sigma_cam[0][0] + J[1] * Sigma_cam[0][1] + J[2] * Sigma_cam[0][2];
        vec2 JS1 = J[0] * Sigma_cam[1][0] + J[1] * Sigma_cam[1][1] + J[2] * Sigma_cam[1][2];
        vec2 JS2 = J[0] * Sigma_cam[2][0] + J[1] * Sigma_cam[2][1] + J[2] * Sigma_cam[2][2];

        // Sigma2D = (J * Sigma_cam) * J^T
        // For M = J*Sigma (cols JS0, JS1, JS2), compute M * J^T:
        // (M*J^T)[i,j] = sum_k M[i,k] * J[j,k] = sum_k JS_k[i] * J[k][j]
        mat2 Sigma2D;
        Sigma2D[0][0] = JS0.x * J[0].x + JS1.x * J[1].x + JS2.x * J[2].x;
        Sigma2D[1][0] = JS0.x * J[0].y + JS1.x * J[1].y + JS2.x * J[2].y;
        Sigma2D[0][1] = Sigma2D[1][0];  // Symmetric (J*S*J^T preserves symmetry)
        Sigma2D[1][1] = JS0.y * J[0].y + JS1.y * J[1].y + JS2.y * J[2].y;

        // Compute ray variance for amplitude boost
        // Ray direction: from camera origin (0,0,0) to splat center in camera space
        vec3 rayDir = normalize(centerCam);
        float sigmaRaySq = dot(rayDir, Sigma_cam * rayDir);
        float sigmaRay = sqrt(max(sigmaRaySq, 1e-8));

        // Projection mode determines amplitude calculation (BRANCHLESS):
        // - Sum projection (uProjectionMode = 0): Integrate Gaussian along ray → ray boost
        // - Max projection (uProjectionMode = 1): Use peak Gaussian value → no boost
        //
        // Branchless implementation using mix():
        //   uProjectionMode = 0 → useSumProjection = 1.0 → rayBoost = sigmaRay * c(s)
        //   uProjectionMode = 1 → useSumProjection = 0.0 → rayBoost = 1.0
        //
        float useSumProjection = 1.0 - float(uProjectionMode); // 1.0 for sum, 0.0 for max
        float c_s = sharpnessIntegralFactor(aSharpness);
        float voxelSpacing = 1.0;
        float rayIntegrationBoost = sigmaRay * c_s / voxelSpacing;
        float rayBoost = mix(1.0, rayIntegrationBoost, useSumProjection);
        vAmplitude2D = aAmplitude * rayBoost;

        // NOTE: Sum projection still ~10x too bright empirically. May need additional normalization.
        // Consider: divide by truncate radius, or by sqrt(2π), or apply global scale

        // Compute 2D Cholesky for fragment shader
        vL2D = cholesky2x2(Sigma2D);

        // Eigenvalues of Σ_2D for quad extents (oriented quad)
        float trace = Sigma2D[0][0] + Sigma2D[1][1];
        float det = Sigma2D[0][0] * Sigma2D[1][1] - Sigma2D[0][1] * Sigma2D[1][0];
        float disc = max(trace * trace - 4.0 * det, 0.0);  // Clamp for numerical stability
        float sqrtDisc = sqrt(disc);
        float lambda1 = max(0.5 * (trace + sqrtDisc), 1e-6);
        float lambda2 = max(0.5 * (trace - sqrtDisc), 1e-6);

        // Eigenvector for major axis (for oriented quad)
        vec2 majorAxis;
        if (abs(Sigma2D[0][1]) > 1e-6) {
            majorAxis = normalize(vec2(lambda1 - Sigma2D[1][1], Sigma2D[0][1]));
        } else {
            majorAxis = vec2(1.0, 0.0);
        }
        vec2 minorAxis = vec2(-majorAxis.y, majorAxis.x);

        // Quad extents: truncation radius × sqrt(eigenvalue) × sharpness factor
        // For generalized Gaussian, adjust truncation for sharpness
        float effectiveTruncate = pow(uTruncate, 2.0 / max(aSharpness, 0.1));
        float extent1 = effectiveTruncate * sqrt(lambda1);
        float extent2 = effectiveTruncate * sqrt(lambda2);

        // Project center to screen (pixels)
        vCenterScreen = vec2(
            uFx * centerCam.x / z + uResolution.x * 0.5,
            uFy * centerCam.y / z + uResolution.y * 0.5
        );

        // Expand quad vertex in screen space (oriented)
        vec2 quadOffset = aQuadCorner.x * majorAxis * extent1
                        + aQuadCorner.y * minorAxis * extent2;
        vec2 screenPos = vCenterScreen + quadOffset;

        // Convert screen pixels to NDC (xy only)
        vec2 ndcXY = (screenPos / uResolution) * 2.0 - 1.0;

        // Pass through other varyings
        vColor = aColor;
        vSharpness = aSharpness;
        vQuadCoord = aQuadCorner * vec2(extent1, extent2);  // For fragment shader

        // Compute proper clip-space depth using projection matrix
        // This ensures correct depth buffer behavior for overlapping splats
        vec4 centerClip = projectionMatrix * centerCam4;
        float ndcZ = centerClip.z / centerClip.w;

        // Output final clip position
        gl_Position = vec4(ndcXY, ndcZ, 1.0);
    }
  `;

  /**
   * Fragment shader with generalized Gaussian falloff.
   *
   * Uses the 2D Cholesky factor passed from vertex shader to compute
   * Mahalanobis distance, then applies generalized Gaussian falloff.
   */
  private static readonly FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    varying vec3 vColor;
    varying float vAmplitude2D;
    varying float vSharpness;
    varying vec3 vL2D;          // 2D Cholesky [L00, L10, L11]
    varying vec2 vCenterScreen;
    varying vec2 vQuadCoord;

    uniform float uOpacity;
    uniform float uHDRMultiplier;

    void main() {
        // Pixel offset from splat center
        vec2 d = gl_FragCoord.xy - vCenterScreen;

        // Forward substitution: solve L · y = d
        float y0 = d.x / vL2D.x;  // L00
        float y1 = (d.y - vL2D.y * y0) / vL2D.z;  // L10, L11

        // Squared Mahalanobis distance
        float mahalSq = y0 * y0 + y1 * y1;

        // Generalized Gaussian falloff: exp(-½ · r^s) where r = ||y||
        // r^s = (r²)^(s/2) = mahalSq^(s/2)
        // Branchless: pow(x, 1.0) == x, so this handles s=2 correctly
        float rToTheS = pow(max(mahalSq, 1e-8), vSharpness * 0.5);
        float intensity = vAmplitude2D * exp(-0.5 * rToTheS);

        // Early discard for negligible contribution
        if (intensity < 1e-6) discard;

        // HDR color output
        vec3 finalColor = vColor * intensity * uHDRMultiplier;

        // Output for blending (additive/normal/max handled by WebGL blend equation)
        gl_FragColor = vec4(finalColor, intensity * uOpacity);
    }
  `;

  /**
   * Create a new GSplatMaterial with the specified configuration.
   *
   * @param materialConfig - Material configuration options
   */
  constructor(materialConfig: GSplatMaterialConfig = {}) {
    const blendingMode = materialConfig.blendingMode ?? 'additive';
    const blending =
      blendingMode === 'additive'
        ? THREE.AdditiveBlending
        : blendingMode === 'max'
          ? THREE.CustomBlending
          : THREE.NormalBlending;

    super({
      uniforms: {
        uResolution: { value: new THREE.Vector2(1, 1) },
        uFx: { value: 500 }, // Default focal length in pixels
        uFy: { value: 500 },
        uTruncate: { value: materialConfig.truncationRadius ?? 3.0 },
        uHDRMultiplier: {
          value: materialConfig.hdrMultiplier ?? config.shader.points.hdrMultiplier,
        },
        uOpacity: { value: materialConfig.opacity ?? 1.0 },
        uProjectionMode: { value: blendingMode === 'max' ? 1 : 0 }, // 0=sum, 1=max
      },

      vertexShader: GSplatMaterial.VERTEX_SHADER,
      fragmentShader: GSplatMaterial.FRAGMENT_SHADER,

      transparent: true,
      depthWrite: blendingMode !== 'additive' && blendingMode !== 'max', // No depth write for additive/max
      toneMapped: false, // HDR values pass through to post-processing
      blending: blending,
      side: THREE.DoubleSide, // Splats visible from both sides
    });

    // Configure custom blending for max mode
    if (blendingMode === 'max') {
      this.blendEquation = THREE.MaxEquation; // Max(source, destination)
      this.blendSrc = THREE.OneFactor;
      this.blendDst = THREE.OneFactor;
    }
  }

  /**
   * Update camera parameters for perspective projection.
   *
   * @param fov - Field of view in radians
   * @param resolution - Viewport resolution
   */
  updateCameraParams(fov: number, resolution: THREE.Vector2): void {
    this.uniforms.uResolution.value.copy(resolution);

    // Compute focal lengths in pixels from FOV
    // f = height / (2 * tan(fov/2)) for vertical FOV
    const tanHalfFov = Math.tan(fov / 2);
    const fy = resolution.y / (2 * tanHalfFov);
    // Assume square pixels (fx = fy based on aspect ratio)
    const fx = fy;

    this.uniforms.uFx.value = fx;
    this.uniforms.uFy.value = fy;
  }

  /**
   * Update HDR multiplier.
   */
  updateHDRMultiplier(multiplier: number): void {
    this.uniforms.uHDRMultiplier.value = multiplier;
  }

  /**
   * Update opacity.
   */
  updateOpacity(opacity: number): void {
    this.uniforms.uOpacity.value = opacity;
  }

  /**
   * Update truncation radius.
   */
  updateTruncationRadius(radius: number): void {
    this.uniforms.uTruncate.value = radius;
  }

  /**
   * Clone this material.
   */
  clone(): this {
    const cloned = new GSplatMaterial({
      opacity: this.uniforms.uOpacity.value,
      hdrMultiplier: this.uniforms.uHDRMultiplier.value,
      truncationRadius: this.uniforms.uTruncate.value,
      blendingMode:
        this.blending === THREE.AdditiveBlending
          ? 'additive'
          : this.blending === THREE.CustomBlending && this.blendEquation === THREE.MaxEquation
            ? 'max'
            : 'normal',
    });

    cloned.uniforms.uFx.value = this.uniforms.uFx.value;
    cloned.uniforms.uFy.value = this.uniforms.uFy.value;
    cloned.uniforms.uResolution.value.copy(this.uniforms.uResolution.value);
    cloned.uniforms.uProjectionMode.value = this.uniforms.uProjectionMode.value;

    return cloned as this;
  }

  /**
   * Dispose this material and unregister from MaterialManager.
   */
  dispose(): void {
    materialManager.unregister(this);
    super.dispose();
  }
}

// ============================================================================
// Instanced Geometry Creation
// ============================================================================

/**
 * Create the base quad geometry for gsplat instances.
 *
 * Each gsplat is rendered as a quad with 4 vertices:
 * - (-1, -1): Bottom-left
 * - ( 1, -1): Bottom-right
 * - (-1,  1): Top-left
 * - ( 1,  1): Top-right
 *
 * The vertex shader expands these based on the 2D covariance eigenvalues.
 *
 * @returns THREE.BufferGeometry for instanced rendering
 */
export function createGSplatQuadGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();

  // Quad corners for oriented quad expansion
  const quadCorners = new Float32Array([
    -1,
    -1, // Bottom-left
    1,
    -1, // Bottom-right
    -1,
    1, // Top-left
    1,
    1, // Top-right
  ]);

  // Triangle indices for the quad
  const indices = new Uint16Array([
    0,
    1,
    2, // First triangle
    2,
    1,
    3, // Second triangle
  ]);

  geometry.setAttribute('aQuadCorner', new THREE.BufferAttribute(quadCorners, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  return geometry;
}

/**
 * Configuration for instanced gsplats mesh
 */
export interface InstancedGSplatsMeshConfig {
  /** Splat centers (splatCount * 3) */
  centers: Float32Array;
  /** Packed Cholesky01 [L00, L10] (splatCount * 2) */
  cholesky01: Float32Array;
  /** Packed Cholesky23 [L11, L20] (splatCount * 2) */
  cholesky23: Float32Array;
  /** Packed Cholesky45 [L21, L22] (splatCount * 2) */
  cholesky45: Float32Array;
  /** Amplitudes (splatCount) */
  amplitudes: Float32Array;
  /** Sharpness values (splatCount) */
  sharpness: Float32Array;
  /** Colors RGB (splatCount * 3) */
  colors: Float32Array;
  /** Number of splats */
  splatCount: number;
}

/**
 * Pack 3D Cholesky factors from flat array into attribute format.
 *
 * Input: choleskyFactors with 6 elements per splat [L00, L10, L11, L20, L21, L22]
 * Output: Three arrays for shader attributes:
 * - cholesky01: [L00, L10] per splat
 * - cholesky23: [L11, L20] per splat
 * - cholesky45: [L21, L22] per splat
 *
 * @param choleskyFactors - Flat array of packed Cholesky factors (N * 6)
 * @param splatCount - Number of splats
 * @returns Object with three packed arrays for shader attributes
 */
export function packCholeskyForShader(
  choleskyFactors: Float32Array,
  splatCount: number
): { cholesky01: Float32Array; cholesky23: Float32Array; cholesky45: Float32Array } {
  const cholesky01 = new Float32Array(splatCount * 2);
  const cholesky23 = new Float32Array(splatCount * 2);
  const cholesky45 = new Float32Array(splatCount * 2);

  for (let i = 0; i < splatCount; i++) {
    const srcOffset = i * 6;
    const dstOffset = i * 2;

    // L00, L10
    cholesky01[dstOffset] = choleskyFactors[srcOffset];
    cholesky01[dstOffset + 1] = choleskyFactors[srcOffset + 1];

    // L11, L20
    cholesky23[dstOffset] = choleskyFactors[srcOffset + 2];
    cholesky23[dstOffset + 1] = choleskyFactors[srcOffset + 3];

    // L21, L22
    cholesky45[dstOffset] = choleskyFactors[srcOffset + 4];
    cholesky45[dstOffset + 1] = choleskyFactors[srcOffset + 5];
  }

  return { cholesky01, cholesky23, cholesky45 };
}

/**
 * Create an instanced mesh for gsplats rendering.
 *
 * Sets up the instanced geometry with all per-splat attributes.
 *
 * Note: We use THREE.Mesh instead of THREE.InstancedMesh because:
 * - InstancedMesh adds instanceMatrix (mat4 = 4 attribute locations)
 * - Our shader computes positions from custom attributes, not matrices
 * - This avoids exceeding WebGL's 16 attribute location limit
 * - InstancedBufferGeometry with Mesh still uses instanced drawing
 *
 * @param meshConfig - Configuration with all splat data
 * @param material - GSplatMaterial to use for rendering
 * @returns THREE.Mesh with InstancedBufferGeometry ready for scene addition
 */
export function createInstancedGSplatsMesh(
  meshConfig: InstancedGSplatsMeshConfig,
  material: GSplatMaterial
): THREE.Mesh {
  const baseGeometry = createGSplatQuadGeometry();

  // Create instanced buffer geometry
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = baseGeometry.index;
  geometry.setAttribute('aQuadCorner', baseGeometry.getAttribute('aQuadCorner'));

  // Set instanced attributes
  geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(meshConfig.centers, 3));
  geometry.setAttribute(
    'aCholesky01',
    new THREE.InstancedBufferAttribute(meshConfig.cholesky01, 2)
  );
  geometry.setAttribute(
    'aCholesky23',
    new THREE.InstancedBufferAttribute(meshConfig.cholesky23, 2)
  );
  geometry.setAttribute(
    'aCholesky45',
    new THREE.InstancedBufferAttribute(meshConfig.cholesky45, 2)
  );
  geometry.setAttribute('aAmplitude', new THREE.InstancedBufferAttribute(meshConfig.amplitudes, 1));
  geometry.setAttribute('aSharpness', new THREE.InstancedBufferAttribute(meshConfig.sharpness, 1));
  geometry.setAttribute('aColor', new THREE.InstancedBufferAttribute(meshConfig.colors, 3));

  // Set instance count
  geometry.instanceCount = meshConfig.splatCount;

  // Compute bounding box from centers
  const tempGeometry = new THREE.BufferGeometry();
  tempGeometry.setAttribute('position', new THREE.BufferAttribute(meshConfig.centers, 3));
  tempGeometry.computeBoundingBox();
  tempGeometry.computeBoundingSphere();

  if (tempGeometry.boundingBox) {
    geometry.boundingBox = tempGeometry.boundingBox.clone();
  }
  if (tempGeometry.boundingSphere) {
    geometry.boundingSphere = tempGeometry.boundingSphere.clone();
  }

  tempGeometry.dispose();

  // Create mesh with instanced geometry
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = true;

  return mesh;
}

/**
 * Update an existing gsplats mesh with new data.
 *
 * This efficiently updates the instanced attributes without recreating the geometry.
 *
 * @param mesh - Existing gsplats mesh to update
 * @param meshConfig - New splat data
 */
export function updateInstancedGSplatsMesh(
  mesh: THREE.Mesh,
  meshConfig: InstancedGSplatsMeshConfig
): void {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;

  // Update or recreate attributes based on size change
  const currentCount = geometry.instanceCount;

  if (meshConfig.splatCount !== currentCount) {
    // Size changed, recreate attributes
    geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(meshConfig.centers, 3));
    geometry.setAttribute(
      'aCholesky01',
      new THREE.InstancedBufferAttribute(meshConfig.cholesky01, 2)
    );
    geometry.setAttribute(
      'aCholesky23',
      new THREE.InstancedBufferAttribute(meshConfig.cholesky23, 2)
    );
    geometry.setAttribute(
      'aCholesky45',
      new THREE.InstancedBufferAttribute(meshConfig.cholesky45, 2)
    );
    geometry.setAttribute(
      'aAmplitude',
      new THREE.InstancedBufferAttribute(meshConfig.amplitudes, 1)
    );
    geometry.setAttribute(
      'aSharpness',
      new THREE.InstancedBufferAttribute(meshConfig.sharpness, 1)
    );
    geometry.setAttribute('aColor', new THREE.InstancedBufferAttribute(meshConfig.colors, 3));
    geometry.instanceCount = meshConfig.splatCount;
  } else {
    // Same size, update in place
    const centerAttr = geometry.getAttribute('aCenter') as THREE.InstancedBufferAttribute;
    const chol01Attr = geometry.getAttribute('aCholesky01') as THREE.InstancedBufferAttribute;
    const chol23Attr = geometry.getAttribute('aCholesky23') as THREE.InstancedBufferAttribute;
    const chol45Attr = geometry.getAttribute('aCholesky45') as THREE.InstancedBufferAttribute;
    const ampAttr = geometry.getAttribute('aAmplitude') as THREE.InstancedBufferAttribute;
    const sharpAttr = geometry.getAttribute('aSharpness') as THREE.InstancedBufferAttribute;
    const colorAttr = geometry.getAttribute('aColor') as THREE.InstancedBufferAttribute;

    centerAttr.set(meshConfig.centers);
    chol01Attr.set(meshConfig.cholesky01);
    chol23Attr.set(meshConfig.cholesky23);
    chol45Attr.set(meshConfig.cholesky45);
    ampAttr.set(meshConfig.amplitudes);
    sharpAttr.set(meshConfig.sharpness);
    colorAttr.set(meshConfig.colors);

    centerAttr.needsUpdate = true;
    chol01Attr.needsUpdate = true;
    chol23Attr.needsUpdate = true;
    chol45Attr.needsUpdate = true;
    ampAttr.needsUpdate = true;
    sharpAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;
  }

  // Update bounding box
  const tempGeometry = new THREE.BufferGeometry();
  tempGeometry.setAttribute('position', new THREE.BufferAttribute(meshConfig.centers, 3));
  tempGeometry.computeBoundingBox();
  tempGeometry.computeBoundingSphere();

  if (tempGeometry.boundingBox) {
    geometry.boundingBox = tempGeometry.boundingBox.clone();
  }
  if (tempGeometry.boundingSphere) {
    geometry.boundingSphere = tempGeometry.boundingSphere.clone();
  }

  tempGeometry.dispose();
}
