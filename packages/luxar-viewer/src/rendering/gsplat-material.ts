/**
 * GSplat Material for Luxar
 *
 * Specialized THREE.ShaderMaterial for rendering Gaussian splats using instanced quads.
 * Implements projection-aware volumetric rendering with oriented, anisotropic Gaussian density functions.
 *
 * Key features:
 * - Instanced oriented quad geometry (4 vertices per splat)
 * - Full 3D covariance via Cholesky factors
 * - Perspective-correct projection of covariance to 2D
 * - Blending-mode-aware projection (sum for additive/normal, max for max blending)
 * - Standard Gaussian falloff: exp(-½ · r²)
 * - Per-splat attributes (center, cholesky, amplitude, color)
 *
 * GPU Optimizations (GLSL ES 3.0 / WebGL2):
 * - flat interpolation: skips GPU interpolation for per-instance varyings
 * - Reciprocal precomputation: DIV→MUL in vertex and fragment shaders
 * - Early discard at 3σ before expensive exp()
 * - Higher intensity threshold (1e-4) for fewer blended pixels
 * - mediump precision for color/amplitude to reduce register pressure
 *
 * Mathematical basis:
 * - GSplat density: G(x) = a · exp(-½ · ‖L⁻¹(x - μ)‖²)
 * - Sum projection: amplitude boost by σ_ray · c_s (ray integration)
 * - Max projection: amplitude = a (peak value, no integration)
 * - 2D covariance: Σ_2D = J · Σ_cam · Jᵀ (perspective Jacobian projection)
 *
 * @module rendering/gsplat-material
 */

import * as THREE from 'three';
import { materialManager } from './material-manager';

/**
 * Configuration for gsplat material creation
 */
export interface GSplatMaterialConfig {
  /** Opacity multiplier (0.0 to 1.0) */
  opacity?: number;
  /** Gamma correction (0.1 to 10.0, default 1.0) */
  gamma?: number;
  /** Intensity (linear color multiplier / gain), default 1.0 */
  intensity?: number;
  /** Offset (additive brightness shift / black level), default 0.0 */
  offset?: number;
  /** Truncation radius in sigmas (default 3.0) */
  truncationRadius?: number;
  /** Blending mode */
  blendingMode?: 'additive' | 'normal' | 'max' | 'opaque' | 'luminous';
  /** Whether material is transparent (default true) */
  transparent?: boolean;
  /** Whether to test against depth buffer (default true; additive sets false) */
  depthTest?: boolean;
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
  /** Opacity multiplier */
  uOpacity: { value: number };
  /** Projection mode: 0=sum (additive/normal), 1=max (max blending) */
  uProjectionMode: { value: number };
  /** Pre-computed 1/gamma for performance */
  uInvGamma: { value: number };
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
   * - aColor: RGB color
   *
   * The shader:
   * 1. Transforms center and Cholesky to camera space
   * 2. Projects 3D covariance to 2D using perspective Jacobian
   * 3. Computes ray variance for amplitude boost
   * 4. Expands oriented quad based on 2D covariance eigenvalues
   *
   * OPTIMIZATION: Uses GLSL ES 3.0 with flat qualifier for per-instance varyings.
   * All varyings use flat to skip GPU interpolation hardware.
   */
  private static readonly VERTEX_SHADER = /* glsl */ `
    precision highp float;

    // Quad corner attribute (static geometry)
    in vec2 aQuadCorner;  // (-1,-1), (1,-1), (-1,1), (1,1)

    // Per-instance attributes
    in vec3 aCenter;           // 3D center (after nD slicing)
    in vec2 aCholesky01;       // [L00, L10]
    in vec2 aCholesky23;       // [L11, L20]
    in vec2 aCholesky45;       // [L21, L22]
    in float aAmplitude;       // Already attenuated by hidden dims
    in vec3 aColor;

    // Uniforms (modelViewMatrix and projectionMatrix are built-in THREE.js uniforms)
    uniform vec2 uResolution;
    uniform float uFx, uFy;           // Focal lengths in pixels
    uniform float uTruncate;          // Truncation radius (in sigmas)
    uniform int uProjectionMode;      // 0 = sum projection (additive), 1 = max projection (max blending)
    uniform int uIsOrtho;             // 0 = perspective, 1 = orthographic

    // Varyings to fragment - all per-instance varyings use "flat" (no interpolation needed)
    // OPTIMIZATION: flat qualifier skips GPU interpolation hardware for constant values
    flat out mediump vec3 vColor;
    flat out mediump float vAmplitude2D;
    // These need highp for screen-space calculations
    // OPTIMIZATION: vL2D stores [1/L00, L10, 1/L11] to replace fragment divisions with multiplications
    flat out highp vec3 vL2D;                // 2D Cholesky packed as [invL00, L10, invL11]
    flat out highp vec2 vCenterScreen;       // Splat center in screen pixels
    flat out highp float vAspectRatio;       // Ray elongation ratio for projection correction
    flat out int vProjectionMode;            // 0=sum (additive), 1=max

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
    // OPTIMIZATION: Returns [1/L00, L10, 1/L11] for faster fragment shader (MUL instead of DIV)
    vec3 cholesky2x2(mat2 S) {
        float L00 = sqrt(max(S[0][0], 1e-8));
        float invL00 = 1.0 / L00;
        float L10 = S[1][0] * invL00;  // Use reciprocal here too
        float L11 = sqrt(max(S[1][1] - L10 * L10, 1e-8));
        float invL11 = 1.0 / L11;
        return vec3(invL00, L10, invL11);  // Pack reciprocals for fragment shader
    }

    void main() {
        // Transform center to camera space
        vec4 centerCam4 = modelViewMatrix * vec4(aCenter, 1.0);
        vec3 centerCam = centerCam4.xyz;

        // Reject splats behind the camera (camera looks down -Z axis)
        // centerCam.z >= 0 means the splat is on the +Z side (behind camera)
        if (centerCam.z >= 0.0) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);  // Behind camera
            return;
        }

        // Transform Cholesky to camera space (rotation only)
        mat3 R = mat3(modelViewMatrix);
        mat3 L3D = unpackCholesky3D();
        mat3 L_cam = R * L3D;
        mat3 Sigma_cam = L_cam * transpose(L_cam);

        // Two-stage near-plane culling for performance
        // Stage 1: Fixed threshold catches most cases (fast path: 1 cycle)
        if (-centerCam.z < 0.1) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);  // Too close to camera
            return;
        }

        // Stage 2: Adaptive threshold for large splats only (slow path: ~6 cycles)
        // Only execute for splats with sigma > 0.1 (rare cases)
        float sigmaTraceSq = Sigma_cam[0][0] + Sigma_cam[1][1] + Sigma_cam[2][2];
        if (sigmaTraceSq > 0.01) {  // sigma > 0.1
            float sigmaTrace = sqrt(sigmaTraceSq);
            if (-centerCam.z < sigmaTrace * uTruncate) {
                gl_Position = vec4(0.0, 0.0, -2.0, 1.0);  // Large splat too close
                return;
            }
        }

        // Precompute depth reciprocals (used by perspective Jacobian and screen projection)
        float z = -centerCam.z;  // Positive depth (camera looks down -Z)
        float invZ = 1.0 / z;
        float invZ2 = invZ * invZ;

        // Projection Jacobian at splat center
        mat3x2 J;
        if (uIsOrtho == 1) {
            // Orthographic: no depth dependence (parallel projection)
            J[0] = vec2(uFx, 0.0);
            J[1] = vec2(0.0, uFy);
            J[2] = vec2(0.0, 0.0);
        } else {
            // Perspective projection Jacobian
            // For camera looking down -Z, with z = -centerCam.z (positive depth):
            // x_s = fx * centerCam.x / z, y_s = fy * centerCam.y / z
            // ∂x_s/∂(centerCam.z) = fx * centerCam.x / z² (since z = -centerCam.z)
            J[0] = vec2(uFx * invZ, 0.0);
            J[1] = vec2(0.0, uFy * invZ);
            J[2] = vec2(uFx * centerCam.x * invZ2, uFy * centerCam.y * invZ2);
        }

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

        // Projection mode determines amplitude calculation:
        // - Sum projection (uProjectionMode = 0): Integrate Gaussian along ray → ray boost
        // - Max projection (uProjectionMode = 1): Use peak Gaussian value → no boost
        //
        // OPTIMIZATION: Use branch instead of branchless mix() to skip expensive operations
        // (normalize, sqrt, exp) when in max mode. Warps are typically coherent on this uniform.
        float sigmaRay = 1.0;  // Default for max mode (no ray integration)
        if (uProjectionMode == 0) {
            // Sum projection: compute ray integration boost
            vec3 rayDir = (uIsOrtho == 1) ? vec3(0.0, 0.0, -1.0) : normalize(centerCam);
            float sigmaRaySq = dot(rayDir, Sigma_cam * rayDir);
            sigmaRay = sqrt(max(sigmaRaySq, 1e-8));
            // c_s = sharpnessIntegralFactor(2.0) ≈ sqrt(2π) ≈ 2.507 (standard Gaussian ray integral)
            float rayIntegrationBoost = sigmaRay * 2.507;  // voxelSpacing = 1.0
            vAmplitude2D = aAmplitude * rayIntegrationBoost;
        } else {
            // Max projection: no boost needed
            vAmplitude2D = aAmplitude;
        }

        // Compute 2D Cholesky for fragment shader
        vL2D = cholesky2x2(Sigma2D);

        // Eigenvalues of Σ_2D for quad extents (oriented quad)
        float trace = Sigma2D[0][0] + Sigma2D[1][1];
        float det = Sigma2D[0][0] * Sigma2D[1][1] - Sigma2D[0][1] * Sigma2D[1][0];
        float disc = max(trace * trace - 4.0 * det, 0.0);  // Clamp for numerical stability
        float sqrtDisc = sqrt(disc);
        float lambda1 = max(0.5 * (trace + sqrtDisc), 1e-6);
        float lambda2 = max(0.5 * (trace - sqrtDisc), 1e-6);

        // Compute aspect ratio for projection correction factor
        // aspectRatio = sigmaRay / sigma2D_avg measures elongation along viewing direction
        // For sum projection: used to correct non-separability of generalized Gaussian integral
        // For max projection: not used (set to 1.0 for safety)
        float sigma2D_avg = sqrt(0.5 * (lambda1 + lambda2));
        vAspectRatio = (uProjectionMode == 0) ? sigmaRay / max(sigma2D_avg, 1e-8) : 1.0;
        vProjectionMode = uProjectionMode;

        // Eigenvector for major axis (for oriented quad)
        vec2 majorAxis;
        if (abs(Sigma2D[0][1]) > 1e-6) {
            majorAxis = normalize(vec2(lambda1 - Sigma2D[1][1], Sigma2D[0][1]));
        } else {
            majorAxis = vec2(1.0, 0.0);
        }
        vec2 minorAxis = vec2(-majorAxis.y, majorAxis.x);

        // Quad extents: truncation radius × sqrt(eigenvalue)
        // Standard Gaussian (sharpness=2): effectiveTruncate = uTruncate
        float extent1 = uTruncate * sqrt(lambda1);
        float extent2 = uTruncate * sqrt(lambda2);

        // Project center to screen (pixels)
        if (uIsOrtho == 1) {
            // Orthographic: direct linear mapping (no depth division)
            vCenterScreen = vec2(
                uFx * centerCam.x + uResolution.x * 0.5,
                uFy * centerCam.y + uResolution.y * 0.5
            );
        } else {
            // Perspective: reuse invZ from earlier computation
            vCenterScreen = vec2(
                uFx * centerCam.x * invZ + uResolution.x * 0.5,
                uFy * centerCam.y * invZ + uResolution.y * 0.5
            );
        }

        // Expand quad vertex in screen space (oriented)
        vec2 quadOffset = aQuadCorner.x * majorAxis * extent1
                        + aQuadCorner.y * minorAxis * extent2;
        vec2 screenPos = vCenterScreen + quadOffset;

        // Convert screen pixels to NDC (xy only)
        vec2 ndcXY = (screenPos / uResolution) * 2.0 - 1.0;

        // Pass through other varyings
        vColor = aColor;

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
   *
   * Optimizations:
   * - GLSL ES 3.0 with flat qualifier: skips GPU interpolation for per-instance values
   * - Reciprocal precomputation: 2 divisions replaced with 2 multiplications
   * - Standard Gaussian: no pow() needed
   * - mediump precision for color/amplitude (sufficient for visual quality)
   * - Early discard at 3σ before exp()
   * - Higher discard threshold (1e-4 is still invisible)
   */
  private static readonly FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    // All varyings use flat - no interpolation needed (constant per instance)
    // OPTIMIZATION: flat qualifier skips GPU interpolation hardware
    flat in mediump vec3 vColor;
    flat in mediump float vAmplitude2D;
    // These need highp for screen-space calculations
    // OPTIMIZATION: vL2D stores [1/L00, L10, 1/L11] for MUL instead of DIV
    flat in highp vec3 vL2D;          // 2D Cholesky packed as [invL00, L10, invL11]
    flat in highp vec2 vCenterScreen;
    flat in highp float vAspectRatio; // Ray elongation ratio for projection correction
    flat in int vProjectionMode;      // 0=sum (additive), 1=max

    uniform mediump float uOpacity;
    uniform mediump float uInvGamma; // Pre-computed 1/gamma for performance
    uniform mediump float uIntensity; // Per-node linear color multiplier (gain)
    uniform mediump float uOffset; // Per-node additive brightness shift (black level)

    // GLSL ES 3.0 requires explicit fragment output declaration
    out vec4 fragColor;

    void main() {
        // Pixel offset from splat center
        vec2 d = gl_FragCoord.xy - vCenterScreen;

        // Forward substitution: solve L · y = d
        // OPTIMIZATION: vL2D contains [invL00, L10, invL11] - use MUL instead of DIV
        float y0 = d.x * vL2D.x;  // d.x * invL00
        float y1 = (d.y - vL2D.y * y0) * vL2D.z;  // (d.y - L10 * y0) * invL11

        // Squared Mahalanobis distance
        float mahalSq = y0 * y0 + y1 * y1;

        // EARLY DISCARD: Skip pixels beyond ~3σ before expensive pow()
        // At mahalSq=9 (3σ), Gaussian value is exp(-4.5) ≈ 0.011, negligible
        // This saves the expensive pow() and exp() for edge pixels
        if (mahalSq > 9.0) discard;

        // Standard Gaussian falloff: exp(-½ · r²)
        // Same formula for both sum and max projection modes
        float intensity = vAmplitude2D * exp(-0.5 * mahalSq);

        // Early discard for negligible contribution (raised threshold for performance)
        if (intensity < 1e-4) discard;

        // Per-node GOG (Gain-Offset-Gamma) color adjustment
        vec3 adjusted = vColor * uIntensity + uOffset;
        adjusted = max(adjusted, vec3(0.0));

        // Early discard for zero-contribution fragments after offset
        if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;

        vec3 gammaColor = pow(adjusted, vec3(uInvGamma));

        // HDR color output for linear additive blending
        // With OneFactor blending (additive/luminous/max modes), alpha is ignored,
        // so apply opacity to RGB directly. This gives correct LINEAR sum projection
        // without the intensity-squaring bug that AdditiveBlending (SrcAlpha) would cause.
        //
        // NOTE: For 'normal' blending mode, this shader outputs alpha=1.0, which means
        // the background won't show through (effectively opaque). This is a known
        // limitation - proper transparent normal blending for gsplats would require
        // premultiplied alpha with ONE, ONE_MINUS_SRC_ALPHA blend func.
        vec3 finalColor = gammaColor * intensity * uOpacity;
        fragColor = vec4(finalColor, 1.0);
    }
  `;

  /**
   * Create a new GSplatMaterial with the specified configuration.
   *
   * @param materialConfig - Material configuration options
   */
  constructor(materialConfig: GSplatMaterialConfig = {}) {
    const blendingMode = materialConfig.blendingMode ?? 'additive';
    const isOpaque = blendingMode === 'opaque';
    const isAdditive = blendingMode === 'additive';
    const gammaValue = Math.max(0.001, materialConfig.gamma ?? 1.0); // Prevent division by zero

    // Determine THREE.js blending mode
    // CRITICAL: For sum projection, we need LINEAR addition of intensities.
    // THREE.AdditiveBlending uses SrcAlpha which SQUARES the intensity - WRONG!
    // We use CustomBlending with OneFactor for correct linear sum projection.
    let blending: THREE.Blending;
    if (isOpaque || blendingMode === 'normal') {
      blending = THREE.NormalBlending;
    } else if (
      blendingMode === 'additive' ||
      blendingMode === 'luminous' ||
      blendingMode === 'max'
    ) {
      // All additive-style modes use CustomBlending for correct linear contribution
      blending = THREE.CustomBlending;
    } else {
      blending = THREE.NormalBlending;
    }

    super({
      uniforms: {
        uResolution: { value: new THREE.Vector2(1, 1) },
        uFx: { value: 500 }, // Default focal length in pixels
        uFy: { value: 500 },
        uTruncate: { value: materialConfig.truncationRadius ?? 3.0 },
        uOpacity: { value: materialConfig.opacity ?? 1.0 },
        uProjectionMode: { value: blendingMode === 'max' ? 1 : 0 }, // 0=sum, 1=max
        uInvGamma: { value: 1.0 / gammaValue }, // Pre-computed inverse for performance
        uIntensity: { value: materialConfig.intensity ?? 1.0 },
        uOffset: { value: materialConfig.offset ?? 0.0 },
        uIsOrtho: { value: 0 }, // 0 = perspective, 1 = orthographic
      },

      vertexShader: GSplatMaterial.VERTEX_SHADER,
      fragmentShader: GSplatMaterial.FRAGMENT_SHADER,

      // GLSL ES 3.0 for flat interpolation and modern syntax
      glslVersion: THREE.GLSL3,

      transparent: materialConfig.transparent ?? !isOpaque,
      depthWrite:
        isOpaque || (blendingMode === 'normal' && (materialConfig.opacity ?? 1.0) >= 0.99),
      // Additive ignores depth (renders on top), luminous respects depth occlusion
      depthTest: materialConfig.depthTest ?? !isAdditive,
      toneMapped: false, // HDR values pass through to post-processing
      blending: blending,
      side: THREE.DoubleSide, // Splats visible from both sides
    });

    // Configure custom blending for additive-style modes
    // CRITICAL: Use OneFactor to avoid squaring intensity (SrcAlpha would square it)
    if (blendingMode === 'additive' || blendingMode === 'luminous') {
      // Linear additive: final = src + dst (no alpha multiplication)
      this.blendEquation = THREE.AddEquation;
      this.blendSrc = THREE.OneFactor;
      this.blendDst = THREE.OneFactor;
      // Prevent alpha accumulation that causes bloom/postprocessing artifacts.
      // With AddEquation, alpha would sum: 1.0 + 1.0 + ... = N per overlapping splat,
      // overflowing HalfFloat16 and causing dark halos via premultipliedAlpha compositing.
      // MaxEquation keeps alpha = max(1.0, existing) = 1.0, preventing accumulation.
      this.blendEquationAlpha = THREE.MaxEquation;
      this.blendSrcAlpha = THREE.OneFactor;
      this.blendDstAlpha = THREE.OneFactor;
    } else if (blendingMode === 'max') {
      this.blendEquation = THREE.MaxEquation; // Max(source, destination)
      this.blendSrc = THREE.OneFactor;
      this.blendDst = THREE.OneFactor;
    }

    // Store blendingMode and gamma in userData for clone()
    this.userData.blendingMode = blendingMode;
    this.userData.gamma = gammaValue;
    this.userData.depthTest = materialConfig.depthTest ?? !isAdditive;
  }

  /**
   * Update camera parameters for perspective projection.
   *
   * @param fov - Field of view in radians
   * @param resolution - Viewport resolution
   */
  updateCameraParams(fov: number, resolution: THREE.Vector2, isOrtho: boolean = false): void {
    this.uniforms.uResolution.value.copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;

    if (isOrtho) {
      // fov = frustumHeight in world units; direct linear mapping
      const fy = resolution.y / fov;
      this.uniforms.uFx.value = fy;
      this.uniforms.uFy.value = fy;
    } else {
      // Compute focal lengths in pixels from FOV
      // f = height / (2 * tan(fov/2)) for vertical FOV
      const tanHalfFov = Math.tan(fov / 2);
      const fy = resolution.y / (2 * tanHalfFov);
      this.uniforms.uFx.value = fy;
      this.uniforms.uFy.value = fy;
    }
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
   * Update gamma correction.
   * Only invGamma is used in shader; gamma value stored in userData for clone()
   */
  updateGamma(gamma: number): void {
    const safeGamma = Math.max(0.001, gamma); // Prevent division by zero
    this.userData.gamma = safeGamma;
    this.uniforms.uInvGamma.value = 1.0 / safeGamma;
  }

  /**
   * Update intensity (linear color multiplier)
   */
  updateIntensity(intensity: number): void {
    this.uniforms.uIntensity.value = intensity;
  }

  /**
   * Update offset (additive brightness shift)
   */
  updateOffset(offset: number): void {
    this.uniforms.uOffset.value = offset;
  }

  /**
   * Clone this material.
   */
  clone(): this {
    const cloned = new GSplatMaterial({
      opacity: this.uniforms.uOpacity.value,
      gamma: this.userData.gamma ?? 1.0,
      intensity: this.uniforms.uIntensity.value,
      offset: this.uniforms.uOffset.value,
      truncationRadius: this.uniforms.uTruncate.value,
      blendingMode: this.userData.blendingMode ?? 'additive',
      transparent: this.transparent,
      depthTest: this.userData.depthTest ?? true,
    });

    // Copy blend equation settings for custom blending (additive/luminous/max modes)
    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
      cloned.blendEquationAlpha = this.blendEquationAlpha;
      cloned.blendSrcAlpha = this.blendSrcAlpha;
      cloned.blendDstAlpha = this.blendDstAlpha;
    }

    cloned.uniforms.uFx.value = this.uniforms.uFx.value;
    cloned.uniforms.uFy.value = this.uniforms.uFy.value;
    cloned.uniforms.uResolution.value.copy(this.uniforms.uResolution.value);
    cloned.uniforms.uProjectionMode.value = this.uniforms.uProjectionMode.value;
    cloned.uniforms.uInvGamma.value = this.uniforms.uInvGamma.value;

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
    geometry.setAttribute('aColor', new THREE.InstancedBufferAttribute(meshConfig.colors, 3));
    geometry.instanceCount = meshConfig.splatCount;

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount from the new attributes.
    // When a mesh is initially created with 0 instances (e.g., a gsplat node not at the
    // current time slice), THREE.js caches _maxInstanceCount=0. Later updates that add
    // instances via setAttribute won't trigger recalculation, so the renderer still draws
    // min(instanceCount, 0) = 0 instances. Deleting the cached value forces recalculation
    // on the next render frame. (THREE.js r163+ internal property)

    delete (geometry as any)._maxInstanceCount;
  } else {
    // Same size, update in place
    const centerAttr = geometry.getAttribute('aCenter') as THREE.InstancedBufferAttribute;
    const chol01Attr = geometry.getAttribute('aCholesky01') as THREE.InstancedBufferAttribute;
    const chol23Attr = geometry.getAttribute('aCholesky23') as THREE.InstancedBufferAttribute;
    const chol45Attr = geometry.getAttribute('aCholesky45') as THREE.InstancedBufferAttribute;
    const ampAttr = geometry.getAttribute('aAmplitude') as THREE.InstancedBufferAttribute;
    const colorAttr = geometry.getAttribute('aColor') as THREE.InstancedBufferAttribute;

    centerAttr.set(meshConfig.centers);
    chol01Attr.set(meshConfig.cholesky01);
    chol23Attr.set(meshConfig.cholesky23);
    chol45Attr.set(meshConfig.cholesky45);
    ampAttr.set(meshConfig.amplitudes);
    colorAttr.set(meshConfig.colors);

    centerAttr.needsUpdate = true;
    chol01Attr.needsUpdate = true;
    chol23Attr.needsUpdate = true;
    chol45Attr.needsUpdate = true;
    ampAttr.needsUpdate = true;
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
