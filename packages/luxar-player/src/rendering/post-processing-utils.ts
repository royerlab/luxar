/**
 * Pure utility functions for post-processing calculations and validations
 *
 * This module contains side-effect-free functions extracted from post-processing.ts
 * to improve testability and maintainability. These functions handle parameter
 * validation, quality calculations, and effect configuration without external
 * dependencies.
 */

/**
 * SSAA (Super-Sample Anti-Aliasing) configuration
 */
export interface SSAAConfig {
  enabled: boolean;
  multiplier: number;
}

/**
 * SMAA (Subpixel Morphological Anti-Aliasing) configuration
 */
export interface SMAAConfig {
  enabled: boolean;
  threshold: number;
  searchSteps: number;
}

/**
 * Bloom effect configuration
 */
export interface BloomConfig {
  enabled: boolean;
  intensity: number;
  threshold: number;
  radius: number;
}

/**
 * Tone mapping configuration
 */
export interface ToneMappingConfig {
  type: 'none' | 'linear' | 'reinhard' | 'cineon' | 'aces' | 'agx' | 'neutral';
  exposure: number;
}

/**
 * Anti-aliasing quality preset
 */
export type AAQuality = 'none' | 'low' | 'medium' | 'high' | 'ultra';

/**
 * Validates and clamps SSAA multiplier to valid range
 *
 * @param multiplier - Requested SSAA multiplier
 * @returns Clamped multiplier value between 1.0 and 4.0
 */
export function validateSSAAMultiplier(multiplier: number): number {
  return Math.max(1.0, Math.min(4.0, multiplier));
}

/**
 * Validates and clamps SMAA threshold to valid range
 *
 * @param threshold - Edge detection threshold
 * @returns Clamped threshold value between 0.05 and 0.2
 */
export function validateSMAAThreshold(threshold: number): number {
  return Math.max(0.05, Math.min(0.2, threshold));
}

/**
 * Validates SMAA search steps to supported values
 *
 * @param steps - Requested search steps
 * @returns Valid search steps (4, 8, 16, or 32)
 */
export function validateSMAASearchSteps(steps: number): number {
  const validSteps = [4, 8, 16, 32];

  // Find closest valid value
  if (validSteps.includes(steps)) {
    return steps;
  }

  // Return closest valid step count
  if (steps < 4) return 4;
  if (steps < 8) return 8;
  if (steps < 16) return 16;
  if (steps < 32) return 16;
  return 32;
}

/**
 * Calculates effective resolution based on SSAA settings
 *
 * @param baseWidth - Base render width
 * @param baseHeight - Base render height
 * @param ssaaConfig - SSAA configuration
 * @returns Effective render resolution
 */
export function calculateSSAAResolution(
  baseWidth: number,
  baseHeight: number,
  ssaaConfig: SSAAConfig
): { width: number; height: number } {
  if (!ssaaConfig.enabled) {
    return { width: baseWidth, height: baseHeight };
  }

  const multiplier = validateSSAAMultiplier(ssaaConfig.multiplier);
  return {
    width: Math.round(baseWidth * multiplier),
    height: Math.round(baseHeight * multiplier),
  };
}

/**
 * Determines MSAA sample count based on quality preset and device capabilities
 *
 * @param quality - Anti-aliasing quality preset
 * @param maxSamples - Maximum samples supported by device
 * @returns MSAA sample count
 */
export function calculateMSAASamples(quality: AAQuality, maxSamples: number = 16): number {
  switch (quality) {
    case 'none':
      return 0;
    case 'low':
      return Math.min(2, maxSamples);
    case 'medium':
      return Math.min(4, maxSamples);
    case 'high':
      return Math.min(8, maxSamples);
    case 'ultra':
      return Math.min(16, maxSamples);
    default:
      return 0;
  }
}

/**
 * Validates bloom parameters
 *
 * @param config - Bloom configuration
 * @returns Validated bloom configuration
 */
export function validateBloomConfig(config: Partial<BloomConfig>): BloomConfig {
  return {
    enabled: config.enabled ?? true,
    intensity: Math.max(0, Math.min(5, config.intensity ?? 1.0)),
    threshold: Math.max(0, Math.min(2, config.threshold ?? 0.9)),
    radius: Math.max(0, Math.min(2, config.radius ?? 0.4)),
  };
}

/**
 * Calculates bloom resolution based on quality settings
 *
 * @param baseWidth - Base render width
 * @param baseHeight - Base render height
 * @param quality - Quality level (0-1)
 * @returns Bloom buffer resolution
 */
export function calculateBloomResolution(
  baseWidth: number,
  baseHeight: number,
  quality: number = 0.5
): { width: number; height: number } {
  // Bloom can use lower resolution for performance
  const scale = 0.25 + 0.75 * quality; // 0.25 to 1.0 scale

  return {
    width: Math.max(256, Math.round(baseWidth * scale)),
    height: Math.max(256, Math.round(baseHeight * scale)),
  };
}

/**
 * Validates tone mapping configuration
 *
 * @param config - Tone mapping configuration
 * @returns Validated configuration
 */
export function validateToneMappingConfig(config: Partial<ToneMappingConfig>): ToneMappingConfig {
  const validTypes: ToneMappingConfig['type'][] = [
    'none',
    'linear',
    'reinhard',
    'cineon',
    'aces',
    'agx',
    'neutral',
  ];

  return {
    type: validTypes.includes(config.type as any) ? config.type! : 'aces',
    exposure: Math.max(0.1, Math.min(10, config.exposure ?? 1.0)),
  };
}

/**
 * Determines if HDR pipeline should be enabled based on configuration
 *
 * @param hasHDRContent - Whether scene contains HDR content
 * @param bloomEnabled - Whether bloom is enabled
 * @param toneMappingType - Tone mapping type
 * @returns Whether HDR pipeline should be used
 */
export function shouldUseHDRPipeline(
  hasHDRContent: boolean,
  bloomEnabled: boolean,
  toneMappingType: string
): boolean {
  // Use HDR pipeline if:
  // 1. Scene has HDR content
  // 2. Bloom is enabled (needs HDR values for proper threshold)
  // 3. Advanced tone mapping is used
  return (
    hasHDRContent || bloomEnabled || ['aces', 'agx', 'reinhard', 'cineon'].includes(toneMappingType)
  );
}

/**
 * Calculates render target format based on HDR requirements
 *
 * @param hdrEnabled - Whether HDR pipeline is enabled
 * @returns WebGL render target type constant
 */
export function getRenderTargetType(hdrEnabled: boolean): number {
  // Return THREE.js constants (we use numbers to avoid importing THREE)
  // THREE.UnsignedByteType = 1009
  // THREE.HalfFloatType = 1016
  return hdrEnabled ? 1016 : 1009;
}

/**
 * Estimates GPU memory usage for post-processing pipeline
 *
 * @param width - Render width
 * @param height - Render height
 * @param msaaSamples - MSAA sample count
 * @param ssaaMultiplier - SSAA multiplier
 * @param hdrEnabled - Whether HDR is enabled
 * @returns Estimated memory usage in MB
 */
export function estimatePostProcessingMemory(
  width: number,
  height: number,
  msaaSamples: number = 0,
  ssaaMultiplier: number = 1.0,
  hdrEnabled: boolean = false
): number {
  // Calculate effective resolution
  const effectiveWidth = width * ssaaMultiplier;
  const effectiveHeight = height * ssaaMultiplier;
  const pixels = effectiveWidth * effectiveHeight;

  // Bytes per pixel (RGBA)
  const bytesPerPixel = hdrEnabled ? 8 : 4; // Half float vs unsigned byte

  // Base render target
  let totalBytes = pixels * bytesPerPixel;

  // MSAA multiplies memory usage
  if (msaaSamples > 0) {
    totalBytes *= msaaSamples;
  }

  // Additional buffers for effects (depth, intermediate passes)
  totalBytes *= 2.5; // Rough estimate for additional buffers

  return totalBytes / (1024 * 1024); // Convert to MB
}

/**
 * Determines optimal anti-aliasing strategy based on performance budget
 *
 * @param targetFPS - Target frame rate
 * @param currentFPS - Current frame rate
 * @param gpuMemoryMB - Available GPU memory in MB
 * @returns Recommended AA configuration
 */
export function optimizeAAStrategy(
  targetFPS: number,
  currentFPS: number,
  gpuMemoryMB: number
): { msaa: boolean; smaa: boolean; fxaa: boolean; ssaa: boolean } {
  const performanceRatio = currentFPS / targetFPS;

  if (performanceRatio > 1.5 && gpuMemoryMB > 2000) {
    // Excellent performance, use best quality
    return { msaa: true, smaa: false, fxaa: false, ssaa: true };
  } else if (performanceRatio > 1.2 && gpuMemoryMB > 1000) {
    // Good performance, use MSAA or SMAA
    return { msaa: true, smaa: false, fxaa: false, ssaa: false };
  } else if (performanceRatio > 0.9) {
    // Adequate performance, use SMAA
    return { msaa: false, smaa: true, fxaa: false, ssaa: false };
  } else {
    // Poor performance, use FXAA or disable AA
    return { msaa: false, smaa: false, fxaa: performanceRatio > 0.7, ssaa: false };
  }
}
