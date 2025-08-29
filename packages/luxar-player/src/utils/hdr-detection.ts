/**
 * HDR Detection and Configuration Utilities
 *
 * Provides functions to detect HDR display capabilities and configure
 * Three.js for optimal HDR rendering including 10-bit color depth.
 */

import * as THREE from 'three';
import { log, Modules, LogEmoji } from './log';

/**
 * HDR capability detection results
 */
export interface HDRCapabilities {
  /** Display supports P3 wide color gamut */
  p3Gamut: boolean;
  /** Display supports Rec2020 color gamut */
  rec2020Gamut: boolean;
  /** Display supports high dynamic range */
  hdr: boolean;
  /** Display supports 10-bit or higher color depth */
  deepColor: boolean;
  /** WebGL supports float textures */
  floatTextures: boolean;
  /** WebGL color buffer bit depth */
  colorDepth: {
    red: number;
    green: number;
    blue: number;
  };
  /** Recommended color space for this display */
  recommendedColorSpace: 'srgb' | 'display-p3' | 'rec2020';
}

/**
 * Detect HDR and wide gamut display capabilities
 */
export function detectHDRCapabilities(renderer?: THREE.WebGLRenderer): HDRCapabilities {
  // Check CSS media queries for display capabilities
  const p3Gamut = window.matchMedia('(color-gamut: p3)').matches;
  const rec2020Gamut = window.matchMedia('(color-gamut: rec2020)').matches;
  const hdr = window.matchMedia('(dynamic-range: high)').matches;

  // Check for deep color support (10-bit or higher)
  // 48-bit total = 16 bits per channel (including alpha)
  // For 10-bit RGB, we'd see at least 30 bits
  const deepColor =
    window.matchMedia('(color: 48)').matches || window.matchMedia('(color: 30)').matches;

  // Check WebGL capabilities if renderer is provided
  let floatTextures = false;
  let colorDepth = { red: 8, green: 8, blue: 8 };

  if (renderer) {
    const gl = renderer.getContext();

    // Check for float texture extension (required for HDR)
    floatTextures = !!(
      gl.getExtension('EXT_color_buffer_float') ||
      gl.getExtension('EXT_color_buffer_half_float') ||
      gl.getExtension('WEBGL_color_buffer_float')
    );

    // Get actual color buffer bit depth
    colorDepth = {
      red: gl.getParameter(gl.RED_BITS),
      green: gl.getParameter(gl.GREEN_BITS),
      blue: gl.getParameter(gl.BLUE_BITS),
    };
  }

  // Determine recommended color space
  let recommendedColorSpace: 'srgb' | 'display-p3' | 'rec2020' = 'srgb';
  if (rec2020Gamut && hdr) {
    recommendedColorSpace = 'rec2020';
  } else if (p3Gamut) {
    recommendedColorSpace = 'display-p3';
  }

  return {
    p3Gamut,
    rec2020Gamut,
    hdr,
    deepColor,
    floatTextures,
    colorDepth,
    recommendedColorSpace,
  };
}

/**
 * Configure Three.js renderer for HDR output
 */
export function configureHDRRenderer(
  _renderer: THREE.WebGLRenderer,
  capabilities: HDRCapabilities
): void {
  // IMPORTANT: When using pmndrs/postprocessing library, we should NOT set
  // outputColorSpace or toneMapping on the renderer here. The post-processing
  // library handles these internally. Setting them here causes conflicts.
  //
  // The PostProcessingManager will:
  // 1. Set renderer.outputColorSpace = THREE.SRGBColorSpace
  // 2. Set renderer.toneMapping = THREE.NoToneMapping
  // 3. Handle all color space conversions and tone mapping in its pipeline
  
  // Log detected capabilities for informational purposes only
  if (capabilities.rec2020Gamut && capabilities.hdr) {
    log.success(
      Modules.HDR,
      'HDR display with Rec2020 gamut detected - post-processing will handle color management'
    );
  } else if (capabilities.p3Gamut) {
    log.success(Modules.HDR, 'Display P3 gamut detected - post-processing will handle color management');
  } else {
    log.info(Modules.HDR, 'Standard sRGB display detected');
  }
  
  // Note: The actual tone mapping and color space configuration is handled by
  // PostProcessingManager to avoid conflicts with the pmndrs library.
  // These settings will be overridden when PostProcessingManager is initialized.
}

/**
 * Log HDR capabilities to console with color coding
 */
export function logHDRCapabilities(capabilities: HDRCapabilities): void {
  log.custom(LogEmoji.RENDER, Modules.HDR, 'Display Capabilities:');
  log.info(Modules.HDR, `  ${capabilities.p3Gamut ? '✅' : '❌'} P3 Wide Gamut`);
  log.info(Modules.HDR, `  ${capabilities.rec2020Gamut ? '✅' : '❌'} Rec2020 Gamut`);
  log.info(Modules.HDR, `  ${capabilities.hdr ? '✅' : '❌'} High Dynamic Range`);
  log.info(Modules.HDR, `  ${capabilities.deepColor ? '✅' : '❌'} 10-bit+ Deep Color`);
  log.info(Modules.HDR, `  ${capabilities.floatTextures ? '✅' : '❌'} Float Textures`);
  log.data(
    Modules.LUXAR,
    `Color Buffer Depth: R${capabilities.colorDepth.red} G${capabilities.colorDepth.green} B${capabilities.colorDepth.blue}`
  );
  log.info(Modules.LUXAR, `Recommended Color Space: ${capabilities.recommendedColorSpace}`);
}

/**
 * Check if current display supports true HDR
 */
export function isHDRDisplay(capabilities: HDRCapabilities): boolean {
  return (
    capabilities.hdr &&
    capabilities.deepColor &&
    capabilities.floatTextures &&
    (capabilities.p3Gamut || capabilities.rec2020Gamut)
  );
}

/**
 * Get optimal render target type for current display
 */
export function getOptimalRenderTargetType(capabilities: HDRCapabilities): THREE.TextureDataType {
  if (capabilities.floatTextures && capabilities.hdr) {
    // Use half float for HDR displays (good balance of quality and performance)
    return THREE.HalfFloatType;
  } else if (capabilities.floatTextures) {
    // Use half float even for SDR if supported (better gradients)
    return THREE.HalfFloatType;
  } else {
    // Fallback to unsigned byte
    return THREE.UnsignedByteType;
  }
}
