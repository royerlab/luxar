/**
 * HDR Detection and Configuration Utilities
 *
 * Provides functions to detect HDR display capabilities and configure
 * Three.js for optimal HDR rendering including 10-bit color depth.
 */

import * as THREE from 'three';

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
  renderer: THREE.WebGLRenderer,
  capabilities: HDRCapabilities
): void {
  // Set appropriate output color space based on display capabilities
  if (capabilities.rec2020Gamut && capabilities.hdr) {
    // Full HDR with Rec2020 gamut
    // Note: Three.js doesn't have Rec2020 color space yet, using Linear as closest
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    console.log(
      '✓ [Luxar] HDR: Configured for HDR with Linear color space (Rec2020 display detected)'
    );
  } else if (capabilities.p3Gamut) {
    // Wide gamut P3 (common on Apple displays)
    // Note: DisplayP3ColorSpace might not be available in all Three.js versions
    // Using SRGBColorSpace as fallback but noting P3 capability
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    console.log('✓ [Luxar] HDR: Display P3 gamut detected, using sRGB color space');
  } else {
    // Standard sRGB
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    console.log('✓ [Luxar] HDR: Using standard sRGB color space');
  }

  // Configure tone mapping for HDR
  if (capabilities.hdr) {
    // Use ACES for HDR displays
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.4; // Slightly boost for HDR headroom
    console.log('✓ [Luxar] HDR: ACES tone mapping enabled with HDR exposure');
  } else {
    // Use ACES for SDR with standard exposure
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    console.log('✓ [Luxar] HDR: ACES tone mapping for SDR display');
  }
}

/**
 * Log HDR capabilities to console with color coding
 */
export function logHDRCapabilities(capabilities: HDRCapabilities): void {
  const style = (supported: boolean) =>
    supported ? 'color: #4CAF50; font-weight: bold' : 'color: #f44336';

  console.group('%c🎨 HDR Display Capabilities', 'font-size: 14px; font-weight: bold');

  console.log(
    '%c' + (capabilities.p3Gamut ? '✅' : '❌') + ' P3 Wide Gamut',
    style(capabilities.p3Gamut)
  );

  console.log(
    '%c' + (capabilities.rec2020Gamut ? '✅' : '❌') + ' Rec2020 Gamut',
    style(capabilities.rec2020Gamut)
  );

  console.log(
    '%c' + (capabilities.hdr ? '✅' : '❌') + ' High Dynamic Range',
    style(capabilities.hdr)
  );

  console.log(
    '%c' + (capabilities.deepColor ? '✅' : '❌') + ' 10-bit+ Deep Color',
    style(capabilities.deepColor)
  );

  console.log(
    '%c' + (capabilities.floatTextures ? '✅' : '❌') + ' Float Textures',
    style(capabilities.floatTextures)
  );

  console.log(
    '📊 [Luxar] Color Buffer Depth: ' +
      `R${capabilities.colorDepth.red} G${capabilities.colorDepth.green} B${capabilities.colorDepth.blue}`
  );

  console.log('🎯 [Luxar] Recommended Color Space:', capabilities.recommendedColorSpace);

  console.groupEnd();
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
