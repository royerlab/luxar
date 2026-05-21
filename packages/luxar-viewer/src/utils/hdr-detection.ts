/**
 * HDR Detection and Configuration Utilities
 *
 * Display-side capability detection (CSS media queries) plus pure
 * decision logic over an `HDRCapabilities` snapshot. The renderer-
 * side probes (`gl.getExtension`, color-buffer bit-depth) live in
 * `src/rendering/renderer-capabilities.ts` — the single seam where
 * raw GL is allowed.
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
 * Detect *display-side* HDR / wide-gamut capabilities via CSS media
 * queries. Renderer-side probes (float-texture extension, color
 * buffer bit depth) are NOT done here — they live in
 * `createRendererCapabilities` so the raw-GL surface stays
 * concentrated in one module.
 *
 * Returned values for `floatTextures` and `colorDepth` are
 * defaults; `createRendererCapabilities` overwrites them with real
 * probes.
 */
export function detectDisplayCapabilities(): HDRCapabilities {
  // Check CSS media queries for display capabilities
  const p3Gamut = window.matchMedia('(color-gamut: p3)').matches;
  const rec2020Gamut = window.matchMedia('(color-gamut: rec2020)').matches;
  const hdr = window.matchMedia('(dynamic-range: high)').matches;

  // Check for deep color support (10-bit or higher).
  // Per CSS Media Queries Level 4, `(color: N)` matches if the device uses
  // at least N bits *per color component*. A true 10-bit display should
  // match `(color: 10)`. The `(color: 48)` / `(color: 30)` thresholds below
  // are empirical: in practice some shipping browsers report the *total*
  // bit depth (~48 for 16-bpc, ~30 for 10-bpc RGB) rather than per-component,
  // so both probes are kept as a belt-and-braces feature test.
  const deepColor =
    window.matchMedia('(color: 48)').matches || window.matchMedia('(color: 30)').matches;

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
    floatTextures: false, // overwritten by createRendererCapabilities
    colorDepth: { red: 8, green: 8, blue: 8 }, // overwritten by createRendererCapabilities
    recommendedColorSpace,
  };
}

/**
 * Configure Three.js renderer for HDR output.
 *
 * Param `_renderer` is kept for the (future) case where the
 * function needs to read renderer-specific HDR capability fields;
 * today it consults the capabilities snapshot only. Typed loosely
 * as `unknown` to avoid coupling the signature to either
 * `WebGLRenderer` or `WebGPURenderer`.
 */
export function configureHDRRenderer(_renderer: unknown, capabilities: HDRCapabilities): void {
  // Do NOT set renderer.outputColorSpace or renderer.toneMapping here.
  // The post-processing pipeline owns both: the mega-shader applies
  // tone mapping internally and the host pins
  // outputColorSpace = SRGB / toneMapping = NoToneMapping at
  // PostProcessingManager construction.
  if (capabilities.rec2020Gamut && capabilities.hdr) {
    log.success(
      Modules.HDR,
      'HDR display with Rec2020 gamut detected - post-processing will handle color management'
    );
  } else if (capabilities.p3Gamut) {
    log.success(
      Modules.HDR,
      'Display P3 gamut detected - post-processing will handle color management'
    );
  } else {
    log.info(Modules.HDR, 'Standard sRGB display detected');
  }
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
