/**
 * Colormap texture management for Luxar.
 *
 * Creates and caches THREE.DataTexture objects for colormap LUT lookup in shaders.
 * Each colormap is a 256x1 RGB texture used for scalar-to-color mapping.
 *
 * @module rendering/colormap-textures
 */

import * as THREE from 'three';
import { BUILTIN_COLORMAPS } from './colormap-data';

/** Cache of built-in colormap textures (name → texture) */
const builtinCache = new Map<string, THREE.DataTexture>();

/** Cache of custom colormap textures (hash → texture) */
const customCache = new Map<string, THREE.DataTexture>();

/**
 * Convert RGB (768 bytes) to RGBA (1024 bytes) for WebGL2 compatibility.
 *
 * WebGL2 with texStorage2D requires sized internal formats. THREE.RGBFormat
 * maps to unsized gl.RGB which causes silent upload failures. RGBA is
 * universally supported.
 */
function rgbToRgba(rgb: Uint8Array): Uint8Array {
  const rgba = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    rgba[i * 4] = rgb[i * 3];
    rgba[i * 4 + 1] = rgb[i * 3 + 1];
    rgba[i * 4 + 2] = rgb[i * 3 + 2];
    rgba[i * 4 + 3] = 255; // Full opacity
  }
  return rgba;
}

/**
 * Create a 256x1 RGBA DataTexture from raw RGB LUT data.
 *
 * @param data - Uint8Array of 768 bytes (256 entries × 3 channels RGB)
 * @returns THREE.DataTexture ready for use as a shader uniform
 */
function createTexture(data: Uint8Array): THREE.DataTexture {
  const rgbaData = rgbToRgba(data);
  const texture = new THREE.DataTexture(rgbaData, 256, 1, THREE.RGBAFormat);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Get a built-in colormap texture by name.
 *
 * Textures are cached — the same THREE.DataTexture is returned for repeated calls
 * with the same name. This is safe because colormap textures are read-only.
 *
 * @param name - Built-in colormap name (e.g., "viridis", "green", "magenta")
 * @returns THREE.DataTexture, or undefined if name is not a built-in colormap
 */
export function getBuiltinColormapTexture(name: string): THREE.DataTexture | undefined {
  const cached = builtinCache.get(name);
  if (cached) return cached;

  const data = BUILTIN_COLORMAPS[name];
  if (!data) return undefined;

  const texture = createTexture(data);
  builtinCache.set(name, texture);
  return texture;
}

/**
 * Get a colormap texture by name (built-in) or from custom LUT data.
 *
 * @param name - Colormap name. "custom" means use customLut parameter.
 * @param customLut - Optional Uint8Array(768) for custom colormaps
 * @returns THREE.DataTexture for shader use
 */
export function getColormapTexture(
  name: string,
  customLut?: Uint8Array
): THREE.DataTexture | undefined {
  if (name === 'custom' && customLut) {
    return createCustomColormapTexture(customLut);
  }
  return getBuiltinColormapTexture(name);
}

/**
 * Create a custom colormap texture from raw LUT data.
 *
 * @param lut - Uint8Array of 768 bytes (256 × 3 RGB)
 * @returns THREE.DataTexture
 */
export function createCustomColormapTexture(lut: Uint8Array): THREE.DataTexture {
  // Hash the full LUT for caching (sample every 8th byte for speed)
  let hash = 0;
  for (let i = 0; i < lut.length; i += 8) {
    hash = ((hash << 5) - hash + lut[i]) | 0;
  }
  // Also include last few bytes to catch suffix differences
  for (let i = Math.max(0, lut.length - 12); i < lut.length; i++) {
    hash = ((hash << 5) - hash + lut[i]) | 0;
  }
  const key = `custom_${hash}`;

  const cached = customCache.get(key);
  if (cached) return cached;

  const texture = createTexture(lut);
  customCache.set(key, texture);
  return texture;
}

/**
 * Check if a colormap name is a built-in colormap.
 */
export function isBuiltinColormap(name: string): boolean {
  return name in BUILTIN_COLORMAPS;
}

/**
 * Dispose all cached colormap textures.
 * Call this when the application is shutting down.
 */
export function disposeColormapTextures(): void {
  for (const tex of builtinCache.values()) {
    tex.dispose();
  }
  builtinCache.clear();

  for (const tex of customCache.values()) {
    tex.dispose();
  }
  customCache.clear();
}
