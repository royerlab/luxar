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
import { log, Modules } from '../utils/log';

/**
 * valid LUT sizes.
 *   - 768 bytes = 256 entries × 3 channels (RGB) — the standard layout from
 *     `compiler.py:_resolve_colormap_lut` (Python writer).
 *   - 1024 bytes = 256 × 4 (RGBA) — supported as a forward-compatible
 *     alternative; the alpha channel is preserved as-is.
 */
const VALID_LUT_LENGTHS = new Set<number>([768, 1024]);

/** Cache of built-in colormap textures (name → texture) */
const builtinCache = new Map<string, THREE.DataTexture>();

/** Cache of custom colormap textures (hash → texture). Bounded LRU. */
const customCache = new Map<string, THREE.DataTexture>();

/**
 * Per-process counters for the custom-LUT cache. Used to validate the
 * LRU is doing useful work (high hit rate) and to spot fingerprint-
 * collision induced misses. Counters are cumulative since process
 * start; surfaced via {@link getCustomColormapCacheStats}.
 */
const customCacheStats = {
  /** Cache lookups that returned a stored texture (post-fingerprint validate). */
  hits: 0,
  /** Cache lookups that missed and built a fresh texture. */
  misses: 0,
  /** Hits that were rejected by fingerprint mismatch (hash collision). */
  collisions: 0,
  /** LRU evictions due to `CUSTOM_LUT_CACHE_MAX` overflow. */
  evictions: 0,
};

/**
 * Maximum number of custom LUT textures kept in the per-app cache.
 * Once exceeded, the oldest entry is disposed and dropped. 16 covers
 * realistic use (a single dataset rarely uses more than 2–3 unique
 * custom LUTs at once); higher values would just delay an inevitable
 * dispose for users that swap many unique LUTs over a long session.
 */
const CUSTOM_LUT_CACHE_MAX = 16;

function customCacheLruGet(key: string): THREE.DataTexture | undefined {
  const value = customCache.get(key);
  if (value !== undefined) {
    // Re-insert to bump to most-recently-used (Map preserves insertion order).
    customCache.delete(key);
    customCache.set(key, value);
  }
  return value;
}

function customCacheLruSet(key: string, value: THREE.DataTexture): void {
  // If the same key was just promoted (somehow), avoid double-set.
  if (customCache.has(key)) {
    customCache.delete(key);
  }
  while (customCache.size >= CUSTOM_LUT_CACHE_MAX) {
    const lruKey = customCache.keys().next().value;
    if (lruKey === undefined) break;
    const lruTex = customCache.get(lruKey);
    customCache.delete(lruKey);
    customCacheStats.evictions++;
    if (lruTex) {
      try {
        lruTex.dispose();
      } catch {
        // Ignore — texture may already be disposed.
      }
    }
  }
  customCache.set(key, value);
}

/**
 * Convert RGB (768 bytes) to RGBA (1024 bytes) for WebGL2 compatibility.
 *
 * WebGL2 with texStorage2D requires sized internal formats. THREE.RGBFormat
 * maps to unsized gl.RGB which causes silent upload failures. RGBA is
 * universally supported.
 */
function rgbToRgba(rgb: Uint8Array): Uint8Array<ArrayBuffer> {
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
 * when `name === 'custom'`, `customLut` MUST be a Uint8Array of length
 * 768 (RGB) or 1024 (RGBA). Invalid lengths log a warning and the function
 * falls back to the `viridis` built-in so the viewer keeps rendering
 * something sensible rather than silently disabling colormap mode.
 *
 * @param name - Colormap name. "custom" means use customLut parameter.
 * @param customLut - Optional Uint8Array(768) or Uint8Array(1024) for custom colormaps
 * @returns THREE.DataTexture for shader use
 */
export function getColormapTexture(
  name: string,
  customLut?: Uint8Array
): THREE.DataTexture | undefined {
  if (name === 'custom') {
    if (!customLut) {
      log.warning(
        Modules.RENDERER,
        "colormap='custom' requested without LUT bytes; falling back to 'viridis'."
      );
      return getBuiltinColormapTexture('viridis');
    }
    if (!VALID_LUT_LENGTHS.has(customLut.length)) {
      log.warning(
        Modules.RENDERER,
        `Custom colormap LUT has invalid length ${customLut.length} (expected 768 RGB or 1024 RGBA); falling back to 'viridis'.`
      );
      return getBuiltinColormapTexture('viridis');
    }
    return createCustomColormapTexture(customLut);
  }
  return getBuiltinColormapTexture(name);
}

/**
 * Create a custom colormap texture from raw LUT data.
 *
 * @param lut - Uint8Array of 768 bytes (256 × 3 RGB) or 1024 bytes (256 × 4 RGBA)
 * @returns THREE.DataTexture
 */
/**
 * G.5: capture a small fingerprint (first/last 16 bytes + length) of
 * the LUT so a cache hit can be content-validated before being
 * returned. Defends against the (unlikely but possible) DJB2 hash
 * collision that would otherwise map two distinct LUTs to the same
 * cache key. Total fingerprint memory is ~33 B per cached entry.
 */
function lutFingerprint(lut: Uint8Array): Uint8Array {
  const fp = new Uint8Array(32 + 1);
  fp[0] = lut.length & 0xff;
  // Length is 768 or 1024 — fits in one byte mod 256, but we use it as
  // a low-confidence pre-check anyway; the byte-comparison below is
  // what catches collisions.
  const head = Math.min(16, lut.length);
  for (let i = 0; i < head; i++) fp[1 + i] = lut[i];
  const tail = Math.min(16, lut.length);
  for (let i = 0; i < tail; i++) fp[17 + i] = lut[lut.length - tail + i];
  return fp;
}

function fingerprintMatches(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export function createCustomColormapTexture(lut: Uint8Array): THREE.DataTexture {
  // Hash ALL bytes of the LUT for caching (DJB2 hash over every byte)
  let hash = 0;
  for (let i = 0; i < lut.length; i++) {
    hash = ((hash << 5) - hash + lut[i]) | 0;
  }
  // include length in the cache key so two LUTs with the same bytes but
  // different lengths can't collide (extremely unlikely but cheap).
  const key = `custom_${lut.length}_${hash}`;
  const incomingFp = lutFingerprint(lut);

  const cached = customCacheLruGet(key);
  if (cached) {
    // G.5: validate cache hit against fingerprint. If two distinct LUTs
    // hashed to the same key, the fingerprints will (with overwhelming
    // probability) diverge — dispose the stale entry and re-create.
    const cachedFp = (cached.userData as { originalLutSample?: Uint8Array }).originalLutSample;
    if (cachedFp && fingerprintMatches(cachedFp, incomingFp)) {
      customCacheStats.hits++;
      return cached;
    }
    // Fingerprint mismatch → collision. Dispose and fall through.
    customCacheStats.collisions++;
    try {
      cached.dispose();
    } catch {
      // Ignore disposal errors — texture may already be invalid.
    }
    // The LRU map kept the key on hit; we need to delete before
    // re-insert so the new texture takes the slot.
    // Note: customCacheLruSet handles this by checking has() first.
  }
  customCacheStats.misses++;

  // 1024-byte RGBA LUTs upload directly; 768-byte RGB LUTs go through
  // rgbToRgba (preserves the existing path).
  let texture: THREE.DataTexture;
  if (lut.length === 1024) {
    // Copy into a fresh Uint8Array<ArrayBuffer> so the THREE typings (which
    // reject SharedArrayBuffer-backed views) accept it without a cast.
    const rgbaCopy = new Uint8Array(1024);
    rgbaCopy.set(lut);
    texture = new THREE.DataTexture(rgbaCopy, 256, 1, THREE.RGBAFormat);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.needsUpdate = true;
  } else {
    texture = createTexture(lut);
  }
  // G.5: stash the fingerprint on userData so future lookups can
  // validate. ~33 B per cache entry is negligible vs the 1024 B LUT.
  (texture.userData as { originalLutSample?: Uint8Array }).originalLutSample = incomingFp;
  customCacheLruSet(key, texture);
  return texture;
}

/**
 * Check if a colormap name is a built-in colormap.
 */
export function isBuiltinColormap(name: string): boolean {
  return name in BUILTIN_COLORMAPS;
}

/**
 * Dispose ONLY the built-in colormap texture cache.
 *
 * Built-in colormaps live for the lifetime of the app — disposing
 * them mid-session would force re-allocation for every subsequent
 * dataset. Call this only at app teardown via the combined
 * `disposeColormapTextures()` below.
 */
export function disposeBuiltinColormapTextures(): void {
  for (const tex of builtinCache.values()) {
    tex.dispose();
  }
  builtinCache.clear();
}

/**
 * Dispose ONLY the custom-LUT colormap texture cache.
 *
 * Custom LUTs are scene/dataset-scoped. SceneLoader.dispose()
 * calls this on dataset unload so unused custom textures don't
 * accumulate across many dataset switches in a long-lived app.
 * Built-ins are not touched.
 */
export function disposeCustomColormapTextures(): void {
  for (const tex of customCache.values()) {
    tex.dispose();
  }
  customCache.clear();
}

/**
 * Dispose ALL cached colormap textures (built-in + custom).
 * Call this when the application is shutting down.
 */
export function disposeColormapTextures(): void {
  disposeBuiltinColormapTextures();
  disposeCustomColormapTextures();
}

/**
 * Test helper exposing the custom cache size so tests can assert the
 * bounded-LRU invariant without poking at module internals.
 */
export function _customColormapCacheSize(): number {
  return customCache.size;
}

/**
 * Snapshot of the custom-LUT cache counters. Cumulative since process
 * start. Exposed for the data monitor and diagnostics. The shape is
 * stable: `{ size, maxSize, hits, misses, collisions, evictions }`.
 *
 * Hit rate = `hits / (hits + misses)`. A low hit rate after many
 * lookups means the LRU isn't doing much (the working set exceeds
 * `maxSize`) or fingerprint collisions are forcing rebuilds.
 */
export function getCustomColormapCacheStats(): {
  size: number;
  maxSize: number;
  hits: number;
  misses: number;
  collisions: number;
  evictions: number;
} {
  return {
    size: customCache.size,
    maxSize: CUSTOM_LUT_CACHE_MAX,
    hits: customCacheStats.hits,
    misses: customCacheStats.misses,
    collisions: customCacheStats.collisions,
    evictions: customCacheStats.evictions,
  };
}

/**
 * Reset the custom-LUT cache counters. Intended for tests and
 * dev-mode diagnostics where a per-session view of the rate is more
 * useful than the lifetime count.
 */
export function _resetCustomColormapCacheStatsForTests(): void {
  customCacheStats.hits = 0;
  customCacheStats.misses = 0;
  customCacheStats.collisions = 0;
  customCacheStats.evictions = 0;
}
