/**
 * `viewer_config.environment.source = "hdri"`: an equirectangular image as the
 * environment.
 *
 * `.hdr` (Radiance RGBE) loads through three's `HDRLoader` — imported DYNAMICALLY so it
 * stays off the eager payload (`scripts/check-eager-chunks.mjs`); the loader imports
 * only from `three`, so it serves both backends. Anything else is treated as an LDR
 * image through `TextureLoader`. Either way the texture gets the equirectangular
 * reflection mapping and three prefilters it on assignment to `scene.environment`.
 *
 * @module rendering/environment/hdri
 */

import * as THREE from 'three';

/** Resolve a store-relative `url` against the scene's base URL; absolute URLs pass through. */
export function resolveEnvironmentUrl(url: string, baseUrl: string | undefined): string {
  if (/^https?:/i.test(url) || !baseUrl) return url;
  const base = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  const baseParsed = new URL(base);
  const resolved = new URL(url.replace(/^\//, ''), baseParsed);
  if (resolved.origin !== baseParsed.origin) {
    throw new Error('Store-relative environment URL resolved outside the store origin');
  }
  return resolved.toString();
}

/** Load an equirectangular environment texture; rejects on a network or decode error. */
export async function loadEquirectangularTexture(url: string): Promise<THREE.Texture> {
  const isHdr = /\.hdr(\?|#|$)/i.test(url);
  let texture: THREE.Texture;
  if (isHdr) {
    const { HDRLoader } = await import('three/examples/jsm/loaders/HDRLoader.js');
    texture = await new HDRLoader().loadAsync(url);
    texture.colorSpace = THREE.LinearSRGBColorSpace;
  } else {
    texture = await new THREE.TextureLoader().loadAsync(url);
    texture.colorSpace = THREE.SRGBColorSpace;
  }
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.needsUpdate = true;
  return texture;
}
