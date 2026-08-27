import type * as THREE from 'three';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';

import type { KTX2TextureDecoder } from '../types/mesh';
import type { Renderer } from './renderer-capabilities';

function hasCompressedTextureSupport(renderer: Renderer): boolean {
  if ((renderer as { isWebGPURenderer?: boolean }).isWebGPURenderer === true) {
    const hasFeature = (renderer as { hasFeature?: (name: string) => boolean }).hasFeature;
    return Boolean(
      hasFeature?.('texture-compression-astc') ||
      hasFeature?.('texture-compression-etc2') ||
      hasFeature?.('texture-compression-s3tc') ||
      hasFeature?.('texture-compression-bc') ||
      hasFeature?.('texture-compression-pvrtc')
    );
  }
  const extensions = (renderer as THREE.WebGLRenderer).extensions;
  return [
    'WEBGL_compressed_texture_astc',
    'WEBGL_compressed_texture_etc',
    'WEBGL_compressed_texture_s3tc',
    'EXT_texture_compression_bptc',
    'WEBGL_compressed_texture_pvrtc',
    'WEBKIT_WEBGL_compressed_texture_pvrtc',
  ].some((name) => extensions.has(name));
}

export function createKTX2TextureDecoder(renderer: Renderer): KTX2TextureDecoder {
  const supported = hasCompressedTextureSupport(renderer);
  const loader = supported ? new KTX2Loader().detectSupport(renderer) : null;

  return async (path, bytes) => {
    if (!loader) {
      throw new Error(
        `${path}: texture encoding 'ktx2' requires ASTC, ETC2, S3TC/BC, or PVRTC ` +
          "GPU compressed-texture support. Use 'raw' or 'jpeg' for a portable texture."
      );
    }
    return await new Promise<THREE.CompressedTexture>((resolve, reject) => {
      loader.parse(new Uint8Array(bytes).buffer, resolve, reject);
    });
  };
}
