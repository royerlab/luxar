import type * as THREE from 'three';
import { RedFormat, RGBAFormat, RGFormat, RGBFormat } from 'three';
import type { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';

import type { KTX2TextureDecoder } from '../types/mesh';
import type { Renderer } from './renderer-capabilities';

const UNCOMPRESSED_FORMATS = new Set<number>([RGBAFormat, RGBFormat, RGFormat, RedFormat]);

/** Return whether the renderer exposes a native Basis transcode target. */
function hasCompressedTextureSupport(renderer: Renderer | null | undefined): boolean {
  if (!renderer) return false;
  if ((renderer as { isWebGPURenderer?: boolean }).isWebGPURenderer === true) {
    const hasFeature = (renderer as { hasFeature?: (name: string) => boolean }).hasFeature;
    if (typeof hasFeature !== 'function') return false;
    return [
      'texture-compression-astc',
      'texture-compression-etc1',
      'texture-compression-etc2',
      'texture-compression-s3tc',
      'texture-compression-bc',
      'texture-compression-pvrtc',
    ].some((name) => hasFeature.call(renderer, name));
  }
  const extensions = (renderer as THREE.WebGLRenderer).extensions;
  if (!extensions?.has) return false;
  return [
    'WEBGL_compressed_texture_astc',
    'WEBGL_compressed_texture_etc1',
    'WEBGL_compressed_texture_etc',
    'WEBGL_compressed_texture_s3tc',
    'EXT_texture_compression_bptc',
    'WEBGL_compressed_texture_pvrtc',
    'WEBKIT_WEBGL_compressed_texture_pvrtc',
  ].some((name) => extensions.has(name));
}

/**
 * Create an app-lifetime KTX2 decoder for one initialized renderer.
 *
 * The Three.js loader and its worker pool are created lazily on the first KTX2
 * payload, so applications that never load KTX2 do not initialize the Basis
 * transcoder. Missing renderers and devices without a compressed target fail
 * closed with the portable texture alternatives in the error.
 */
export function createKTX2TextureDecoder(
  renderer: Renderer | null | undefined
): KTX2TextureDecoder {
  let loader: KTX2Loader | null = null;
  let loaderPending: Promise<KTX2Loader> | null = null;
  let disposed = false;

  const decode = async (path: string, bytes: Uint8Array) => {
    if (disposed) {
      throw new Error(`${path}: KTX2 decoder was disposed before parsing began`);
    }
    if (!renderer || !hasCompressedTextureSupport(renderer)) {
      throw new Error(
        `${path}: texture encoding 'ktx2' requires ASTC, ETC1/2, S3TC/BC, or PVRTC ` +
          "GPU compressed-texture support. Use 'raw' or 'jpeg' for a portable texture."
      );
    }
    if (!loaderPending) {
      const pending = import('three/examples/jsm/loaders/KTX2Loader.js').then((module) => {
        const created = new module.KTX2Loader().detectSupport(renderer);
        if (disposed) {
          created.dispose();
          throw new Error(`${path}: KTX2 decoder was disposed before parsing began`);
        }
        loader = created;
        return created;
      });
      loaderPending = pending;
    }
    const pending = loaderPending;
    let activeLoader: KTX2Loader;
    try {
      activeLoader = await pending;
    } catch (error) {
      if (loaderPending === pending) loaderPending = null;
      throw error;
    }
    if (disposed) {
      throw new Error(`${path}: KTX2 decoder was disposed before parsing began`);
    }
    const texture = await new Promise<THREE.CompressedTexture>((resolve, reject) => {
      activeLoader.parse(new Uint8Array(bytes).buffer, resolve, reject);
    });
    const image = texture.image as { depth?: number } | undefined;
    const isUncompressedTexture =
      (texture as THREE.CompressedTexture & { isCompressedTexture?: boolean })
        .isCompressedTexture !== true || UNCOMPRESSED_FORMATS.has(texture.format);
    const isCubeTexture = (texture as THREE.CompressedTexture & { isCubeTexture?: boolean })
      .isCubeTexture;
    if (isUncompressedTexture || isCubeTexture || (image?.depth ?? 1) > 1) {
      texture.dispose();
      const reason = isUncompressedTexture
        ? 'would produce an uncompressed texture'
        : isCubeTexture
          ? 'is a cubemap but mesh textures require one 2D image'
          : `contains ${image?.depth ?? '?'} layers but mesh textures require one 2D image`;
      throw new Error(
        `${path}: texture encoding 'ktx2' ${reason}. ` +
          "Use 'raw' or 'jpeg' for a portable texture."
      );
    }
    return texture;
  };
  decode.dispose = () => {
    if (disposed) return;
    disposed = true;
    loader?.dispose();
    loader = null;
  };
  return decode;
}
