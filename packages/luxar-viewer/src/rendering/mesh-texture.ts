/**
 * Upload a decoded mesh texture to the GPU.
 *
 * The first *image* texture in the tree, and it differs from every existing one
 * in ways that are easy to get wrong silently:
 *
 * - the colormap LUTs (`rendering/colormap-textures.ts`) are 256x1 8-bit and
 *   always `LinearFilter`/`ClampToEdge` — no wrap, no mipmaps, no colour space;
 * - the element store (`rendering/element-storage.ts`) is float but always
 *   `NearestFilter`, because it carries *data* rather than an image.
 *
 * So this is the first texture that is simultaneously float-capable, linearly
 * filtered, mipmapped and colour-managed, and each of those four is a place where
 * the wrong choice renders plausibly rather than failing.
 *
 * ## Three decisions worth reading before changing anything here
 *
 * **1. Three-channel data is expanded to RGBA.** Not an oversight and not
 * laziness: WebGPU has no 3-channel texture formats at all, so `RGBFormat` would
 * work under WebGL2 and break the moment the same scene loads on the WebGPU
 * backend. `rgbToRgba` in `colormap-textures.ts` expands for the same reason.
 * Single-channel is kept as `RedFormat`, which both backends do support.
 *
 * **2. HDR uploads as `HalfFloatType` unless the device says otherwise.** Core
 * WebGL2 lets a `FloatType` texture exist and be sampled with NEAREST, but LINEAR
 * filtering on one requires `OES_texture_float_linear` (`float32-filterable` on
 * WebGPU). When it is absent the sampler *silently* drops to nearest, so the
 * texture looks blocky with nothing to attribute it to. Half-float is linearly
 * filterable in core WebGL2 and carries ~11 bits of mantissa, which is ample for
 * an image; float32 is used only where the capability probe confirms it.
 *
 * **3. `v = 0` is the FIRST row of the image, on all three arms.** `texture.flipY`
 * is set to `false` explicitly rather than left at its default, because the
 * defaults disagree AND two of them are inert or misleading:
 *
 * - `DataTexture` defaults `flipY = false`;
 * - a `Texture` over an `ImageBitmap` defaults `flipY = true`, and WebGL
 *   **silently ignores it** — `UNPACK_FLIP_Y_WEBGL` has no effect on an
 *   `ImageBitmap` upload, so the flag reads `true` while nothing is flipped.
 * - a `CompressedTexture` defaults `flipY = false`, and WebGL likewise cannot
 *   apply `UNPACK_FLIP_Y_WEBGL` to its already-compressed mip payloads.
 *
 * Left alone, that made `texture_encoding` change the MEANING of a UV: the same
 * coordinates over the same pixels rendered upside down as raw vs as JPEG. Since
 * the raw path takes an `(h, w, c)` array, the convention that matches it —
 * `v = 0` is `arr[0]` — is the one an author can predict, and it is also what
 * `sample_equirect` uses (row 0 is +90 latitude). So `flipY` is pinned off and
 * the authored `v` runs top-down.
 *
 * Found by eye, not by the numeric check: a UV-vs-position residual test passed
 * because it MODELLED the shader's sampling from `texture.flipY`, and the flag it
 * trusted was the thing that was wrong.
 *
 * **4. sRGB is decoded HERE for float textures, and by the sampler for 8-bit
 * ones.** THREE's `SRGBColorSpace` maps to the hardware `SRGB8_ALPHA8` sampler,
 * which is exact and free — but only exists for 8-bit textures. Rather than
 * depend on what THREE does with `colorSpace` on a float texture (which has
 * varied across versions, and whose failure mode is a washed-out or over-dark
 * surface that still looks plausible), the float path linearizes explicitly with
 * the piecewise sRGB EOTF and declares itself linear. That makes the conversion
 * a known-value unit test instead of a visual judgement — which is the whole
 * reason to prefer it, since "looks about right" is exactly the check this class
 * of bug passes.
 *
 * @module rendering/mesh-texture
 */

import * as THREE from 'three';
import { log, Modules } from '../utils/log';
import type { MeshMetadata, MeshTextureData } from '../types/mesh';

/** What the upload needs to know about the running renderer. */
export interface MeshTextureCapabilities {
  /**
   * Whether a 32-bit float texture can be sampled with LINEAR filtering.
   *
   * `caps.hdr.filterableFloatTextures`. NOT `floatTextures`, which reports
   * whether the renderer can draw *into* a float buffer — a different extension
   * that a device may have without this one.
   */
  filterableFloatTextures: boolean;
  /** Upload-time anisotropy ceiling supplied by the material manager. */
  maxAnisotropy?: number;
}

/** Decode one sRGB-encoded channel value in `[0, 1]` to linear light. */
function srgbToLinearChannel(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/**
 * Linearize an sRGB-encoded float texture in place, leaving alpha alone.
 *
 * Alpha is coverage, not colour: it is never gamma-encoded, and running it
 * through the EOTF would make every partially transparent texel more transparent
 * than authored. The same exclusion the writer applies when it computes
 * `texture_data_range` over RGB only.
 */
export function linearizeSRGBFloat(pixels: Float32Array, channels: number): Float32Array {
  const colourChannels = Math.min(channels, 3);
  for (let i = 0; i < pixels.length; i += channels) {
    for (let c = 0; c < colourChannels; c++) {
      pixels[i + c] = srgbToLinearChannel(pixels[i + c]);
    }
  }
  return pixels;
}

/**
 * Widen raw texels to float32 in `[0, 1]`, normalizing an integer source.
 *
 * `Uint16Array` reaches here only from an UNENCODED 16-bit store — a quantized
 * HDR texture was already widened to float by the decoder — so it is genuine
 * 16-bit image data and is normalized by 65535 rather than truncated to 8 bits.
 */
function toFloat32(pixels: Float32Array | Uint8Array | Uint16Array): Float32Array {
  if (pixels instanceof Float32Array) return pixels;
  const divisor = pixels instanceof Uint16Array ? 65535 : 255;
  const out = new Float32Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) out[i] = pixels[i] / divisor;
  return out;
}

/** Expand 3-channel texels to 4, with an opaque alpha. See decision (1) above. */
function expandToRGBA<T extends Float32Array | Uint8Array>(pixels: T, opaque: number): T {
  const texels = pixels.length / 3;
  const out = new (pixels.constructor as new (n: number) => T)(texels * 4);
  for (let t = 0; t < texels; t++) {
    out[t * 4] = pixels[t * 3];
    out[t * 4 + 1] = pixels[t * 3 + 1];
    out[t * 4 + 2] = pixels[t * 3 + 2];
    out[t * 4 + 3] = opaque;
  }
  return out;
}

/** THREE filter constants for the authored `texture_filter`, defaulting linear. */
function resolveFilters(attrs: MeshMetadata): {
  magFilter: THREE.MagnificationTextureFilter;
  minFilter: THREE.MinificationTextureFilter;
  mipmaps: boolean;
} {
  if (attrs.texture_filter === 'nearest') {
    // No mipmaps on the nearest path. A mipmap chain is built by averaging, which
    // is precisely what `nearest` was chosen to avoid — on a categorical texture
    // it would blend two class ids into a third that means nothing, so the
    // minified levels would invent categories the data does not contain.
    return {
      magFilter: THREE.NearestFilter,
      minFilter: THREE.NearestFilter,
      mipmaps: false,
    };
  }
  return {
    magFilter: THREE.LinearFilter,
    minFilter: THREE.LinearMipmapLinearFilter,
    mipmaps: true,
  };
}

/**
 * THREE wrap constants for the authored `texture_wrap`.
 *
 * The default is deliberately ASYMMETRIC — repeat in u, clamp in v — because the
 * shape this feature exists for is an equirectangular basemap. Longitude is
 * genuinely periodic, so u must wrap for the dateline seam to close; latitude is
 * not, so a wrapping v bleeds the north pole into the south. A symmetric default
 * either breaks the seam or corrupts the poles, and both look like a UV bug.
 */
function resolveWrap(attrs: MeshMetadata): { wrapS: THREE.Wrapping; wrapT: THREE.Wrapping } {
  if (attrs.texture_wrap === 'clamp') {
    return { wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping };
  }
  if (attrs.texture_wrap === 'repeat') {
    return { wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping };
  }
  return { wrapS: THREE.RepeatWrapping, wrapT: THREE.ClampToEdgeWrapping };
}

/**
 * Build a GPU texture from a decoded mesh texture payload.
 *
 * The caller owns disposal. For the `bitmap` arm the returned texture holds the
 * `ImageBitmap`, so disposing the texture is what releases it. The `compressed`
 * arm returns the decoder-owned texture instance after applying mesh sampling
 * state; `applyMeshTexture` transfers that instance to geometry-owned disposal.
 */
export function createMeshTexture(
  data: MeshTextureData,
  attrs: MeshMetadata,
  caps: MeshTextureCapabilities,
  nodePath = ''
): THREE.Texture {
  const { magFilter, minFilter, mipmaps } = resolveFilters(attrs);
  const { wrapS, wrapT } = resolveWrap(attrs);
  const srgb = attrs.texture_color_space !== 'linear';

  let texture: THREE.Texture;

  if (data.kind === 'compressed') {
    texture = data.texture;
    texture.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  } else if (data.kind === 'bitmap') {
    // The browser already produced an 8-bit RGBA surface, so the hardware sRGB
    // sampler is available and exact — no CPU pass, and no precision lost.
    texture = new THREE.Texture(data.bitmap);
    texture.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  } else {
    const isFloat = !(data.pixels instanceof Uint8Array);
    if (!isFloat) {
      // 8-bit: keep it 8-bit. The GPU normalizes to [0, 1] for free and the sRGB
      // sampler decodes for free, so widening would cost 4x the memory to reach
      // the same sampled value.
      const bytes = data.pixels as Uint8Array;
      const rgba = data.channels === 3 ? expandToRGBA(bytes, 255) : bytes;
      const format = data.channels === 1 ? THREE.RedFormat : THREE.RGBAFormat;
      texture = new THREE.DataTexture(
        rgba,
        data.width,
        data.height,
        format,
        THREE.UnsignedByteType
      );
      texture.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    } else {
      let floats = toFloat32(data.pixels);
      if (srgb) {
        // Decision (3): decoded here, not by the sampler, so the conversion is
        // pinned by a known-value test rather than by how it looks.
        floats = linearizeSRGBFloat(floats, data.channels);
      }
      const expanded = data.channels === 3 ? expandToRGBA(floats, 1) : floats;
      const format = data.channels === 1 ? THREE.RedFormat : THREE.RGBAFormat;
      // Half-float unless the device confirms it can LINEARLY filter float32 —
      // the silent-fallback-to-nearest trap in decision (2).
      const useFloat32 = caps.filterableFloatTextures && magFilter === THREE.LinearFilter;
      texture = new THREE.DataTexture(
        useFloat32 ? expanded : toHalfFloat(expanded),
        data.width,
        data.height,
        format,
        useFloat32 ? THREE.FloatType : THREE.HalfFloatType
      );
      // Already linear light, whichever branch ran: either the source declared
      // itself linear, or `linearizeSRGBFloat` made it so. Declaring sRGB here
      // would apply the transfer function a second time.
      texture.colorSpace = THREE.NoColorSpace;
    }
  }

  // Decision (3): pinned off on ALL arms, so `texture_encoding` cannot change
  // what a UV means. Not left to the defaults — they differ by arm, and the
  // ImageBitmap/compressed upload flags are ignored by WebGL.
  texture.flipY = false;
  texture.magFilter = magFilter;
  texture.minFilter =
    data.kind === 'compressed' && texture.mipmaps.length <= 1 && mipmaps
      ? THREE.LinearFilter
      : minFilter;
  texture.wrapS = wrapS;
  texture.wrapT = wrapT;
  texture.generateMipmaps = data.kind === 'compressed' ? false : mipmaps;
  // Anisotropy matters more here than for any existing texture: a globe is viewed
  // at grazing incidence near its silhouette, which is exactly where isotropic
  // mipmapping blurs along the wrong axis. Clamped to the material manager's
  // conservative upload ceiling.
  texture.anisotropy = mipmaps ? Math.max(1, Math.min(8, caps.maxAnisotropy ?? 1)) : 1;
  texture.needsUpdate = true;

  log.info(
    Modules.RENDERER,
    `${nodePath || 'mesh'}: ${data.width}x${data.height}x${data.channels} ` +
      `${data.kind}, ${srgb ? 'sRGB' : 'linear'}, ${magFilter === THREE.NearestFilter ? 'nearest' : 'linear'}` +
      `${mipmaps ? '+mips' : ''}`
  );
  return texture;
}

/**
 * Convert float32 texels to half-float bit patterns.
 *
 * `THREE.DataUtils.toHalfFloat` clamps to the half-float range and is the same
 * helper THREE's own loaders use, so an HDR value above 65504 saturates rather
 * than becoming Infinity.
 */
function toHalfFloat(floats: Float32Array): Uint16Array {
  const out = new Uint16Array(floats.length);
  for (let i = 0; i < floats.length; i++) {
    out[i] = THREE.DataUtils.toHalfFloat(floats[i]);
  }
  return out;
}
