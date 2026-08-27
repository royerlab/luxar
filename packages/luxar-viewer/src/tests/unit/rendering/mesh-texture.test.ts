/**
 * Mesh texture upload.
 *
 * The sRGB assertions are known-value rather than visual on purpose. A colour
 * space mistake here produces a washed-out or over-dark surface that still looks
 * like a plausible Earth, so "checked it by eye" is exactly the review this class
 * of bug passes — see the module docstring's decision (3).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { createMeshTexture, linearizeSRGBFloat } from '../../../rendering/mesh-texture';
import type { MeshMetadata, MeshTextureData } from '../../../types/mesh';

const attrs = (o: Partial<MeshMetadata> = {}): MeshMetadata =>
  ({
    type: 'mesh',
    n_vertices: 4,
    n_faces: 2,
    ndim: 3,
    has_normals: false,
    has_colors: false,
    has_scalars: false,
    has_uvs: true,
    has_texture: true,
    shading: 'none',
    double_sided: false,
    ordering: 'none',
    texture_encoding: 'raw',
    texture_width: 2,
    texture_height: 1,
    texture_channels: 4,
    texture_color_space: 'srgb',
    ...o,
  }) as MeshMetadata;

const raw = (pixels: Float32Array | Uint8Array | Uint16Array, channels = 4): MeshTextureData => ({
  kind: 'raw',
  pixels,
  width: 2,
  height: 1,
  channels,
});

const CAPS = { filterableFloatTextures: false, maxAnisotropy: 16 };

describe('linearizeSRGBFloat', () => {
  it.each([
    ['black', 0, 0],
    ['white', 1, 1],
    // Both sides of the piecewise breakpoint at 0.04045. A single-branch
    // implementation (pure 2.2 power, or pure linear) matches one and misses the
    // other, so testing only mid-grey would not distinguish them.
    ['just below the knee', 0.04, 0.04 / 12.92],
    ['just above the knee', 0.05, Math.pow((0.05 + 0.055) / 1.055, 2.4)],
    // The canonical check: sRGB 0.5 is ~0.2140 linear, NOT 0.5 and not 0.25.
    ['mid grey', 0.5, 0.21404114],
  ])('maps %s correctly', (_label, input, expected) => {
    const out = linearizeSRGBFloat(new Float32Array([input, input, input, 1]), 4);
    expect(out[0]).toBeCloseTo(expected, 6);
  });

  it('leaves alpha untouched', () => {
    // Alpha is coverage, never gamma-encoded. Running it through the EOTF would
    // make every partially transparent texel more transparent than authored —
    // a plausible-looking bug, since it only shows on translucent regions.
    const out = linearizeSRGBFloat(new Float32Array([0.5, 0.5, 0.5, 0.5]), 4);
    expect(out[3]).toBe(0.5);
    expect(out[0]).toBeCloseTo(0.21404114, 6);
  });

  it('converts only the colour channels of a 3-channel texture', () => {
    const out = linearizeSRGBFloat(new Float32Array([0.5, 0.5, 0.5]), 3);
    expect([...out].every((v) => Math.abs(v - 0.21404114) < 1e-6)).toBe(true);
  });
});

describe('createMeshTexture — formats', () => {
  it('uses a transcoded compressed texture directly and preserves its mip chain', () => {
    const compressed = new THREE.CompressedTexture([], 2, 1, THREE.RGBA_S3TC_DXT5_Format);
    compressed.generateMipmaps = false;
    const tex = createMeshTexture(
      { kind: 'compressed', texture: compressed, width: 2, height: 1, channels: 4 },
      attrs({ texture_encoding: 'ktx2' }),
      CAPS
    );
    expect(tex).toBe(compressed);
    expect(tex.generateMipmaps).toBe(false);
    expect(tex.colorSpace).toBe(THREE.SRGBColorSpace);
  });

  it('uses a non-mipmapped filter for a single-level compressed texture', () => {
    const compressed = new THREE.CompressedTexture([], 2, 1, THREE.RGBA_S3TC_DXT5_Format);
    compressed.mipmaps = [{ data: new Uint8Array(16), width: 2, height: 1 }];
    compressed.minFilter = THREE.LinearFilter;
    const tex = createMeshTexture(
      { kind: 'compressed', texture: compressed, width: 2, height: 1, channels: 4 },
      attrs({ texture_encoding: 'ktx2' }),
      CAPS
    );
    expect(tex.minFilter).toBe(THREE.LinearFilter);
    expect(tex.generateMipmaps).toBe(false);
  });

  it('keeps an 8-bit texture 8-bit and lets the sampler decode sRGB', () => {
    // Widening would cost 4x the memory to reach the identical sampled value:
    // the GPU normalizes uint8 to [0, 1] and decodes sRGB in hardware, both free.
    const tex = createMeshTexture(raw(new Uint8Array(8)), attrs(), CAPS);
    expect((tex as THREE.DataTexture).type).toBe(THREE.UnsignedByteType);
    expect(tex.colorSpace).toBe(THREE.SRGBColorSpace);
  });

  it('linearizes a float texture itself and declares no colour space', () => {
    // The float path must NOT also hand THREE an sRGB colorSpace — that would
    // apply the transfer function twice, darkening everything.
    const tex = createMeshTexture(
      raw(new Float32Array([0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 1])),
      attrs(),
      CAPS
    );
    expect(tex.colorSpace).toBe(THREE.NoColorSpace);
  });

  it.each([
    ['half-float without the capability', false, THREE.HalfFloatType],
    ['float32 with it', true, THREE.FloatType],
  ])('uploads HDR as %s', (_label, filterable, expected) => {
    // Core WebGL2 can hold a FloatType texture but cannot LINEARLY filter one;
    // when the extension is missing the sampler silently drops to nearest, so
    // the default has to be the type that filters everywhere.
    const tex = createMeshTexture(
      raw(new Float32Array([2, 3, 4, 1, 5, 6, 7, 1])),
      attrs({ texture_color_space: 'linear' }),
      { ...CAPS, filterableFloatTextures: filterable }
    );
    expect((tex as THREE.DataTexture).type).toBe(expected);
  });

  it('never uses float32 on the nearest path, capability or not', () => {
    // The capability only covers LINEAR filtering of float32, so it is not a
    // licence to use float32 when the author asked for nearest — but it is also
    // not harmful there. Pinned because the condition reads as if the filter were
    // incidental, and it is not.
    const tex = createMeshTexture(
      raw(new Float32Array([2, 3, 4, 1, 5, 6, 7, 1])),
      attrs({ texture_color_space: 'linear', texture_filter: 'nearest' }),
      { ...CAPS, filterableFloatTextures: true }
    );
    expect((tex as THREE.DataTexture).type).toBe(THREE.HalfFloatType);
  });

  it.each([
    ['3 channels to RGBA', 3, THREE.RGBAFormat],
    ['4 channels as RGBA', 4, THREE.RGBAFormat],
    ['1 channel as Red', 1, THREE.RedFormat],
  ])('expands %s', (_label, channels, expected) => {
    // RGBFormat is deliberately never used: WebGPU has no 3-channel texture
    // format, so a scene that worked under WebGL2 would break on the other
    // backend.
    const tex = createMeshTexture(
      raw(new Uint8Array(2 * channels), channels),
      attrs({ texture_channels: channels }),
      CAPS
    );
    expect((tex as THREE.DataTexture).format).toBe(expected);
  });

  it('gives an expanded 3-channel texture an opaque alpha', () => {
    const tex = createMeshTexture(
      raw(new Uint8Array([10, 20, 30, 40, 50, 60]), 3),
      attrs({ texture_channels: 3 }),
      CAPS
    );
    const data = (tex as THREE.DataTexture).image.data as Uint8Array;
    expect([...data]).toEqual([10, 20, 30, 255, 40, 50, 60, 255]);
  });

  it('normalizes a 16-bit source by 65535, not 255', () => {
    // A uint16 store reaches here only unencoded, so it is genuine 16-bit image
    // data — dividing by 255 would clip everything above 1/257 to white.
    const tex = createMeshTexture(
      raw(new Uint16Array([65535, 32768, 0, 65535, 0, 0, 0, 65535])),
      attrs({ texture_color_space: 'linear' }),
      { ...CAPS, filterableFloatTextures: true }
    );
    const data = (tex as THREE.DataTexture).image.data as Float32Array;
    expect(data[0]).toBeCloseTo(1, 6);
    expect(data[1]).toBeCloseTo(0.5, 3);
  });
});

describe('createMeshTexture — sampling', () => {
  it('defaults to repeat in u and clamp in v', () => {
    // Asymmetric because the target case is equirectangular: longitude is
    // periodic and must wrap for the dateline seam to close, latitude is not and
    // a wrapping v bleeds the north pole into the south.
    const tex = createMeshTexture(raw(new Uint8Array(8)), attrs(), CAPS);
    expect(tex.wrapS).toBe(THREE.RepeatWrapping);
    expect(tex.wrapT).toBe(THREE.ClampToEdgeWrapping);
  });

  it.each([
    ['repeat', THREE.RepeatWrapping, THREE.RepeatWrapping],
    ['clamp', THREE.ClampToEdgeWrapping, THREE.ClampToEdgeWrapping],
  ])('honours an authored %s wrap on both axes', (wrap, s, t) => {
    const tex = createMeshTexture(
      raw(new Uint8Array(8)),
      attrs({ texture_wrap: wrap as 'repeat' | 'clamp' }),
      CAPS
    );
    expect(tex.wrapS).toBe(s);
    expect(tex.wrapT).toBe(t);
  });

  it('mipmaps and filters linearly by default', () => {
    const tex = createMeshTexture(raw(new Uint8Array(8)), attrs(), CAPS);
    expect(tex.magFilter).toBe(THREE.LinearFilter);
    expect(tex.minFilter).toBe(THREE.LinearMipmapLinearFilter);
    expect(tex.generateMipmaps).toBe(true);
    expect(tex.anisotropy).toBeGreaterThan(1);
  });

  it('builds no mipmaps on the nearest path', () => {
    // A mipmap chain is built by AVERAGING, which is what `nearest` was chosen to
    // avoid: on a categorical texture the minified levels would blend two class
    // ids into a third that means nothing.
    const tex = createMeshTexture(
      raw(new Uint8Array(8)),
      attrs({ texture_filter: 'nearest' }),
      CAPS
    );
    expect(tex.magFilter).toBe(THREE.NearestFilter);
    expect(tex.minFilter).toBe(THREE.NearestFilter);
    expect(tex.generateMipmaps).toBe(false);
    expect(tex.anisotropy).toBe(1);
  });

  it('clamps anisotropy to the supplied upload ceiling', () => {
    const tex = createMeshTexture(raw(new Uint8Array(8)), attrs(), {
      ...CAPS,
      maxAnisotropy: 2,
    });
    expect(tex.anisotropy).toBe(2);
  });
});
