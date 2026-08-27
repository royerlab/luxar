import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import vm from 'node:vm';
import { RGBAFormat, RGBA_S3TC_DXT5_Format } from 'three';
import { describe, expect, it, vi } from 'vitest';

import { createKTX2TextureDecoder } from '../../../rendering/ktx2-texture-decoder';
import type { Renderer } from '../../../rendering/renderer-capabilities';

const loaderState = vi.hoisted(() => ({
  texture: null as import('three').CompressedTexture | null,
  dispose: vi.fn(),
}));

vi.mock('three/examples/jsm/loaders/KTX2Loader.js', () => ({
  KTX2Loader: class {
    detectSupport() {
      return this;
    }
    parse(_bytes: ArrayBuffer, resolve: (texture: import('three').CompressedTexture) => void) {
      if (!loaderState.texture) throw new Error('test texture not configured');
      resolve(loaderState.texture);
    }
    dispose() {
      loaderState.dispose();
    }
  },
}));

const supportedRenderer = () =>
  ({
    isWebGPURenderer: true,
    hasFeature: () => true,
  }) as unknown as Renderer;

const compressedTexture = (overrides: Record<string, unknown> = {}) =>
  ({
    format: RGBA_S3TC_DXT5_Format,
    image: { width: 8, height: 8 },
    dispose: vi.fn(),
    ...overrides,
  }) as unknown as import('three').CompressedTexture;

describe('createKTX2TextureDecoder', () => {
  it('rejects clearly when the renderer has no compressed-texture target', async () => {
    const renderer = {
      isWebGPURenderer: true,
      hasFeature: () => false,
    } as unknown as Renderer;

    const decode = createKTX2TextureDecoder(renderer);

    await expect(decode('mesh/texture', new Uint8Array(0))).rejects.toThrow(
      /mesh\/texture.*ktx2.*raw.*jpeg/i
    );
  });

  it('checks renderer support lazily after backend initialization', async () => {
    let initialized = false;
    const hasFeature = vi.fn(() => {
      if (!initialized) throw new Error('backend not initialized');
      return true;
    });
    const renderer = { isWebGPURenderer: true, hasFeature } as unknown as Renderer;
    const decode = createKTX2TextureDecoder(renderer);
    initialized = true;
    loaderState.texture = compressedTexture();

    await expect(decode('mesh/texture', new Uint8Array(1))).resolves.toBe(loaderState.texture);
    expect(hasFeature).toHaveBeenCalled();
  });

  it('rejects and disposes an uncompressed RGBA8 transcode fallback', async () => {
    const dispose = vi.fn();
    loaderState.texture = compressedTexture({ format: RGBAFormat, dispose });
    const decode = createKTX2TextureDecoder(supportedRenderer());

    await expect(decode('mesh/texture', new Uint8Array(1))).rejects.toThrow(
      /uncompressed RGBA8.*raw.*jpeg/i
    );
    expect(dispose).toHaveBeenCalledOnce();
  });

  it('rejects and disposes layered containers', async () => {
    const dispose = vi.fn();
    loaderState.texture = compressedTexture({ image: { width: 8, height: 8, depth: 2 }, dispose });
    const decode = createKTX2TextureDecoder(supportedRenderer());

    await expect(decode('mesh/texture', new Uint8Array(1))).rejects.toThrow(/2 layers.*2D image/i);
    expect(dispose).toHaveBeenCalledOnce();
  });

  it('rejects and disposes cubemap containers', async () => {
    const dispose = vi.fn();
    loaderState.texture = compressedTexture({ isCubeTexture: true, dispose });
    const decode = createKTX2TextureDecoder(supportedRenderer());

    await expect(decode('mesh/texture', new Uint8Array(1))).rejects.toThrow(/cubemap.*2D image/i);
    expect(dispose).toHaveBeenCalledOnce();
  });

  it('disposes the lazily-created loader worker pool', async () => {
    loaderState.dispose.mockClear();
    loaderState.texture = compressedTexture();
    const decode = createKTX2TextureDecoder(supportedRenderer());
    await decode('mesh/texture', new Uint8Array(1));

    decode.dispose();
    expect(loaderState.dispose).toHaveBeenCalledOnce();
  });
});

describe('KTX2 fixtures', () => {
  it('parses the authored UASTC+zstd RGBA container with its alpha slice and mip chain', async () => {
    // Three ships this parser without TypeScript declarations.
    // @ts-expect-error untyped Three.js example module
    const { read } = await import('three/examples/jsm/libs/ktx-parse.module.js');
    const bytes = new Uint8Array(
      readFileSync(new URL('../data/mesh/fixtures/2d_uastc_zstd_rgba.ktx2', import.meta.url))
    );
    const container = read(bytes);

    expect(container).toMatchObject({
      pixelWidth: 8,
      pixelHeight: 8,
      levelCount: 4,
      supercompressionScheme: 2,
      vkFormat: 0,
    });
    expect(container.dataFormatDescriptor[0].samples).toHaveLength(1);
    expect(container.dataFormatDescriptor[0].samples[0].channelType & 0xf).toBe(3);

    const require = createRequire(import.meta.url);
    const basisScript = require.resolve('three/examples/jsm/libs/basis/basis_transcoder.js');
    const basisSource = readFileSync(basisScript, 'utf8');
    const basisFactory = vm.runInNewContext(`${basisSource}\nBASIS;`, {
      require,
      process,
      __filename: basisScript,
      __dirname: dirname(basisScript),
      console,
      WebAssembly,
      Buffer,
      URL,
      setTimeout,
      clearTimeout,
      TextDecoder,
      TextEncoder,
    }) as (options: { wasmBinary: Uint8Array }) => Promise<{
      initializeBasis(): void;
      KTX2File: new (data: Uint8Array) => {
        isValid(): boolean;
        isUASTC(): boolean;
        getHasAlpha(): boolean;
        startTranscoding(): number;
        getImageTranscodedSizeInBytes(
          level: number,
          layer: number,
          face: number,
          format: number
        ): number;
        transcodeImage(
          target: Uint8Array,
          level: number,
          layer: number,
          face: number,
          format: number,
          decodeFlags: number,
          channel0: number,
          channel1: number
        ): number;
        close(): void;
        delete(): void;
      };
    }>;
    const basis = await basisFactory({
      wasmBinary: readFileSync(join(dirname(basisScript), 'basis_transcoder.wasm')),
    });
    basis.initializeBasis();
    const texture = new basis.KTX2File(bytes);
    try {
      expect(texture.isValid()).toBe(true);
      expect(texture.isUASTC()).toBe(true);
      expect(texture.getHasAlpha()).toBe(true);
      expect(texture.startTranscoding()).toBe(1);
      const rgba32TranscoderFormat = 13;
      const pixels = new Uint8Array(
        texture.getImageTranscodedSizeInBytes(0, 0, 0, rgba32TranscoderFormat)
      );
      expect(texture.transcodeImage(pixels, 0, 0, 0, rgba32TranscoderFormat, 0, -1, -1)).toBe(1);
      const alphas = Array.from(pixels.filter((_, index) => index % 4 === 3));
      const expected = Array.from({ length: 64 }, (_, index) =>
        (Math.floor(index / 8) + (index % 8)) % 2 === 0 ? 255 : 32
      );
      expect(alphas).toEqual(expected);
    } finally {
      texture.close();
      texture.delete();
    }
  });
});
