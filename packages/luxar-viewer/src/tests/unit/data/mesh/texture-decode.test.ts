/**
 * Mesh texture fetch + decode, and the Stage 2 dimension check.
 *
 * The check worth testing hardest is the one Stage 1 structurally CANNOT make:
 * the preflight admits an encoded texture on the strength of its declared
 * dimensions, because a codec payload's stored length says nothing about what it
 * expands to. That admission is only as true as the declaration, so the moment
 * real dimensions exist they have to be compared against it — otherwise a store
 * declares 16x16, is admitted for a kilobyte, and detonates a 30000x30000 decode.
 */

import { readFileSync } from 'node:fs';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { decodeMeshTexture } from '../../../../data/mesh/texture-decode';
import { ArrayDecoder, ArrayRefRegistry } from '../../../../data/array-decoder/decoder';
import { LoaderError } from '../../../../data/scene-loader/nodes/load-leaf-error-dispatch';
import type { TextureDeclaration } from '../../../../data/mesh/preflight';
import type * as zarr from '../../../../data/zarr';

const PATH = '/globe';

/** A declaration as the preflight would have validated it. */
const decl = (o: Partial<TextureDeclaration> = {}): TextureDeclaration => ({
  encoding: 'png',
  width: 4,
  height: 2,
  channels: 4,
  decode: 'codec',
  ...o,
});

/**
 * A zarr array handle whose read returns `data`.
 *
 * `decodeMeshTexture` reads through `zarr.readArray`, which is mocked below, so
 * the handle only has to carry the metadata the function inspects.
 */
const handle = (dtype: string, attrs: Record<string, unknown> = {}) =>
  ({ shape: [8], chunks: [8], dtype, attrs }) as unknown as zarr.Array<
    zarr.DataType,
    zarr.Readable
  >;

const storeRoot = {} as zarr.Location<zarr.Readable>;

/** A decoder over an empty ref registry — no `array_ref` hops in these fixtures. */
const newDecoder = () => new ArrayDecoder(new ArrayRefRegistry());

/** What the mocked `zarr.readArray` will hand back next. */
let nextRead: { data: ArrayBufferView } = { data: new Uint8Array(8) };

vi.mock('../../../../data/zarr', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../../data/zarr')>();
  return {
    ...actual,
    readArray: vi.fn(async () => nextRead),
    abortOptions: vi.fn(() => ({})),
  };
});

/** A stub `ImageBitmap` recording whether it was closed. */
function fakeBitmap(width: number, height: number) {
  return {
    width,
    height,
    closed: false,
    close(this: { closed: boolean }) {
      this.closed = true;
    },
  };
}

describe('decodeMeshTexture — encoded payloads', () => {
  let bitmap: ReturnType<typeof fakeBitmap>;

  beforeEach(() => {
    nextRead = { data: new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1, 2, 3, 4]) };
    bitmap = fakeBitmap(4, 2);
    vi.stubGlobal(
      'createImageBitmap',
      vi.fn(async () => bitmap as unknown as ImageBitmap)
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it('decodes to a bitmap payload carrying the verified dimensions', async () => {
    const result = await decodeMeshTexture(PATH, handle('|u1'), decl(), newDecoder(), storeRoot);
    expect(result).toMatchObject({ kind: 'bitmap', width: 4, height: 2, channels: 4 });
  });

  it.each([
    ['larger than declared', 30000, 30000],
    ['smaller than declared', 2, 1],
    ['transposed', 2, 4],
  ])('refuses a bitmap %s', async (_label, w, h) => {
    bitmap = fakeBitmap(w, h);
    await expect(
      decodeMeshTexture(PATH, handle('|u1'), decl(), newDecoder(), storeRoot)
    ).rejects.toThrow(/decoded to .* but declares/);
  });

  it('closes the bitmap it is about to reject', async () => {
    // Otherwise the rejection leaks the very allocation it exists to complain
    // about — and an ImageBitmap's surface is outside the JS heap, so nothing
    // else will reclaim it.
    bitmap = fakeBitmap(8, 8);
    await expect(
      decodeMeshTexture(PATH, handle('|u1'), decl(), newDecoder(), storeRoot)
    ).rejects.toThrow(LoaderError);
    expect(bitmap.closed).toBe(true);
  });

  it('picks the MIME type from the declared encoding', async () => {
    // Typed with the Blob parameter so the recorded call is indexable — a
    // zero-arg mock records a `[]` tuple and `calls[0][0]` will not compile.
    const spy = vi.fn(async (_blob: Blob) => bitmap as unknown as ImageBitmap);
    vi.stubGlobal('createImageBitmap', spy);
    await decodeMeshTexture(
      PATH,
      handle('|u1'),
      decl({ encoding: 'jpeg' }),
      newDecoder(),
      storeRoot
    );
    expect(spy.mock.calls[0]?.[0].type).toBe('image/jpeg');
  });

  it('refuses an encoded payload that is not uint8 bytes', async () => {
    nextRead = { data: new Float32Array([1, 2, 3, 4]) };
    await expect(
      decodeMeshTexture(PATH, handle('<f4'), decl(), newDecoder(), storeRoot)
    ).rejects.toThrow(/opaque byte blob/);
  });

  it('reports a codec failure as a validation error, not a crash', async () => {
    vi.stubGlobal(
      'createImageBitmap',
      vi.fn(async () => {
        throw new Error('unsupported image format');
      })
    );
    await expect(
      decodeMeshTexture(PATH, handle('|u1'), decl(), newDecoder(), storeRoot)
    ).rejects.toThrow(/failed to decode as png/);
  });
});

describe('decodeMeshTexture — KTX2 payloads', () => {
  const fixture = new Uint8Array(
    readFileSync(new URL('./fixtures/2d_etc1s.ktx2', import.meta.url))
  );
  const ktxDecl = decl({
    encoding: 'ktx2',
    decode: 'ktx2',
    width: 40,
    height: 40,
    channels: 3,
  });

  beforeEach(() => {
    nextRead = { data: fixture };
  });

  it('returns the renderer-transcoded compressed texture without a bitmap decode', async () => {
    const texture = {
      image: { width: 40, height: 40 },
      dispose: vi.fn(),
    } as unknown as import('three').CompressedTexture;
    const decodeKTX2 = Object.assign(
      vi.fn(async () => texture),
      { dispose: vi.fn() }
    );
    const result = await decodeMeshTexture(
      PATH,
      handle('|u1'),
      ktxDecl,
      newDecoder(),
      storeRoot,
      decodeKTX2
    );
    expect(result).toMatchObject({
      kind: 'compressed',
      texture,
      width: 40,
      height: 40,
      channels: 3,
    });
    expect(decodeKTX2).toHaveBeenCalledWith(PATH, expect.any(Uint8Array));
  });

  it('rejects clearly when the host did not configure a KTX2 decoder', async () => {
    await expect(
      decodeMeshTexture(PATH, handle('|u1'), ktxDecl, newDecoder(), storeRoot)
    ).rejects.toThrow(/no renderer-owned decoder configured.*raw.*jpeg/);
  });

  it('wraps transcode failures with path and payload context', async () => {
    const decodeKTX2 = Object.assign(
      vi.fn(async () => {
        throw new Error('invalid KTX2 identifier');
      }),
      { dispose: vi.fn() }
    );
    await expect(
      decodeMeshTexture(PATH, handle('|u1'), ktxDecl, newDecoder(), storeRoot, decodeKTX2)
    ).rejects.toThrow(/failed to decode as ktx2.*966 bytes.*invalid KTX2 identifier/i);
  });

  it('disposes a transcode whose dimensions contradict the declaration', async () => {
    const dispose = vi.fn();
    const decodeKTX2 = Object.assign(
      vi.fn(
        async () =>
          ({
            image: { width: 8, height: 8 },
            dispose,
          }) as unknown as import('three').CompressedTexture
      ),
      { dispose: vi.fn() }
    );
    await expect(
      decodeMeshTexture(PATH, handle('|u1'), ktxDecl, newDecoder(), storeRoot, decodeKTX2)
    ).rejects.toThrow(/decoded to 8x8 but declares 40x40/);
    expect(dispose).toHaveBeenCalledOnce();
  });
});

describe('decodeMeshTexture — raw payloads', () => {
  const rawDecl = decl({ encoding: 'raw', decode: 'raw', width: 2, height: 2, channels: 3 });

  it('keeps a plain uint8 texture in its native dtype', async () => {
    // Widening to float32 would cost 4x the memory to reach the identical
    // sampled value, since the GPU normalizes uint8 to [0, 1] for free. At
    // 4096x4096 RGBA that is the difference between inside the budget and over.
    nextRead = { data: new Uint8Array(12) };
    const result = await decodeMeshTexture(PATH, handle('|u1'), rawDecl, newDecoder(), storeRoot);
    expect(result.kind).toBe('raw');
    if (result.kind === 'raw') expect(result.pixels).toBeInstanceOf(Uint8Array);
  });

  it('refuses a raw texture whose value count disagrees with its declaration', async () => {
    nextRead = { data: new Uint8Array(9) };
    await expect(
      decodeMeshTexture(PATH, handle('|u1'), rawDecl, newDecoder(), storeRoot)
    ).rejects.toThrow(LoaderError);
  });

  it('routes an ENCODED raw texture through the decoder, not a native read', async () => {
    // The HDR case. A quantized store holds u16 codes, so reading its bytes
    // natively would hand the GPU quantization codes and render a garbage image
    // — the dtype alone cannot decide this, only the encoding attr can.
    const decoder = newDecoder();
    const spy = vi.spyOn(decoder, 'decode').mockResolvedValue(new Float32Array(12) as never);
    const result = await decodeMeshTexture(
      PATH,
      handle('<u2', {
        encoding: {
          name: 'geolog_perchannel_u16',
          original_dtype: 'float32',
          original_shape: [2, 2, 3],
        },
      }),
      rawDecl,
      decoder,
      storeRoot
    );
    expect(spy).toHaveBeenCalledOnce();
    // The expected value count is derived from the DECLARATION, so the decoder
    // is told the same number the byte budget was checked against.
    expect(spy.mock.calls[0]?.[2]).toBe(12);
    if (result.kind === 'raw') expect(result.pixels).toBeInstanceOf(Float32Array);
  });
});
