/**
 * Mesh texture fetch + decode, and the Stage 2 check that the decoded surface is
 * the one the store declared.
 *
 * Its own module rather than a branch of `mesh-whole-node-loader.ts` for the
 * reason the writer split `_write_mesh_texture_arrays` out on the Python side: a
 * texture is the only array in a mesh that is not per-vertex, so it shares none
 * of the loader's `nVertices`-shaped machinery and every one of its steps is
 * conditional on an encoding the other arrays do not have.
 *
 * ## Three arms, and why they cannot share a decode
 *
 * `raw` is a numeric `(h, w, c)` array, so it goes through the normal Luxar
 * decode stack — which matters, because the raw arm is the one that carries HDR
 * and an HDR texture is stored as `geolog_perchannel_u16` codes under `AUTO`.
 * Reading its bytes directly would hand back quantization codes and render a
 * garbage image.
 *
 * A codec arm (`png` / `webp` / `jpeg`) is an opaque blob no Luxar encoder
 * touches, decoded by the browser via `createImageBitmap` — the same mechanism
 * `loaders/picking/image-label-loader.ts` already uses for hover thumbnails.
 *
 * The `ktx2` arm is also an opaque blob, but it must bypass browser image decode:
 * the renderer-owned KTX2 decoder transcodes it directly to a native compressed
 * GPU texture while preserving the authored mip chain.
 *
 * ## Stage 2 exists because Stage 1 cannot finish the job
 *
 * The preflight charges the decoded surface against the byte budget from the
 * DECLARED `texture_width`/`texture_height`, because a compressed image's stored
 * size says nothing about what it expands to. That admission is therefore only as
 * true as the declaration. So the moment real dimensions exist — after
 * `createImageBitmap` reports them — they are compared against what was declared
 * and a mismatch rejects the node. Without that, a store declares `16 x 16`, is
 * admitted for 1 KB, and detonates a 30000 x 30000 decode.
 *
 * @module data/mesh/texture-decode
 */

import * as zarr from '../zarr';
import { ArrayDecoder, type ArrayMetadata } from '../array-decoder/decoder';
import { LoaderError } from '../scene-loader/nodes/load-leaf-error-dispatch';
import { validateMaterializedLength } from './validate';
import type { TextureDeclaration } from './preflight';
import type {
  KTX2TextureDecoder,
  MeshColorArray,
  MeshTextureData,
  MeshTextureEncoding,
} from '../../types/mesh';

/**
 * MIME type per codec encoding, taken from the DECLARED encoding rather than
 * sniffed from the bytes.
 *
 * The declaration is the store's own contract and the preflight has already
 * closed it to this vocabulary, so sniffing would only add a second opinion for
 * the two to disagree about. A payload whose bytes contradict its declaration
 * fails in `createImageBitmap` and is reported as a decode failure, which is the
 * right outcome and a clearer one than silently decoding as something else.
 */
const CODEC_MIME: Record<MeshTextureEncoding, string | null> = {
  raw: null,
  png: 'image/png',
  webp: 'image/webp',
  jpeg: 'image/jpeg',
  ktx2: null,
};

/**
 * Dtypes a raw texture may keep in its native width.
 *
 * The same set and the same reasoning as `UNENCODED_COLOR_DTYPES` in
 * `preflight.ts`: the GPU normalizes `uint8`/`uint16` to `[0, 1]` for free, so
 * widening them to float32 would cost 4x/2x the memory to reach the identical
 * sampled value. It matters more here than for per-vertex colours — a 4096x4096
 * RGBA texture is 67 MB native and 268 MB widened, which is the difference
 * between comfortably inside the per-node budget and over it.
 *
 * Only reachable when the array is UNENCODED. Anything the encoder touched
 * (quantized HDR, a LUT) must go through the decoder, whose output is float32 by
 * construction.
 */
const NATIVE_TEXTURE_DTYPES = new Set(['uint8', '|u1', 'uint16', '<u2', 'float32', '<f4']);

/** Build the typed array a native read produced, in its own dtype. */
function asNativeTexels(data: ArrayLike<number> | ArrayBufferView): MeshColorArray | null {
  if (data instanceof Uint8Array || data instanceof Uint16Array || data instanceof Float32Array) {
    return data;
  }
  return null;
}

/**
 * Fetch and decode a mesh texture into an upload-ready payload.
 *
 * @param path Scene path of the node, for error attribution.
 * @param handle The `texture` zarr array.
 * @param declared The preflight's VALIDATED declaration — never `attrs`, so the
 *   dimensions this allocates from are the same ones the budget was checked
 *   against.
 * @param decoder Shared array decoder, for the encoded raw arm.
 * @param storeRoot Store root, for an `array_ref` hop.
 * @param signal Abort signal, honoured by the zarr reads.
 */
export async function decodeMeshTexture(
  path: string,
  handle: zarr.Array<zarr.DataType, zarr.Readable>,
  declared: TextureDeclaration,
  decoder: ArrayDecoder,
  storeRoot: zarr.Location<zarr.Readable>,
  decodeKTX2?: KTX2TextureDecoder,
  signal?: AbortSignal
): Promise<MeshTextureData> {
  const { width, height, channels } = declared;

  if (declared.decode === 'codec' || declared.decode === 'ktx2') {
    const raw = await zarr.readArray(handle, undefined, zarr.abortOptions(signal));
    const bytes = asNativeTexels(raw.data as ArrayBufferView);
    if (!(bytes instanceof Uint8Array)) {
      throw new LoaderError(
        'Validation',
        path,
        new Error(
          `texture is ${declared.encoding}-encoded but its array is not uint8 bytes ` +
            `(dtype '${String(handle.dtype)}'). An encoded payload is an opaque byte ` +
            'blob; any other dtype means the store is describing something else.'
        )
      );
    }
    if (declared.decode === 'ktx2') {
      if (!decodeKTX2) {
        throw new LoaderError(
          'Validation',
          path,
          new Error(
            "texture encoding 'ktx2' has no renderer-owned decoder configured. " +
              "Initialize the viewer renderer before loading the scene, or use 'raw' or " +
              "'jpeg' for a portable texture."
          )
        );
      }
      let texture: import('three').CompressedTexture;
      try {
        texture = await decodeKTX2(path, new Uint8Array(bytes));
      } catch (error) {
        throw new LoaderError(
          'Validation',
          path,
          new Error(
            `texture failed to decode as ktx2 (${bytes.length.toLocaleString()} bytes). ` +
              `${error instanceof Error ? error.message : String(error)}`
          )
        );
      }
      const image = texture.image as { width?: number; height?: number } | undefined;
      if (image?.width !== width || image?.height !== height) {
        texture.dispose();
        throw new LoaderError(
          'Validation',
          path,
          new Error(
            `texture decoded to ${image?.width ?? '?'}x${image?.height ?? '?'} but declares ` +
              `${width}x${height}; refusing a declaration that would invalidate the preflight budget.`
          )
        );
      }
      return { kind: 'compressed', texture, width, height, channels: channels as 3 | 4 };
    }

    // A fresh copy so the Blob owns a plain ArrayBuffer, exactly as
    // `image-label-loader.ts` does — a typed-array view over a larger buffer
    // would hand the decoder the wrong bytes.
    // Keyed over EVERY encoding, with `raw` and `ktx2` mapped to null, so adding
    // a new encoding to the contract is a compile error here rather than an
    // `undefined` MIME the browser silently sniffs around. The null case is
    // unreachable — `raw` is excluded by `decode === 'codec'`, while `ktx2`
    // returns through the renderer-owned decoder above — but is checked rather
    // than asserted, since the two tables agreeing is an invariant across two
    // files and nothing else enforces it.
    const mime = CODEC_MIME[declared.encoding];
    if (mime === null) {
      throw new LoaderError(
        'Unexpected',
        path,
        new Error(`texture encoding '${declared.encoding}' has no codec MIME type`)
      );
    }
    const blob = new Blob([new Uint8Array(bytes)], { type: mime });
    let bitmap: ImageBitmap;
    try {
      bitmap = await createImageBitmap(blob);
    } catch (error) {
      throw new LoaderError(
        'Validation',
        path,
        new Error(
          `texture failed to decode as ${declared.encoding} (${bytes.length.toLocaleString()} ` +
            `bytes). ${error instanceof Error ? error.message : String(error)}`
        )
      );
    }

    // Stage 2. The bitmap is already allocated at this point — the browser
    // decoded it — so this cannot prevent the allocation, and it is not trying
    // to: the BUDGET prevented it, using the declared numbers. What this stops is
    // the declaration being a lie that goes unnoticed, which is what would make
    // the budget meaningless for every future load.
    if (bitmap.width !== width || bitmap.height !== height) {
      const actual = `${bitmap.width}x${bitmap.height}`;
      bitmap.close();
      throw new LoaderError(
        'Validation',
        path,
        new Error(
          `texture decoded to ${actual} but declares ${width}x${height}. The declared ` +
            'dimensions are what the per-node byte budget admitted this node on, so a ' +
            'store whose real image is larger would bypass the ceiling entirely. ' +
            'Re-write the texture so its attrs match its data.'
        )
      );
    }
    return { kind: 'bitmap', bitmap, width, height, channels };
  }

  // `raw`. Native dtype when nothing encoded it, decoder otherwise.
  const attrs = (handle.attrs ?? {}) as unknown as ArrayMetadata;
  const expected = width * height * channels;
  if (!ArrayDecoder.isEncoded(attrs) && NATIVE_TEXTURE_DTYPES.has(String(handle.dtype))) {
    const raw = await zarr.readArray(handle, undefined, zarr.abortOptions(signal));
    const texels = asNativeTexels(raw.data as ArrayBufferView);
    if (texels) {
      validateMaterializedLength(path, 'texture', texels.length, expected);
      return { kind: 'raw', pixels: texels, width, height, channels };
    }
    // The dtype was in the native set but the read produced some other view.
    // Fall through to the decoder rather than trusting a mismatch — it always
    // yields Float32Array, so the payload stays well-formed.
  }
  const pixels = await decoder.decode(handle, attrs, expected, storeRoot, signal);
  validateMaterializedLength(path, 'texture', pixels.length, expected);
  return { kind: 'raw', pixels, width, height, channels };
}
