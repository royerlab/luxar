/**
 * Decode per-channel quantized data (uint8/uint16) to float32 using WASM.
 *
 * The per-column-scale family (`linear_perchannel_*` for COORDINATE
 * positions/centers, `log_perchannel_*` for the Cholesky diagonal,
 * `signed_log_perchannel_*` for the off-diagonal, `geolog_perchannel_*`
 * for HDR colors). Each element's column is
 * its global flattened index modulo the column count — `colOffset` carries
 * the column phase of the range's first element so a mid-array range decodes
 * with the exact same column assignment as a whole-array decode.
 *
 * `zeroLevel: true` (the current writer for the log/signed-log pair) reserves
 * code 0 for exact zeros over nonzero-anchored scales; without the flag the
 * legacy all-levels mapping applies. Linear never carries the flag (mirrors
 * the Python `_decode_linear_perchannel`).
 */

import { transfer } from 'comlink';
import { requireWasm, type WasmCtx } from '../state';
import { validateDecodeArgs } from '../validation';

export type PerChannelKind = 'linear' | 'log' | 'signed_log' | 'geolog';

export async function decodePerChannel(
  ctx: WasmCtx,
  params: {
    data: Uint8Array | Uint16Array;
    kind: PerChannelKind;
    /** Per-column companded-domain scales (f64, straight from the attrs). */
    colLo: Float64Array;
    colHi: Float64Array;
    /** Reserved-zero layout flag (log/signed-log only; ignored for linear). */
    zeroLevel: boolean;
    /** Column phase of the first element (`flatStart % cols`). */
    colOffset: number;
    /** Quantization bits from the encoding attrs (authoritative over dtype). */
    bits: 8 | 16;
  }
): Promise<Float32Array> {
  const wasmModule = requireWasm(ctx);

  const { data, kind, colLo, colHi, zeroLevel, colOffset, bits } = params;
  validateDecodeArgs('decodePerChannel', data, {
    positiveInt: { name: 'cols', value: colLo.length },
  });
  // The scales were validated (finite, hi >= lo, length == cols) on the main
  // thread by `makePerChannelDequant` before dispatch; re-check the structural
  // invariants the WASM kernel indexes by, so a malformed payload fails loud
  // at the boundary instead of reading garbage columns.
  if (colHi.length !== colLo.length) {
    throw new Error(
      `decodePerChannel: colLo/colHi length mismatch (${colLo.length} vs ${colHi.length})`
    );
  }
  if (!Number.isInteger(colOffset) || colOffset < 0) {
    throw new Error(`decodePerChannel: colOffset=${colOffset} must be a non-negative integer`);
  }
  // `bits` (from the encoding attrs) is authoritative for the level count and
  // must match the stored container — the kernels hardcode 255/65535 per dtype.
  const expected8 = bits === 8;
  if (expected8 !== data instanceof Uint8Array) {
    throw new Error(
      `decodePerChannel: bits=${bits} does not match data container ${data.constructor.name}`
    );
  }

  const result = new Float32Array(data.length);

  if (kind === 'linear') {
    if (expected8) {
      wasmModule.decode_linear_perchannel_u8(data as Uint8Array, colLo, colHi, colOffset, result);
    } else {
      wasmModule.decode_linear_perchannel_u16(data as Uint16Array, colLo, colHi, colOffset, result);
    }
  } else if (kind === 'geolog') {
    // TRUE-log (HDR colors): reserved zero level is the name contract, so
    // the kernels take no zeroLevel flag.
    if (expected8) {
      wasmModule.decode_geolog_perchannel_u8(data as Uint8Array, colLo, colHi, colOffset, result);
    } else {
      wasmModule.decode_geolog_perchannel_u16(data as Uint16Array, colLo, colHi, colOffset, result);
    }
  } else if (kind === 'log') {
    if (expected8) {
      wasmModule.decode_log_perchannel_u8(
        data as Uint8Array,
        colLo,
        colHi,
        zeroLevel,
        colOffset,
        result
      );
    } else {
      wasmModule.decode_log_perchannel_u16(
        data as Uint16Array,
        colLo,
        colHi,
        zeroLevel,
        colOffset,
        result
      );
    }
  } else if (kind === 'signed_log') {
    if (expected8) {
      wasmModule.decode_signed_log_perchannel_u8(
        data as Uint8Array,
        colLo,
        colHi,
        zeroLevel,
        colOffset,
        result
      );
    } else {
      wasmModule.decode_signed_log_perchannel_u16(
        data as Uint16Array,
        colLo,
        colHi,
        zeroLevel,
        colOffset,
        result
      );
    }
  } else {
    throw new Error(`decodePerChannel: unknown kind '${kind as string}'`);
  }

  // Transfer ownership to main thread (zero-copy)
  return transfer(result, [result.buffer]);
}
