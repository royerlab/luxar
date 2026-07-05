/**
 * Decode geometric-log quantized data (uint8/uint16) to float32 using WASM.
 *
 * The min/max-anchored ("rescale first") log encoding for wide-dynamic-range
 * positive scalars (e.g. gsplat amplitudes): level 0 is RESERVED for exact
 * zeros; levels [1, 2^bits - 1] decode to
 * `exp(minLog + (u - 1)/(2^bits - 2) * (maxLog - minLog))` — uniform relative
 * precision across the array's own nonzero range.
 */

import { transfer } from 'comlink';
import { requireWasm, type WasmCtx } from '../state';
import { validateDecodeArgs } from '../validation';

export async function decodeGeologScalar(
  ctx: WasmCtx,
  params: {
    data: Uint8Array | Uint16Array;
    minLog: number;
    maxLog: number;
    dtype: 'uint8' | 'uint16';
  }
): Promise<Float32Array> {
  const wasmModule = requireWasm(ctx);

  const { data, minLog, maxLog, dtype } = params;
  validateDecodeArgs('decodeGeologScalar', data, {
    finiteScalar: { name: 'minLog', value: minLog },
  });
  validateDecodeArgs('decodeGeologScalar', data, {
    finiteScalar: { name: 'maxLog', value: maxLog },
  });

  const result = new Float32Array(data.length);

  if (dtype === 'uint8') {
    wasmModule.decode_geolog_scalar_u8(data as Uint8Array, minLog, maxLog, result);
  } else {
    wasmModule.decode_geolog_scalar_u16(data as Uint16Array, minLog, maxLog, result);
  }

  // Transfer ownership to main thread (zero-copy)
  return transfer(result, [result.buffer]);
}
