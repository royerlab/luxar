/**
 * Decode log-space quantized data (uint8/uint16) to float32 using WASM.
 *
 * Used for positive scalars with wide dynamic range (e.g., radii).
 * Decoding: expm1(normalized * maxLog)
 *
 * WASM provides 2-3x speedup for large arrays.
 */

import { transfer } from 'comlink';
import { requireWasm, type WasmCtx } from '../state';
import { validateDecodeArgs } from '../validation';

export async function decodeLogScalar(
  ctx: WasmCtx,
  params: {
    data: Uint8Array | Uint16Array;
    maxLog: number;
    dtype: 'uint8' | 'uint16';
  }
): Promise<Float32Array> {
  const wasmModule = requireWasm(ctx);

  const { data, maxLog, dtype } = params;
  validateDecodeArgs('decodeLogScalar', data, {
    finiteScalar: { name: 'maxLog', value: maxLog },
  });

  const result = new Float32Array(data.length);

  // Use WASM for decoding
  if (dtype === 'uint8') {
    wasmModule.decode_log_scalar_u8(data as Uint8Array, maxLog, result);
  } else {
    wasmModule.decode_log_scalar_u16(data as Uint16Array, maxLog, result);
  }

  // Transfer ownership to main thread (zero-copy)
  return transfer(result, [result.buffer]);
}
