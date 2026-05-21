/**
 * Decode quantized data (uint8/uint16) to float32 using WASM.
 *
 * Main thread fetches raw bytes from cache, worker dequantizes via WASM.
 * Uses Comlink.transfer for zero-copy return.
 *
 * WASM provides 2-3x speedup for large arrays (>10K elements).
 */

import { transfer } from 'comlink';
import { requireWasm, type WasmCtx } from '../state';
import { validateDecodeArgs } from '../validation';

export async function decodeQuantized(
  ctx: WasmCtx,
  params: {
    data: Uint8Array | Uint16Array;
    bounds: [number, number];
    dtype: 'uint8' | 'uint16';
  }
): Promise<Float32Array> {
  const wasmModule = requireWasm(ctx);

  const { data, bounds, dtype } = params;
  validateDecodeArgs('decodeQuantized', data, {
    boundsPair: { name: 'bounds', bounds },
  });
  const [minVal, maxVal] = bounds;

  const result = new Float32Array(data.length);

  // Use WASM for decoding
  if (dtype === 'uint8') {
    wasmModule.decode_quantized_u8(data as Uint8Array, minVal, maxVal, result);
  } else {
    wasmModule.decode_quantized_u16(data as Uint16Array, minVal, maxVal, result);
  }

  // Transfer ownership to main thread (zero-copy)
  return transfer(result, [result.buffer]);
}
