/**
 * Decode broadcasted data (single value replicated to all points) using WASM.
 *
 * Used for uniform attributes (e.g., all points same color).
 *
 * WASM provides speedup for large point counts.
 */

import { transfer } from 'comlink';
import { requireWasm, type WasmCtx } from '../state';

export async function decodeBroadcasted(
  ctx: WasmCtx,
  params: {
    value: Float32Array;
    numPoints: number;
    elementsPerPoint: number;
  }
): Promise<Float32Array> {
  const wasmModule = requireWasm(ctx);

  const { value, numPoints, elementsPerPoint } = params;
  if (!Number.isInteger(numPoints) || numPoints < 0) {
    throw new Error(`decodeBroadcasted: numPoints=${numPoints} must be a non-negative integer`);
  }
  if (!Number.isInteger(elementsPerPoint) || elementsPerPoint < 1) {
    throw new Error(
      `decodeBroadcasted: elementsPerPoint=${elementsPerPoint} must be a positive integer`
    );
  }
  if (value.length < elementsPerPoint) {
    throw new Error(
      `decodeBroadcasted: value too short (got ${value.length}, expected ≥ ${elementsPerPoint})`
    );
  }
  const result = new Float32Array(numPoints * elementsPerPoint);

  // Use WASM for broadcasting
  wasmModule.decode_broadcasted(value, numPoints, elementsPerPoint, result);

  return transfer(result, [result.buffer]);
}
