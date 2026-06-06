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
  // Strict contract (matches the Rust kernel + TS reference): value must be a
  // scalar (length 1) or exactly elementsPerPoint values. Every other shape is
  // rejected here — this is the single enforcement point, since the WASM kernel
  // assumes a valid shape and the old "mixed broadcast" pad/truncate behavior
  // was removed for Rust↔TS parity.
  if (value.length !== 1 && value.length !== elementsPerPoint) {
    throw new Error(
      `decodeBroadcasted: value.length must be 1 (scalar broadcast) or elementsPerPoint (${elementsPerPoint}), got ${value.length}`
    );
  }
  const result = new Float32Array(numPoints * elementsPerPoint);

  // Use WASM for broadcasting
  wasmModule.decode_broadcasted(value, numPoints, elementsPerPoint, result);

  return transfer(result, [result.buffer]);
}
