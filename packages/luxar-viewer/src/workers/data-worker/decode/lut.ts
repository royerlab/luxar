/**
 * Decode LUT-encoded data (indices → values via lookup table) using WASM.
 *
 * Supports two modes:
 * - "row": One index per row → k values (e.g., one index per point → xyz)
 * - "scalar": One index per element → 1 value
 *
 * WASM provides 2-4x speedup for large arrays.
 */

import { transfer } from 'comlink';
import { requireWasm, type WasmCtx } from '../state';
import { validateDecodeArgs } from '../validation';

export async function decodeLUT(
  ctx: WasmCtx,
  params: {
    indices: Uint8Array | Uint16Array;
    lut: number[];
    k: number;
    lutMode: 'row' | 'scalar';
    dtype?: 'uint8' | 'uint16'; // Optional - inferred from indices type if not provided
  }
): Promise<Float32Array> {
  const wasmModule = requireWasm(ctx);

  const { indices, lut, k, lutMode } = params;
  validateDecodeArgs('decodeLUT', indices, {
    positiveInt: { name: 'k', value: k },
    lut: {
      name: 'lut',
      values: lut,
      // Row mode requires the LUT to have ≥ k columns per entry; the
      // table is flat with `lut.length` total entries (= rows × k).
      minLength: lutMode === 'row' ? k : 1,
    },
  });
  if (lutMode !== 'row' && lutMode !== 'scalar') {
    throw new Error(`decodeLUT: lutMode='${lutMode}' must be 'row' or 'scalar'`);
  }

  // Scan indices for out-of-range values BEFORE handing off to WASM.
  // Rust functions (decode_lut_scalar_*, decode_lut_row_*) index
  // `lut[indices[i] as usize]` directly; an out-of-range index panics
  // or traps inside WASM. JS-side rejection turns malformed encoded
  // data into a clear error at the worker boundary.
  if (lutMode === 'row' && lut.length % k !== 0) {
    throw new Error(`decodeLUT: row-mode lut length ${lut.length} is not divisible by k=${k}`);
  }
  const entryCount = lutMode === 'row' ? Math.floor(lut.length / k) : lut.length;
  // Early-return guard: an empty indices array (or an empty LUT) has no
  // out-of-range scan work to do. Skipping the loop avoids the O(n) JS
  // overhead on the common no-op call path; the WASM call below also
  // becomes a no-op against the zero-length output buffer.
  if (indices.length > 0 && entryCount > 0) {
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] >= entryCount) {
        throw new Error(
          `decodeLUT: indices[${i}]=${indices[i]} out of range for ${entryCount} LUT ` +
            `entr${entryCount === 1 ? 'y' : 'ies'} (lutMode=${lutMode})`
        );
      }
    }
  } else if (indices.length > 0 && entryCount === 0) {
    // Empty LUT with non-empty indices is unambiguously a producer bug:
    // every index would be out-of-range. Report the first one explicitly
    // so the caller gets the same error shape as the in-loop branch.
    throw new Error(
      `decodeLUT: indices[0]=${indices[0]} out of range for 0 LUT entries (lutMode=${lutMode})`
    );
  }

  // Infer dtype from indices type if not explicitly provided
  const dtype = params.dtype ?? (indices instanceof Uint8Array ? 'uint8' : 'uint16');
  const n = indices.length;

  // Convert LUT to Float32Array for WASM
  const lutF32 = new Float32Array(lut);

  if (lutMode === 'scalar') {
    // Scalar mode: one index per element, output size = n
    const result = new Float32Array(n);

    if (dtype === 'uint8') {
      wasmModule.decode_lut_scalar_u8(indices as Uint8Array, lutF32, result);
    } else {
      wasmModule.decode_lut_scalar_u16(indices as Uint16Array, lutF32, result);
    }

    return transfer(result, [result.buffer]);
  } else {
    // Row mode: one index per row, output size = n * k
    const result = new Float32Array(n * k);

    if (dtype === 'uint8') {
      wasmModule.decode_lut_row_u8(indices as Uint8Array, lutF32, k, result);
    } else {
      wasmModule.decode_lut_row_u16(indices as Uint16Array, lutF32, k, result);
    }

    return transfer(result, [result.buffer]);
  }
}
