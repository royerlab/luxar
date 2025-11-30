/**
 * Array Decoder for Luxar Encoded Arrays
 *
 * This module implements decoding of arrays encoded by the Python luxar.encoding system.
 * It supports all encoding modes: broadcasting, LUT, quantization, and array references.
 *
 * **Critical for Compatibility**: Must decode all formats written by Python ArrayEncoder.
 *
 * Reference: luxar.encoding specification (../../luxar/src/luxar/encoding/SPECIFICATIONS.md)
 */

import * as zarr from 'zarrita';
import { get } from 'zarrita';
import { log, Modules } from '../utils/log';

/**
 * Array metadata from zarr .zattrs
 */
export interface ArrayMetadata {
  /** Shape of the array as written */
  shape?: number[];

  /** Data type */
  dtype?: string;

  /** Number of elements this array represents (for broadcasting) */
  n_elements?: number;

  /** Encoding mode */
  encoding_mode?: 'direct' | 'lut' | 'broadcasted' | 'array_ref';

  /** Lookup table for LUT encoding */
  lut?: number[];

  /** Quantization bounds [min, max] */
  quantization_bounds?: [number, number];

  /** Array reference hash (for deduplication) */
  array_ref?: string;

  /** Original array hash (for debugging) */
  array_hash?: string;
}

/**
 * Global registry for array references (deduplication)
 */
export class ArrayRefRegistry {
  private registry = new Map<string, Float32Array>();

  /**
   * Register an array with its hash
   */
  register(hash: string, array: Float32Array): void {
    if (!this.registry.has(hash)) {
      this.registry.set(hash, array);
      log.info(Modules.ZARR_LOADER, `Registered array ref: ${hash} (${array.length} elements)`);
    }
  }

  /**
   * Get an array by hash
   */
  get(hash: string): Float32Array | undefined {
    return this.registry.get(hash);
  }

  /**
   * Check if hash exists
   */
  has(hash: string): boolean {
    return this.registry.has(hash);
  }

  /**
   * Clear all registered arrays
   */
  clear(): void {
    this.registry.clear();
  }

  /**
   * Get statistics
   */
  getStats(): { count: number; totalBytes: number } {
    let totalBytes = 0;
    for (const arr of this.registry.values()) {
      totalBytes += arr.byteLength;
    }
    return {
      count: this.registry.size,
      totalBytes,
    };
  }
}

/**
 * Main array decoder class
 */
export class ArrayDecoder {
  constructor(private refRegistry: ArrayRefRegistry) {}

  /**
   * Decode an array from zarr, handling all encoding modes
   *
   * @param zarrArray - Zarr array handle
   * @param attrs - Array metadata from .zattrs
   * @param expectedElements - Expected total elements (for validation)
   * @returns Decoded Float32Array
   */
  async decode(
    zarrArray: zarr.Array<zarr.DataType, zarr.Readable>,
    attrs: ArrayMetadata,
    expectedElements?: number
  ): Promise<Float32Array> {
    // Check for array reference first (highest priority)
    if (attrs.array_ref) {
      return this.decodeArrayRef(attrs.array_ref, expectedElements);
    }

    // Load raw data from zarr
    const rawData = await get(zarrArray);
    // Convert zarr data to Float32Array
    const rawArray = rawData.data;
    const data =
      rawArray instanceof Float32Array
        ? rawArray
        : new Float32Array(rawArray as ArrayBuffer | number[]);

    // Check for broadcasting
    if (attrs.n_elements && attrs.n_elements > data.length) {
      return this.decodeBroadcasted(data, attrs, expectedElements);
    }

    // Check for LUT encoding
    if (attrs.encoding_mode === 'lut' && attrs.lut) {
      return this.decodeLUT(data, attrs.lut, expectedElements);
    }

    // Check for quantization
    if (
      attrs.quantization_bounds &&
      (attrs.dtype === 'uint8' ||
        attrs.dtype === '<u1' ||
        attrs.dtype === 'uint16' ||
        attrs.dtype === '<u2')
    ) {
      return this.dequantize(data, attrs.quantization_bounds, attrs.dtype);
    }

    // Direct mode (no encoding)
    // Register for potential array ref usage
    if (attrs.array_hash) {
      this.refRegistry.register(attrs.array_hash, data);
    }

    return data;
  }

  /**
   * Decode broadcasted array (uniform values)
   *
   * Format: Shape (1, k) replicated to (n_elements, k)
   */
  private decodeBroadcasted(
    data: Float32Array,
    attrs: ArrayMetadata,
    _expectedElements?: number
  ): Float32Array {
    const n_elements = attrs.n_elements!;
    const shape = attrs.shape || [];

    // Validate: first dimension should be 1
    if (shape[0] !== 1) {
      throw new Error(`Broadcasting: expected shape[0] == 1, got ${shape[0]}`);
    }

    // Feature dimension (typically 3 for positions/colors, 1 for radii)
    const k = shape.length > 1 ? shape[1] : 1;

    log.info(
      Modules.ZARR_LOADER,
      `Broadcasting: (1, ${k}) → (${n_elements}, ${k}) = ${n_elements * k} elements`
    );

    // Create output array
    const result = new Float32Array(n_elements * k);

    // Replicate single value across all points
    for (let i = 0; i < n_elements; i++) {
      for (let j = 0; j < k; j++) {
        result[i * k + j] = data[j];
      }
    }

    // Register for potential array ref usage
    if (attrs.array_hash) {
      this.refRegistry.register(attrs.array_hash, result);
    }

    return result;
  }

  /**
   * Decode LUT-encoded array
   *
   * Format: Indices (uint8/uint16) + lookup table
   */
  private decodeLUT(indices: Float32Array, lut: number[], expectedElements?: number): Float32Array {
    const n = indices.length;

    // Determine feature dimension from LUT size
    // LUT contains all unique values flattened
    // For RGB: lut.length = num_unique * 3
    let k = 1; // Default to scalar
    if (expectedElements) {
      k = Math.round(expectedElements / n);
      if (expectedElements % n !== 0) {
        log.warning(
          Modules.ZARR_LOADER,
          `LUT: expectedElements ${expectedElements} not divisible by n=${n}`
        );
      }
    }

    log.info(Modules.ZARR_LOADER, `LUT: ${n} indices → ${n * k} elements (k=${k})`);
    log.info(Modules.ZARR_LOADER, `  LUT size: ${lut.length} values (${lut.length / k} unique)`);

    const result = new Float32Array(n * k);

    // Map each index to its LUT value
    for (let i = 0; i < n; i++) {
      const idx = Math.round(indices[i]); // Indices should be integers

      for (let j = 0; j < k; j++) {
        result[i * k + j] = lut[idx * k + j];
      }
    }

    return result;
  }

  /**
   * Dequantize integer array to floats
   *
   * Format: uint8 or uint16 → float with bounds [min, max]
   */
  private dequantize(data: Float32Array, bounds: [number, number], dtype: string): Float32Array {
    const [min_val, max_val] = bounds;

    // Determine max integer value from dtype
    let max_int: number;
    if (dtype === 'uint8' || dtype === '<u1') {
      max_int = 255;
    } else if (dtype === 'uint16' || dtype === '<u2') {
      max_int = 65535;
    } else {
      throw new Error(`Unsupported quantization dtype: ${dtype}`);
    }

    log.info(
      Modules.ZARR_LOADER,
      `Dequantizing: ${dtype} [0, ${max_int}] → float [${min_val.toFixed(3)}, ${max_val.toFixed(3)}]`
    );

    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      // Normalize to [0, 1]
      const normalized = data[i] / max_int;

      // Map to [min, max]
      result[i] = min_val + normalized * (max_val - min_val);
    }

    return result;
  }

  /**
   * Resolve array reference (deduplicated array)
   *
   * Format: array_ref hash points to previously loaded array
   */
  private decodeArrayRef(hash: string, expectedElements?: number): Float32Array {
    const cached = this.refRegistry.get(hash);

    if (!cached) {
      throw new Error(`Array reference not found: ${hash}`);
    }

    log.info(Modules.ZARR_LOADER, `Array ref resolved: ${hash} (${cached.length} elements)`);

    // Validate size if expected
    if (expectedElements && cached.length !== expectedElements) {
      log.warning(
        Modules.ZARR_LOADER,
        `Array ref size mismatch: expected ${expectedElements}, got ${cached.length}`
      );
    }

    return cached;
  }

  /**
   * Helper: Determine if array is encoded
   */
  static isEncoded(attrs: ArrayMetadata): boolean {
    return !!(
      attrs.array_ref ||
      attrs.n_elements ||
      attrs.encoding_mode === 'lut' ||
      attrs.quantization_bounds
    );
  }

  /**
   * Helper: Get encoding mode from metadata
   */
  static getEncodingMode(attrs: ArrayMetadata): string {
    if (attrs.array_ref) return 'array_ref';
    if (attrs.n_elements) return 'broadcasted';
    if (attrs.encoding_mode === 'lut') return 'lut';
    if (attrs.quantization_bounds) return 'quantized';
    return 'direct';
  }
}

/**
 * Convenience function: Load and decode an optional array
 *
 * @param location - Zarr location
 * @param arrayName - Name of array (e.g., 'colors', 'radii')
 * @param decoder - ArrayDecoder instance
 * @param expectedElements - Expected total elements
 * @returns Decoded array or null if not present
 */
export async function loadAndDecodeOptionalArray(
  location: zarr.Location<zarr.Readable>,
  arrayName: string,
  decoder: ArrayDecoder,
  expectedElements?: number
): Promise<Float32Array | null> {
  try {
    // Try to open array
    const array = await zarr.open(location.resolve(arrayName), { kind: 'array' });

    // Load attributes
    const attrs = array.attrs as unknown as ArrayMetadata;

    // Decode
    const decoded = await decoder.decode(array, attrs, expectedElements);

    log.success(Modules.ZARR_LOADER, `Loaded ${arrayName}: ${decoded.length} elements`);

    return decoded;
  } catch {
    // Array doesn't exist (this is OK for optional arrays)
    return null;
  }
}
