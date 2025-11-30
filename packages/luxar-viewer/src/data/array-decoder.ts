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
 * Encoding metadata (nested under "encoding" key per Python spec)
 */
export interface EncodingMetadata {
  /** Encoding name/type */
  name?: string;

  /** Number of elements (for broadcasting) */
  n_elements?: number;

  /** Lookup table values */
  lut?: number[];

  /** Quantization bounds [min, max] */
  bounds?: [number, number];

  /** Array reference target path */
  target?: string;

  /** Array reference hash */
  hash?: string;

  /** Quantization bits */
  bits?: number;
}

/**
 * Array metadata from zarr .zattrs
 *
 * NOTE: Per Python encoding spec, encoding metadata is nested under "encoding" key
 */
export interface ArrayMetadata {
  /** Shape of the array as written */
  shape?: number[];

  /** Data type */
  dtype?: string;

  /** Encoding metadata (nested) */
  encoding?: EncodingMetadata;
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
    const enc = attrs.encoding;

    // Check for array reference first (highest priority)
    if (enc?.target && enc?.hash) {
      return this.decodeArrayRef(enc.hash, expectedElements);
    }

    // Load raw data from zarr
    const rawData = await get(zarrArray);
    // Convert zarr data to Float32Array
    const rawArray = rawData.data;
    const data =
      rawArray instanceof Float32Array
        ? rawArray
        : new Float32Array(rawArray as ArrayBuffer | number[]);

    // Check for broadcasting (name: "broadcasted", n_elements > data.length)
    if (enc?.name === 'broadcasted' && enc?.n_elements && enc.n_elements > data.length) {
      const shape = attrs.shape || [];
      const k = shape.length > 1 ? shape[1] : 1;
      const result = this.decodeBroadcasted(data, enc.n_elements, k, expectedElements);

      // Register for potential array ref usage
      if (enc?.hash) {
        this.refRegistry.register(enc.hash, result);
      }

      return result;
    }

    // Check for LUT encoding (name: "lut", lut array present)
    if (enc?.name === 'lut' && enc?.lut) {
      return this.decodeLUT(data, enc.lut, expectedElements);
    }

    // Check for quantization (name contains "uint", bounds present)
    if (enc?.bounds && enc?.name && (enc.name.includes('uint') || enc.name.includes('scalar'))) {
      const dtype = attrs.dtype || enc.name; // Use encoding name as fallback dtype
      return this.dequantize(data, enc.bounds, dtype);
    }

    // Direct mode (no encoding or name: "none" / "float16" / "float32")
    // Register for potential array ref usage
    if (enc?.hash) {
      this.refRegistry.register(enc.hash, data);
    }

    return data;
  }

  /**
   * Decode broadcasted array (uniform values)
   *
   * Format: Shape (1, k) replicated to (n_elements, k)
   *
   * @param data - Raw data array (shape: (1, k))
   * @param n_elements - Target number of elements to replicate to
   * @param k - Feature dimension (e.g., 3 for RGB, 1 for scalar)
   * @param _expectedElements - Expected total elements (for validation, currently unused)
   */
  private decodeBroadcasted(
    data: Float32Array,
    n_elements: number,
    k: number,
    _expectedElements?: number
  ): Float32Array {
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
   *
   * Checks for any encoding metadata in attrs.encoding structure
   */
  static isEncoded(attrs: ArrayMetadata): boolean {
    const enc = attrs.encoding;
    if (!enc) return false;

    // Check for any encoding mode
    return !!(
      enc.target || // array reference
      enc.name === 'broadcasted' || // broadcasting
      enc.name === 'lut' || // LUT encoding
      enc.bounds // quantization
    );
  }

  /**
   * Helper: Get encoding mode from metadata
   *
   * Returns a string describing the encoding mode
   */
  static getEncodingMode(attrs: ArrayMetadata): string {
    const enc = attrs.encoding;
    if (!enc || !enc.name) return 'direct';

    // Map encoding name to mode
    if (enc.target) return 'array_ref';
    if (enc.name === 'broadcasted') return 'broadcasted';
    if (enc.name === 'lut') return 'lut';
    if (enc.bounds) return 'quantized';

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
