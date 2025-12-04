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

  /** Lookup table values (can be flat array or array of arrays) */
  lut?: number[] | number[][];

  /** LUT storage mode: "row" (one index per row) or "scalar" (one index per element) */
  lut_mode?: 'row' | 'scalar' | string;

  /** Original shape before encoding [n, k] */
  original_shape?: number[];

  /** Quantization bounds [min, max] (legacy format) */
  bounds?: [number, number];

  /** Quantization min (current format) */
  min?: number;

  /** Quantization max (current format) */
  max?: number;

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
   * @param zarrRootLoc - Zarr root location for resolving array_ref paths (required for array_ref)
   * @returns Decoded Float32Array
   */
  async decode(
    zarrArray: zarr.Array<zarr.DataType, zarr.Readable>,
    attrs: ArrayMetadata,
    expectedElements?: number,
    zarrRootLoc?: zarr.Location<zarr.Readable>
  ): Promise<Float32Array> {
    const enc = attrs.encoding;

    // Log decoding start for debugging
    console.log('[ArrayDecoder] Decoding array:', {
      shape: zarrArray.shape,
      dtype: zarrArray.dtype,
      encodingName: enc?.name || 'none',
      expectedElements,
      hasRootLoc: !!zarrRootLoc,
    });

    // ENCODING PRIORITY ORDER (CRITICAL - must match spec):
    // 1. Broadcasting → 2. Array Reference → 3. LUT → 4. Dtype

    // PRIORITY 1: Check for broadcasting FIRST (highest priority per spec)
    // Broadcasting detection: name="broadcasted" AND (shape[0]==1 OR scalar input)
    if (enc?.name === 'broadcasted' && expectedElements) {
      // Load the single value
      const rawData = await get(zarrArray);
      const rawArray = rawData.data;
      const data =
        rawArray instanceof Float32Array
          ? rawArray
          : new Float32Array(rawArray as ArrayBuffer | number[]);

      // Broadcast if needed
      if (expectedElements > data.length) {
        const shape = zarrArray.shape;
        const k = shape.length > 1 ? shape[1] : 1;
        const result = this.decodeBroadcasted(data, expectedElements, k, expectedElements);

        // Register for potential array ref usage
        if (enc?.hash) {
          this.refRegistry.register(enc.hash, result);
        }

        return result;
      }
    }

    // PRIORITY 2: Check for array reference (second priority per spec)
    if (enc?.name === 'array_ref' && enc?.target) {
      // Early warning if zarrRootLoc missing (will fail later if not cached)
      if (!zarrRootLoc && !enc.hash) {
        log.warning(
          Modules.ZARR_LOADER,
          'Array reference detected without zarrRootLoc or hash. ' +
            'This will fail if the reference is not already cached.'
        );
      }
      return this.decodeArrayRef(enc.target, enc.hash, expectedElements, zarrRootLoc);
    }

    // Load raw data from zarr (needed for LUT, quantization, dtype)
    const rawData = await get(zarrArray);
    const rawArray = rawData.data;

    // VALIDATION: Log raw data type for debugging
    console.log('[ArrayDecoder] Raw data loaded:', {
      dataType: rawArray.constructor.name,
      length: (rawArray as any).length,
      zarrDtype: zarrArray.dtype,
    });

    // Type validation: Ensure we can convert to Float32Array
    if (
      !(rawArray instanceof Float32Array) &&
      !(rawArray instanceof ArrayBuffer) &&
      !Array.isArray(rawArray)
    ) {
      const actualType = rawArray.constructor.name;
      console.warn(`[ArrayDecoder] Unexpected data type: ${actualType}, attempting conversion`);
    }

    const data =
      rawArray instanceof Float32Array
        ? rawArray
        : new Float32Array(rawArray as ArrayBuffer | number[]);

    // PRIORITY 3: Check for LUT encoding (third priority per spec)
    // Python generates names like: lut_uint8, lut_uint16
    if (enc?.name?.startsWith('lut') && enc?.lut) {
      // Get k (feature dimension) from original_shape in encoding metadata
      // original_shape is [n, k] where n is number of points, k is feature dimension
      // CRITICAL: Default to 1 for scalar mode, but MUST check original_shape for vector data
      const k = enc.original_shape && enc.original_shape.length > 1 ? enc.original_shape[1] : 1;

      // Get LUT mode from metadata (scalar vs row)
      const lutMode = (enc as any).lut_mode || 'row'; // Default to row mode

      // Decode LUT with correct mode
      const decoded = this.decodeLUT(data, enc.lut, k, lutMode);

      // Validate decoded size matches expected (only warn, don't fail)
      if (expectedElements && decoded.length !== expectedElements) {
        log.warning(
          Modules.ZARR_LOADER,
          `LUT decode size mismatch: got ${decoded.length}, expected ${expectedElements}. ` +
            `This is normal for ${lutMode} mode. Using decoded size.`
        );
      }

      return decoded;
    }

    // Check for quantization (name contains "uint", bounds present OR implicit)
    // NOTE: Some encodings like rgb_uint8 have implicit bounds [0, 1]
    if (enc?.name && (enc.name.includes('uint') || enc.name.includes('scalar'))) {
      // Bounds can be stored as:
      // 1. Array: bounds = [min, max] (legacy format)
      // 2. Separate fields: min, max (current format)
      // 3. Inferred from encoding name (implicit for known types)
      let bounds: [number, number] | null = null;

      if (enc.bounds) {
        bounds = enc.bounds;
      } else if (enc.min !== undefined && enc.max !== undefined) {
        bounds = [enc.min, enc.max];
      } else {
        bounds = this.inferBounds(enc.name);
        if (bounds) {
          log.warning(
            Modules.ZARR_LOADER,
            `Using inferred bounds ${JSON.stringify(bounds)} for ${enc.name}. ` +
              'Consider storing explicit bounds in metadata for clarity.'
          );
        }
      }

      // VALIDATION: Bounds are required for quantized data
      if (!bounds) {
        throw new Error(
          `[ArrayDecoder] Missing bounds for quantized encoding: ${enc.name}. ` +
            'Quantized arrays require either:\n' +
            '  1. Explicit bounds field: encoding.bounds = [min, max]\n' +
            '  2. Separate min/max fields: encoding.min, encoding.max\n' +
            '  3. Implicit bounds for known types (rgb_uint8, hdr_uint8)\n' +
            `Got encoding: ${JSON.stringify(enc)}`
        );
      }

      // Get actual dtype from zarr metadata (e.g., '<u1', 'uint8')
      // Do NOT use enc.name (e.g., 'rgb_uint8') - that's the encoding name, not the dtype
      const actualDtype = attrs.dtype || 'uint8';
      return this.dequantize(data, bounds, actualDtype);
    }

    // Direct mode (no encoding or name: "none" / "float16" / "float32")
    // Register for potential array ref usage
    if (enc?.hash) {
      this.refRegistry.register(enc.hash, data);
    }

    // Log successful decoding
    console.log('[ArrayDecoder] ✅ Decode complete:', {
      encodingName: enc?.name || 'none',
      outputLength: data.length,
      outputType: data.constructor.name,
      min: Math.min(...Array.from(data)),
      max: Math.max(...Array.from(data)),
    });

    return data;
  }

  /**
   * Infer quantization bounds from encoding name
   *
   * Some encodings have implicit bounds that don't need to be stored:
   * - rgb_uint8/rgb_uint16: [0, 1] (standard RGB range)
   * - hdr_uint8/hdr_uint16: [0, 10] (HDR range, though bounds may be explicit)
   * - Others: return null (must have explicit bounds field)
   *
   * @param encName - Encoding name (e.g., "rgb_uint8", "bounded_scalar_uint16")
   * @returns Bounds [min, max] or null if not inferrable
   */
  private inferBounds(encName: string): [number, number] | null {
    if (encName === 'rgb_uint8' || encName === 'rgb_uint16') {
      return [0, 1]; // Standard RGB range
    }
    if (encName === 'hdr_uint8' || encName === 'hdr_uint16') {
      return [0, 10]; // HDR range (may be overridden by explicit bounds)
    }
    return null; // Must have explicit bounds field
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
   * Modes:
   * - "row": One index per row → k values (e.g., positions: one index per point → (x,y,z))
   * - "scalar": One index per element → 1 value (e.g., positions: one index per coordinate)
   *
   * @param indices - Array of indices into the lookup table
   * @param lut - Lookup table containing unique values
   * @param k - Feature dimension (ignored for scalar mode)
   * @param lutMode - "row" or "scalar" mode
   */
  private decodeLUT(
    indices: Float32Array,
    lut: number[] | number[][],
    k: number,
    lutMode: string = 'row'
  ): Float32Array {
    const n = indices.length;

    // Flatten LUT if it's an array of arrays (row mode)
    let flatLUT: number[];
    if (Array.isArray(lut[0])) {
      // LUT is array of arrays - flatten it
      flatLUT = (lut as number[][]).flat();
    } else {
      // LUT is already flat
      flatLUT = lut as number[];
    }

    // CRITICAL: Handle scalar vs row mode
    if (lutMode === 'scalar') {
      // Scalar mode: one index per element, LUT contains scalar values
      // Output size = n (NOT n * k)
      log.info(Modules.ZARR_LOADER, `LUT (scalar): ${n} indices → ${n} elements`);
      log.info(Modules.ZARR_LOADER, `  LUT size: ${flatLUT.length} unique scalar values`);

      const result = new Float32Array(n);

      for (let i = 0; i < n; i++) {
        const idx = Math.round(indices[i]);
        result[i] = flatLUT[idx];
      }

      return result;
    } else {
      // Row mode: one index per row, LUT contains k-dimensional vectors
      // Output size = n * k
      log.info(Modules.ZARR_LOADER, `LUT (row): ${n} indices → ${n * k} elements (k=${k})`);
      log.info(
        Modules.ZARR_LOADER,
        `  LUT size: ${flatLUT.length} values (${flatLUT.length / k} unique vectors)`
      );

      const result = new Float32Array(n * k);

      for (let i = 0; i < n; i++) {
        const idx = Math.round(indices[i]);

        for (let j = 0; j < k; j++) {
          result[i * k + j] = flatLUT[idx * k + j];
        }
      }

      return result;
    }
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
   * Per spec section 7.6, this method:
   * 1. Checks hash-based cache first (fast path for already-loaded arrays)
   * 2. If not cached, loads the target array from zarr using target path
   * 3. Recursively decodes the target (it may also be encoded)
   * 4. Caches the result by hash for future references
   *
   * @param targetPath - Path to the target array (from encoding.target)
   * @param hash - Content hash for caching/verification (from encoding.hash)
   * @param expectedElements - Expected total elements (for validation)
   * @param zarrRootLoc - Zarr root location for path resolution
   */
  private async decodeArrayRef(
    targetPath: string,
    hash: string | undefined,
    expectedElements?: number,
    zarrRootLoc?: zarr.Location<zarr.Readable>
  ): Promise<Float32Array> {
    // Fast path: check cache by hash
    if (hash) {
      const cached = this.refRegistry.get(hash);
      if (cached) {
        log.info(Modules.ZARR_LOADER, `Array ref cache hit: ${hash} (${cached.length} elements)`);
        return cached;
      }
    }

    // Slow path: load from target path
    if (!zarrRootLoc) {
      throw new Error(
        `Array reference to "${targetPath}" requires zarrRootLoc parameter for path resolution`
      );
    }

    log.info(Modules.ZARR_LOADER, `Loading array ref target: ${targetPath}`);

    // Resolve and load target array
    const targetLoc = zarrRootLoc.resolve(targetPath);
    const targetArray = await zarr.open(targetLoc, { kind: 'array' });
    const targetAttrs = targetArray.attrs as unknown as ArrayMetadata;

    // Recursively decode target (it may also be encoded, e.g., LUT or quantized)
    const decoded = await this.decode(targetArray, targetAttrs, expectedElements, zarrRootLoc);

    // Cache by hash for future references
    if (hash) {
      this.refRegistry.register(hash, decoded);
      log.info(
        Modules.ZARR_LOADER,
        `Array ref resolved and cached: ${targetPath} -> ${hash} (${decoded.length} elements)`
      );
    }

    // Validate size if expected
    if (expectedElements && decoded.length !== expectedElements) {
      log.warning(
        Modules.ZARR_LOADER,
        `Array ref size mismatch: expected ${expectedElements}, got ${decoded.length}`
      );
    }

    return decoded;
  }

  /**
   * Helper: Determine if array is encoded
   *
   * Checks for any encoding metadata in attrs.encoding structure
   */
  static isEncoded(attrs: ArrayMetadata): boolean {
    if (!attrs) return false;
    const enc = attrs.encoding;
    if (!enc || !enc.name) return false;

    // Check for any encoding mode
    return !!(
      enc.target || // array reference
      enc.name === 'broadcasted' || // broadcasting
      enc.name?.startsWith('lut') || // LUT encoding (lut_uint8, lut_uint16)
      enc.name?.includes('uint') || // quantization (rgb_uint8, bounded_scalar_uint16, etc.)
      enc.bounds // explicit quantization bounds
    );
  }

  /**
   * Helper: Get encoding mode from metadata
   *
   * Returns a string describing the encoding mode
   */
  static getEncodingMode(attrs: ArrayMetadata): string {
    if (!attrs) return 'direct';
    const enc = attrs.encoding;
    if (!enc || !enc.name) return 'direct';

    // Map encoding name to mode
    if (enc.target) return 'array_ref';
    if (enc.name === 'broadcasted') return 'broadcasted';
    if (enc.name?.startsWith('lut')) return 'lut';
    if (enc.name?.includes('uint') || enc.bounds) return 'quantized';

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
