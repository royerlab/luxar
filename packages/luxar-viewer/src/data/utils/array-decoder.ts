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
import { log, Modules } from '../../utils/log';

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

  /**
   * Original dtype before encoding (e.g., "uint8", "float32")
   * CRITICAL: The decoder must restore this dtype for correct rendering.
   * - uint8 colors: THREE.js normalizes (0-255 → 0-1) with normalized=true
   * - float32 colors: Expected to be 0-1, no normalization
   */
  original_dtype?: string;

  /** Quantization bounds [min, max] */
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

  /** Log-space encoding max (for log_scalar_uint8/uint16) */
  max_log?: number;
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
    ArrayDecoder.validateEncodingMetadata(enc);

    // ENCODING PRIORITY ORDER (CRITICAL - must match spec):
    // 1. Broadcasting → 2. Array Reference → 3. LUT → 4. Dtype

    // PRIORITY 1: Check for broadcasting FIRST (highest priority per spec)
    // Broadcasting detection: name="broadcasted" AND (shape[0]==1 OR scalar input)
    // IMPORTANT: Only handle broadcasting when expectedElements is provided AND broadcasting
    // is actually needed. Otherwise, fall through to dtype/quantization handlers below.
    if (enc?.name === 'broadcasted' && expectedElements !== undefined) {
      if (expectedElements === 0) {
        return new Float32Array(0);
      }
      // Load the single value
      const rawData = await get(zarrArray);
      const rawArray = rawData.data;
      const data =
        rawArray instanceof Float32Array
          ? rawArray
          : new Float32Array(rawArray as ArrayBuffer | number[]);

      // Broadcast if needed (data.length < expectedElements means we need to replicate)
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
      // If no broadcasting needed, fall through to other handlers (quantization, etc.)
    }

    // PRIORITY 2: Check for array reference (second priority per spec)
    if (enc?.name === 'array_ref') {
      // Early warning if zarrRootLoc missing (will fail later if not cached)
      if (!zarrRootLoc && !enc.hash) {
        log.warning(
          Modules.ZARR_LOADER,
          'Array reference detected without zarrRootLoc or hash. ' +
            'This will fail if the reference is not already cached.'
        );
      }
      return this.decodeArrayRef(enc.target!, enc.hash, expectedElements, zarrRootLoc);
    }

    // Load raw data from zarr (needed for LUT, quantization, dtype)
    const rawData = await get(zarrArray);
    const rawArray = rawData.data;

    // Convert raw zarr data to Float32Array safely.
    // If rawArray is a typed array view, new Float32Array(view) does element-wise conversion.
    // If rawArray is a raw ArrayBuffer, we must NOT reinterpret bytes — create a typed view first.
    const data = (() => {
      if (rawArray instanceof Float32Array) return rawArray;
      if (rawArray instanceof Uint8Array) return new Float32Array(rawArray);
      if (rawArray instanceof Uint16Array) return new Float32Array(rawArray);
      if (rawArray instanceof Uint32Array) return new Float32Array(rawArray);
      if (rawArray instanceof Int8Array) return new Float32Array(rawArray);
      if (rawArray instanceof Int16Array) return new Float32Array(rawArray);
      if (rawArray instanceof Int32Array) return new Float32Array(rawArray);
      if (rawArray instanceof Float64Array) return new Float32Array(rawArray);
      if (rawArray instanceof ArrayBuffer) {
        // Raw ArrayBuffer — interpret bytes based on zarr dtype, then convert to Float32
        const dtype = String(zarrArray.dtype);
        if (dtype.includes('f4') || dtype === 'float32') return new Float32Array(rawArray);
        if (dtype.includes('f8') || dtype === 'float64')
          return new Float32Array(new Float64Array(rawArray));
        if (dtype.includes('u1') || dtype === 'uint8')
          return new Float32Array(new Uint8Array(rawArray));
        if (dtype.includes('u2') || dtype === 'uint16')
          return new Float32Array(new Uint16Array(rawArray));
        if (dtype.includes('u4') || dtype === 'uint32')
          return new Float32Array(new Uint32Array(rawArray));
        if (dtype.includes('i1') || dtype === 'int8')
          return new Float32Array(new Int8Array(rawArray));
        if (dtype.includes('i2') || dtype === 'int16')
          return new Float32Array(new Int16Array(rawArray));
        if (dtype.includes('i4') || dtype === 'int32')
          return new Float32Array(new Int32Array(rawArray));
        // Fallback: assume float32 layout (preserves existing behavior)
        log.warning(
          Modules.ZARR_LOADER,
          `Unknown dtype "${dtype}" for ArrayBuffer, assuming float32`
        );
        return new Float32Array(rawArray);
      }
      // number[] — element-wise conversion
      return new Float32Array(rawArray as number[]);
    })();

    // PRIORITY 3: Check for LUT encoding (third priority per spec)
    if (enc && ArrayDecoder.isLUTEncodingName(enc.name)) {
      if (!enc.lut) {
        throw new Error(`[ArrayDecoder] Missing LUT metadata for encoding: ${enc.name}`);
      }
      // Get k (feature dimension) from original_shape in encoding metadata
      // original_shape is [n, k] where n is number of points, k is feature dimension
      // CRITICAL: Default to 1 for scalar mode, but MUST check original_shape for vector data
      const k = enc.original_shape && enc.original_shape.length > 1 ? enc.original_shape[1] : 1;

      // Get LUT mode from metadata (scalar vs row)
      const lutMode = enc.lut_mode || 'row'; // Default to row mode

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

    // Check for LOG-SPACE scalar encoding.
    // MUST be checked BEFORE generic quantization since both contain 'uint'/'scalar'.
    if (ArrayDecoder.isLogScalarEncodingName(enc?.name) && enc?.max_log !== undefined) {
      // CRITICAL: Use zarrArray.dtype, NOT attrs.dtype (which may be undefined)
      // Python encoder writes uint8/uint16 data but attrs.dtype may not be set.
      const actualDtype = zarrArray.dtype;
      if (actualDtype === undefined || actualDtype === null || String(actualDtype) === '') {
        throw new Error(`[ArrayDecoder] Missing zarr dtype for quantized encoding: ${enc.name}`);
      }
      return this.decodeLogScalar(data, enc.max_log, String(actualDtype));
    }

    // Check for quantization (known quantized encodings, bounds present OR implicit)
    // NOTE: Dtype encodings like "uint8"/"uint16" are direct storage, not quantization.
    if (enc?.name && ArrayDecoder.isQuantizedEncoding(attrs)) {
      // Bounds can be stored as:
      // 1. Array: bounds = [min, max]
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
            '  3. Implicit bounds for known types (rgb_uint8/rgb_uint16)\n' +
            `Got encoding: ${JSON.stringify(enc)}`
        );
      }

      // Get actual dtype from zarr array, NOT from attrs (Python encoder doesn't write attrs.dtype)
      // zarrArray.dtype has the real storage dtype (e.g., '|u1', '<u1', 'uint8')
      // Do NOT use enc.name (e.g., 'rgb_uint8') - that's the encoding name, not the dtype
      const actualDtype = zarrArray.dtype;
      if (actualDtype === undefined || actualDtype === null || String(actualDtype) === '') {
        throw new Error(`[ArrayDecoder] Missing zarr dtype for quantized encoding: ${enc.name}`);
      }
      log.info(
        Modules.ZARR_LOADER,
        `Quantized array: encoding=${enc.name}, zarr_dtype=${zarrArray.dtype}, actualDtype=${String(actualDtype)}`
      );
      return this.dequantize(data, bounds, String(actualDtype));
    }

    // Direct mode (no encoding or name: "none" / dtype names)
    if (enc?.name && !ArrayDecoder.isDirectEncodingName(enc.name)) {
      throw new Error(`[ArrayDecoder] Unknown encoding name: ${enc.name}`);
    }

    // Register for potential array ref usage
    if (enc?.hash) {
      this.refRegistry.register(enc.hash, data);
    }

    return data;
  }

  /**
   * Infer quantization bounds from encoding name
   *
   * Some encodings have implicit bounds that don't need to be stored:
   * - rgb_uint8/rgb_uint16: [0, 1] (standard RGB range)
   * - Others: return null (must have explicit bounds field)
   *
   * @param encName - Encoding name (e.g., "rgb_uint8", "bounded_scalar_uint16")
   * @returns Bounds [min, max] or null if not inferrable
   */
  private inferBounds(encName: string): [number, number] | null {
    if (encName === 'rgb_uint8' || encName === 'rgb_uint16') {
      return [0, 1]; // Standard RGB range
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

      if (flatLUT.length === 0) {
        throw new Error('LUT (scalar) is empty');
      }

      const result = new Float32Array(n);

      for (let i = 0; i < n; i++) {
        const idx = Math.round(indices[i]);
        if (!Number.isFinite(idx) || idx < 0 || idx >= flatLUT.length) {
          throw new Error(`LUT (scalar) index out of range: idx=${idx}, lutSize=${flatLUT.length}`);
        }
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

      if (k <= 0) {
        throw new Error(`LUT (row) has invalid feature dimension k=${k}`);
      }
      if (flatLUT.length % k !== 0) {
        log.warning(
          Modules.ZARR_LOADER,
          `LUT (row) size ${flatLUT.length} is not divisible by k=${k}`
        );
      }

      const result = new Float32Array(n * k);
      const lutRows = Math.floor(flatLUT.length / k);

      for (let i = 0; i < n; i++) {
        const idx = Math.round(indices[i]);
        if (!Number.isFinite(idx) || idx < 0 || idx >= lutRows) {
          throw new Error(`LUT (row) index out of range: idx=${idx}, lutRows=${lutRows}`);
        }

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
    const [min_val, max_val] = ArrayDecoder.validateQuantizationBounds(
      bounds,
      'quantization bounds'
    );

    // Determine max integer value from dtype
    // NumPy dtype formats: 'uint8', '<u1' (little-endian), '|u1' (native byte order for single-byte)
    let max_int: number;
    if (dtype === 'uint8' || dtype === '<u1' || dtype === '|u1') {
      max_int = 255;
    } else if (dtype === 'uint16' || dtype === '<u2' || dtype === '>u2' || dtype === '|u2') {
      max_int = 65535;
    } else {
      throw new Error(`Unsupported quantization dtype: ${dtype}`);
    }

    // Note: Removed per-chunk dequantization logging - too verbose for production
    // Each array load can trigger 100+ log messages, causing performance issues

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
   * Decode log-space encoded scalar array
   *
   * Format: log1p(value)/max_log → uint8/uint16
   * Decoding: expm1(normalized * max_log)
   *
   * Used for positive scalars with wide dynamic range (e.g., radii)
   */
  private decodeLogScalar(data: Float32Array, maxLog: number, dtype: string): Float32Array {
    if (!Number.isFinite(maxLog) || maxLog <= 0) {
      throw new Error(
        `[ArrayDecoder] Invalid log_scalar max_log: ${maxLog}. Must be finite and > 0.`
      );
    }

    // Determine max integer value from dtype
    // NumPy dtype formats: 'uint8', '<u1' (little-endian), '|u1' (native byte order for single-byte)
    let max_int: number;
    if (dtype === 'uint8' || dtype === '<u1' || dtype === '|u1') {
      max_int = 255;
    } else if (dtype === 'uint16' || dtype === '<u2' || dtype === '>u2' || dtype === '|u2') {
      max_int = 65535;
    } else {
      throw new Error(`Unsupported log_scalar dtype: ${dtype}`);
    }

    log.info(
      Modules.ZARR_LOADER,
      `Decoding log_scalar: ${dtype} [0, ${max_int}] → float (max_log=${maxLog.toFixed(4)})`
    );

    const result = new Float32Array(data.length);
    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;

    for (let i = 0; i < data.length; i++) {
      // Normalize to [0, 1]
      const normalized = data[i] / max_int;

      // Apply inverse log1p transform: expm1(normalized * max_log)
      // This reverses: log1p(value) / max_log → normalized
      const value = Math.expm1(normalized * maxLog);
      result[i] = value;
      if (value < minValue) minValue = value;
      if (value > maxValue) maxValue = value;
    }

    if (result.length > 0) {
      log.info(
        Modules.ZARR_LOADER,
        `  Decoded range: [${minValue.toFixed(4)}, ${maxValue.toFixed(4)}]`
      );
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
      enc.name === 'array_ref' || // array reference
      enc.name === 'broadcasted' || // broadcasting
      ArrayDecoder.isLUTEncodingName(enc.name) || // LUT encoding (lut_uint8, lut_uint16)
      ArrayDecoder.isQuantizedEncoding(attrs) // quantization (rgb_uint8, bounded_scalar_uint16, etc.)
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
    if (enc.name === 'array_ref') return 'array_ref';
    if (enc.name === 'broadcasted') return 'broadcasted';
    if (ArrayDecoder.isLUTEncodingName(enc.name)) return 'lut';
    if (ArrayDecoder.isLogScalarEncodingName(enc.name)) return 'log_scalar';
    if (ArrayDecoder.isQuantizedEncoding(attrs)) return 'quantized';

    return 'direct';
  }

  /**
   * Helper: Check if encoding is LUT (lookup table)
   *
   * LUT encoding is special because the lookup table is stored in metadata,
   * so we can load only a range of indices from zarr and still decode correctly.
   * This enables efficient range-based loading instead of loading the full array.
   */
  static isLUTEncoded(attrs: ArrayMetadata): boolean {
    if (!attrs) return false;
    const enc = attrs.encoding;
    if (!enc || !enc.name) return false;
    return !!(ArrayDecoder.isLUTEncodingName(enc.name) && enc.lut);
  }

  /**
   * Helper: Check if encoding is broadcasted (uniform value)
   *
   * Broadcasted encoding stores a single value that's replicated to all points.
   * For range loading, we just need the single value - no need to extract ranges.
   */
  static isBroadcasted(attrs: ArrayMetadata): boolean {
    if (!attrs) return false;
    const enc = attrs.encoding;
    if (!enc || !enc.name) return false;
    return enc.name === 'broadcasted';
  }

  /**
   * Helper: Check if encoding is array reference
   *
   * Array references point to another array for deduplication.
   * For range loading, we can apply optimized loading to the target array.
   */
  static isArrayRef(attrs: ArrayMetadata): boolean {
    if (!attrs) return false;
    const enc = attrs.encoding;
    if (!enc || !enc.name) return false;
    return enc.name === 'array_ref' && !!enc.target;
  }

  /**
   * Helper: Check if an encoding name means direct stored values.
   */
  static isDirectEncodingName(name: string | undefined): boolean {
    return (
      name === undefined ||
      name === 'none' ||
      name === 'float16' ||
      name === 'float32' ||
      name === 'uint8' ||
      name === 'uint16' ||
      name === 'uint32' ||
      name === 'uint64'
    );
  }

  /**
   * Helper: Check if an encoding name is a known LUT encoding.
   */
  static isLUTEncodingName(name: string | undefined): boolean {
    return name === 'lut_uint8' || name === 'lut_uint16';
  }

  /**
   * Helper: Check if an encoding name is a known log-scalar encoding.
   */
  static isLogScalarEncodingName(name: string | undefined): boolean {
    return name === 'log_scalar_uint8' || name === 'log_scalar_uint16';
  }

  /**
   * Validate encoding metadata shape before dispatch.
   */
  static validateEncodingMetadata(enc: EncodingMetadata | undefined): void {
    if (!enc) return;
    if (!enc.name) {
      throw new Error('[ArrayDecoder] encoding.name is required when encoding metadata is present');
    }
    if (enc.name === 'array_ref' && !enc.target) {
      throw new Error('[ArrayDecoder] array_ref encoding requires encoding.target');
    }
    if (enc.name !== 'array_ref' && enc.target) {
      throw new Error('[ArrayDecoder] encoding.target is only valid for array_ref');
    }
    if (!ArrayDecoder.isKnownEncodingName(enc.name)) {
      throw new Error(`[ArrayDecoder] Unknown encoding name: ${enc.name}`);
    }
    if (enc.name === 'broadcasted' && enc.n_elements === undefined) {
      throw new Error('[ArrayDecoder] broadcasted encoding requires encoding.n_elements');
    }
    if (ArrayDecoder.isLUTEncodingName(enc.name) && enc.lut === undefined) {
      throw new Error('[ArrayDecoder] LUT encoding requires encoding.lut');
    }
    const hasBounds = enc.bounds !== undefined || enc.min !== undefined || enc.max !== undefined;
    if (!ArrayDecoder.isQuantizedEncodingName(enc.name) && hasBounds) {
      throw new Error(
        '[ArrayDecoder] bounds/min/max metadata is only valid for quantized encodings'
      );
    }
    if (!ArrayDecoder.isLogScalarEncodingName(enc.name) && enc.max_log !== undefined) {
      throw new Error('[ArrayDecoder] max_log metadata is only valid for log_scalar encodings');
    }
    if (
      (enc.name === 'bounded_scalar_uint8' || enc.name === 'bounded_scalar_uint16') &&
      !hasBounds
    ) {
      throw new Error('[ArrayDecoder] bounded_scalar encoding requires bounds or min/max');
    }
    if ((enc.min === undefined) !== (enc.max === undefined)) {
      throw new Error('[ArrayDecoder] encoding.min and encoding.max must be provided together');
    }
    if (ArrayDecoder.isQuantizedEncodingName(enc.name)) {
      if (enc.bounds !== undefined) {
        ArrayDecoder.validateQuantizationBounds(enc.bounds, 'encoding.bounds');
      } else if (enc.min !== undefined && enc.max !== undefined) {
        ArrayDecoder.validateQuantizationBounds([enc.min, enc.max], 'encoding.min/max');
      }
    }
    if (ArrayDecoder.isLogScalarEncodingName(enc.name) && enc.max_log === undefined) {
      throw new Error('[ArrayDecoder] log_scalar encoding requires encoding.max_log');
    }
    if (
      ArrayDecoder.isLogScalarEncodingName(enc.name) &&
      enc.max_log !== undefined &&
      (!Number.isFinite(enc.max_log) || enc.max_log <= 0)
    ) {
      throw new Error(
        `[ArrayDecoder] Invalid log_scalar max_log: ${enc.max_log}. Must be finite and > 0.`
      );
    }
    if (
      (ArrayDecoder.isLUTEncodingName(enc.name) ||
        ArrayDecoder.isQuantizedEncodingName(enc.name)) &&
      !enc.original_dtype
    ) {
      throw new Error('[ArrayDecoder] encoded arrays require encoding.original_dtype');
    }
  }

  /**
   * Validate quantization bounds before dequantization.
   */
  private static validateQuantizationBounds(
    bounds: [number, number],
    context: string
  ): [number, number] {
    const [minVal, maxVal] = bounds;
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) {
      throw new Error(
        `[ArrayDecoder] Invalid ${context}: [${minVal}, ${maxVal}]. Bounds must be finite.`
      );
    }
    if (maxVal <= minVal) {
      throw new Error(
        `[ArrayDecoder] Invalid ${context}: max (${maxVal}) must be greater than min (${minVal}).`
      );
    }
    return bounds;
  }

  /**
   * Helper: Check if an encoding name is known by the decoder.
   */
  static isKnownEncodingName(name: string): boolean {
    return (
      ArrayDecoder.isDirectEncodingName(name) ||
      name === 'broadcasted' ||
      name === 'array_ref' ||
      ArrayDecoder.isLUTEncodingName(name) ||
      ArrayDecoder.isQuantizedEncodingName(name)
    );
  }

  /**
   * Helper: Check if an encoding name is quantized (uint8/uint16 with bounds).
   */
  static isQuantizedEncodingName(name: string | undefined): boolean {
    return !!(
      name === 'log_scalar_uint8' ||
      name === 'log_scalar_uint16' ||
      name === 'bounded_scalar_uint8' ||
      name === 'bounded_scalar_uint16' ||
      name === 'rgb_uint8' ||
      name === 'rgb_uint16'
    );
  }

  /**
   * Helper: Check if encoding is quantized (uint8/uint16 with bounds)
   *
   * Quantized encodings store data in reduced precision (uint8/uint16) with
   * bounds metadata for dequantization. These can be efficiently range-loaded:
   * load only the needed ranges of quantized data, then dequantize.
   *
   * Includes:
   * - rgb_uint8, rgb_uint16 (colors)
   * - bounded_scalar_uint8, bounded_scalar_uint16 (radii, sharpness)
   * - log_scalar_uint8, log_scalar_uint16 (log-space radii)
   */
  static isQuantizedEncoding(attrs: ArrayMetadata): boolean {
    if (!attrs) return false;
    const enc = attrs.encoding;
    if (!enc || !enc.name) return false;

    return ArrayDecoder.isQuantizedEncodingName(enc.name);
  }

  /**
   * Get quantization metadata for range-based decoding
   *
   * Extracts the bounds needed to dequantize a subset of quantized data.
   * Returns null if not quantized.
   *
   * @param attrs - Array metadata from .zattrs
   * @param zarrDtype - Actual zarr array dtype (e.g., 'uint16', '<u2').
   *                    Required because Python stores the quantized dtype on
   *                    the zarr array, not in attrs.dtype.
   */
  static getQuantizationMetadata(
    attrs: ArrayMetadata,
    zarrDtype: string
  ): { bounds: [number, number]; dtype: 'uint8' | 'uint16'; isLogSpace: boolean } | null {
    if (!attrs) return null;
    const enc = attrs.encoding;
    if (!enc || !enc.name) return null;

    // Check for log-space encoding first (special case)
    if (ArrayDecoder.isLogScalarEncodingName(enc.name) && enc.max_log !== undefined) {
      return {
        bounds: ArrayDecoder.validateQuantizationBounds([0, enc.max_log], 'encoding.max_log'),
        dtype: ArrayDecoder.normalizeQuantizedDtype(zarrDtype),
        isLogSpace: true,
      };
    }

    // Check for regular quantization (rgb, bounded_scalar)
    if (ArrayDecoder.isQuantizedEncoding(attrs)) {
      // Extract bounds from encoding metadata
      let bounds: [number, number] | null = null;

      if (enc.bounds) {
        bounds = enc.bounds;
      } else if (enc.min !== undefined && enc.max !== undefined) {
        bounds = [enc.min, enc.max];
      } else {
        // Try to infer bounds for known types
        if (enc.name === 'rgb_uint8' || enc.name === 'rgb_uint16') {
          bounds = [0, 1];
        }
      }

      if (bounds) {
        return {
          bounds: ArrayDecoder.validateQuantizationBounds(bounds, 'quantization metadata'),
          dtype: ArrayDecoder.normalizeQuantizedDtype(zarrDtype),
          isLogSpace: false,
        };
      }
    }

    return null;
  }

  /**
   * Normalize dtype string to canonical format
   *
   * Handles all NumPy dtype string variants:
   * - 'uint8', '|u1', '<u1', '>u1' → 'uint8'
   * - 'uint16', '|u2', '<u2', '>u2' → 'uint16'
   */
  private static normalizeQuantizedDtype(dtype: string): 'uint8' | 'uint16' {
    if (dtype === 'uint8' || dtype === '|u1' || dtype === '<u1' || dtype === '>u1') return 'uint8';
    if (dtype === 'uint16' || dtype === '|u2' || dtype === '<u2' || dtype === '>u2')
      return 'uint16';
    throw new Error(`[ArrayDecoder] Unsupported quantized zarr dtype: ${dtype}`);
  }

  /**
   * Get LUT encoding metadata for range-based decoding
   *
   * Returns the LUT metadata needed to decode a subset of indices.
   * Returns null if not LUT-encoded.
   */
  static getLUTMetadata(
    attrs: ArrayMetadata
  ): { lut: number[] | number[][]; lutMode: string; k: number } | null {
    if (!attrs) return null;
    const enc = attrs.encoding;
    if (!enc || !ArrayDecoder.isLUTEncodingName(enc.name) || !enc.lut) return null;

    const k = enc.original_shape && enc.original_shape.length > 1 ? enc.original_shape[1] : 1;
    const lutMode = enc.lut_mode || 'row';

    return { lut: enc.lut, lutMode, k };
  }

  /**
   * Decode LUT indices directly (for range-based loading)
   *
   * This allows decoding a subset of indices without loading the full array.
   * The indices can be loaded from zarr using range slicing, then decoded
   * using the LUT from metadata.
   *
   * @param indices - Indices loaded from a range of the zarr array
   * @param lutMetadata - LUT metadata from getLUTMetadata()
   * @returns Decoded values (Float32Array)
   */
  decodeLUTIndices(
    indices: Float32Array | Uint8Array | Uint16Array,
    lutMetadata: { lut: number[] | number[][]; lutMode: string; k: number }
  ): Float32Array {
    const { lut, lutMode, k } = lutMetadata;
    return this.decodeLUT(
      indices instanceof Float32Array ? indices : new Float32Array(indices),
      lut,
      k,
      lutMode
    );
  }

  /**
   * Dequantize quantized data directly (for range-based loading)
   *
   * This allows dequantizing a subset of quantized data without loading the full array.
   * The quantized ranges can be loaded from zarr using range slicing, then dequantized
   * using the bounds from metadata.
   *
   * @param quantizedData - Quantized data loaded from ranges (uint8/uint16)
   * @param quantMetadata - Quantization metadata from getQuantizationMetadata()
   * @returns Dequantized values (Float32Array)
   */
  dequantizeRange(
    quantizedData: Float32Array | Uint8Array | Uint16Array,
    quantMetadata: { bounds: [number, number]; dtype: string; isLogSpace: boolean }
  ): Float32Array {
    const { bounds, dtype, isLogSpace } = quantMetadata;

    if (isLogSpace) {
      // Log-space quantization: dequantize then exponentiate
      return this.decodeLogScalar(
        quantizedData instanceof Float32Array ? quantizedData : new Float32Array(quantizedData),
        bounds[1], // max_log
        dtype
      );
    } else {
      // Linear quantization: standard dequantization
      return this.dequantize(
        quantizedData instanceof Float32Array ? quantizedData : new Float32Array(quantizedData),
        bounds,
        dtype
      );
    }
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
 *
 * @internal — used by loader internals; not part of the public API.
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
