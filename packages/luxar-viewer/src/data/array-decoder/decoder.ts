/**
 * Array Decoder for Luxar Encoded Arrays
 *
 * This module implements decoding of arrays encoded by the Python luxar.encoding system.
 * It supports all encoding modes: broadcasting, LUT, quantization, and array references.
 *
 * **Critical for Compatibility**: Must decode all formats written by Python ArrayEncoder.
 *
 * Reference: see the Python `luxar.encoding` package
 * (../../luxar/src/luxar/encoding/README.md).
 */

import * as zarr from '../zarr';
import { readArray } from '../zarr';
import { log, Modules } from '../../utils/log';
import { ArrayRefRegistry } from './ref-registry';
import type { ArrayMetadata, EncodingMetadata } from './types';

export { ArrayRefRegistry } from './ref-registry';
export type { ArrayMetadata, EncodingMetadata } from './types';
export { loadAndDecodeOptionalArray } from './load-and-decode';

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

    // PRIORITY 1: Check for broadcasting FIRST (highest priority per spec).
    // Python stores the logical broadcast count in encoding.n_elements; callers
    // can optionally override it with expectedElements, but metadata alone must
    // be sufficient for full-array decoding and Python parity.
    if (enc?.name === 'broadcasted') {
      const broadcastElements = expectedElements ?? enc.n_elements;
      if (broadcastElements === undefined) {
        // Include zarr path + encoding shape so the diagnostic points
        // the user at the exact array with broken metadata.
        const path = (zarrArray as { path?: string }).path ?? '<unknown>';
        throw new Error(
          `[ArrayDecoder] broadcasted encoding requires encoding.n_elements (zarr path: ${path}, encoding: ${JSON.stringify(enc)})`
        );
      }
      if (broadcastElements === 0) {
        return new Float32Array(0);
      }

      // Load the single value
      const rawData = await readArray(zarrArray);
      const rawArray = rawData.data;
      const data =
        rawArray instanceof Float32Array
          ? rawArray
          : new Float32Array(rawArray as ArrayBuffer | number[]);

      const shape = zarrArray.shape;
      const k = shape.length > 1 ? shape[1] : 1;
      const result = this.decodeBroadcasted(data, broadcastElements, k, broadcastElements);

      // Register for potential array ref usage
      if (enc?.hash) {
        this.refRegistry.register(enc.hash, result);
      }

      return result;
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

    // Zarrita cannot materialize empty arrays with `readArray()` in Node. Direct
    // empty arrays are valid Python encoder output, so return the decoded empty
    // buffer before touching chunk indexing. Array refs are handled above even
    // though their physical zarr shape is also empty.
    if (zarrArray.shape.some((dim) => dim === 0)) {
      return new Float32Array(0);
    }

    // Load raw data from zarr (needed for LUT, quantization, dtype)
    const rawData = await readArray(zarrArray);
    const rawArray = rawData.data;

    // Convert raw zarr data to Float32Array safely.
    // If rawArray is a typed array view, new Float32Array(view) does element-wise conversion.
    // If rawArray is a raw ArrayBuffer, we must NOT reinterpret bytes — create a typed view first.
    const bigIntArrayToFloat32 = (values: BigInt64Array | BigUint64Array): Float32Array => {
      const result = new Float32Array(values.length);
      for (let i = 0; i < values.length; i++) {
        result[i] = Number(values[i]);
      }
      return result;
    };

    const data = (() => {
      if (rawArray instanceof Float32Array) return rawArray;
      if (rawArray instanceof Uint8Array) return new Float32Array(rawArray);
      if (rawArray instanceof Uint16Array) return new Float32Array(rawArray);
      if (rawArray instanceof Uint32Array) return new Float32Array(rawArray);
      if (rawArray instanceof Int8Array) return new Float32Array(rawArray);
      if (rawArray instanceof Int16Array) return new Float32Array(rawArray);
      if (rawArray instanceof Int32Array) return new Float32Array(rawArray);
      if (rawArray instanceof Float64Array) return new Float32Array(rawArray);
      if (typeof BigUint64Array !== 'undefined' && rawArray instanceof BigUint64Array)
        return bigIntArrayToFloat32(rawArray);
      if (typeof BigInt64Array !== 'undefined' && rawArray instanceof BigInt64Array)
        return bigIntArrayToFloat32(rawArray);
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
        if (dtype.includes('u8') || dtype === 'uint64')
          return bigIntArrayToFloat32(new BigUint64Array(rawArray));
        if (dtype.includes('i1') || dtype === 'int8')
          return new Float32Array(new Int8Array(rawArray));
        if (dtype.includes('i2') || dtype === 'int16')
          return new Float32Array(new Int16Array(rawArray));
        if (dtype.includes('i4') || dtype === 'int32')
          return new Float32Array(new Int32Array(rawArray));
        if (dtype.includes('i8') || dtype === 'int64')
          return bigIntArrayToFloat32(new BigInt64Array(rawArray));
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

      // Get LUT mode from metadata (scalar vs row). ABSENT defaults to row
      // (the Python encoder omits lut_mode for 1-D arrays by contract), but
      // a present-yet-unknown mode must fail loud — the worker decoder
      // already throws here, and silently decoding garbage as row mode
      // would corrupt every value. Keep both paths aligned.
      const lutMode = ArrayDecoder.validatedLutMode(enc.lut_mode);

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

    // Geometric-log scalar (min/max-anchored, reserved zero level) — the
    // rescale-first encoding for wide-dynamic-range positive scalars
    // (gsplat amplitudes). MUST also be checked before generic quantization.
    if (
      ArrayDecoder.isGeologScalarEncodingName(enc?.name) &&
      enc?.min_log !== undefined &&
      enc?.max_log !== undefined
    ) {
      const actualDtype = zarrArray.dtype;
      if (actualDtype === undefined || actualDtype === null || String(actualDtype) === '') {
        throw new Error(`[ArrayDecoder] Missing zarr dtype for quantized encoding: ${enc.name}`);
      }
      return this.decodeGeologScalar(data, enc.min_log, enc.max_log, String(actualDtype));
    }

    // Per-channel quantization (log / signed-log / linear `*_perchannel_*`) —
    // fully self-decoded here for full-array reads, mirroring the RangeLoader's
    // 'perchannel' path and Python's `_decode_*_perchannel`. `data` holds the raw
    // integer levels as float; apply the per-column inverse (col = index % C).
    if (enc?.name && ArrayDecoder.isPerChannelQuantEncodingName(enc.name)) {
      // Column count comes from the ARRAY's own last dimension (like Python's
      // `data.shape[-1]` in `_perchannel_scales`), NOT from col_lo.length —
      // deriving it from the metadata would make makePerChannelDequant's
      // length guard a tautology, silently misaligning axes on a corrupt file
      // whose col_lo/col_hi length disagrees with the stored data.
      const arrShape = zarrArray.shape;
      const cols =
        Array.isArray(arrShape) && arrShape.length > 1 ? Number(arrShape[arrShape.length - 1]) : 1;
      const dequant = ArrayDecoder.makePerChannelDequant(
        enc as { name?: string; bits?: number; col_lo?: number[]; col_hi?: number[] },
        cols
      );
      const out = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) out[i] = dequant(data[i], i % cols);
      return out;
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
        throw new Error('[ArrayDecoder] LUT (scalar) is empty');
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
   * Decode geometric-log scalar (min/max-anchored, reserved zero level).
   *
   * Level 0 decodes to exactly 0; levels [1, 2^bits - 1] decode to
   * exp(minLog + (u - 1)/(2^bits - 2) * (maxLog - minLog)) — uniform
   * relative precision across the array's own nonzero range. Mirrors
   * Python `_decode_geolog_scalar` and the worker/WASM kernels exactly.
   */
  private decodeGeologScalar(
    data: Float32Array,
    minLog: number,
    maxLog: number,
    dtype: string
  ): Float32Array {
    if (!Number.isFinite(minLog) || !Number.isFinite(maxLog) || maxLog < minLog) {
      throw new Error(
        `[ArrayDecoder] Invalid geolog_scalar min_log/max_log: ${minLog}/${maxLog}. ` +
          'Must be finite with max_log >= min_log.'
      );
    }

    let top: number;
    if (dtype === 'uint8' || dtype === '<u1' || dtype === '|u1') {
      top = 255;
    } else if (dtype === 'uint16' || dtype === '<u2' || dtype === '>u2' || dtype === '|u2') {
      top = 65535;
    } else {
      throw new Error(`Unsupported geolog_scalar dtype: ${dtype}`);
    }

    const inv = Math.max(maxLog - minLog, 0) / Math.max(top - 1, 1);
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      const u = data[i];
      result[i] = u === 0 ? 0 : Math.exp(minLog + (u - 1) * inv);
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
      ArrayDecoder.isQuantizedEncoding(attrs) || // quantization (rgb_uint8, bounded_scalar_uint16, etc.)
      ArrayDecoder.isPerChannelQuantEncodingName(enc.name) // per-channel (log/signed-log/linear) — decoded to float32 by the loader
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
    if (ArrayDecoder.isGeologScalarEncodingName(enc.name)) return 'geolog_scalar';
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
   * Helper: Check if an encoding name is a geometric-log scalar encoding
   * (min/max-anchored true-log with a reserved zero level).
   */
  static isGeologScalarEncodingName(name: string | undefined): boolean {
    return name === 'geolog_scalar_uint8' || name === 'geolog_scalar_uint16';
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
    if (
      !ArrayDecoder.isLogScalarEncodingName(enc.name) &&
      !ArrayDecoder.isGeologScalarEncodingName(enc.name) &&
      enc.max_log !== undefined
    ) {
      throw new Error(
        '[ArrayDecoder] max_log metadata is only valid for log_scalar/geolog_scalar encodings'
      );
    }
    if (!ArrayDecoder.isGeologScalarEncodingName(enc.name) && enc.min_log !== undefined) {
      throw new Error('[ArrayDecoder] min_log metadata is only valid for geolog_scalar encodings');
    }
    if (
      ArrayDecoder.isGeologScalarEncodingName(enc.name) &&
      (enc.min_log === undefined || enc.max_log === undefined)
    ) {
      throw new Error(
        '[ArrayDecoder] geolog_scalar encoding requires encoding.min_log and encoding.max_log'
      );
    }
    if (
      ArrayDecoder.isGeologScalarEncodingName(enc.name) &&
      enc.min_log !== undefined &&
      enc.max_log !== undefined &&
      (!Number.isFinite(enc.min_log) || !Number.isFinite(enc.max_log) || enc.max_log < enc.min_log)
    ) {
      throw new Error(
        `[ArrayDecoder] Invalid geolog_scalar min_log/max_log: ${enc.min_log}/${enc.max_log}. ` +
          'Must be finite with max_log >= min_log.'
      );
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
      ArrayDecoder.isQuantizedEncodingName(name) ||
      ArrayDecoder.isPerChannelQuantEncodingName(name)
    );
  }

  /**
   * Generic per-channel quantization encodings: per-column (per-channel) log
   * (`log_perchannel_u8/u16`, non-negative), signed-log
   * (`signed_log_perchannel_u8/u16`, signed), linear/fixed-point
   * (`linear_perchannel_u8/u16`, identity — COORDINATE positions/centers/vertices),
   * and TRUE-log (`geolog_perchannel_u8/u16`, wide-range positive — HDR colors).
   * They carry per-channel scale arrays (`col_lo/col_hi`) rather than global
   * bounds, so they are NOT in {@link isQuantizedEncodingName} (the global-scale
   * dequant path); they have their own `'perchannel'` load path in the RangeLoader
   * which fully decodes them to float32 (see `range-loader/perchannel.ts`). The
   * scheme is geometry-agnostic; consumers receive decoded float32.
   */
  static isPerChannelQuantEncodingName(name: string | undefined): boolean {
    return !!(
      name === 'log_perchannel_u8' ||
      name === 'log_perchannel_u16' ||
      name === 'signed_log_perchannel_u8' ||
      name === 'signed_log_perchannel_u16' ||
      name === 'linear_perchannel_u8' ||
      name === 'linear_perchannel_u16' ||
      name === 'geolog_perchannel_u8' ||
      name === 'geolog_perchannel_u16'
    );
  }

  /**
   * Build the per-channel dequantizer for a `log_perchannel_*` /
   * `signed_log_perchannel_*` / `linear_perchannel_*` / `geolog_perchannel_*`
   * array. Mirrors the Python decoders (`_decode_*_perchannel`):
   *   log:        `x = expm1(lo[c] + level/levels·(hi[c]-lo[c]))`
   *   signed-log: `y = lo[c] + level/levels·(hi[c]-lo[c]); x = sign(y)·expm1(|y|)`
   *   linear:     `x = lo[c] + level/levels·(hi[c]-lo[c])`  (identity; coordinates)
   *   geolog:     `x = 0` at level 0, else `exp(lo[c] + (level-1)/(levels-1)·rng)`
   *               (TRUE-log grid, ln-domain scales; HDR colors — always zero-level)
   * With `zero_level: true` (current writer for the log/signed-log pair),
   * level 0 is a RESERVED ZERO (decodes to exactly 0) and nonzero levels
   * `1..2^bits-1` span the nonzero-anchored `[lo, hi]` with denominator
   * `2^bits-2` — same layout as `geolog_scalar`. Arrays without the flag keep
   * the legacy all-levels mapping above. For any other / float32 / direct
   * encoding it returns the identity, so a raw value passes through unchanged.
   *
   * @returns `(level, col) => value` — `level` is the raw stored integer (as float).
   */
  static makePerChannelDequant(
    encoding:
      | {
          name?: string;
          bits?: number;
          col_lo?: number[];
          col_hi?: number[];
          zero_level?: boolean;
        }
      | undefined,
    numCols: number
  ): (level: number, col: number) => number {
    const name = encoding?.name;
    const isLog = name === 'log_perchannel_u8' || name === 'log_perchannel_u16';
    const isSlog = name === 'signed_log_perchannel_u8' || name === 'signed_log_perchannel_u16';
    const isLinear = name === 'linear_perchannel_u8' || name === 'linear_perchannel_u16';
    const isGeolog = name === 'geolog_perchannel_u8' || name === 'geolog_perchannel_u16';
    if (!isLog && !isSlog && !isLinear && !isGeolog) {
      return (level: number) => level; // float32 / direct: identity
    }
    // Per-channel scales are mandatory and must match the column count. Failing
    // loud beats silently dequantizing against zero/undefined scales (which would
    // produce NaN or all-zero covariance with no error). Mirrors the Python
    // decoder, which requires col_lo/col_hi and errors on a length mismatch.
    const lo = encoding!.col_lo;
    const hi = encoding!.col_hi;
    if (
      !Array.isArray(lo) ||
      !Array.isArray(hi) ||
      lo.length !== numCols ||
      hi.length !== numCols
    ) {
      throw new Error(
        `[ArrayDecoder] ${name}: col_lo/col_hi must each have ${numCols} entries ` +
          `(got ${lo?.length} / ${hi?.length}). Corrupt or malformed encoding metadata.`
      );
    }
    // Finiteness + ordering, matching the Python decoder's `_perchannel_scales`:
    // a non-finite scale would propagate NaN into every covariance, and hi < lo
    // would silently invert the range. Fail loud instead. (hi === lo is valid —
    // a constant column decodes every level to lo.)
    for (let c = 0; c < numCols; c++) {
      if (!Number.isFinite(lo[c]) || !Number.isFinite(hi[c]) || hi[c] < lo[c]) {
        throw new Error(
          `[ArrayDecoder] ${name}: col_lo/col_hi must be finite with col_hi >= col_lo ` +
            `(column ${c}: lo=${lo[c]}, hi=${hi[c]}). Corrupt or malformed encoding metadata.`
        );
      }
    }
    const bits = encoding!.bits ?? (name!.endsWith('u8') ? 8 : 16);
    const levels = (1 << bits) - 1;
    const rng = lo.map((l, i) => Math.max(hi[i] - l, 1e-30));
    if (isLinear) {
      return (level: number, c: number) => lo[c] + (level / levels) * rng[c];
    }
    // zero_level (current writer): level 0 = exact zero; levels 1..2^bits-1
    // span the nonzero-anchored [lo, hi] (denominator 2^bits-2). Legacy
    // arrays (no flag) map all levels 0..2^bits-1 over a zero-anchored scale.
    // geolog_perchannel has NO legacy variant — the reserved zero level is
    // part of the name's contract, so it ignores the flag.
    const zeroLevel = isGeolog || encoding!.zero_level === true;
    const denom = Math.max(levels - 1, 1);
    const companded = zeroLevel
      ? (level: number, c: number) => lo[c] + ((level - 1) / denom) * rng[c]
      : (level: number, c: number) => lo[c] + (level / levels) * rng[c];
    if (isGeolog) {
      // TRUE-log grid: scales are ln(min+)/ln(max+) per column; exp, not expm1.
      return (level: number, c: number) => (level === 0 ? 0 : Math.exp(companded(level, c)));
    }
    if (isLog) {
      return (level: number, c: number) =>
        zeroLevel && level === 0 ? 0 : Math.expm1(companded(level, c));
    }
    return (level: number, c: number) => {
      if (zeroLevel && level === 0) return 0;
      const y = companded(level, c);
      return Math.sign(y) * Math.expm1(Math.abs(y));
    };
  }

  /**
   * Helper: Check if an encoding name is quantized (uint8/uint16 with bounds).
   */
  static isQuantizedEncodingName(name: string | undefined): boolean {
    return !!(
      name === 'log_scalar_uint8' ||
      name === 'log_scalar_uint16' ||
      name === 'geolog_scalar_uint8' ||
      name === 'geolog_scalar_uint16' ||
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
  ): {
    bounds: [number, number];
    dtype: 'uint8' | 'uint16';
    isLogSpace: boolean;
    isGeologSpace?: boolean;
  } | null {
    if (!attrs) return null;
    const enc = attrs.encoding;
    if (!enc || !enc.name) return null;

    // Geometric-log (min/max-anchored, reserved zero level). bounds carry
    // [min_log, max_log]; min_log may be negative and may equal max_log
    // (constant nonzero array), so the strict quantization-bounds validator
    // does not apply here.
    if (
      ArrayDecoder.isGeologScalarEncodingName(enc.name) &&
      enc.min_log !== undefined &&
      enc.max_log !== undefined
    ) {
      if (
        !Number.isFinite(enc.min_log) ||
        !Number.isFinite(enc.max_log) ||
        enc.max_log < enc.min_log
      ) {
        throw new Error(
          `[ArrayDecoder] Invalid geolog_scalar min_log/max_log: ${enc.min_log}/${enc.max_log}`
        );
      }
      return {
        bounds: [enc.min_log, enc.max_log],
        dtype: ArrayDecoder.normalizeQuantizedDtype(zarrDtype),
        isLogSpace: false,
        isGeologSpace: true,
      };
    }

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
    const lutMode = ArrayDecoder.validatedLutMode(enc.lut_mode);

    return { lut: enc.lut, lutMode, k };
  }

  /**
   * Validate a stored `lut_mode`: absent defaults to 'row' (the Python
   * encoder omits it for 1-D arrays), but a present-yet-unknown value throws
   * — matching the worker decoder (`workers/data-worker/decode/lut.ts`),
   * which already rejects it. Previously the main thread silently treated
   * garbage as row mode.
   */
  static validatedLutMode(lutMode: string | undefined): string {
    if (lutMode !== undefined && lutMode !== 'row' && lutMode !== 'scalar') {
      throw new Error(
        `[ArrayDecoder] Invalid lut_mode '${lutMode}' (expected 'row' or 'scalar'). ` +
          'Corrupt or malformed encoding metadata.'
      );
    }
    return lutMode ?? 'row';
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
    quantMetadata: {
      bounds: [number, number];
      dtype: string;
      isLogSpace: boolean;
      isGeologSpace?: boolean;
    }
  ): Float32Array {
    const { bounds, dtype, isLogSpace } = quantMetadata;

    if (quantMetadata.isGeologSpace) {
      // Geometric-log: bounds = [min_log, max_log], reserved zero level.
      return this.decodeGeologScalar(
        quantizedData instanceof Float32Array ? quantizedData : new Float32Array(quantizedData),
        bounds[0],
        bounds[1],
        dtype
      );
    }

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
