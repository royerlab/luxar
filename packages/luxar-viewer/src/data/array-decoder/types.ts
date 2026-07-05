/**
 * Encoding metadata shapes for arrays produced by the Python
 * `luxar.encoding` system. The decoder, the array-ref registry, and
 * the loader helpers all consume these types — kept in a dedicated
 * types module so each layer can import the schema without dragging
 * in the decoder body.
 *
 * Reference: see the Python `luxar.encoding` package
 * (../../../../../luxar/src/luxar/encoding/README.md).
 */

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
  /** Geometric-log scalar lower anchor (geolog_scalar_* encodings). */
  min_log?: number;
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
