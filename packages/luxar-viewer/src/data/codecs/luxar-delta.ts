/**
 * `luxar_delta_v1` — columnar per-chunk delta+zigzag zarr filter (decode twin).
 *
 * The Python writer (`packages/luxar/src/luxar/encoding/_encoders/delta_codec.py`)
 * stores Hilbert-ordered uint8/uint16 quantization codes as per-axis modular
 * deltas, zigzag-mapped to unsigned and laid out COLUMN-major within each
 * chunk (all column-0 residuals, then column-1, ...). Spatial ordering makes
 * consecutive codes a smooth ramp; the residuals compress ~12% better
 * whole-store under the existing Blosc policy. Keep the two implementations
 * in 1:1 wire-format sync — the Python unit tests lock the bytes.
 *
 * Wire format (per chunk of rows x cols codes, per column, modular 2^bits):
 *
 *   encode:  d  = (code - prev) mod 2^bits          // prev = 0 at chunk start
 *            s  = d >= 2^(bits-1) ? d - 2^bits : d  // signed interpretation
 *            zz = (s << 1) ^ (s >> (bits-1))        // zigzag -> uint
 *   decode:  s    = (zz >>> 1) ^ -(zz & 1)
 *            code = (prev + s) mod 2^bits
 *
 * This is a zarrita `array_to_array` codec: it runs inside `getChunk` on the
 * WHOLE chunk, before any sub-chunk slicing — so the stateless per-element
 * decode kernels (WASM/TS) and the range-loader are untouched. zarr v2 stores
 * edge chunks at full chunk shape (fill-padded), so `data.length` is always
 * `rows * cols`.
 *
 * Registered as `numcodecs.luxar_delta_v1` in `../zarr.ts` (module scope, so
 * every context that opens zarr arrays — main thread or worker — has it).
 * Structural types only: `../zarr.ts` must stay the sole zarrita import
 * boundary.
 */

type CodeArray = Uint8Array | Uint16Array;

/** Structural twin of zarrita's `Chunk`: flat typed array + shape/stride. */
interface CodeChunk {
  data: CodeArray;
  shape: number[];
  stride: number[];
}

interface LuxarDeltaConfig {
  cols?: number;
  bits?: number;
}

/** The subset of zarrita's resolved array metadata the codec needs. */
interface ArrayMetadata {
  data_type?: string;
  dataType?: string;
}

export class LuxarDeltaCodec {
  readonly kind = 'array_to_array';
  private readonly cols: number;
  private readonly mask: number;

  constructor(cols: number, bits: number) {
    if (bits !== 8 && bits !== 16) {
      throw new Error(`luxar_delta_v1: bits must be 8 or 16, got ${bits}`);
    }
    if (!Number.isInteger(cols) || cols < 1) {
      throw new Error(`luxar_delta_v1: cols must be >= 1, got ${cols}`);
    }
    this.cols = cols;
    this.mask = (1 << bits) - 1;
  }

  static fromConfig(config: LuxarDeltaConfig, meta: ArrayMetadata): LuxarDeltaCodec {
    const dataType = meta.dataType ?? meta.data_type;
    if (dataType !== 'uint8' && dataType !== 'uint16') {
      throw new Error(`luxar_delta_v1: unsupported data type ${String(dataType)}`);
    }
    const bits = config.bits ?? (dataType === 'uint8' ? 8 : 16);
    const expectedBits = dataType === 'uint8' ? 8 : 16;
    if (bits !== expectedBits) {
      throw new Error(
        `luxar_delta_v1: config bits=${bits} does not match array dtype ${dataType}`
      );
    }
    return new LuxarDeltaCodec(config.cols ?? 1, bits);
  }

  private rowsOf(data: CodeArray): number {
    if (data.length % this.cols !== 0) {
      throw new Error(
        `luxar_delta_v1: chunk of ${data.length} elements is not a multiple of ` +
          `cols=${this.cols} (columns must never be split)`
      );
    }
    return data.length / this.cols;
  }

  /** Row-major codes -> columnar zigzag residuals (viewer never writes; kept for parity tests). */
  encode(chunk: CodeChunk): CodeChunk {
    const { cols, mask } = this;
    const src = chunk.data;
    const rows = this.rowsOf(src);
    const out = new (src.constructor as new (n: number) => CodeArray)(src.length);
    const half = (mask >>> 1) + 1;
    for (let c = 0; c < cols; c++) {
      const base = c * rows;
      let prev = 0;
      for (let r = 0; r < rows; r++) {
        const code = src[r * cols + c];
        const d = (code - prev) & mask;
        const s = d >= half ? d - (mask + 1) : d;
        out[base + r] = ((s << 1) ^ (s >> 31)) & mask;
        prev = code;
      }
    }
    return { data: out, shape: chunk.shape, stride: chunk.stride };
  }

  /** Columnar zigzag residuals -> row-major codes (the hot path). */
  decode(chunk: CodeChunk): CodeChunk {
    const { cols, mask } = this;
    const src = chunk.data;
    const rows = this.rowsOf(src);
    const out = new (src.constructor as new (n: number) => CodeArray)(src.length);
    for (let c = 0; c < cols; c++) {
      const base = c * rows;
      let prev = 0;
      for (let r = 0; r < rows; r++) {
        const zz = src[base + r];
        const s = (zz >>> 1) ^ -(zz & 1);
        prev = (prev + s) & mask;
        out[r * cols + c] = prev;
      }
    }
    return { data: out, shape: chunk.shape, stride: chunk.stride };
  }
}
