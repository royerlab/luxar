/**
 * `luxar_delta_v1` — columnar per-chunk delta+zigzag zarr filter (decode twin).
 *
 * The Python writer (`packages/luxar/src/luxar/encoding/_encoders/delta_codec.py`)
 * stores Hilbert-ordered uint8/uint16 quantization codes as per-axis modular
 * deltas, zigzag-mapped to unsigned and laid out COLUMN-major within each
 * chunk (all column-0 residuals, then column-1, ...). Spatial ordering makes
 * consecutive codes a smooth ramp; the residuals compress 12-16% better
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

/**
 * Buffers whose delta has ALREADY been undone — by a data worker that ran it
 * fused with the preceding blosc decode (see `./worker-blosc.ts`). The worker
 * hands back finished codes; the main-thread pipeline still walks its codec
 * chain (blosc -> bytes -> luxar_delta), and this is how the delta step knows
 * to pass that one chunk through instead of undoing the delta a second time.
 *
 * Keyed on the ArrayBuffer because the `bytes` codec between the two re-wraps
 * the data in a new typed-array VIEW over the same buffer. An entry is
 * consumed by the delta step it was meant for, so it applies exactly once;
 * nothing else ever carries one of these buffers (they are freshly received
 * from a worker), and a WeakSet keeps an unconsumed entry from leaking.
 */
const DELTA_ALREADY_DECODED = new WeakSet<ArrayBufferLike>();

/** Mark `buffer` as holding already delta-decoded codes (fused worker decode). */
export function markDeltaDecoded(buffer: ArrayBufferLike): void {
  DELTA_ALREADY_DECODED.add(buffer);
}

/** True — and forgets the mark — when `buffer` was marked by {@link markDeltaDecoded}. */
function consumeDeltaDecoded(buffer: ArrayBufferLike): boolean {
  return DELTA_ALREADY_DECODED.delete(buffer);
}

/** Codec-config fields of a fused delta, as handed to a worker. */
export interface LuxarDeltaSpec {
  cols: number;
  bits: 8 | 16;
}

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
  /** Chunk shape (zarrita passes the chunk grid's chunk_shape here). */
  shape?: number[];
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
      throw new Error(`luxar_delta_v1: config bits=${bits} does not match array dtype ${dataType}`);
    }
    const cols = config.cols ?? 1;
    // Fail-loud cols/shape cross-check: a corrupted `cols` that still divides
    // the chunk size would otherwise decode silently to garbage. The Luxar
    // writer only emits 1D (cols=1) and 2D (cols = trailing chunk dim) arrays.
    const shape = meta.shape;
    if (shape && (shape.length === 1 || shape.length === 2)) {
      const expectedCols = shape.length === 1 ? 1 : shape[1];
      if (cols !== expectedCols) {
        throw new Error(
          `luxar_delta_v1: config cols=${cols} does not match chunk shape ` +
            `[${shape.join(', ')}] (expected ${expectedCols})`
        );
      }
    }
    return new LuxarDeltaCodec(cols, bits);
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

  /** The config this instance decodes with (for a fused worker decode). */
  get spec(): LuxarDeltaSpec {
    return { cols: this.cols, bits: this.mask === 0xff ? 8 : 16 };
  }

  /**
   * Columnar zigzag residuals -> row-major codes (the hot path). A chunk a
   * data worker already delta-decoded (fused with blosc) passes through.
   */
  decode(chunk: CodeChunk): CodeChunk {
    if (consumeDeltaDecoded(chunk.data.buffer)) return chunk;
    return this.decodeResiduals(chunk);
  }

  /** The delta decode itself — what a worker runs for a fused chunk. */
  decodeResiduals(chunk: CodeChunk): CodeChunk {
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
