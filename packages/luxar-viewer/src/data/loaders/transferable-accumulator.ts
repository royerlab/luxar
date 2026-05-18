/**
 * TransferableAccumulator - Zero-allocation with worker CPU offload.
 *
 * This module implements the TransferableAccumulator pattern that enables
 * BOTH zero-allocation operation AND worker CPU offload simultaneously.
 *
 * The key insight is that we can use `Comlink.transfer()` for bidirectional
 * zero-copy buffer exchange between main thread and workers:
 *
 * ```
 * ┌─────────────────┐                    ┌─────────────────┐
 * │   MAIN THREAD   │                    │     WORKER      │
 * │                 │                    │                 │
 * │  Accumulator    │ ── transfer() ──► │  Receives       │
 * │  (owns buffers) │                    │  buffers        │
 * │                 │                    │                 │
 * │                 │                    │  Fills buffers  │
 * │                 │                    │  (zero alloc!)  │
 * │                 │                    │                 │
 * │  Adopts         │ ◄── transfer() ── │  Returns        │
 * │  buffers back   │                    │  buffers        │
 * └─────────────────┘                    └─────────────────┘
 * ```
 *
 * Key properties:
 * - Zero-copy transfer: `transfer()` moves ownership without copying
 * - Buffer reuse: Same buffers cycle between main thread and worker
 * - Zero allocation (after warmup): No `new Float32Array()` in steady state
 * - CPU offload: Heavy computation runs on worker thread
 *
 * @module data/loaders/transferable-accumulator
 */

import { log, Modules } from '../../utils/log';
import type { AccumulatorStats } from './base-types';

// ============================================================================
// Transferable Buffer Types
// ============================================================================

/**
 * Buffer shape descriptor for typed array allocation.
 */
export interface BufferShape {
  /** Buffer name (e.g., 'positions', 'colors') */
  name: string;

  /** Elements per item (e.g., 3 for RGB colors, 1 for radii) */
  elementsPerItem: number;

  /** Array type */
  type: 'float32' | 'uint8' | 'uint16';

  /** Whether this buffer is optional */
  optional?: boolean;
}

/**
 * Generic transferable buffer set.
 */
export interface TransferableBuffers {
  /** Map of buffer names to typed arrays */
  [name: string]: Float32Array | Uint8Array | Uint16Array | undefined;
}

/**
 * Points-specific buffer set.
 */
export interface PointsBuffers extends TransferableBuffers {
  positions: Float32Array;
  colors?: Float32Array | Uint8Array | Uint16Array;
  radii?: Float32Array;
  sharpness?: Float32Array;
}

/**
 * Lines-specific buffer set.
 */
export interface LinesBuffers extends TransferableBuffers {
  startPositions: Float32Array;
  endPositions: Float32Array;
  startColors?: Float32Array | Uint8Array;
  endColors?: Float32Array | Uint8Array;
  widths?: Float32Array;
}

/**
 * GSplats-specific buffer set.
 */
export interface GSplatsBuffers extends TransferableBuffers {
  positions: Float32Array;
  amplitudes: Float32Array;
  choleskyFactors: Float32Array;
  colors?: Float32Array | Uint8Array;
}

// ============================================================================
// TransferableAccumulator Class
// ============================================================================

/**
 * Generic accumulator that supports zero-copy transfer to/from workers.
 *
 * Buffers can be "detached" for worker use and "adopted" back.
 *
 * Usage:
 * ```typescript
 * const accumulator = new TransferableAccumulator<PointsBuffers>(shapes);
 * accumulator.ensureCapacity(10000);
 *
 * // Detach buffers for transfer to worker
 * const buffers = accumulator.detach();
 * const transferables = accumulator.getTransferables(buffers);
 *
 * // Transfer to worker (zero-copy)
 * const result = await worker.project(
 *   Comlink.transfer({ params, inputBuffers: buffers }, transferables)
 * );
 *
 * // Adopt returned buffers (zero-copy back)
 * accumulator.adopt(result.buffers);
 * ```
 */
export class TransferableAccumulator<TBuffers extends TransferableBuffers> {
  private buffers: TBuffers | null = null;
  private shapes: BufferShape[];
  private capacity: number;
  private isDetached = false;

  // Statistics
  private stats: AccumulatorStats = {
    allocations: 0,
    bytesAllocated: 0,
    reuseCount: 0,
    peakMemoryBytes: 0,
  };

  /**
   * Create a new TransferableAccumulator.
   *
   * @param shapes - Buffer shape descriptors
   * @param initialCapacity - Initial capacity in items
   */
  constructor(shapes: BufferShape[], initialCapacity = 1024) {
    this.shapes = shapes;
    this.capacity = initialCapacity;
    this.allocateBuffers();
  }

  /**
   * Ensure capacity (grow if needed).
   *
   * @param needed - Required capacity in items
   * @returns True if capacity was grown
   */
  ensureCapacity(needed: number): boolean {
    if (needed <= this.capacity) return false;

    if (this.isDetached) {
      throw new Error(
        'Cannot grow TransferableAccumulator while buffers are detached. ' +
          'Adopt buffers back first.'
      );
    }

    // Calculate new capacity with 1.5x growth factor
    let newCapacity = this.capacity;
    while (newCapacity < needed) {
      newCapacity = Math.ceil(newCapacity * 1.5);
    }

    log.info(
      Modules.DATA_ACCUMULATOR,
      `Growing TransferableAccumulator: ${this.capacity} → ${newCapacity} items`
    );

    const oldBuffers = this.buffers;
    this.capacity = newCapacity;
    this.allocateBuffers();

    // Copy old data if exists
    if (oldBuffers && this.buffers) {
      for (const shape of this.shapes) {
        const oldBuf = oldBuffers[shape.name];
        const newBuf = this.buffers[shape.name];
        if (oldBuf && newBuf) {
          (newBuf as Float32Array | Uint8Array | Uint16Array).set(
            oldBuf as Float32Array | Uint8Array | Uint16Array
          );
        }
      }
    }

    return true;
  }

  /**
   * Detach buffers for transfer to worker.
   *
   * After this call, the accumulator has no buffers until `adopt()` is called.
   * This is necessary because transferred ArrayBuffers become "detached" and
   * unusable in the original thread.
   *
   * @returns The buffer set to transfer
   * @throws Error if already detached
   */
  detach(): TBuffers {
    if (this.isDetached) {
      throw new Error(
        'TransferableAccumulator is already detached. ' +
          'Adopt buffers back before detaching again.'
      );
    }

    if (!this.buffers) {
      // First call - allocate buffers
      this.allocateBuffers();
    }

    const detached = this.buffers!;
    this.buffers = null;
    this.isDetached = true;

    log.info(
      Modules.DATA_ACCUMULATOR,
      `Detached ${this.shapes.length} buffers (${this.capacity} items)`
    );

    return detached;
  }

  /**
   * Adopt buffers returned from worker.
   *
   * Takes ownership of the transferred buffers. The buffers should be
   * the same ones that were detached (same capacity and structure).
   *
   * @param buffers - Buffers to adopt
   */
  adopt(buffers: TBuffers): void {
    // Validate that we're adopting compatible buffers
    for (const shape of this.shapes) {
      if (!shape.optional && !buffers[shape.name]) {
        throw new Error(`Adopted buffers missing required buffer: ${shape.name}`);
      }
    }

    this.buffers = buffers;
    this.isDetached = false;
    this.stats.reuseCount++;

    log.info(
      Modules.DATA_ACCUMULATOR,
      `Adopted ${this.shapes.length} buffers (reuse #${this.stats.reuseCount})`
    );
  }

  /**
   * Get list of ArrayBuffers for `Comlink.transfer()`.
   *
   * @param buffers - Buffer set to get transferables from
   * @returns Array of ArrayBuffer objects
   */
  getTransferables(buffers: TBuffers): ArrayBuffer[] {
    const transferables: ArrayBuffer[] = [];

    for (const shape of this.shapes) {
      const buf = buffers[shape.name];
      if (buf && buf.buffer && buf.buffer instanceof ArrayBuffer) {
        transferables.push(buf.buffer);
      }
    }

    return transferables;
  }

  /**
   * Get direct access to buffers (for main thread use).
   *
   * @returns Buffer set (or null if detached)
   */
  getBuffers(): TBuffers | null {
    return this.buffers;
  }

  /**
   * Check if buffers are currently detached.
   */
  isBuffersDetached(): boolean {
    return this.isDetached;
  }

  /**
   * Get current capacity in items.
   */
  getCapacity(): number {
    return this.capacity;
  }

  /**
   * Get statistics.
   */
  getStats(): AccumulatorStats {
    return { ...this.stats };
  }

  /**
   * Dispose accumulator and release memory.
   */
  dispose(): void {
    this.buffers = null;
    this.isDetached = false;
    this.capacity = 0;
  }

  /**
   * Allocate buffers based on shapes.
   */
  private allocateBuffers(): void {
    const buffers: TransferableBuffers = {};
    let totalBytes = 0;

    for (const shape of this.shapes) {
      if (shape.optional) {
        // Skip allocation for optional buffers — use enableBuffer() when needed
        continue;
      }

      const elements = this.capacity * shape.elementsPerItem;
      let buf: Float32Array | Uint8Array | Uint16Array;

      switch (shape.type) {
        case 'float32':
          buf = new Float32Array(elements);
          totalBytes += elements * 4;
          break;
        case 'uint8':
          buf = new Uint8Array(elements);
          totalBytes += elements;
          break;
        case 'uint16':
          buf = new Uint16Array(elements);
          totalBytes += elements * 2;
          break;
      }

      buffers[shape.name] = buf;
    }

    this.buffers = buffers as TBuffers;
    this.stats.allocations++;
    this.stats.bytesAllocated += totalBytes;
    this.stats.peakMemoryBytes = Math.max(this.stats.peakMemoryBytes, totalBytes);
  }

  /**
   * Enable an optional buffer.
   *
   * @param name - Buffer name
   */
  enableBuffer(name: string): void {
    const shape = this.shapes.find((s) => s.name === name);
    if (!shape) {
      throw new Error(`Unknown buffer: ${name}`);
    }

    if (!this.buffers) {
      throw new Error('[Accumulator] Buffers not allocated');
    }

    if (this.buffers[name]) {
      return; // Already enabled
    }

    const elements = this.capacity * shape.elementsPerItem;
    let buf: Float32Array | Uint8Array | Uint16Array;

    switch (shape.type) {
      case 'float32':
        buf = new Float32Array(elements);
        break;
      case 'uint8':
        buf = new Uint8Array(elements);
        break;
      case 'uint16':
        buf = new Uint16Array(elements);
        break;
    }

    (this.buffers as TransferableBuffers)[name] = buf;
  }
}

// ============================================================================
// Pre-configured Accumulators
// ============================================================================

/**
 * Create a TransferableAccumulator configured for Points data.
 */
export function createPointsAccumulator(
  initialCapacity = 1024,
  colorType: 'float32' | 'uint8' | 'uint16' = 'float32'
): TransferableAccumulator<PointsBuffers> {
  const shapes: BufferShape[] = [
    { name: 'positions', elementsPerItem: 3, type: 'float32' },
    { name: 'colors', elementsPerItem: 3, type: colorType, optional: true },
    { name: 'radii', elementsPerItem: 1, type: 'float32', optional: true },
    { name: 'sharpness', elementsPerItem: 1, type: 'float32', optional: true },
  ];

  return new TransferableAccumulator<PointsBuffers>(shapes, initialCapacity);
}

/**
 * Create a TransferableAccumulator configured for Lines data.
 */
export function createLinesAccumulator(
  initialCapacity = 1024
): TransferableAccumulator<LinesBuffers> {
  const shapes: BufferShape[] = [
    { name: 'startPositions', elementsPerItem: 3, type: 'float32' },
    { name: 'endPositions', elementsPerItem: 3, type: 'float32' },
    { name: 'startColors', elementsPerItem: 3, type: 'float32', optional: true },
    { name: 'endColors', elementsPerItem: 3, type: 'float32', optional: true },
    { name: 'widths', elementsPerItem: 1, type: 'float32', optional: true },
  ];

  return new TransferableAccumulator<LinesBuffers>(shapes, initialCapacity);
}

/**
 * Create a TransferableAccumulator configured for GSplats data.
 *
 * @param initialCapacity - Initial capacity in splats
 * @param choleskySize - Packed Cholesky size (k = ndim * (ndim + 1) / 2)
 */
export function createGSplatsAccumulator(
  initialCapacity = 1024,
  choleskySize = 6 // 3D: k = 3 * 4 / 2 = 6
): TransferableAccumulator<GSplatsBuffers> {
  const shapes: BufferShape[] = [
    { name: 'positions', elementsPerItem: 3, type: 'float32' },
    { name: 'amplitudes', elementsPerItem: 1, type: 'float32' },
    { name: 'choleskyFactors', elementsPerItem: choleskySize, type: 'float32' },
    { name: 'colors', elementsPerItem: 3, type: 'float32', optional: true },
  ];

  return new TransferableAccumulator<GSplatsBuffers>(shapes, initialCapacity);
}

// ============================================================================
// Worker Protocol Types
// ============================================================================

/**
 * Request to worker with transferable buffers.
 *
 * @internal — preserved for future use; no current consumer.
 */
export interface WorkerProjectionRequest<TParams, TBuffers> {
  /** Projection parameters */
  params: TParams;

  /** Pre-allocated output buffers */
  inputBuffers: TBuffers;
}

/**
 * Response from worker with filled buffers.
 *
 * @internal — preserved for future use; no current consumer.
 */
export interface WorkerProjectionResponse<TBuffers> {
  /** Number of items written */
  itemCount: number;

  /** Filled output buffers (to be adopted back) */
  buffers: TBuffers;
}
