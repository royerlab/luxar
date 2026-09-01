/**
 * Lifecycle hints supplied when acquiring a pooled geometry.
 *
 * @module rendering/gpu-buffer-pool/acquire-options
 */

export interface BufferAcquireOptions {
  /** Whether this node may need a larger capacity after this commit. */
  canRegrow?: boolean;
}
