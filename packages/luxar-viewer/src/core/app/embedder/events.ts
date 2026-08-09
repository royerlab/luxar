/**
 * Public event + value types for the programmatic embedder API.
 *
 * Kept separate from the cross-layer {@link LuxarEventMap}
 * (`utils/cross-layer/event-bus.ts`): that bus is internal frame/UI plumbing
 * routed through a module singleton, whereas these events are app-scoped,
 * part of the public package surface, and emitted on a PER-APP emitter that
 * `LuxarApp` owns. Decoupling them keeps the embedder catalog stable and
 * multi-instance-ready (no dependence on the global singleton).
 *
 * Re-exported from the package root (`src/index.ts`).
 */

import type { DimensionMetadata } from '../../../types/dims';

export type { Unsubscribe } from '../../../utils/cross-layer/event-bus';
export type { CameraSnapshot } from '../snapshot/viewer-snapshot';
export type { DimensionMetadata } from '../../../types/dims';

/**
 * Snapshot of the dataset's nD dimension state returned by
 * `LuxarApp.getDimensions()`. All array fields are CLONED copies of the
 * scene-dims manager's internal state — an embedder may read them freely
 * without mutating viewer internals; use `setDimensionValue()` to change a
 * slice position.
 */
export interface EmbedderDimensions {
  /** Total dimensions in the dataset. */
  ndim: number;
  /** Indices currently shown on screen (length 1–3). */
  displayed: number[];
  /** Current slice position per dimension (length === ndim). */
  currentStep: number[];
  /** Per-dimension metadata (name, unit, range, discrete, step, …). */
  metadata: DimensionMetadata[];
  /** `[min, max]` navigable bounds per dimension. */
  ranges: Array<[number, number]>;
}

/** Options for `LuxarApp.screenshot()`. */
export interface ScreenshotOptions {
  /** Output image format. Defaults to `'png'`. */
  format?: 'png' | 'webp' | 'jpeg';
  /** Quality 0–1 for lossy formats (`jpeg`/`webp`); ignored for PNG. */
  quality?: number;
  /** Composite visible DOM overlays onto the frame. Defaults to `true`. */
  includeOverlays?: boolean;
}

/** A picked element, or `null` when the hover/selection is cleared. */
export interface SelectionPayload {
  /** Scene-node (zarr path) of the picked element. */
  nodeName: string;
  /**
   * Index of the picked element within that node.
   *
   * This is the ON-DISK element index (the one the node's arrays and its
   * label CSR are keyed by) wherever the loader could publish a slot →
   * on-disk map — today: a Points node declaring `has_labels` /
   * `has_image_labels`. Otherwise it is the element's slot in the buffer
   * that reached the GPU, which after spatial range loading or nD
   * compaction is NOT the on-disk index. See
   * `rendering/picking/picking-system/element-id-map.ts`.
   */
  elementIndex: number;
}

/**
 * Events an embedder can subscribe to via `LuxarApp.on(event, listener)`.
 *
 * - `dataset-loaded` / `dataset-error` — fire around every dataset load
 *   (initial `init()`, the built-in browser, and `switchDataset()`).
 * - `dimensions-changed` — fires whenever a slice position changes (slider,
 *   keyboard, or `setDimensionValue()`), carrying a fresh `EmbedderDimensions`.
 * - `selection` — fires on hover-pick changes (the element under the cursor,
 *   or `null` when the hover clears). Works on any dataset — the picking
 *   pipeline is provisioned when a `selection` listener exists at dataset
 *   load time, so subscribe BEFORE `init()` / `switchDataset()` (on scenes
 *   with labels it is always provisioned). Hover-driven, not click-to-select.
 */
export interface LuxarEmbedderEventMap {
  'dataset-loaded': { src: string };
  'dataset-error': { src: string; error: Error };
  'dimensions-changed': EmbedderDimensions;
  selection: SelectionPayload | null;
}
