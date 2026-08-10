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

/**
 * A picked element, or `null` when the hover/selection is cleared.
 *
 * Under a `kind=partition` layer `nodeName` and `elementIndex` are reported
 * against different nodes and are NOT directly joinable: `nodeName` is the
 * outermost partition wrapper (the user-facing layer), while `elementIndex` is
 * local to the `part_<i>` leaf that was actually hit. Use `hitNodeName` — the
 * leaf the index belongs to — to resolve the element; it equals `nodeName`
 * whenever there is no partition wrapper.
 */
export interface SelectionPayload {
  /**
   * Scene-node (zarr path) of the picked layer — the outermost
   * `kind=partition` wrapper when the hit sits under one, otherwise the hit
   * node itself.
   */
  nodeName: string;
  /**
   * Index of the picked element within the *hit leaf* — under a partition
   * that is the `part_<i>` leaf, not `nodeName`. Index it against
   * `hitNodeName`.
   *
   * This is the ON-DISK element index (the one the leaf's arrays and its
   * label CSR are keyed by) wherever the node can resolve one — through a
   * published slot → on-disk map, or trivially where the identity already
   * holds and no map is published. Today: a FLAT Points, GSplats or Lines node
   * declaring `has_labels` / `has_image_labels`, and Mesh, whose
   * `gl_VertexID` already is the on-disk ordinal. For a LINES node that
   * on-disk index is the picked segment's START vertex, not a segment index:
   * line labels are per-vertex, and a segment carries a single pick id, so its
   * start endpoint is the one reported. A multi-additive-LOD (laddered) Points
   * or Lines node is the exception among labelled nodes: its label CSR spans
   * the levels and so declares `has_labels` on the ladder parent (#1422), but
   * no map is composed across a ladder, so it reports the raw committed slot —
   * equal to the union index only on a fully-loaded, unsliced layer (#1439).
   * Otherwise it is the element's slot in the buffer that reached the GPU,
   * which after spatial range loading or nD
   * compaction is NOT the on-disk index — and on an unlabelled Lines node it
   * is a per-segment slot against a per-vertex CSR whatever the slicing. See
   * `rendering/picking/picking-system/element-id-map.ts`.
   */
  elementIndex: number;
  /**
   * Scene node (zarr path) `elementIndex` is local to — the `part_<i>` leaf
   * actually hit under a `kind=partition` layer, and the node itself
   * otherwise. This is the path an embedder should index against; `nodeName`
   * is for display.
   */
  hitNodeName: string;
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
