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

import type { ControlPanelSettings } from '../../../config/zarr-bridge/control-panel';
import type { DimensionMetadata } from '../../../types/dims';

export type { Unsubscribe } from '../../../utils/cross-layer/event-bus';
import type { CameraSnapshot } from '../snapshot/viewer-snapshot';
import type { RenderingSettings } from '../../../config';
import type { BlendingMode } from '../../../types/blending';
import type { LayerType } from '../../../ui/layers/layer-state';
import type { AudioState } from '../../../types/audio';

export type { AudioState, AudioPatch, AudioBusName, PanningModel } from '../../../types/audio';
export type { CameraSnapshot } from '../snapshot/viewer-snapshot';
export type { RenderingSettings } from '../../../config';
export type { BlendingMode } from '../../../types/blending';
export type { LayerType } from '../../../ui/layers/layer-state';
export type { FlyToOptions, FlightResult, FlightEasing } from '../camera/camera-flight';
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

/** A latched fault that leaves the current dataset loaded but unable to update. */
export interface DatasetFaultPayload {
  /** Dataset source passed to `init()` or `switchDataset()`. */
  src: string;
  /** Current error reported by the active scene loader. */
  error: Error;
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
   * per-element string/image CSRs are keyed by) wherever the node can resolve
   * one — through a published slot → on-disk map, or trivially where the
   * identity already holds and no map is published. Today: a Points, GSplats
   * or Lines node declaring `has_labels` / `has_image_labels` / `has_keys` —
   * flat, or, for Points, an additive-LOD ladder (see below) — and Mesh, whose
   * `gl_VertexID` already is the on-disk ordinal. For a LINES node that on-disk
   * index is the picked segment's START vertex, not a segment index:
   * line string/image channels are per-vertex, and a segment carries a single
   * pick id, so its start endpoint is the one reported. A multi-additive-LOD
   * (laddered) node carries one labels/keys CSR per present channel on the
   * ladder parent, spanning the levels, and declares `has_labels` / `has_keys`
   * there (#1422). On POINTS that resolves like a flat node: the per-level maps
   * are composed into the union CSR's index space (#1439), so the index holds
   * under culling and compaction too, degrading to the raw committed slot only
   * when the levels' own metadata is inconsistent. On LINES no map is composed
   * across the levels, so it reports the raw committed slot, which is a
   * per-segment one against the per-vertex union CSR — wrong at the granularity
   * whatever the slicing (#1439 covers Points only). Otherwise it is the
   * element's slot in the buffer that reached the GPU, which after spatial
   * range loading or nD compaction is NOT the on-disk index — and on a Lines
   * node without a per-element string channel it is a per-segment slot in what
   * is a per-vertex element space whatever the slicing. See
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
 * Payload for {@link LuxarEmbedderEventMap.element-click} and
 * `element-contextmenu` (issue #1917).
 *
 * Extends the hover {@link SelectionPayload} fields with the gesture itself,
 * so all of that type's caveats about `elementIndex` apply here unchanged.
 */
export interface ElementPointerPayload extends SelectionPayload {
  /**
   * Which pointer button: 0 = primary, 2 = secondary.
   *
   * Reports the raw DOM value, so a macOS Ctrl+primary-click — the
   * platform's secondary gesture — arrives as `0` on an
   * `element-contextmenu` event. Switch on the event NAME, not on this,
   * to tell the two gestures apart.
   */
  button: number;
  /** Viewport coordinates of the gesture, in CSS pixels. */
  x: number;
  y: number;
  /**
   * The URL the built-in handler resolved from the element's `link` template,
   * or null when it has none, the template could not be made safe, or link
   * opening is disabled. Reported so a host can mirror or override the
   * behaviour without re-implementing template resolution.
   */
  link: string | null;
}

/**
 * One row of the Layers panel as the embedder API reports it: the per-layer
 * appearance knobs a remote controller may read and (via {@link LayerPatch})
 * write. Structural / diagnostic fields of the panel's own `LayerInfo` are
 * deliberately left out; this is the stable subset.
 */
export interface LayerSummary {
  /** Scene-graph path — the key `setLayer()` takes. */
  path: string;
  name: string;
  type: LayerType;
  visible: boolean;
  opacity: number;
  gamma: number;
  /** Display window `[min, max]` in data units. */
  displayRange: [number, number];
  /** Authored data bounds the window is clamped into. */
  dataRange: [number, number];
  /** Active colormap name, or `null` for direct colour. */
  colormap: string | null;
  supportsColormap: boolean;
  blendingMode: BlendingMode;
  /** Volumetric absorption κ (meaningful in `volumetric` blending only). */
  absorption: number;
  /** Explicit compositing order, or `null` when inherited/automatic. */
  layerOrder: number | null;
  /** Live linear gain — present on `type === 'sound'` layers only. */
  gain?: number;
}

/**
 * Partial appearance update for one layer. Every field is optional; only the
 * fields present are applied, each through the same code path the Layers
 * panel's own control uses. `colormap: null` returns to direct colour;
 * `layerOrder: null` releases an explicit order.
 */
export interface LayerPatch {
  visible?: boolean;
  opacity?: number;
  gamma?: number;
  displayRange?: [number, number];
  colormap?: string | null;
  blendingMode?: BlendingMode;
  absorption?: number;
  layerOrder?: number | null;
  /** Sound layers only: live linear gain, clamped to `[0, 2]`. Ignored elsewhere. */
  gain?: number;
}

/**
 * Everything a remote controller needs to mirror the viewer: the dataset,
 * the camera pose, the slice position, the rendering settings and the
 * per-layer appearance. All values are copies — mutate freely.
 */
export interface ViewerState {
  /** Dataset URL currently shown, or `null` before the first load. */
  src: string | null;
  /**
   * What the display calls itself: the scene's authored
   * `viewer_config.title`, else the `?title=` parameter, else the page's own.
   *
   * Here because a remote controller cannot know it any other way — a touch
   * panel is a separate page whose own `document.title` is the bundle's
   * generic string, and putting that in front of an audience is exactly the
   * wrong heading.
   */
  title: string;
  camera: CameraSnapshot;
  dimensions: EmbedderDimensions;
  rendering: RenderingSettings;
  layers: LayerSummary[];
  /** The sound layer: context state, mute, gains, what is playing. */
  audio: AudioState;
  /**
   * The scene's authored `viewer_config.control_panel` block, or `null`.
   *
   * `null` means "derive everything", which is the normal case: the touch
   * panel builds itself from a discrete dimension's `categories` and needs no
   * authoring at all. Here because the panel is a SEPARATE PAGE from the
   * display and has no other way to read the store's attributes.
   */
  controlPanel: ControlPanelSettings | null;
}

/**
 * Events an embedder can subscribe to via `LuxarApp.on(event, listener)`.
 *
 * - `dataset-loaded` / `dataset-error` — fire around every dataset load
 *   (initial `init()`, the built-in browser, and `switchDataset()`).
 * - `dataset-fault` — fires when an already-loaded dataset becomes unable to
 *   update, for example after an archive URL expires. The last complete frame
 *   remains visible; an explicit retry clears the fault, and a recurring fault
 *   fires the event again. Call `getDatasetFault()` to inspect the current state.
 * - `dimensions-changed` — fires whenever a slice position changes (slider,
 *   keyboard, or `setDimensionValue()`), carrying a fresh `EmbedderDimensions`.
 * - `selection` — fires on hover-pick changes (the element under the cursor,
 *   or `null` when the hover clears). Works on any dataset — the picking
 *   pipeline is provisioned when a `selection` listener exists at dataset
 *   load time, so subscribe BEFORE `init()` / `switchDataset()` (on scenes
 *   with labels it is always provisioned). Hover-driven: it reports what is
 *   under the cursor, not what was clicked.
 * - `element-click` / `element-contextmenu` — fire when the user left- or
 *   right-clicks an element without dragging (issue #1917). Unlike
 *   `selection`, these are gesture-driven. They fire ALONGSIDE the built-in
 *   behaviour rather than instead of it: a host that wants exclusive control
 *   should also pass `allowLinks: false` (or load with `?no-links`), which
 *   suppresses navigation while still delivering the events. Like `selection`,
 *   a listener present at dataset-load time provisions the picking pipeline,
 *   so subscribe BEFORE `init()` / `switchDataset()` — on a scene with no
 *   labels and no interaction templates, subscribing afterwards leaves picking
 *   switched off and the event never fires.
 */
export interface LuxarEmbedderEventMap {
  'dataset-loaded': { src: string };
  /**
   * Fires whenever the camera pose changes — interactive orbit/fly input,
   * `setCameraPose()`, `flyTo()` frames, auto-rotate — at frame rate while
   * the camera is moving. Consumers relaying over a network should throttle.
   */
  'camera-changed': CameraSnapshot;
  'dataset-error': { src: string; error: Error };
  'dataset-fault': DatasetFaultPayload;
  'dimensions-changed': EmbedderDimensions;
  selection: SelectionPayload | null;
  'element-click': ElementPointerPayload;
  'element-contextmenu': ElementPointerPayload;
  /** A sound node started playing (after its `delay_ms`). `name` is the node name. */
  'sound-started': { name: string };
  /** A sound node stopped — its `once` clip ran out, or it faded out on the slab edge. */
  'sound-ended': { name: string };
  /** The matched story waypoint changed away from `index` (`viewer_config.waypoints` order). */
  'waypoint-departed': { index: number };
  /**
   * The flight to waypoint `index` resolved (or it was snapped to at load).
   * `completed: false` = the visitor cancelled the flight; it still counts as
   * an arrival from wherever the camera stopped.
   */
  'waypoint-arrived': { index: number; completed: boolean };
}
