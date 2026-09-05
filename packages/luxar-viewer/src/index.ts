/**
 * Luxar Viewer — public package entry point.
 *
 * Importing this module is **side-effect-free**:
 * - No console patching.
 * - No CSS injection (consumers import `@luxar/viewer/styles.css` separately).
 * - No DOM mutation.
 * - No singleton instantiation.
 *
 * That's the contract that lets a third-party host page do
 * `import { LuxarApp } from '@luxar/viewer'` without surprising changes to
 * its environment.
 *
 * The minimal embed shape:
 *
 * ```ts
 * import { LuxarApp } from '@luxar/viewer';
 * import '@luxar/viewer/styles.css';   // optional but typical
 *
 * const canvas = document.querySelector('canvas#viewer') as HTMLCanvasElement;
 * const app = new LuxarApp();
 * app.on('dataset-fault', ({ error }) => showHostError(error.message));
 * await app.init({
 *   canvas,
 *   src: 'https://example.com/cells.zarr',
 *   updateBrowserUrl: false,         // default: do NOT rewrite host page URL
 * });
 *
 * // Later, when the host wants to tear the viewer down:
 * app.dispose();
 * ```
 *
 * For consumers who want the standalone-app behavior verbatim (URL parsing,
 * theme from `?theme`, debug from `?debug`, codec warming, console patching),
 * use {@link bootstrapStandalone} instead of LuxarApp directly.
 */

// Core API.
export { LuxarApp, type LuxarAppOptions } from './core/app';
export { bootstrapStandalone, type BootstrapOptions } from './core/bootstrap';
export { InputContext, KeyAction } from './input';
export type {
  ContextConfig,
  InputContextId,
  KeyActionId,
  KeyBinding,
  ShortcutHelpMetadata,
  ShortcutHelpSectionId,
} from './input';

/**
 * Layer mode — render a Luxar scene inside a host application's own Three.js
 * renderer, camera, and scene graph. The headless sibling of {@link LuxarApp}:
 * the host keeps its pipeline, the layer contributes a `THREE.Group` plus the
 * per-frame LOD / depth-sort bookkeeping.
 *
 * Same single-instance rule as `LuxarApp`, and mutually exclusive with it.
 */
export { LuxarLayer, type LuxarLayerOptions, type ViewportSize } from './core/layer/luxar-layer';

/**
 * Programmatic embedder API — value/event types for the LuxarApp methods
 * (switchDataset, getDimensions/setDimensionValue, camera, resize, screenshot)
 * and the `on(event, listener)` surface. `LuxarEmbedderEventMap` names the
 * event payloads; `Unsubscribe` is the disposer returned by `on`.
 */
export type {
  LuxarEmbedderEventMap,
  EmbedderDimensions,
  DatasetFaultPayload,
  ScreenshotOptions,
  SelectionPayload,
  ElementPointerPayload,
  CameraSnapshot,
  DimensionMetadata,
  Unsubscribe,
} from './core/app/embedder/events';
/**
 * JSON-serializable snapshot of viewer state (camera placement + per-dimension
 * slice position) so an external caller can reproduce a specific view across
 * reloads, e.g. a "share view" link or a regression harness.
 */
export type { ViewerSnapshot } from './core/app/snapshot/viewer-snapshot';

// URL parsing — useful for embedders that want to honor a few of the
// standalone-app's URL flags without taking the whole bootstrap path.
export { normalizeDataSourceUrl, readUrlParams, type UrlParams } from './config/url-params';

// Storage namespacing — exposed so an embedder can clear Luxar-owned keys
// (e.g. on uninstall) without grepping the codebase for prefixes.
export { StorageKeys } from './utils/storage-keys';

// Build identity. Exported because the standalone surfaces (`window.__luxarBuild`,
// the `<meta>` tag, the startup log line) are all attached by
// `bootstrapStandalone`, and the documented common case for an embedder is to
// construct `LuxarApp` directly and skip it — which would leave the one thing a
// bug report needs unreachable.
export { buildInfo, buildInfoLine, type BuildInfo } from './config/build-info';

/**
 * Loader configuration — shape of the cache/prefetch flags accepted by
 * `LuxarAppOptions.loaderConfig`.
 */
export type { LoaderConfig } from './data/data-loader-types';

/**
 * Optional zarr viewer-config shape — data authors may type their own
 * `viewer_config` metadata against this.
 */
export type { ZarrViewerConfig } from './types/zarr';

// Rendering helpers exposed for embedders that build custom materials
// or bespoke colormap pipelines on top of Luxar's geometry. These are
// stable, side-effect-free utilities.
export { getCompleteBlendingState, applyBlendingStateToMaterial } from './rendering/blending-state';
/**
 * Fully-resolved THREE blending pipeline state (blend equation, src/dst
 * factors, depth-write) for a mode — the value produced by
 * `getCompleteBlendingState` and applied via `applyBlendingStateToMaterial`.
 */
export type { CompleteBlendingState } from './rendering/blending-state';
/**
 * Union of the six canonical Luxar blending modes (additive, volumetric,
 * normal, max, opaque, luminous) selectable per geometry node — a mesh node
 * given `volumetric` warns and falls back to `opaque`.
 */
export type { BlendingMode } from './rendering/material-manager';
export {
  supportsScalarColormap,
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from './rendering/material-colormap-helpers';
