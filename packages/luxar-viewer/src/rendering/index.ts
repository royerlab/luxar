/**
 * Barrel export for the rendering layer.
 *
 * Matches the pattern used by `cache/`, `data/`, and `ui/`: the one import
 * point higher layers (`scene/`, `ui/`) use for the rendering symbols they
 * share. Everything else is imported deep, which keeps tree-shaking fully
 * precise. NOT the embedder API — that is `src/index.ts`, the package's
 * only entry point.
 */

export { materialManager, type BlendingMode } from './material-manager';
export { PostProcessingManager } from './post-processing/post-processing-manager';
