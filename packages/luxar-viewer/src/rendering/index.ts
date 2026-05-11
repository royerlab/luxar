/**
 * Barrel export for the rendering layer.
 *
 * Matches the pattern used by `cache/`, `data/`, and `ui/`. Internal
 * imports across the codebase can stay deep (and keep tree-shaking
 * fully precise) — this barrel exists primarily for higher-layer
 * consumers (`core/`, `scene/`) that want one stable import point.
 */

// Materials
export { PointMaterial } from './point-material';
export { LineMaterial } from './line-material';
export { GSplatMaterial } from './gsplat-material';
export { MaterialManager, materialManager } from './material-manager';

// GPU buffer pooling
export { GPUBufferPool } from './gpu-buffer-pool';

// Adaptive resolution
export { AdaptiveDPRManager } from './adaptive-dpr-manager';

// Post-processing entry point
export { PostProcessingManager } from './post-processing/post-processing-manager';

// Shared rendering types / utilities that consumers commonly need.
export type { CameraAwareMaterial } from './camera-aware-material';
export type { BlendingMode } from './material-manager';
