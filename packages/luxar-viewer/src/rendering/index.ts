/**
 * Barrel export for the rendering layer.
 *
 * Matches the pattern used by `cache/`, `data/`, and `ui/`. Internal
 * imports across the codebase can stay deep (and keep tree-shaking
 * fully precise) — this barrel exists primarily for higher-layer
 * consumers (`core/`, `scene/`) that want one stable import point.
 *
 * The blending-state + material-colormap-helpers re-exports here are
 * the **same** symbols re-exported by `src/index.ts` to embedders, so
 * internal callers (`scene/`, `ui/`) and external embedders agree on a
 * single import path.
 */

// Materials
export { PointMaterial } from './materials/point/material-glsl';
export { LineMaterial } from './materials/line/material-glsl';
export { GSplatMaterial } from './materials/gsplat/material-glsl';
export { MaterialManager, materialManager } from './material-manager';

// GPU buffer pooling
export { GPUBufferPool } from './gpu-buffer-pool';

// Adaptive resolution
export { AdaptiveDPRManager } from './adaptive-dpr-manager';

// Post-processing entry point
export { PostProcessingManager } from './post-processing/post-processing-manager';

// Blending helpers (public API; mirrors src/index.ts re-exports).
export {
  getCompleteBlendingState,
  applyBlendingStateToMaterial,
  isAdditiveMode,
  isMaxMode,
  isOpaqueMode,
  isLuminousMode,
  isNormalMode,
  isVolumetricMode,
  needsDepthSort,
} from './blending-state';
export type { CompleteBlendingState } from './blending-state';

// Colormap helpers (public API; mirrors src/index.ts re-exports).
export {
  supportsScalarColormap,
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from './material-colormap-helpers';

// Shared rendering types / utilities that consumers commonly need.
export type { CameraAwareMaterial } from './materials/_shared/camera-aware-material';
export type { BlendingMode } from './material-manager';
