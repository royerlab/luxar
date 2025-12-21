/**
 * Type definitions for luxar-viewer.
 *
 * This module exports all type definitions used throughout the viewer.
 *
 * @module types
 */

// Dimension types
export type { DimensionMetadata, SimpleDims } from './dims';
export { initializeDims, getDimensionRanges } from './dims';

// Lines types
export type {
  OrderingMetadata,
  LineType,
  LinesMetadata,
  LinesChunkSpatialIndex,
  SegmentRange,
  LoadedLinesData,
  ProcessedLinesData,
  ClippedSegment,
  LinesDataLoader,
  LinesViewState,
  LinesUserData,
} from './lines';
export { isLinesMetadata, isLinesUserData, isValidLineType } from './lines';

// GSplats types
export type {
  ValueRange,
  CoordinateBounds,
  GSplatsMetadata,
  GSplatsChunkSpatialIndex,
  SplatRange,
  LoadedGSplatsData,
  ProcessedGSplatsData,
  GSplatsDataLoader,
  GSplatsViewState,
  GSplatsUserData,
} from './gsplats';
export {
  isGSplatsMetadata,
  isGSplatsUserData,
  choleskyPackedSize,
  CHOLESKY_SIZES,
} from './gsplats';

// Zarr types
export type { ZarrSceneAttrs, ZarrNodeAttrs } from './zarr';
export { hasContentsMethod } from './zarr';
