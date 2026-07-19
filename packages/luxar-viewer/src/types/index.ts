/**
 * Type definitions for luxar-viewer.
 *
 * This module exports all type definitions used throughout the viewer.
 *
 * @module types
 */

// Blending modes (canonical mode set — SSOT)
export type { BlendingMode } from './blending';
export { BLENDING_MODES } from './blending';

// Dimension types
export type { DimensionMetadata, SimpleDims } from './dims';
export { initializeDims, getDimensionRanges } from './dims';

// Points types
export type { PointsMetadata, PointsViewState, PointsDataLoader, PointsUserData } from './points';
export { isPointsMetadata, isPointsUserData } from './points';

// Lines types
export type {
  OrderingMetadata,
  LineType,
  LinesMetadata,
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

// Committed-data stamp (memoized-concat no-op contract)
export type { CommittedDataUserData } from './committed-data';
export {
  hasCommittedData,
  getCommittedData,
  setCommittedData,
  clearCommittedData,
} from './committed-data';

// Zarr types
export type { ZarrSceneAttrs, ZarrNodeAttrs } from './zarr';
export { hasContentsMethod } from './zarr';
