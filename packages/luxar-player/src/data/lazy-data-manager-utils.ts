/**
 * Pure utility functions for lazy data management
 *
 * This module contains pure, side-effect-free functions extracted from LazyDataManager
 * to improve testability and maintainability. These functions handle the core logic
 * for chunk management, memory calculations, and data slicing without any external
 * dependencies or state mutations.
 *
 * ## Architecture Benefits:
 * - **Testability**: Pure functions are easy to unit test with predictable inputs/outputs
 * - **Reusability**: Can be used by multiple components without coupling
 * - **Maintainability**: Business logic is separated from state management
 * - **Performance**: Functions can be optimized and memoized independently
 *
 * ## Usage Pattern:
 * ```typescript
 * // In LazyDataManager or other components
 * import { calculateRequiredChunks, getChunkId } from './lazy-data-manager-utils';
 *
 * const chunks = calculateRequiredChunks(shape, chunkShape, position, dims);
 * const id = getChunkId(arrayPath, chunkIndices);
 * ```
 */

/**
 * Calculate which chunks are needed for a given slice position.
 *
 * This is a pure function that determines the optimal set of chunks to load.
 *
 * @param arrayShape - The shape of the full array
 * @param chunkShape - The chunk dimensions
 * @param slicePosition - Current position in array indices
 * @param sliceDimensions - Dimensions that should be fully loaded
 * @param preloadRadius - How many adjacent chunks to preload
 * @returns Array of chunk indices to load
 */
export function calculateRequiredChunks(
  arrayShape: number[],
  chunkShape: number[],
  slicePosition: number[],
  sliceDimensions: number[],
  preloadRadius: number = 1
): number[][] {
  const requiredChunks: number[][] = [];
  const nDims = arrayShape.length;

  // Calculate chunk grid dimensions
  const chunkGrid = arrayShape.map((size, i) => Math.ceil(size / chunkShape[i]));

  // Calculate center chunk for current position
  const centerChunk = slicePosition.map((pos, i) => Math.floor(pos / chunkShape[i]));

  // Build list of chunks to load with preload radius
  const ranges = centerChunk.map((center, dim) => {
    if (sliceDimensions.includes(dim)) {
      // Fully load this dimension
      return Array.from({ length: chunkGrid[dim] }, (_, i) => i);
    } else {
      // Load with radius around center
      const min = Math.max(0, center - preloadRadius);
      const max = Math.min(chunkGrid[dim] - 1, center + preloadRadius);
      return Array.from({ length: max - min + 1 }, (_, i) => min + i);
    }
  });

  // Generate all combinations of chunk indices
  function generateCombinations(dimIndex: number, current: number[]): void {
    if (dimIndex === nDims) {
      requiredChunks.push([...current]);
      return;
    }
    for (const idx of ranges[dimIndex]) {
      current[dimIndex] = idx;
      generateCombinations(dimIndex + 1, current);
    }
  }

  generateCombinations(0, new Array(nDims));
  return requiredChunks;
}

/**
 * Generate a unique chunk ID for caching
 *
 * @param arrayPath - Path to the array in the zarr store
 * @param chunkIndices - Indices of the chunk in the chunk grid
 * @returns Unique string identifier for the chunk
 */
export function getChunkId(arrayPath: string, chunkIndices: number[]): string {
  return `${arrayPath}:${chunkIndices.join(',')}`;
}

/**
 * Calculate the slice of data to extract from a chunk
 *
 * @param chunkIndices - The chunk's position in the chunk grid
 * @param chunkShape - Dimensions of each chunk
 * @param arrayShape - Total dimensions of the array
 * @param startIndex - Starting index in the full array to extract
 * @param endIndex - Ending index in the full array to extract
 * @returns Slice ranges for each dimension or null if no overlap
 */
export function calculateChunkSlice(
  chunkIndices: number[],
  chunkShape: number[],
  arrayShape: number[],
  startIndex: number[],
  endIndex: number[]
): { chunkStart: number[]; chunkEnd: number[]; arrayStart: number[]; arrayEnd: number[] } | null {
  const nDims = chunkIndices.length;
  const chunkStart: number[] = [];
  const chunkEnd: number[] = [];
  const arrayStart: number[] = [];
  const arrayEnd: number[] = [];

  for (let d = 0; d < nDims; d++) {
    const chunkStartInArray = chunkIndices[d] * chunkShape[d];
    const chunkEndInArray = Math.min(chunkStartInArray + chunkShape[d], arrayShape[d]);

    // Check if chunk overlaps with requested range
    if (chunkEndInArray <= startIndex[d] || chunkStartInArray >= endIndex[d]) {
      return null; // No overlap
    }

    // Calculate overlap region
    const overlapStart = Math.max(startIndex[d], chunkStartInArray);
    const overlapEnd = Math.min(endIndex[d], chunkEndInArray);

    chunkStart.push(overlapStart - chunkStartInArray);
    chunkEnd.push(overlapEnd - chunkStartInArray);
    arrayStart.push(overlapStart);
    arrayEnd.push(overlapEnd);
  }

  return { chunkStart, chunkEnd, arrayStart, arrayEnd };
}

/**
 * Calculate total memory size of chunks
 *
 * @param chunks - Map of cached chunks
 * @returns Total size in bytes
 */
export function calculateTotalCacheSize(chunks: Map<string, { sizeBytes: number }>): number {
  let total = 0;
  for (const chunk of chunks.values()) {
    total += chunk.sizeBytes;
  }
  return total;
}

/**
 * Determine which chunks to evict based on LRU strategy
 *
 * @param chunks - Map of cached chunks with lastAccessed timestamps
 * @param targetSize - Target cache size in bytes
 * @param currentSize - Current cache size in bytes
 * @returns Array of chunk IDs to evict
 */
export function selectChunksToEvict(
  chunks: Map<string, { sizeBytes: number; lastAccessed: number }>,
  targetSize: number,
  currentSize: number
): string[] {
  if (currentSize <= targetSize) {
    return [];
  }

  // Sort chunks by last accessed time (oldest first)
  const sortedChunks = Array.from(chunks.entries()).sort(
    (a, b) => a[1].lastAccessed - b[1].lastAccessed
  );

  const toEvict: string[] = [];
  let freedSpace = 0;
  const spaceToFree = currentSize - targetSize;

  for (const [id, chunk] of sortedChunks) {
    if (freedSpace >= spaceToFree) {
      break;
    }
    toEvict.push(id);
    freedSpace += chunk.sizeBytes;
  }

  return toEvict;
}

/**
 * Calculate array slice parameters for efficient data loading
 *
 * @param position - Current position in nD space
 * @param shape - Array shape
 * @param radius - Radius around position to load
 * @returns Start and end indices for slicing
 */
export function calculateSliceParams(
  position: number[],
  shape: number[],
  radius: number[]
): { start: number[]; end: number[] } {
  const start = position.map((pos, i) => Math.max(0, Math.floor(pos - radius[i])));

  const end = position.map((pos, i) => Math.min(shape[i], Math.ceil(pos + radius[i] + 1)));

  return { start, end };
}

/**
 * Check if a chunk index is valid for the given array shape
 *
 * @param chunkIndex - The chunk indices to validate
 * @param chunkGrid - The dimensions of the chunk grid
 * @returns True if the chunk index is valid
 */
export function isValidChunkIndex(chunkIndex: number[], chunkGrid: number[]): boolean {
  if (chunkIndex.length !== chunkGrid.length) {
    return false;
  }

  for (let i = 0; i < chunkIndex.length; i++) {
    if (chunkIndex[i] < 0 || chunkIndex[i] >= chunkGrid[i]) {
      return false;
    }
  }

  return true;
}

/**
 * Calculate the byte size of typed array data
 *
 * @param data - The typed array
 * @returns Size in bytes
 */
export function calculateDataSize(data: ArrayBuffer | Float32Array | Uint8Array): number {
  if (data instanceof ArrayBuffer) {
    return data.byteLength;
  } else if (data instanceof Float32Array) {
    return data.byteLength;
  } else if (data instanceof Uint8Array) {
    return data.byteLength;
  }
  return 0;
}

/**
 * Estimate memory usage for a given array shape and dtype
 *
 * @param shape - Array dimensions
 * @param dtype - Data type (e.g., '<f4', '|u1')
 * @returns Estimated size in bytes
 */
export function estimateArrayMemory(shape: number[], dtype: string): number {
  const totalElements = shape.reduce((acc, dim) => acc * dim, 1);

  // Parse dtype to get bytes per element
  let bytesPerElement = 4; // Default to float32
  if (dtype.includes('f4') || dtype.includes('float32')) {
    bytesPerElement = 4;
  } else if (dtype.includes('f8') || dtype.includes('float64')) {
    bytesPerElement = 8;
  } else if (dtype.includes('u1') || dtype.includes('uint8')) {
    bytesPerElement = 1;
  } else if (dtype.includes('u2') || dtype.includes('uint16')) {
    bytesPerElement = 2;
  } else if (dtype.includes('u4') || dtype.includes('uint32')) {
    bytesPerElement = 4;
  }

  return totalElements * bytesPerElement;
}

/**
 * Determine if lazy loading should be enabled based on dataset size
 *
 * @param arrayShape - Shape of the array
 * @param dtype - Data type
 * @param maxMemoryMB - Maximum memory available in MB
 * @returns True if lazy loading should be enabled
 */
export function shouldEnableLazyLoading(
  arrayShape: number[],
  dtype: string,
  maxMemoryMB: number
): boolean {
  const estimatedSizeMB = estimateArrayMemory(arrayShape, dtype) / (1024 * 1024);

  // Enable lazy loading if dataset is larger than 50% of available memory
  return estimatedSizeMB > maxMemoryMB * 0.5;
}
