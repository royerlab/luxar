/**
 * WASM bindings for spatial queries and nD visibility computation
 *
 * Phase 2: TypeScript fallback implementations
 * Phase 3: Replace with actual WASM module
 *
 * NOTE: These TypeScript implementations are functional but slower than WASM.
 * They enable Phase 2 worker infrastructure without requiring WASM build setup.
 */

export interface WasmModule {
  /**
   * Query chunks whose bounding boxes intersect the nD slice
   *
   * @param chunkBounds - Flattened chunk bounds [numChunks * ndim * 2] (min/max pairs)
   * @param slicePosition - Current slice position in nD space [ndim]
   * @param tolerance - Tolerance per dimension [ndim]
   * @param ndim - Number of dimensions
   * @param numChunks - Total number of chunks
   * @param output - Output buffer for matching chunk indices [numChunks]
   * @returns Number of matching chunks
   */
  query_chunks_for_view(
    chunkBounds: Float32Array,
    slicePosition: Float32Array,
    tolerance: Float32Array,
    ndim: number,
    numChunks: number,
    output: Uint32Array
  ): number;

  /**
   * Compute nD visibility for points using hypersphere intersection
   *
   * @param positions - Point positions [numPoints * ndim]
   * @param radii - Point radii [numPoints]
   * @param slicePosition - Current slice position [ndim]
   * @param tolerance - Tolerance per dimension [ndim]
   * @param ndim - Number of dimensions
   * @param numPoints - Total number of points
   * @param output - Output visibility mask [numPoints] (1=visible, 0=hidden)
   * @returns Number of visible points
   */
  compute_nd_visibility_points(
    positions: Float32Array,
    radii: Float32Array,
    slicePosition: Float32Array,
    tolerance: Float32Array,
    ndim: number,
    numPoints: number,
    output: Uint8Array
  ): number;

  /**
   * Compute nD visibility for line segments (endpoint-based)
   *
   * @param vertices - Vertex positions [numVertices * ndim]
   * @param segments - Segment indices [numSegments * 2] (pairs of vertex indices)
   * @param widths - Per-vertex widths [numVertices]
   * @param slicePosition - Current slice position [ndim]
   * @param tolerance - Tolerance per dimension [ndim]
   * @param ndim - Number of dimensions
   * @param numSegments - Total number of segments
   * @param output - Output visibility mask [numSegments]
   * @returns Number of visible segments
   */
  compute_nd_visibility_lines(
    vertices: Float32Array,
    segments: Uint32Array,
    widths: Float32Array,
    slicePosition: Float32Array,
    tolerance: Float32Array,
    ndim: number,
    numSegments: number,
    output: Uint8Array
  ): number;

  /**
   * Compute nD visibility for GSplats using ellipsoid extent
   *
   * @param centers - Splat centers [numSplats * ndim]
   * @param choleskyFactors - Packed Cholesky factors [numSplats * k] where k = ndim*(ndim+1)/2
   * @param slicePosition - Current slice position [ndim]
   * @param tolerance - Tolerance per dimension [ndim]
   * @param ndim - Number of dimensions
   * @param numSplats - Total number of splats
   * @param output - Output visibility mask [numSplats]
   * @returns Number of visible splats
   */
  compute_nd_visibility_gsplats(
    centers: Float32Array,
    choleskyFactors: Float32Array,
    slicePosition: Float32Array,
    tolerance: Float32Array,
    ndim: number,
    numSplats: number,
    output: Uint8Array
  ): number;
}

/**
 * TypeScript fallback implementation (Phase 2)
 * Will be replaced with actual WASM in Phase 3
 */
class TypeScriptWasmFallback implements WasmModule {
  query_chunks_for_view(
    chunkBounds: Float32Array,
    slicePosition: Float32Array,
    tolerance: Float32Array,
    ndim: number,
    numChunks: number,
    output: Uint32Array
  ): number {
    let count = 0;

    for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
      const boundsOffset = chunkIdx * ndim * 2;
      let intersects = true;

      // Check if chunk bounding box intersects the nD slice
      for (let dim = 0; dim < ndim; dim++) {
        const minBound = chunkBounds[boundsOffset + dim * 2];
        const maxBound = chunkBounds[boundsOffset + dim * 2 + 1];
        const slicePos = slicePosition[dim];
        const tol = tolerance[dim];

        // Check intersection: [minBound, maxBound] ∩ [slicePos - tol, slicePos + tol]
        if (maxBound < slicePos - tol || minBound > slicePos + tol) {
          intersects = false;
          break;
        }
      }

      if (intersects) {
        output[count++] = chunkIdx;
      }
    }

    return count;
  }

  compute_nd_visibility_points(
    positions: Float32Array,
    radii: Float32Array,
    slicePosition: Float32Array,
    tolerance: Float32Array,
    ndim: number,
    numPoints: number,
    output: Uint8Array
  ): number {
    let visibleCount = 0;

    for (let i = 0; i < numPoints; i++) {
      const posOffset = i * ndim;
      const radius = radii[i];
      let distanceSquared = 0;

      // Compute distance in hidden dimensions only
      // (displayed dimensions are always visible)
      for (let dim = 3; dim < ndim; dim++) {
        const pos = positions[posOffset + dim];
        const slice = slicePosition[dim];
        const tol = tolerance[dim];
        const delta = pos - slice;

        // Normalize by tolerance
        const normalizedDelta = delta / tol;
        distanceSquared += normalizedDelta * normalizedDelta;
      }

      // Check if point is within hypersphere
      // Use effective radius (point radius normalized by tolerance)
      const effectiveRadiusSquared = (radius / tolerance[0]) ** 2;
      const visible = distanceSquared <= effectiveRadiusSquared;

      output[i] = visible ? 1 : 0;
      if (visible) visibleCount++;
    }

    return visibleCount;
  }

  compute_nd_visibility_lines(
    vertices: Float32Array,
    segments: Uint32Array,
    widths: Float32Array,
    slicePosition: Float32Array,
    tolerance: Float32Array,
    ndim: number,
    numSegments: number,
    output: Uint8Array
  ): number {
    let visibleCount = 0;

    for (let i = 0; i < numSegments; i++) {
      const idx1 = segments[i * 2];
      const idx2 = segments[i * 2 + 1];

      // Check if EITHER endpoint is visible
      const v1Offset = idx1 * ndim;
      const v2Offset = idx2 * ndim;
      const width1 = widths[idx1];
      const width2 = widths[idx2];

      let visible = false;

      // Check endpoint 1
      let dist1Sq = 0;
      for (let dim = 3; dim < ndim; dim++) {
        const delta = (vertices[v1Offset + dim] - slicePosition[dim]) / tolerance[dim];
        dist1Sq += delta * delta;
      }
      const effectiveRadius1 = width1 / tolerance[0];
      if (dist1Sq <= effectiveRadius1 * effectiveRadius1) {
        visible = true;
      }

      // Check endpoint 2 if endpoint 1 not visible
      if (!visible) {
        let dist2Sq = 0;
        for (let dim = 3; dim < ndim; dim++) {
          const delta = (vertices[v2Offset + dim] - slicePosition[dim]) / tolerance[dim];
          dist2Sq += delta * delta;
        }
        const effectiveRadius2 = width2 / tolerance[0];
        if (dist2Sq <= effectiveRadius2 * effectiveRadius2) {
          visible = true;
        }
      }

      output[i] = visible ? 1 : 0;
      if (visible) visibleCount++;
    }

    return visibleCount;
  }

  compute_nd_visibility_gsplats(
    centers: Float32Array,
    _choleskyFactors: Float32Array, // Not used in Phase 2 fallback (simplified visibility)
    slicePosition: Float32Array,
    tolerance: Float32Array,
    ndim: number,
    numSplats: number,
    output: Uint8Array
  ): number {
    let visibleCount = 0;

    for (let i = 0; i < numSplats; i++) {
      const centerOffset = i * ndim;

      // Simplified visibility check (proper Mahalanobis distance in Phase 3)
      // For now, just check if center is within tolerance
      let visible = true;
      for (let dim = 3; dim < ndim; dim++) {
        const delta = Math.abs(centers[centerOffset + dim] - slicePosition[dim]);
        if (delta > tolerance[dim] * 2) {
          // Use 2x tolerance for ellipsoid extent approximation
          visible = false;
          break;
        }
      }

      output[i] = visible ? 1 : 0;
      if (visible) visibleCount++;
    }

    return visibleCount;
  }
}

/**
 * Initialize WASM module
 *
 * Phase 3: Attempts to load actual WASM module, falls back to TypeScript if unavailable
 *
 * Load order:
 * 1. Try to load compiled WASM from /wasm/luxar_wasm_bg.wasm
 * 2. If fails (not built or browser incompatibility), use TypeScript fallback
 */
export async function initWasm(): Promise<WasmModule> {
  // Phase 3: Try to load actual WASM module
  try {
    // Use Function constructor to avoid TypeScript compile-time module resolution
    // This allows the code to compile even when WASM module doesn't exist yet
    const importWasm = new Function('return import("/wasm/luxar_wasm.js")');
    const wasmModule = await importWasm();

    // Initialize WASM (loads the .wasm binary)
    await wasmModule.default();

    console.log('[WASM] ✅ Loaded compiled WASM module (Phase 3)');

    // Return the WASM module (it already implements WasmModule interface)
    return wasmModule as unknown as WasmModule;
  } catch (error) {
    // WASM not available - use TypeScript fallback
    console.warn('[WASM] ⚠️ Failed to load WASM module, using TypeScript fallback:', error);
    console.log('[WASM] To build WASM module: pnpm build:wasm (or make wasm-build)');
    console.log('[WASM] See src/workers/wasm/README.md for build instructions');

    return new TypeScriptWasmFallback();
  }
}
