/**
 * WASM module interface: projection, effective-radius, and decode kernels.
 *
 * This interface defines the contract between TypeScript and the WASM module.
 * Both the compiled WASM and the TypeScript fallback implement this interface.
 */
export interface WasmModule {
  /**
   * Calculate effective radii for nD points when sliced.
   *
   * When an nD hypersphere is sliced by a hyperplane, the visible cross-section
   * has a smaller radius. This function computes that effective radius.
   *
   * @param positions - Point positions [numPoints * ndim]
   * @param radii - Original point radii [numPoints]
   * @param displayDims - Dimensions to display (typically [0,1,2]) [numDisplayDims]
   * @param slicePosition - Current slice position [ndim]
   * @param spatialExtendDims - Which dims are spatial (1) vs discrete (0) [ndim]
   * @param ndim - Total number of dimensions
   * @param numPoints - Number of points
   * @param output - Output effective radii [numPoints]
   * @returns Number of points with non-zero effective radius (visible points)
   */
  calculate_effective_radii(
    positions: Float32Array,
    radii: Float32Array,
    displayDims: Uint32Array,
    slicePosition: Float32Array,
    spatialExtendDims: Uint8Array,
    ndim: number,
    numPoints: number,
    output: Float32Array
  ): number;

  /**
   * Sort splats back-to-front by camera-space depth.
   *
   * Produces the permutation consumed by the `aSortedIndex` instance
   * attribute: `ordering[j]` is the original splat index drawn at instance
   * slot `j` (slot 0 = farthest). Degenerate depth ranges (single depth
   * plane, everything behind the camera) yield the identity ordering.
   * Input is always projected 3D centers, so no ndim cap applies.
   *
   * @param centers3 - Projected 3D splat centers [count * 3]
   * @param modelView - Column-major 4x4 model-view matrix [16]
   * @param ordering - Output permutation [count]
   * @param count - Number of splats
   * @returns Number of splats placed via depth keys (0 = identity fallback)
   */
  sort_splats_by_depth(
    centers3: Float32Array,
    modelView: Float32Array,
    ordering: Uint32Array,
    count: number
  ): number;

  // ============================================================================
  // DECODE FUNCTIONS - Dequantize compressed data formats
  // ============================================================================

  /** Decode quantized uint8 to float32. Maps [0,255] -> [minVal,maxVal] */
  decode_quantized_u8(data: Uint8Array, minVal: number, maxVal: number, output: Float32Array): void;

  /** Decode quantized uint16 to float32. Maps [0,65535] -> [minVal,maxVal] */
  decode_quantized_u16(
    data: Uint16Array,
    minVal: number,
    maxVal: number,
    output: Float32Array
  ): void;

  /** Decode log-space quantized uint8. Result = expm1(normalized * maxLog) */
  decode_log_scalar_u8(data: Uint8Array, maxLog: number, output: Float32Array): void;

  /** Decode log-space quantized uint16. Result = expm1(normalized * maxLog) */
  decode_log_scalar_u16(data: Uint16Array, maxLog: number, output: Float32Array): void;

  /** Decode geometric-log uint8 (reserved zero level; min/max-anchored). */
  decode_geolog_scalar_u8(
    data: Uint8Array,
    minLog: number,
    maxLog: number,
    output: Float32Array
  ): void;

  /** Decode geometric-log uint16 (reserved zero level; min/max-anchored). */
  decode_geolog_scalar_u16(
    data: Uint16Array,
    minLog: number,
    maxLog: number,
    output: Float32Array
  ): void;

  /**
   * Decode per-channel LINEAR (fixed-point) uint8 codes to float32.
   * Per-column `[lo, hi]` scales (f64, straight from the JSON attrs);
   * `colOffset` is the column phase of the first element.
   */
  decode_linear_perchannel_u8(
    data: Uint8Array,
    colLo: Float64Array,
    colHi: Float64Array,
    colOffset: number,
    output: Float32Array
  ): void;

  /** Decode per-channel LINEAR (fixed-point) uint16 codes to float32. */
  decode_linear_perchannel_u16(
    data: Uint16Array,
    colLo: Float64Array,
    colHi: Float64Array,
    colOffset: number,
    output: Float32Array
  ): void;

  /**
   * Decode per-channel LOG uint8 codes to float32 (`x = expm1(y)`).
   * `zeroLevel: true` = reserved zero code 0 + codes 1..255 over the
   * nonzero-anchored scale; `false` = legacy all-levels mapping.
   */
  decode_log_perchannel_u8(
    data: Uint8Array,
    colLo: Float64Array,
    colHi: Float64Array,
    zeroLevel: boolean,
    colOffset: number,
    output: Float32Array
  ): void;

  /** Decode per-channel LOG uint16 codes to float32. */
  decode_log_perchannel_u16(
    data: Uint16Array,
    colLo: Float64Array,
    colHi: Float64Array,
    zeroLevel: boolean,
    colOffset: number,
    output: Float32Array
  ): void;

  /** Decode per-channel SIGNED-LOG uint8 codes (`x = sign(y)·expm1(|y|)`). */
  decode_signed_log_perchannel_u8(
    data: Uint8Array,
    colLo: Float64Array,
    colHi: Float64Array,
    zeroLevel: boolean,
    colOffset: number,
    output: Float32Array
  ): void;

  /** Decode per-channel SIGNED-LOG uint16 codes to float32. */
  decode_signed_log_perchannel_u16(
    data: Uint16Array,
    colLo: Float64Array,
    colHi: Float64Array,
    zeroLevel: boolean,
    colOffset: number,
    output: Float32Array
  ): void;

  /**
   * Decode per-channel TRUE-log uint8 codes (`x = exp(y)`; HDR colors).
   * Reserved zero level always on (name contract — no flag).
   */
  decode_geolog_perchannel_u8(
    data: Uint8Array,
    colLo: Float64Array,
    colHi: Float64Array,
    colOffset: number,
    output: Float32Array
  ): void;

  /** Decode per-channel TRUE-log uint16 codes to float32. */
  decode_geolog_perchannel_u16(
    data: Uint16Array,
    colLo: Float64Array,
    colHi: Float64Array,
    colOffset: number,
    output: Float32Array
  ): void;

  /** Decode LUT indices (uint8) to scalar float values */
  decode_lut_scalar_u8(indices: Uint8Array, lut: Float32Array, output: Float32Array): void;

  /** Decode LUT indices (uint16) to scalar float values */
  decode_lut_scalar_u16(indices: Uint16Array, lut: Float32Array, output: Float32Array): void;

  /** Decode LUT indices (uint8) to k-element vectors */
  decode_lut_row_u8(indices: Uint8Array, lut: Float32Array, k: number, output: Float32Array): void;

  /** Decode LUT indices (uint16) to k-element vectors */
  decode_lut_row_u16(
    indices: Uint16Array,
    lut: Float32Array,
    k: number,
    output: Float32Array
  ): void;

  /** Broadcast a value to all points */
  decode_broadcasted(
    value: Float32Array,
    numPoints: number,
    elementsPerPoint: number,
    output: Float32Array
  ): void;

  // ============================================================================
  // PROJECTION FUNCTIONS - nD to 3D projection and bounds calculation
  // ============================================================================

  /**
   * Extract 3D positions from nD positions using display dimension indices.
   *
   * @param positionsNd - Input nD positions [numPoints * ndim]
   * @param displayDims - Which dimensions to display as X,Y,Z [3 or fewer]
   * @param ndim - Total number of dimensions
   * @param numPoints - Number of points
   * @param output - Output 3D positions [numPoints * 3]
   */
  extract_3d_positions(
    positionsNd: Float32Array,
    displayDims: Uint32Array,
    ndim: number,
    numPoints: number,
    output: Float32Array
  ): void;

  // ============================================================================
  // GSPLATS PROCESSING - nD to 3D conversion and attenuation
  // ============================================================================

  /**
   * Compute Mahalanobis distance for a single point using packed Cholesky factor.
   *
   * Given L (lower-triangular Cholesky of covariance),
   * Mahalanobis distance = ||L⁻¹ · (x - μ)||
   *
   * @param diff - Difference vector (x - μ) [ndim]
   * @param packedL - Packed Cholesky factor [packedSize]
   * @param ndim - Dimensionality
   * @returns Mahalanobis distance
   */
  mahalanobis_distance(diff: Float32Array, packedL: Float32Array, ndim: number): number;

  /**
   * Fused nD→3D GSplat projection in a single pass: discrete-visibility gate →
   * continuous attenuation (marginal Cholesky + shifted Gaussian) → visibility
   * (`amplitude * attenuation >= minAmplitude`) → COMPACTED outputs. Replaces the
   * 6-call pipeline (`compute_gsplats_attenuation` + `extract_3d_positions` +
   * `compact_by_mask` ×2 + `extract_visible_cholesky_3d` +
   * `compact_attenuated_amplitudes`), eliminating ~5 passes and the repeated
   * large-array boundary copies. Bit-identical visible set + values.
   *
   * Colors must be pre-normalized to f32 (white-filled when absent) — the kernel
   * takes a single `Float32Array` because wasm-bindgen can't accept a typed-array
   * union. Outputs are sized for the `splatCount` worst case; slice each to the
   * returned visible count.
   *
   * @param positions - Splat centers [splatCount * ndim]
   * @param cholesky - Packed Cholesky factors [splatCount * packedSize]
   * @param amplitudes - Splat amplitudes [splatCount]
   * @param colors - Pre-normalized RGB or RGBA [splatCount * colorComponents]
   * @param discreteVisibility - Precomputed discrete-dim gate [splatCount]
   * @param slicePosition - Current slice [ndim]
   * @param continuousHiddenDims - Sorted continuous hidden dims [numContinuous]
   * @param displayDims - Display dims in requested order [2 or 3]
   * @param ndim - Total dimensionality
   * @param splatCount - Number of splats
   * @param colorComponents - Color channel count (3 = RGB, 4 = RGBA); strides every color read/write
   * @param minAmplitude - Visibility threshold
   * @param truncate - Truncation radius in sigmas
   * @param outCenters3d - Output visible centers [splatCount * 3] worst-case
   * @param outCholesky3d - Output visible 3D Cholesky [splatCount * 6] worst-case
   * @param outAmplitudes - Output visible attenuated amplitudes [splatCount] worst-case
   * @param outColors - Output visible colors [splatCount * colorComponents] worst-case
   * @returns Number of visible splats written
   */
  project_gsplats_nd_to_3d(
    positions: Float32Array,
    cholesky: Float32Array,
    amplitudes: Float32Array,
    colors: Float32Array,
    discreteVisibility: Uint8Array,
    slicePosition: Float32Array,
    continuousHiddenDims: Uint32Array,
    displayDims: Uint32Array,
    ndim: number,
    splatCount: number,
    colorComponents: number,
    minAmplitude: number,
    truncate: number,
    outCenters3d: Float32Array,
    outCholesky3d: Float32Array,
    outAmplitudes: Float32Array,
    outColors: Float32Array
  ): number;

  // ============================================================================
  // LINES CLIPPING FUNCTIONS - nD segment clipping and interpolation
  // ============================================================================

  /**
   * Clip a single segment to the nD slice and return interpolation parameters.
   *
   * @param p1 - Start vertex position [ndim]
   * @param p2 - End vertex position [ndim]
   * @param slicePosition - Current slice position [ndim]
   * @param tolerance - Per-dimension tolerance [ndim]
   * @param displayDims - Which dimensions to display [numDisplayDims]
   * @param ndim - Number of dimensions
   * @returns Float32Array [visible, t1, t2] where visible is 1.0 or 0.0
   */
  clip_segment_single(
    p1: Float32Array,
    p2: Float32Array,
    slicePosition: Float32Array,
    tolerance: Float32Array,
    displayDims: Uint32Array,
    ndim: number
  ): Float32Array;

  /**
   * Batch clip all segments and output visibility mask and interpolation parameters.
   *
   * @param positions - Vertex positions [numVertices * ndim]
   * @param segments - Segment indices [numSegments * 2]
   * @param slicePosition - Current slice position [ndim]
   * @param tolerance - Per-dimension tolerance [ndim]
   * @param displayDims - Which dimensions to display [numDisplayDims]
   * @param ndim - Number of dimensions
   * @param numSegments - Number of segments
   * @param outputVisibility - Output visibility mask [numSegments]
   * @param outputT1 - Output t1 parameters [numSegments]
   * @param outputT2 - Output t2 parameters [numSegments]
   * @returns Number of visible segments
   */
  clip_segments_batch(
    positions: Float32Array,
    segments: Uint32Array,
    slicePosition: Float32Array,
    tolerance: Float32Array,
    displayDims: Uint32Array,
    ndim: number,
    numSegments: number,
    outputVisibility: Uint8Array,
    outputT1: Float32Array,
    outputT2: Float32Array
  ): number;

  /**
   * Interpolate clipped positions to 3D display space.
   *
   * @param positions - Vertex positions [numVertices * ndim]
   * @param segments - Segment indices [numSegments * 2]
   * @param visibility - Visibility mask [numSegments]
   * @param t1Params - Start interpolation parameters [numSegments]
   * @param t2Params - End interpolation parameters [numSegments]
   * @param displayDims - Which dimensions to display [3]
   * @param ndim - Number of dimensions
   * @param numSegments - Total number of segments
   * @param outputStart - Output start positions [visibleCount * 3]
   * @param outputEnd - Output end positions [visibleCount * 3]
   * @returns Number of visible segments written
   */
  interpolate_clipped_positions(
    positions: Float32Array,
    segments: Uint32Array,
    visibility: Uint8Array,
    t1Params: Float32Array,
    t2Params: Float32Array,
    displayDims: Uint32Array,
    ndim: number,
    numSegments: number,
    outputStart: Float32Array,
    outputEnd: Float32Array
  ): number;

  /**
   * Batch interpolate scalar attributes for visible segments.
   *
   * @param values - Per-vertex attribute values [numVertices]
   * @param segments - Segment indices [numSegments * 2]
   * @param visibility - Visibility mask [numSegments]
   * @param t1Params - Start interpolation parameters [numSegments]
   * @param t2Params - End interpolation parameters [numSegments]
   * @param numSegments - Total number of segments
   * @param outputStart - Output interpolated start values [visibleCount]
   * @param outputEnd - Output interpolated end values [visibleCount]
   * @returns Number of visible segments written
   */
  interpolate_scalars_batch(
    values: Float32Array,
    segments: Uint32Array,
    visibility: Uint8Array,
    t1Params: Float32Array,
    t2Params: Float32Array,
    numSegments: number,
    outputStart: Float32Array,
    outputEnd: Float32Array
  ): number;

  /**
   * Batch interpolate RGB color attributes for visible segments.
   *
   * @param colors - Per-vertex RGB colors [numVertices * 3]
   * @param segments - Segment indices [numSegments * 2]
   * @param visibility - Visibility mask [numSegments]
   * @param t1Params - Start interpolation parameters [numSegments]
   * @param t2Params - End interpolation parameters [numSegments]
   * @param numSegments - Total number of segments
   * @param outputStart - Output interpolated start colors [visibleCount * 3]
   * @param outputEnd - Output interpolated end colors [visibleCount * 3]
   * @returns Number of visible segments written
   */
  interpolate_colors_batch(
    colors: Float32Array,
    segments: Uint32Array,
    visibility: Uint8Array,
    t1Params: Float32Array,
    t2Params: Float32Array,
    numSegments: number,
    outputStart: Float32Array,
    outputEnd: Float32Array
  ): number;

  /**
   * Calculate 3D segment lengths for visible segments.
   *
   * @param startPositions - Clipped start positions [visibleCount * 3]
   * @param endPositions - Clipped end positions [visibleCount * 3]
   * @param visibleCount - Number of visible segments
   * @param output - Output segment lengths [visibleCount]
   */
  calculate_segment_lengths(
    startPositions: Float32Array,
    endPositions: Float32Array,
    visibleCount: number,
    output: Float32Array
  ): void;

  /**
   * Per-endpoint cap suppression in [0, 1] (drives the shader cap factor).
   *
   * `1.0` suppresses the shader's endpoint dimming (clipped endpoint, or a
   * straight-through interior joint where the neighbouring quad tiles rather
   * than overlaps); `0.0` keeps it (free polyline end, branch point, or a
   * bend sharp enough that the quads genuinely overlap). See
   * `wasm/rust/src/lines_clipping.rs` for the full derivation.
   *
   * @param segments - Vertex index pairs [numSegments * 2]
   * @param visibility - Visibility mask [numSegments]
   * @param t1Params - Start interpolation parameters [numSegments]
   * @param t2Params - End interpolation parameters [numSegments]
   * @param numSegments - Total number of segments
   * @param numVertices - Total number of source vertices
   * @param startPositions - Clipped start positions [visibleCount * 3]
   * @param endPositions - Clipped end positions [visibleCount * 3]
   * @param outputStart - Output start suppression [visibleCount]
   * @param outputEnd - Output end suppression [visibleCount]
   * @returns Number of visible segments written
   */
  compute_cap_suppression(
    segments: Uint32Array,
    visibility: Uint8Array,
    t1Params: Float32Array,
    t2Params: Float32Array,
    numSegments: number,
    numVertices: number,
    startPositions: Float32Array,
    endPositions: Float32Array,
    outputStart: Float32Array,
    outputEnd: Float32Array
  ): number;

  // ============================================================================
  // MESH CULLING - whole-triangle nD visibility for indexed surfaces
  // ============================================================================

  /**
   * Compute per-vertex nD slab membership for a mesh.
   *
   * For each non-displayed ("hidden") dimension `d`, with
   * `sliceMin = slicePosition[d] - tolerance[d]` and
   * `sliceMax = slicePosition[d] + tolerance[d]`, a vertex is in iff
   * `v[d] >= sliceMin && v[d] <= sliceMax` for every such `d`. This is the
   * `p1_in` branch of {@link WasmModule.clip_segment_single} applied per vertex.
   *
   * A `NaN` or `±Inf` coordinate on any hidden dimension makes the vertex
   * invisible (the #806 rule, shared with the lines backends).
   *
   * Unlike lines, nothing is clipped or interpolated — see
   * {@link WasmModule.compact_visible_faces} for the whole-triangle rule and
   * `docs/specs/MESH_NODE_SPEC.md` §5 for why v1 does not clip.
   *
   * The WASM implementation calls `validate_ndim` and therefore panics above 16
   * dimensions; `pickBackend(ctx, ndim)` routes `ndim > 16` to the uncapped
   * TypeScript backend.
   *
   * @param positions - Vertex positions [numVertices * ndim]
   * @param slicePosition - Current slice position [ndim]
   * @param tolerance - Per-dimension tolerance [ndim]
   * @param displayDims - Which dimensions are displayed [numDisplayDims]
   * @param ndim - Number of dimensions
   * @param numVertices - Number of vertices
   * @param output - Output visibility mask [numVertices] (1 = in, 0 = out)
   * @returns Number of visible vertices
   */
  mesh_vertex_visibility_mask(
    positions: Float32Array,
    slicePosition: Float32Array,
    tolerance: Float32Array,
    displayDims: Uint32Array,
    ndim: number,
    numVertices: number,
    output: Uint8Array
  ): number;

  /**
   * Compact `faces` to those whose three vertices are all visible.
   *
   * Writes ORIGINAL (un-remapped) vertex indices, so a slice change rebuilds
   * only the index buffer while the vertex attribute buffers stay uploaded in
   * full. The authored per-face index order is preserved, so this is
   * winding-agnostic.
   *
   * A face index `>= vertexMask.length` drops the whole face rather than reading
   * out of bounds — the indices are store-supplied, and the two backends fail
   * differently without the guard (a Rust out-of-bounds read traps with an
   * uncatchable `RuntimeError: unreachable`; the TS read yields `undefined`).
   *
   * @param faces - Triangle vertex indices [numFaces * 3]
   * @param vertexMask - Per-vertex visibility from
   *   {@link WasmModule.mesh_vertex_visibility_mask}; its LENGTH defines the
   *   valid vertex range, so pass a view sized exactly `numVertices`
   * @param numFaces - Number of triangles
   * @param output - Output indices [numFaces * 3] worst case
   * @returns Number of visible faces written (slice `output` to 3x this)
   */
  compact_visible_faces(
    faces: Uint32Array,
    vertexMask: Uint8Array,
    numFaces: number,
    output: Uint32Array
  ): number;
}
