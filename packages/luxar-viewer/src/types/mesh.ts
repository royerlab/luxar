/**
 * Mesh type definitions for luxar-viewer.
 *
 * These types mirror Python `luxar.core.Mesh` and enable type-safe handling of
 * triangle-surface data throughout the viewer. See
 * `docs/specs/MESH_NODE_SPEC.md`.
 *
 * Two things here differ from the Points/Lines/GSplats siblings, and both are
 * structural rather than incidental:
 *
 * 1. **No primary size scalar.** Points carry `radii`, Lines `widths`, GSplats
 *    `cholesky_factors` — a per-element extent that the loader must know about
 *    because it pads query bounds and drives an instanced quad. A triangle's
 *    extent comes from its own three vertices, so a mesh has nothing of the
 *    kind and adds zero padding to scene bounds (spec §2.2).
 * 2. **Topology, not ranges.** `LoadedPointsData`/`LoadedLinesData` describe a
 *    *subset* selected by a spatial-index range query. A mesh is whole-node
 *    resident (spec §7): {@link LoadedMeshData} is always the entire mesh, and
 *    what varies with the slice is only which faces are *indexed* — see
 *    `data/mesh/projection.ts`.
 *
 * @module types/mesh
 */

import type { BlendingMode } from './blending';
import type { ViewState } from '../data/data-loader-types';
import type { LoaderMetrics, MonitorEventListener, QueryInfo } from './data-monitor-types';
import type { UpdateSession } from '../profiling/update-profiler';

// ============================================================================
// Metadata Types (from zarr .zattrs)
// ============================================================================

/**
 * How a mesh's surface normals are obtained at render time.
 *
 * - `'smooth'` — use the stored per-vertex `normals` array, when it is present
 *   AND its `normal_dims` matches the displayed axes (spec §3.4). Interpolated
 *   across each triangle, so a welded surface reads as curved.
 * - `'flat'` — derive a per-fragment normal from screen-space derivatives of
 *   the interpolated position, giving a faceted surface. Also the fallback
 *   whenever `'smooth'` cannot be honoured.
 *
 * The writer resolves this at write time and always stamps a concrete value
 * (`'smooth'` only when normals were supplied), so the viewer never has to
 * guess. An explicit authored value is stamped as given and is never rewritten
 * — an explicit `'smooth'` with no stored normals is legal and falls back, and
 * an explicit `'flat'` gives a faceted surface even when normals exist.
 */
export type MeshShading = 'smooth' | 'flat';

/**
 * Mesh node metadata from zarr `.zattrs`.
 *
 * Field-for-field what `io/_compiler/geometry_writers/mesh.py` stamps, plus the
 * render attrs every geometry node shares.
 */
export interface MeshMetadata {
  /** Node type identifier */
  type: 'mesh';

  /** Total vertex count. Capped at `MAX_MESH_VERTICES` (spec §3.5, §6.5). */
  n_vertices: number;

  /** Total triangle count */
  n_faces: number;

  /** Vertex position dimensionality */
  ndim: number;

  /** Whether a per-vertex `normals` array is present */
  has_normals: boolean;

  /**
   * Which three dimension indices the 3-component stored normals describe.
   * Present iff `has_normals`.
   *
   * This is never implicit. For a `(t, x, y, z)` mesh the "first three
   * dimensions" are `(t, x, y)` — meaningless as a normal frame — which is the
   * bug this attr exists to prevent. It is also the authored *winding frame*:
   * `sorted(normal_dims)` is the axis triple whose column order the stored
   * face winding is front-facing in (spec §3.2, §5.4).
   */
  normal_dims?: number[];

  /** Whether a per-vertex `colors` array is present */
  has_colors: boolean;

  /** Whether per-vertex scalar values for colormap lookup are present */
  has_scalars: boolean;

  /** How to obtain surface normals; always stamped by the writer */
  shading: MeshShading;

  /**
   * Whether both faces of the surface are drawn (`THREE.DoubleSide`) rather
   * than only the front (`THREE.FrontSide`).
   *
   * Note the viewer may render double-sided even when this is `false`: when
   * projected winding cannot be decided against the authored frame, no index
   * post-pass can restore it, so the epoch falls back to `DoubleSide` with a
   * one-time notice (spec §5.4, §7).
   */
  double_sided: boolean;

  /**
   * Spatial ordering method. Always `'none'` in v1 — the loader is whole-node,
   * so there is no chunk index to skip. Stamped unconditionally so a reader
   * never has to distinguish "no ordering" from "attr missing".
   */
  ordering: 'none';

  /** Named colormap or `'custom'` (paired with a `colormap_lut` zarr array) */
  colormap?: string;

  /** `[min, max]` for normalising scalars before LUT lookup */
  scalar_data_range?: [number, number];

  /** Min/max of the color data, computed at encoding time */
  color_data_range?: [number, number];

  /** Whether per-vertex string labels exist (CSR-encoded, for hover tooltips) */
  has_labels?: boolean;

  /** Whether per-vertex image labels exist (CSR-encoded, for hover thumbnails) */
  has_image_labels?: boolean;

  /** 4x4 transform matrix (column-major for THREE.js) */
  transform?: number[];

  /** Opacity multiplier */
  opacity?: number;

  /** Absorption coefficient (volumetric blending; mesh refuses that mode, §6.3) */
  absorption?: number;

  /** Gamma correction */
  gamma?: number;

  /** Intensity (linear color multiplier / gain) */
  intensity?: number;

  /** Offset (additive brightness shift / black level subtraction) */
  offset?: number;

  /** Blending mode */
  blending_mode?: BlendingMode;

  /** Whether this node is exposed as a layer in the Layers panel */
  layer?: boolean;

  /**
   * Dimension names to extend visibility across — a mesh with `extend_to_all`
   * is visible regardless of slice position in those dimensions.
   */
  extend_to_all?: string[];
}

// ============================================================================
// Loaded Data Types
// ============================================================================

/**
 * Per-vertex color buffer, in whichever dtype the store used.
 *
 * Native dtype is deliberately preserved rather than widened to float32: the
 * GPU normalizes `uint8`/`uint16` to `[0, 1]` for free, at 4 bytes per RGBA
 * vertex instead of 12. Matches `LoadedLinesData.colors` /
 * `LoadedPointsData.colors`.
 */
export type MeshColorArray = Float32Array | Uint8Array | Uint16Array;

/**
 * A whole decoded mesh, before display-space projection.
 *
 * Unlike the range-query siblings this is *always* the complete mesh: the
 * loader is whole-node (spec §7). Slicing does not change what is loaded, only
 * which faces `data/mesh/projection.ts` writes into the index buffer.
 *
 * Vertices are still in full nD space here — `extract_3d_positions` projects
 * them to the displayed triple downstream.
 */
export interface LoadedMeshData {
  /** Vertex positions (`vertexCount * ndim`), flattened row-major */
  vertices: Float32Array;

  /**
   * Triangle vertex indices (`faceCount * 3`), indexing into `vertices`.
   *
   * Always `Uint32Array` here regardless of the store's dtype. Externally
   * produced stores legitimately use `int8`..`int64`, and the writer's own
   * `INDEX` encoder narrows to the smallest unsigned dtype that fits, so this
   * is a widening the loader owns. Every value has been range-checked in
   * `[0, vertexCount)` against the *source-typed* values, before coercion —
   * each side of the cast hides its own wrap-around (spec §3.5 Stage 2).
   */
  faces: Uint32Array;

  /**
   * Per-vertex normals (`vertexCount * 3`), or `null` when absent.
   *
   * Only meaningful in the frame named by {@link MeshMetadata.normal_dims}.
   */
  normals: Float32Array | null;

  /** Per-vertex colors (`vertexCount * colorComponents`), or `null` when absent */
  colors: MeshColorArray | null;

  /**
   * Channels per color entry: 3 (RGB) or 4 (RGBA, the alpha column being
   * per-vertex opacity). Undefined when `colors` is `null`.
   */
  colorComponents?: 3 | 4;

  /** Per-vertex scalars for colormap lookup (`vertexCount`), or `undefined` */
  scalars?: Float32Array;

  /** Number of vertices loaded */
  vertexCount: number;

  /** Number of triangles loaded */
  faceCount: number;

  /** Dimensionality for interpreting `vertices` */
  ndim: number;

  /**
   * Loader-owned scratch for the display-space projection, reused across epochs.
   *
   * The Mesh counterpart of the Points accumulator's reusable target buffers
   * (`points-spatial-index-loader.ts` hands `positions3D` to the projection rather
   * than letting it allocate). Mesh's home for it is here because `updateView`
   * returns this same object for the node's whole life and drops it on dispose, so
   * the buffer inherits exactly the right lifetime with no separate cache to
   * invalidate.
   *
   * Without it, `projectMeshTo3D` allocated a fresh `vertexCount * 3` buffer on every
   * slice move — so the geometry's array-identity check never matched and every
   * scrub copied and re-uploaded the whole vertex buffer and recomputed bounds,
   * defeating the "only the index changes on a pure slice move" design (#1245).
   *
   * `displayDimsKey` records which axis triple `position` currently holds, because
   * once the buffer is reused its identity can no longer signal a change. Optional
   * so a caller may still project without one (it then allocates, as before).
   */
  projection?: MeshProjectionTargetBuffers;
}

/**
 * Display-space AABB of a mesh projection, over the vertices the emitted index
 * actually references.
 *
 * The Mesh member of the `<Geometry>ProjectionBounds` family
 * (`LinesProjectionBounds`, `GSplatsProjectionBounds`): same `min`/`max` 3-tuples,
 * computed by the projection and consumed by `computeMeshBounds` to set the
 * geometry's box and sphere.
 *
 * It carries NO per-element extent term, and that absence is the point. Lines add
 * `maxWidth` and gsplats `maxRowNorm` because their elements are sprites whose
 * rendered footprint exceeds their center. A triangle's extent IS its vertices, so a
 * mesh needs no expansion — the same reason a mesh contributes zero padding to scene
 * bounds on the write side.
 *
 * Bounding the INDEXED vertices rather than the whole position buffer is what makes
 * this useful: under the no-compaction design `position` holds every vertex of the
 * whole nD mesh, including those whose triangles the slab cull removed, so a 4D
 * surface that translates over time would otherwise frame its entire trajectory
 * (#1252).
 */
export interface MeshProjectionBounds {
  /** AABB min corner over the indexed display-space vertices [x, y, z]. */
  min: [number, number, number];
  /** AABB max corner over the indexed display-space vertices [x, y, z]. */
  max: [number, number, number];
}

/**
 * Reusable projection output buffer plus the epoch it currently holds.
 *
 * Named after Points' `ProjectionTargetBuffers` (`data/points/projection.ts`), which
 * is the same idea: the loader owns the output memory and the projection writes
 * through it instead of allocating. Mesh carries one extra field its sibling does not
 * need — `displayDimsKey` — because reuse makes buffer identity useless as a
 * "did the displayed axes change" signal, and mesh must answer that to know whether
 * to re-extract at all.
 */
export interface MeshProjectionTargetBuffers {
  /** Display-space positions (`vertexCount * 3`), rewritten per `displayDims` epoch */
  position: Float32Array;
  /** `displayDims.join()` of what `position` holds; `null` before the first extract */
  displayDimsKey: string | null;
  /**
   * Per-vertex slab-membership mask (`vertexCount`), rewritten every cull.
   *
   * Unlike `position` this genuinely changes on every slice move, so there is nothing to
   * skip — but there is also no reason to reallocate it each time. One byte per vertex is
   * small per epoch and pure garbage at scrub rates.
   */
  mask: Uint8Array;
  /**
   * Worst-case face-compaction scratch (`faceCount * 3`), rewritten every cull.
   *
   * The largest per-epoch allocation the cull had: 12 bytes per face, so ~12 MB for a
   * 1M-face mesh, discarded and reallocated on every slice move. Sized for the worst case
   * because every face may survive; the kernel reports how many actually did.
   */
  faceScratch: Uint32Array;
  /**
   * Fast-path bounds cache, and the `displayDims` epoch they were measured in.
   *
   * ONLY the no-hidden-dims fast path may use this, and the distinction is the whole
   * reason it is a cache rather than a plain field. On that path nothing is culled, so
   * the emitted index is the authored `faces` verbatim and the box depends solely on
   * `position` — which changes only when `displayDims` does. On the CULL path the
   * visible set changes with every slice move, so its box must be recomputed every
   * epoch and this cache is never consulted.
   *
   * Worth having because the recompute is O(faces), not O(1): measured 5.3 ms at 1M
   * faces and 26 ms at 5M, per sweep. A 3D mesh sitting in a scene that has navigable
   * dimensions elsewhere gets swept on every scrub frame, so at 5M faces that alone
   * overruns a 60 Hz frame budget while computing a box that cannot have changed.
   */
  fastPathBounds?: MeshProjectionBounds | null;
  /** `displayDims.join()` that {@link fastPathBounds} was measured in; unset before the first */
  fastPathBoundsKey?: string | null;
}

// ============================================================================
// Loader Interface
// ============================================================================

/**
 * Data loader interface for Mesh nodes.
 *
 * Mirrors `DataLoader` / `LinesDataLoader` / `GSplatsDataLoader` so the
 * scene-loader machinery treats all four kinds uniformly — and so a
 * spatial-index implementation can be swapped in behind this interface later
 * with no caller change (spec §7).
 *
 * The `viewState` parameters are honoured but do not change *what is fetched*:
 * a mesh is whole-node resident, so both methods return the same complete
 * {@link LoadedMeshData} and the view state matters only downstream, where the
 * cull decides which faces are indexed.
 */
export interface MeshDataLoader {
  /** Load the mesh for the given view state */
  loadMesh(viewState: MeshViewState, session?: UpdateSession): Promise<LoadedMeshData>;

  /** Re-serve the mesh for a new view state (see `DataLoader.updateView` for `signal`) */
  updateView(
    viewState: MeshViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedMeshData>;

  /** Clean up resources */
  dispose(): void;

  /**
   * LoaderMonitor surface (optional, for the data-loading-monitor UI).
   * Mirrors the surface the sibling loaders expose — implementations that
   * don't track metrics may omit these.
   */
  addEventListener?(listener: MonitorEventListener): void;
  removeEventListener?(listener: MonitorEventListener): void;
  getMetrics?(): LoaderMetrics;
  getActiveQueries?(): QueryInfo[];
}

/**
 * View state for mesh loading.
 *
 * Identical shape to the points/lines/gsplats `ViewState` — kept as a named
 * alias for documentation, exactly as `LinesViewState` is.
 */
export type MeshViewState = ViewState;

/**
 * User data attached to the `THREE.Mesh` object for a Mesh node in the scene.
 *
 * Enables runtime type checking and provides access to the loader/metadata.
 * Dimension info is NOT stored here — it lives only at the Scene level.
 */
export interface MeshUserData {
  /** Node type identifier for runtime type checking */
  nodeType: 'mesh';

  /** Data loader instance */
  loader: MeshDataLoader;

  /** Zarr group attributes */
  attrs: MeshMetadata;

  /** Scene path of this node */
  path: string;

  /** Triangles the current slice indexes (0 until the first commit) */
  visibleTriangleCount: number;

  /** Vertices that passed the nD slab test; diagnostic, not a draw bound */
  visibleVertexCount: number;
}

/** Runtime guard for {@link MeshUserData}, mirroring `isPointsUserData`. */
export function isMeshUserData(userData: unknown): userData is MeshUserData {
  return (
    typeof userData === 'object' &&
    userData !== null &&
    (userData as { nodeType?: unknown }).nodeType === 'mesh'
  );
}
