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

import type * as THREE from 'three';
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
 * - `'none'` — no lighting at all: the base colour reaches the screen
 *   unmodulated, with the diffuse and specular terms suppressed. This is what
 *   the other three geometry types always do (they are purely emissive), and
 *   what a *data basemap* needs — a textured globe whose colours carry meaning
 *   must not be reshaded by a view-anchored key. Never a default, only ever
 *   explicit, since defaulting to it would un-light every existing mesh.
 *
 * The writer resolves this at write time and always stamps a concrete value
 * (`'smooth'` only when normals were supplied), so the viewer never has to
 * guess. An explicit authored value is stamped as given and is never rewritten
 * — an explicit `'smooth'` with no stored normals is legal and falls back, and
 * an explicit `'flat'` gives a faceted surface even when normals exist.
 */
export type MeshShading = 'smooth' | 'flat' | 'none';

/**
 * How a mesh's `texture` array is stored.
 *
 * `'raw'` is an `(H, W, C)` numeric array the decoder materializes directly —
 * the only arm that can carry HDR, since no browser-native image codec stores
 * floats. PNG, WebP, and JPEG are 1-D `uint8` codec bytes decoded with
 * `createImageBitmap`, exactly as `image_label_bytes` already does for hover
 * thumbnails. KTX2 is also opaque bytes, but delegates to THREE's Basis
 * transcoder and stays GPU-compressed after upload.
 *
 * A closed vocabulary on purpose: an unrecognised value must be a rejection and
 * not a fall-through to "probably an image", because the decode path and the
 * byte budget differ between the raw, bitmap-codec, and GPU-compressed arms.
 */
export type MeshTextureEncoding = 'raw' | 'png' | 'webp' | 'jpeg' | 'ktx2';

/**
 * Colour space the texture's values are in.
 *
 * `'srgb'` for a photographic basemap (every JPEG/PNG off the shelf); `'linear'`
 * for values that are already linear-light, which is what Luxar's own authored
 * colours are. Getting this wrong gives a washed-out or over-dark surface that
 * still looks plausible, so it is stored explicitly rather than guessed from the
 * encoding.
 */
export type MeshTextureColorSpace = 'srgb' | 'linear';

/** Texture magnification/minification filter. `'linear'` also enables mipmaps. */
export type MeshTextureFilter = 'linear' | 'nearest';

/** Texture wrap mode, applied per-axis by the upload. */
export type MeshTextureWrap = 'repeat' | 'clamp';

/**
 * The two mesh material families. `'luxar'` is the house shader (spec §6.2) and
 * what an absent attr means; `'physical'` opts into three's physically based
 * material (`MESH_PHYSICAL_MATERIALS_SPEC.md`).
 */
export type MeshMaterialKind = 'luxar' | 'physical';

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

  /**
   * Whether a per-vertex `uvs` array is present.
   *
   * Paired with {@link has_texture}: the writer refuses either alone, because
   * each is inert without the other (UVs index into nothing; a texture with no
   * mapping samples one arbitrary texel across every triangle).
   */
  has_uvs: boolean;

  /** Whether a `texture` array is present */
  has_texture: boolean;

  /** How `texture` is stored. Present iff {@link has_texture}. */
  texture_encoding?: MeshTextureEncoding;

  /**
   * Declared texture width in pixels. Present iff {@link has_texture}.
   *
   * These three dimension attrs are **load-bearing, not descriptive**. For an
   * encoded texture they are the only thing that bounds the decode before any
   * bytes are fetched — a 200 KB JPEG can declare 30000x30000 and expand to
   * 3.6 GB — so the preflight charges the decoded surface from these numbers and
   * Stage 2 re-checks the decoded bitmap against them. See `data/mesh/preflight.ts`.
   */
  texture_width?: number;

  /** Declared texture height in pixels. Present iff {@link has_texture}. */
  texture_height?: number;

  /** Channels per texel: 1, 3 or 4. Present iff {@link has_texture}. */
  texture_channels?: number;

  /** Colour space of the texture's values. Present iff {@link has_texture}. */
  texture_color_space?: MeshTextureColorSpace;

  /**
   * `[min, max]` of an HDR texture's RGB values, stamped only when the raw
   * payload carried a value above 1.0.
   *
   * The texture peer of {@link color_data_range}, computed over RGB only (alpha
   * excluded) so the viewer's window derivation reads one shape whatever the
   * source.
   */
  texture_data_range?: [number, number];

  /** Sampling filter; viewer-defaulted to `'linear'` when absent. */
  texture_filter?: MeshTextureFilter;

  /**
   * Wrap mode; viewer-defaulted when absent to repeat-in-u / clamp-in-v, which
   * is what an equirectangular basemap needs — tiling across the dateline seam
   * without bleeding the north pole into the south.
   */
  texture_wrap?: MeshTextureWrap;

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

  /** Whether per-vertex stable string keys exist (CSR-encoded, for element actions) */
  has_keys?: boolean;

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
  /**
   * Offset-key shade floor in `[0, 1]` — what a face-away fragment keeps (§6.2).
   * `1.0` removes the diffuse gradient; set `specular = 0.0` as well for a fully
   * emissive look.
   *
   * Optional and viewer-defaulted: the writer never stamps it, so it is present only
   * when an author passed it through `add_mesh(**attrs)`. Authoring support and the
   * material read landed together so accepted values affect the rendered mesh.
   */
  ambient?: number;

  /** Wrapped-diffuse exponent (§6.2). Clamped positive — `pow(0, 0)` is undefined GLSL. */
  shade_exponent?: number;

  /** Additive specular strength in `[0, 1]` (§6.2). */
  specular?: number;

  /** Specular highlight exponent (> 0) (§6.2). */
  shininess?: number;

  /** `opaque`-mode alpha cutout threshold in `[0, 1]` (§6.2). */
  alpha_cutoff?: number;

  /**
   * Which material family renders this mesh
   * (`docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md` §3.1).
   *
   * Absent or `'luxar'`: the house shader (§6.2). `'physical'`: three's own
   * physically based material, lit by the scene environment the viewer builds
   * lazily on the first such mesh. Written only when authored, so every
   * pre-existing store reads as the house shader.
   */
  material?: MeshMaterialKind;

  /** Physical: microfacet roughness in `[0, 1]`; three's default `1`. */
  roughness?: number;

  /** Physical: metalness in `[0, 1]`; three's default `0`. */
  metalness?: number;

  /** Physical: clearcoat layer strength in `[0, 1]`; three's default `0`. */
  clearcoat?: number;

  /** Physical: clearcoat roughness in `[0, 1]`; three's default `0`. */
  clearcoat_roughness?: number;

  /** Physical: thin-film iridescence strength in `[0, 1]`; three's default `0`. */
  iridescence?: number;

  /** Physical: sheen strength in `[0, 1]`; three's default `0`. */
  sheen?: number;

  /**
   * Physical: sheen tint as `#rrggbb`. Viewer-defaulted to WHITE rather than
   * three's black, because a black sheen is a no-op and `sheen` alone would then
   * render nothing.
   */
  sheen_color?: string;

  /**
   * Physical (glass, spec §3.4): share of light transmitted in `[0, 1]`; three's
   * default `0`. Above zero the mesh composites as translucent and refracts the
   * background and other meshes — NOT points, lines or splats, which three's
   * transmission pass never sees, unless {@link refract_data} opts it in.
   */
  transmission?: number;

  /** Physical: index of refraction in `[1, 2.333]`; three's default `1.5`. */
  ior?: number;

  /** Physical: refraction volume thickness in scene units, `>= 0`; three's default `0`. */
  thickness?: number;

  /**
   * Physical: `#rrggbb` tint reached after `attenuation_distance` through the volume
   * (Beer–Lambert); three's default white = no tint.
   */
  attenuation_color?: string;

  /** Physical: attenuation length in scene units, `> 0`; absent = `Infinity` (none). */
  attenuation_distance?: number;

  /** Physical: chromatic dispersion strength, `>= 0`; three's default `0`. */
  dispersion?: number;

  /**
   * Physical (glass, spec §3.4 Phase 3): draw this glass AFTER the emissive data so
   * it refracts the points, lines and splats behind it. Absent = false (Phase 2
   * glass-first). Authoring refuses it without `transmission > 0`. Data in front of
   * the glass stays crisp: the refraction split partitions every data fragment by depth
   * against the glass's front surface (`materials/_shared/glass-partition.ts`).
   */
  refract_data?: boolean;

  /**
   * Half-width, IN CELLS, of the nD membership slab a CONTINUOUS hidden dimension
   * is culled against (§5.2.1): a vertex is inside when it is within
   * `slab_tolerance` cells of the slice. Strictly positive; defaults to one cell.
   *
   * The only authored control mesh has over its nD approximation, and the only
   * per-node input to `computeTolerance('mesh', …)`. Mesh culls whole triangles
   * against the slab — a triangle draws only when all three of its vertices are
   * inside — so on a continuous hidden axis it shows "the surface near this
   * slice" as a thick slab, never a planar cut (§5.3). This says how far from
   * the slice — the slab spans twice that.
   *
   * Mesh is the only geometry type whose continuous arm is tunable, and the
   * reason is that it has nothing to measure: the other three derive their
   * tolerance from a per-element extent (`radii` / `widths` / the truncated
   * `sigma`) that a mesh vertex does not have, so the slab thickness is invented
   * rather than read off the data.
   *
   * No effect on a DISCRETE hidden dimension (time, channel — the dominant real
   * case), which takes the half-cell membership rule instead.
   */
  slab_tolerance?: number;

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
 * A decoded texture, in whichever form its encoding produced.
 *
 * A discriminated union rather than one struct with optional fields, because the
 * three arms upload through genuinely different THREE.js classes (`DataTexture`,
 * `Texture` over an `ImageBitmap`, or `CompressedTexture`) and an exhaustive
 * branch on `kind` prevents a future arm from becoming a silently untextured mesh.
 *
 * `width`/`height`/`channels` are the *verified* dimensions, not the declared
 * ones: the loader compares what it decoded against
 * {@link MeshMetadata.texture_width} and friends and rejects a mismatch, so by
 * the time a payload exists these two agree. Carrying them here rather than
 * re-reading the attrs keeps the upload from having to trust the attrs again.
 */
export type MeshTextureData =
  | {
      kind: 'raw';
      /**
       * `height * width * channels` texels, row-major.
       *
       * `Float32Array` for an HDR or otherwise float-encoded texture — note the
       * decoder always widens `geolog_perchannel_u16` back to float32, so the
       * quantized-HDR case arrives here as floats too. `Uint8Array`/`Uint16Array`
       * are kept native, exactly as per-vertex colours are, since the GPU
       * normalizes them to `[0, 1]` for free.
       */
      pixels: MeshColorArray;
      width: number;
      height: number;
      channels: number;
    }
  | {
      kind: 'bitmap';
      /**
       * A decoded `ImageBitmap`, always 4-channel 8-bit regardless of what the
       * source codec stored — which is why the byte budget charges `w * h * 4`
       * for this arm no matter the declared `texture_channels`.
       */
      bitmap: ImageBitmap;
      width: number;
      height: number;
      channels: number;
    }
  | {
      kind: 'compressed';
      texture: THREE.CompressedTexture;
      width: number;
      height: number;
      channels: 3 | 4;
    };

/** Renderer-owned KTX2 transcode seam injected into the data loader. */
export type KTX2TextureDecoder = ((
  path: string,
  bytes: Uint8Array
) => Promise<THREE.CompressedTexture>) & { dispose: () => void };

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

  /**
   * Per-vertex texture coordinates (`vertexCount * 2`), or `undefined`.
   *
   * Deliberately NOT clamped to `[0, 1]`: values outside it are how a texture
   * tiles under `texture_wrap: 'repeat'`, so clamping would break the feature it
   * looks like it protects. Non-finite values ARE refused, on both sides — a NaN
   * UV pulls its whole triangle into an undefined sample.
   */
  uvs?: Float32Array;

  /** The decoded texture, or `undefined` when the node has none */
  texture?: MeshTextureData;

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
    signal?: AbortSignal,
    residencyAllowanceBytes?: number
  ): Promise<LoadedMeshData>;

  /** Clean up resources */
  dispose(): void;

  /**
   * LoaderMonitor surface (optional, for the data-loading-monitor UI).
   * Mirrors the surface the sibling loaders expose — implementations that
   * don't track metrics may omit these. Both shipped implementations
   * (`MeshWholeNodeLoader`, `MeshProgressiveLoader`) provide all four:
   * `connectLoaderToMonitor` duck-types the complete set, so a partial
   * implementation is silently skipped rather than partially reported.
   */
  addEventListener?(listener: MonitorEventListener): void;
  removeEventListener?(listener: MonitorEventListener): void;
  getMetrics?(): LoaderMetrics;
  getActiveQueries?(): QueryInfo[];

  /**
   * Report how many triangles the just-committed slice indexes, for
   * `LoaderMetrics.visibleElements`.
   *
   * Pushed IN (by `commit-mesh-geometry.ts`) rather than read out, because the
   * count is produced downstream of the loader: a mesh is resident in full and
   * projection decides which faces reach the index buffer. The three sibling
   * geometries set `visibleElements` inside their loaders instead, where the
   * range query that produced the visible set lives.
   *
   * Optional for the same reason as the four above — a metrics-free
   * implementation may omit it.
   */
  recordVisibleElements?(triangles: number): void;
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

  /**
   * Vertices this commit actually received (absent until the first commit).
   *
   * `mesh-geometry.ts`'s `position` attribute is capacity-sized for a reveal
   * ladder (#1521), so it can no longer answer this question — `debug-state.ts`
   * reads this stamp instead of `position.count`. `create-mesh-node.ts` never
   * seeds it on the placeholder, and `debug-state.ts`'s `??` fallback to
   * `position.count` depends on that absence to report the placeholder's true
   * (degenerate) count rather than a stale `0`.
   */
  committedVertexCount?: number;

  /**
   * Marks the material as already node-owned, so the layers panel and the LOD
   * cross-fade mutate it directly instead of cloning on first interaction.
   *
   * Always `true` for mesh, as for the other three types since their materials went
   * per node: a mesh material carries the node's own shading variant and epoch
   * `side`, so a shared one would let one node's appearance follow another's.
   */
  _layerMaterialCloned?: boolean;
}

/** Runtime guard for {@link MeshUserData}, mirroring `isPointsUserData`. */
export function isMeshUserData(userData: unknown): userData is MeshUserData {
  return (
    typeof userData === 'object' &&
    userData !== null &&
    (userData as { nodeType?: unknown }).nodeType === 'mesh'
  );
}
