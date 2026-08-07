/**
 * Which geometry types support which viewer features.
 *
 * The geometry *vocabulary* is single-sourced from the format contract
 * ({@link GEOMETRY_TYPES} / {@link GeometryTypeName}, generated from
 * `format-contract/contract.yaml`). But plenty of viewer code does not want the
 * whole vocabulary — it wants "the types that can be a `kind=lod` child", or
 * "the types stored in the instanced-quad element texture". Those are *subsets*,
 * and before this module each one was an inline `x === 'points' || x === 'lines'
 * || x === 'gsplats'` chain or a hand-written `Set`.
 *
 * Inline chains are the wrong shape for two reasons:
 *
 *  1. They all *look* identical, so a sweep that widens the vocabulary widens
 *     the capability sets with it — silently enabling a feature for a type that
 *     does not support it. (This has already been caught twice in review.)
 *  2. They are invisible to the compiler, so adding a geometry type to the
 *     contract produces no error and no prompt to classify it.
 *
 * {@link GEOMETRY_CAPABILITIES} fixes both: it is a
 * `Record<GeometryTypeName, GeometryCapabilities>`, so **adding a geometry type
 * to the contract is a compile error here** until its capabilities are declared
 * — and every runtime consumer then behaves correctly without further edits.
 *
 * Type *annotations* cannot read this table (a union cannot be derived from a
 * value without more machinery than it is worth), so the handful of interface
 * fields that spell a subset out by hand carry a comment pointing here instead.
 *
 * @module types/geometry-capabilities
 */

import { GEOMETRY_TYPES, type GeometryTypeName } from './format-contract';
import type { BlendingMode } from './blending';

/** Membership set for {@link isGeometryType}; `Set` so lookup is not a scan. */
const GEOMETRY_TYPE_SET: ReadonlySet<string> = new Set(GEOMETRY_TYPES);

/**
 * Whether `value` is one of the contract's leaf geometry types.
 *
 * Use this — never a hand-written `=== 'points' || ...` chain — wherever the
 * question is "is this node a geometry leaf?". Accepts `unknown` because the
 * callers read it out of `userData` / zarr attrs, where it is untyped.
 */
export function isGeometryType(value: unknown): value is GeometryTypeName {
  return typeof value === 'string' && GEOMETRY_TYPE_SET.has(value);
}

/**
 * The viewer features a geometry type may participate in.
 *
 * Every flag is a capability question that some call site asks at runtime.
 * A `false` means there is no viewer path for that pairing today, so admitting
 * the type would take a code path that cannot represent it. *Why* there is no
 * path differs from flag to flag — see the per-flag notes on the `mesh` row.
 */
export interface GeometryCapabilities {
  /**
   * May appear as a `kind=lod` group's `display_type`, i.e. can be a level in a
   * substitutive LOD ladder. Implies the leaf stamps `loadedViewVersion` so the
   * LOD registry can judge per-slice freshness.
   */
  readonly lod: boolean;

  /**
   * May appear as a `kind=partition` group's `display_type`, i.e. can be split
   * into spatially-culled parts. Mirrors the Python-side allowlist in
   * `core/node/specialized_groups.py`.
   */
  readonly partition: boolean;

  /**
   * Rendered through the instanced-quad path: per-element attributes live in a
   * GPU element texture drawn from the buffer pool, and the data monitor tracks
   * a per-type accumulator. A type rendered from a plain `BufferGeometry` is
   * not pooled. The compile-time keying of the monitor's per-type records is
   * `POOLED_GEOMETRY_TYPES` in `types/data-monitor-types.ts` — the test suite
   * pins the two to agree.
   */
  readonly pooled: boolean;

  /**
   * Registers per-element centers with the depth-sort coordinator, so switching
   * blending mode has to start or stop sorting for the layer.
   */
  readonly depthSortable: boolean;
}

/**
 * Capability matrix for the geometry vocabulary.
 *
 * **Adding a geometry type to `format-contract/contract.yaml` will fail to
 * compile here.** That is deliberate: the new type's support for LOD,
 * partitioning, pooled storage and depth sorting is a decision that must be
 * made explicitly, not inherited by accident from a widened literal.
 *
 * Points / Lines / GSplats are uniformly capable — they are all soft, emissive,
 * per-element primitives drawn as instanced quads. `mesh` is the row that proves
 * the table earns its keep: its flags are MIXED, and each is false (or true) for
 * its own reason. `pooled` is architectural — a mesh is an indexed
 * `BufferGeometry`, so there is nothing to pool. `lod` is not: substitutive
 * levels need only a decimator on the writer side and a widened
 * `LODGroupMetadata.display_type` here. And `partition` was false only for want
 * of the bookkeeping to duplicate vertices across a cut, which is exactly why it
 * is now true. The row comment below gives each flag its own reason; do not read
 * "impossible" into a column that means "not yet".
 *
 * Readonly + frozen: every predicate reads this object live, so a mutation
 * would globally flip a capability for the whole session. (The record is
 * frozen; the rows are readonly at the type level only, so the wiring test
 * can flip one flag at a time through a deliberate cast.)
 */
export const GEOMETRY_CAPABILITIES: Readonly<Record<GeometryTypeName, GeometryCapabilities>> =
  Object.freeze({
    points: { lod: true, partition: true, pooled: true, depthSortable: true },
    lines: { lod: true, partition: true, pooled: true, depthSortable: true },
    gsplats: { lod: true, partition: true, pooled: true, depthSortable: true },
    // Mesh: a connected surface, not a set of independent elements — see
    // docs/specs/MESH_NODE_SPEC.md §2.1 and §9.
    //   lod           the ADDITIVE prefix ladder reduces independent elements
    //                 and cannot apply (a prefix of an index buffer is a holed
    //                 surface, not a coarser one); SUBSTITUTIVE levels assume
    //                 nothing of the sort and are missing only a producer, the
    //                 mesh analogue of which is QEM decimation.
    //   partition     TRUE. A BSP cut runs between faces, never through one,
    //                 and each part gathers + renumbers the vertices its own
    //                 faces use (luxar.mesh.split on the writer side). Vertices
    //                 on the cut are duplicated, which is what makes each part
    //                 independently drawable; the seam stays invisible because
    //                 both copies carry identical position AND normal.
    //   pooled        mesh renders as an indexed BufferGeometry, NOT through the
    //                 instanced-quad element-texture stack.
    //   depthSortable sorting a mesh means permuting an index buffer, not an
    //                 instance list, so it registers no per-element centers.
    // Flip a flag here when the corresponding path lands — never at a call site.
    mesh: { lod: false, partition: true, pooled: false, depthSortable: false },
  });

/** Look up one capability of an untyped node-type value. Non-types are `false`. */
function hasCapability(value: unknown, capability: keyof GeometryCapabilities): boolean {
  return isGeometryType(value) && GEOMETRY_CAPABILITIES[value][capability];
}

/** Whether `nodeType` can be a level of a substitutive LOD ladder. */
export function supportsLod(nodeType: unknown): boolean {
  return hasCapability(nodeType, 'lod');
}

/** Whether `nodeType` can be decomposed into `kind=partition` parts. */
export function supportsPartition(nodeType: unknown): boolean {
  return hasCapability(nodeType, 'partition');
}

/** Whether `nodeType` stores its elements in a pooled GPU element texture. */
export function isPooledGeometry(nodeType: unknown): boolean {
  return hasCapability(nodeType, 'pooled');
}

/** Whether `nodeType` registers element centers with the depth sorter. */
export function isDepthSortable(nodeType: unknown): boolean {
  return hasCapability(nodeType, 'depthSortable');
}

/**
 * Per-type default blending mode — the mode a leaf renders with when NO level
 * of its scene-graph ancestry sets one. The three emissive primitives sum light
 * (`additive`); `mesh` is the one shaded surface type and defaults to `opaque`
 * (docs/specs/MESH_NODE_SPEC.md §6.3). Kept as its own table rather than a
 * `GeometryCapabilities` column because that interface is boolean feature-flags
 * and this is a value; the exhaustive `Record<GeometryTypeName, …>` still forces
 * a decision when a geometry type is added. The material factories
 * (`create-{points,lines,gsplats,mesh}-node.ts`) inline the same literal for
 * their own statically-known type; this helper serves the runtime-dispatch sites
 * (layers panel) where the type is a variable.
 */
const DEFAULT_BLENDING_MODE: Readonly<Record<GeometryTypeName, BlendingMode>> = Object.freeze({
  points: 'additive',
  lines: 'additive',
  gsplats: 'additive',
  mesh: 'opaque',
});

/**
 * The blending mode a node of `nodeType` renders with when its ancestry sets
 * none. Non-geometry node types (`group`/`scene`/`lod`/`partition`) fall back to
 * `'additive'`, matching the historical composed default.
 */
export function defaultBlendingMode(nodeType: unknown): BlendingMode {
  return isGeometryType(nodeType) ? DEFAULT_BLENDING_MODE[nodeType] : 'additive';
}
