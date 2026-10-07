/**
 * What each geometry type does — the one table to read.
 *
 * Behaviour written once per type drifts. Points and Lines lacked the
 * sorted-append hold GSplats had; a Lines ladder picked against the wrong
 * index space; a mesh ladder silently dropped its texture; a bare GSplats leaf
 * was refused a colormap its wrapper offered. Each was a cell of this table
 * that nobody had written down, so nothing could notice it was empty.
 *
 * Every row is one behaviour; every column is a contract geometry type. A
 * cell is either `'yes'` or an {@link Absent} that names WHY the type does not
 * do it, HOW that is enforced, and WHERE the decision is written. Keyed by
 * `GeometryTypeName`, so adding a type to the format contract is a compile
 * error here until every row has decided what the new type does.
 *
 * Each row names the test file that probes it (`probedIn`). That file calls
 * `defineBehaviourConformance` (`./define-behaviour-conformance.ts`) with the row's id and a per-type probe,
 * which runs every cell against real code: a `'yes'` cell asserts the
 * behaviour, an absent cell asserts its declared enforcement. So a cell that
 * claims `'yes'` but silently does nothing fails, and so does an absent cell
 * whose type quietly started doing the thing — the table cannot rot in either
 * direction. `tests/unit/conformance/geometry-behaviours.test.ts` checks the
 * table itself: every row probed exactly once, where it says; every spec
 * reference resolvable; the capability rows equal to `GEOMETRY_CAPABILITIES`.
 *
 * Deliberate asymmetries are as much a part of this table as the symmetries.
 * Mesh is a connected, shaded, indexed surface rather than a set of
 * independent emissive instances, so most of its `absent` cells trace back to
 * `MESH_NODE_SPEC.md` §2 (the storage layer does not transfer) and §9
 * (explicitly out of scope).
 *
 * @module tests/_conformance/geometry-behaviours
 */

import type { GeometryTypeName } from '../../types/format-contract';

/**
 * How an absent cell is kept honest.
 *
 * - `refuse`: the type cannot take the behaviour's input, and says so — a
 *   throw or a warning with a message the probe asserts.
 * - `hidden`: the UI or the scene graph does not offer it — a predicate the
 *   probe asserts returns false.
 * - `no-op`: the shared path runs and leaves this type's state unchanged — the
 *   probe asserts the observable the behaviour would have moved did not move.
 */
export type Enforcement = 'refuse' | 'hidden' | 'no-op';

/** A type that deliberately does not do a behaviour. */
export interface Absent {
  /** Why, in one sentence a contributor can check against the code. */
  readonly absent: string;
  readonly enforcedBy: Enforcement;
  /**
   * Where the decision is written: a doc or source path, with an optional
   * `§N.N` section (`docs/specs/MESH_NODE_SPEC.md §9`). `src/…` paths are the
   * viewer package's own tree; every other path is repo-relative. Checked to
   * resolve — the file must exist, and a section must be a heading in it.
   */
  readonly spec: string;
}

/** One cell of the matrix. */
export type Support = 'yes' | Absent;

/** One behaviour, across the four geometry types. */
export interface BehaviourRow {
  /** What the behaviour is, as the probe checks it. */
  readonly what: string;
  /** The test file (relative to `src/tests/unit/`) whose probe runs this row. */
  readonly probedIn: string;
  readonly cells: Readonly<Record<GeometryTypeName, Support>>;
}

const MESH_SPEC = 'docs/specs/MESH_NODE_SPEC.md';

/** Every behaviour, by id. The order is the order the rows read in. */
export const GEOMETRY_BEHAVIOURS = {
  // ── Committing geometry ──────────────────────────────────────────────────
  commitRequestsRender: {
    what:
      'A commit wakes the idle render loop and asks for a redraw; a commit to a hidden ' +
      'node only wakes it.',
    probedIn: 'data/scene-loader/commit-request-render.test.ts',
    cells: { points: 'yes', lines: 'yes', gsplats: 'yes', mesh: 'yes' },
  },
  sortedAppendHold: {
    what:
      'In a sorted mode, an append or a pool grow keeps drawing the previously sorted ' +
      'population until the grown ordering lands (planInstancedOrdering).',
    probedIn: 'conformance/sorted-append-hold.test.ts',
    cells: {
      points: 'yes',
      lines: 'yes',
      gsplats: 'yes',
      mesh: {
        absent:
          'A mesh’s draw order IS its index buffer, which a commit rewrites whole and the ' +
          'sort permutes atomically (triangle-ordering.ts): there is no aSortedIndex prefix ' +
          'to hold on.',
        enforcedBy: 'no-op',
        spec: `${MESH_SPEC} §2.1`,
      },
    },
  },
  depthSortRegistration: {
    what: 'A commit registers the node’s element centers with the depth-sort coordinator.',
    probedIn: 'conformance/depth-sort-registration.test.ts',
    cells: { points: 'yes', lines: 'yes', gsplats: 'yes', mesh: 'yes' },
  },
  monitorVisibleCount: {
    what:
      'The committed visible-element count reaches the data monitor’s per-type total ' +
      'and its headline card.',
    probedIn: 'conformance/monitor-visible-count.test.ts',
    cells: { points: 'yes', lines: 'yes', gsplats: 'yes', mesh: 'yes' },
  },

  // ── Loading, retry and slicing ───────────────────────────────────────────
  retryForwardsNoPreimage: {
    what: 'A failed-load retry forwards the derived view state’s noPreimage flag to the loader.',
    probedIn: 'data/scene-loader/lifecycle/retry.test.ts',
    cells: { points: 'yes', lines: 'yes', gsplats: 'yes', mesh: 'yes' },
  },
  failedRetryUnwindsLadderPass: {
    what: 'A retry that throws rolls the progressive loader back to its pass start.',
    probedIn: 'data/scene-loader/lifecycle/retry.test.ts',
    cells: { points: 'yes', lines: 'yes', gsplats: 'yes', mesh: 'yes' },
  },
  partialExtendTolerance: {
    what:
      'deriveNodeViewState widens the slice tolerance across dimensions the node only ' +
      'partially extends through.',
    probedIn: 'data/scene-loader/partial-extend-tolerance.test.ts',
    cells: {
      points: 'yes',
      lines: {
        absent:
          'A segment’s bounds already encode its non-displayed extent, so widening again ' +
          'would double-count it during clipping.',
        enforcedBy: 'no-op',
        spec: 'src/data/scene-loader/partial-extend-tolerance.ts',
      },
      gsplats: 'yes',
      mesh: 'yes',
    },
  },
  l0ChunkCache: {
    what:
      'Every zarr array the loader opens is read through the shared L0 decompressed-chunk ' +
      'cache, keyed under the node path.',
    probedIn: 'data/loaders/l0-cache-wiring.test.ts',
    cells: {
      points: 'yes',
      lines: 'yes',
      gsplats: 'yes',
      mesh: {
        absent:
          'A mesh loads whole and its loader keeps the decoded mesh for its lifetime, so ' +
          'there is no per-slice re-read for a chunk cache to save; the factory hands ' +
          'MeshWholeNodeLoader no L0 cache.',
        enforcedBy: 'no-op',
        spec: `${MESH_SPEC} §9`,
      },
    },
  },
  lodGroupChildActivation: {
    what:
      'A non-leaf child (a nested kind=partition / kind=lod wrapper) of a kind=lod group is ' +
      'deferred at init and loads its whole subtree on activation.',
    probedIn: 'data/scene-loader/nodes/load-lod-group-node.test.ts',
    cells: { points: 'yes', lines: 'yes', gsplats: 'yes', mesh: 'yes' },
  },

  // ── Structure: LOD, partition, ladders ───────────────────────────────────
  lodLevel: {
    what: 'May be the display_type of a kind=lod group (GEOMETRY_CAPABILITIES.lod).',
    probedIn: 'conformance/structural-membership.test.ts',
    cells: { points: 'yes', lines: 'yes', gsplats: 'yes', mesh: 'yes' },
  },
  partitionPart: {
    what: 'May be the display_type of a kind=partition group (GEOMETRY_CAPABILITIES.partition).',
    probedIn: 'conformance/structural-membership.test.ts',
    cells: { points: 'yes', lines: 'yes', gsplats: 'yes', mesh: 'yes' },
  },
  ladderPickingLayout: {
    what:
      'A labelled progressive ladder composes per-level picking maps into the parent’s ' +
      'union CSR (CSR-style level offsets, level label flags on).',
    probedIn: 'data/scene-loader/loaders/loader-factory.test.ts',
    cells: {
      points: 'yes',
      lines: 'yes',
      gsplats: {
        absent:
          'The gsplat authoring path has no labels channel at any level, so the factory ' +
          'clears every level’s label flags and passes no offsets.',
        enforcedBy: 'no-op',
        spec: 'src/data/scene-loader/loaders/loader-factory.ts',
      },
      mesh: {
        absent:
          'write_mesh_multi_lod refuses labels on a laddered mesh (a face partition ' +
          'duplicates boundary vertices, so the union index space is ill-defined); the ' +
          'factory clears the level flags too.',
        enforcedBy: 'no-op',
        spec: `${MESH_SPEC} §9.1`,
      },
    },
  },
  ladderTexture: {
    what: 'A progressive ladder whose levels declare a texture renders it.',
    probedIn: 'data/scene-loader/loaders/loader-factory.test.ts',
    cells: {
      points: {
        absent: 'Points have no texture channel; a level’s has_texture attr is ignored.',
        enforcedBy: 'no-op',
        spec: 'docs/guides/user/LUXAR_ZARR_FORMAT.md',
      },
      lines: {
        absent: 'Lines have no texture channel; a level’s has_texture attr is ignored.',
        enforcedBy: 'no-op',
        spec: 'docs/guides/user/LUXAR_ZARR_FORMAT.md',
      },
      gsplats: {
        absent: 'GSplats have no texture channel; a level’s has_texture attr is ignored.',
        enforcedBy: 'no-op',
        spec: 'docs/guides/user/LUXAR_ZARR_FORMAT.md',
      },
      mesh: {
        absent:
          'The reveal-ladder concat carries no uvs or texture (the writer refuses texture= ' +
          'with additive_lod=), so a hand-written textured ladder is refused before any ' +
          'level is fetched rather than rendered untextured.',
        enforcedBy: 'refuse',
        spec: `${MESH_SPEC} §9.1`,
      },
    },
  },

  // ── Appearance ───────────────────────────────────────────────────────────
  colormapSupport: {
    what:
      'The Layers panel offers a colormap exactly when the material can feed one ' +
      '(supportsScalarColormap), for a bare leaf and for its kind=lod wrapper alike.',
    probedIn: 'conformance/colormap-support.test.ts',
    cells: { points: 'yes', lines: 'yes', gsplats: 'yes', mesh: 'yes' },
  },
  densityGuard: {
    what:
      'Every visual and pick material (GLSL and TSL) carries uDensityDrop, and the ' +
      'projected-density guard thins an over-dense additive node through it.',
    probedIn: 'rendering/materials/density-drop.test.ts',
    cells: {
      points: 'yes',
      lines: 'yes',
      gsplats: 'yes',
      mesh: {
        absent:
          'A shaded surface tiles the screen rather than stacking emissive energy, so ' +
          'neither mesh material declares uDensityDrop and the guard skips it.',
        enforcedBy: 'no-op',
        spec: 'src/scene/density-guard.ts',
      },
    },
  },
  glassPartitionGuard: {
    what:
      'Every visual fragment shader runs the glass depth partition before its first ' +
      'discard, and every visual material (GLSL and TSL) declares the pair; no pick ' +
      'material does.',
    probedIn: 'rendering/materials/glass-partition.test.ts',
    cells: { points: 'yes', lines: 'yes', gsplats: 'yes', mesh: 'yes' },
  },

  // ── Picking ──────────────────────────────────────────────────────────────
  pickRegistration: {
    what:
      'The retro-registration pass gives an existing node a pick node with its own ' +
      'type’s pick material.',
    probedIn: 'rendering/node-factory/register-existing-scene-nodes.test.ts',
    cells: { points: 'yes', lines: 'yes', gsplats: 'yes', mesh: 'yes' },
  },
} as const satisfies Record<string, BehaviourRow>;

/** A row id. */
export type BehaviourId = keyof typeof GEOMETRY_BEHAVIOURS;

/** Every row id, in table order. */
export const BEHAVIOUR_IDS = Object.keys(GEOMETRY_BEHAVIOURS) as BehaviourId[];

/** A row, typed as the general shape. */
export function behaviourRow(id: BehaviourId): BehaviourRow {
  return GEOMETRY_BEHAVIOURS[id];
}

/** Whether `support` is an absent cell. */
export function isAbsent(support: Support): support is Absent {
  return support !== 'yes';
}

/**
 * Exports that one sibling of a parallel per-type module family has and the
 * others do not, on purpose. Keyed by family, then by the export's name with
 * the type token replaced by `<T>` (`concatenate<T>Data`). The meta-check in
 * `geometry-behaviours.test.ts` diffs each family's exports and fails on any
 * asymmetry not listed here — which is how "added to GSplats only" is caught.
 */
export interface ExportAsymmetry {
  /** The types whose module exports the symbol. */
  readonly presentIn: readonly GeometryTypeName[];
  readonly reason: string;
}

export const MODULE_FAMILY_ASYMMETRIES: Readonly<
  Record<string, Readonly<Record<string, ExportAsymmetry>>>
> = {
  'commit-<t>-geometry.ts': {
    '<T>CommitCtx': {
      presentIn: ['mesh'],
      reason:
        'The three instanced commits share GeometryCommitHost (commit-host.ts); the mesh ' +
        'commit also needs the current view version and is not pooled.',
    },
  },
  '<t>-progressive-loader.ts': {
    '<T>ProgressiveLoader.prefetchChunks': {
      presentIn: ['points', 'lines', 'gsplats'],
      reason: 'A mesh level loads whole; there are no spatial-index chunks to prefetch.',
    },
    '<T>ProgressiveLoader.prefetchChunkBoundary': {
      presentIn: ['points', 'lines', 'gsplats'],
      reason: 'A mesh level loads whole; there are no spatial-index chunks to prefetch.',
    },
    '<T>ProgressiveLoader.recordVisibleElements': {
      presentIn: ['mesh'],
      reason:
        'A mesh is resident in full, so its visible count is decided by the projection and ' +
        'pushed by the commit; the other ladders report it from their current payload.',
    },
    'concatenate<T>Data': {
      presentIn: ['gsplats', 'mesh'],
      reason:
        'Every ladder concatenates; gsplats and mesh export theirs because their own tests ' +
        'drive it directly, while points and lines keep it module-private.',
    },
  },
};
