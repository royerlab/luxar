/**
 * Partition gating: per-part frustum and slice culling of ``kind=partition``
 * groups, the targeted resync of parts that re-enter the frustum, and lazy
 * (B4) activation of deferred parts.
 *
 * A ``kind=partition`` group's parts are tested every frame against a frustum
 * padded by ``config.lod.partitionFrustumMargin``; a part outside it (or
 * outside the COMMITTED slice) is hidden and stamped
 * ``userData.partitionFrustumVisible = false``, which the scene loader's sweep
 * and refinement read to skip its loaders. Because a culled part misses slice
 * updates, its RE-ENTRY requests a resync of exactly that part's loaders
 * (``deps.requestReprocess(partPaths)``, coalesced per wrapper across frames
 * and gated on ``isUpdateInProgress``) — unless every leaf of the part is
 * already committed for the current view with a complete ladder. The resync
 * re-sweeps under the UNCHANGED view version: bumping it would read every lazy
 * fine level scene-wide as stale and drop all groups to coarse on camera
 * motion. A deferred part's activation rides the same resync.
 *
 * The registry owns one gate and drives it each frame: {@link
 * PartitionGate.beginFrame} gates every visible wrapper before the LOD pass (so
 * a lod_group nested in a part sees this frame's cull), {@link
 * PartitionGate.endFrame} re-gates the wrappers the LOD pass revealed and
 * flushes the pending resyncs.
 *
 * Liveness ({@link TickDemand}): the loop must keep ticking while a visible
 * resync waits for the loader — only when ``requestReprocess`` is wired, since
 * nothing else can serve it. A deferred part's activation request that no pass
 * reaches needs no ticks: one wake at its expiry
 * (``config.lod.lazyActivationRequestTimeoutMs``, backed off per unanswered
 * request — ``retry-wakes.ts``) asks again.
 *
 * @module scene/partition-gate
 */

import * as THREE from 'three';

import type { BoundingBox } from './scene-manager/clipping/bounds-math';
import type { ViewContext } from './view-context';
import type { LODGroupRegistryDeps } from './lod-group-registry';
import { log, Modules } from '../utils/log';
import { isEffectivelyVisible, isPartitionFrustumCulled } from '../utils/object-visibility';
import { subtreeSweepSettled, type SweepNode } from './lod-freshness';
import { subtreeDisplayProgress, type ProgressNode } from './lod-display-gate';
import { config } from '../config';
import { computeEntryWorldBox, type WorldBoxOptions } from './lod-selector-math';
import { perfCounters } from '../profiling/perf-counters';
import {
  partBoundsIntersectSlice,
  type PartitionSliceView,
} from '../data/scene-loader/view-state/partition-slice-gate';
import { NO_TICK, UNTIL_RESOLVED, type TickDemand } from './tick-demand';
import type { RetryWakes } from './retry-wakes';

/** Perf counter: deferred partition parts initialised on activation (B4). */
const S_PARTS_ACTIVATED = perfCounters.slot('partition.partsActivated');

export interface PartitionGroupChild {
  /** Stable node path for a part that may emit more than one scene object. */
  path: string;
  objects: THREE.Object3D[];
  positionBounds: { min: readonly number[]; max: readonly number[] };
  /**
   * Deferred part (B4): the part's subtree has NOT been loaded — `objects` is
   * the empty slot it will load into. The registry runs this once, inside a
   * loader pass (``LODGroupRegistry.activatePartitionParts``), when the
   * part is in the padded frustum and in the pass's slice. Absent ⇒ loaded.
   */
  activate?: () => Promise<void>;
  /**
   * `true` ⇒ never slice-gate this part (its bounds are not in the space of the
   * world slice, e.g. an `nd_transform` on its path). See `partition-slice-gate.ts`.
   */
  sliceExempt?: boolean;
  /** Names of the dimensions the part extends across (`extend_to_all`): never gated. */
  extendDims?: readonly string[];
}

export interface PartitionGroupEntry {
  path: string;
  groupObject: THREE.Object3D;
  children: PartitionGroupChild[];
}

/** What the gate needs from the registry that owns it. */
export interface PartitionGateHost {
  /** The frame's view snapshot. */
  view(): ViewContext;
  /** Keep the loop ticking (``LODGroupRegistryDeps.requestTick``). */
  keepTicking(): void;
  /** A part was culled or restored: the next frame must be redrawn. */
  markDrawn(): void;
  /** The registry's per-frame clock reading (``frame.nowMs``). */
  readonly frame: { readonly nowMs: number };
  /** The registry's one-shot retry wakes (unanswered activation requests). */
  readonly retryWakes: RetryWakes;
}

/** Reused per frame: one scratch box for the frustum tests. */
const WORLD_BOX3_SCRATCH = new THREE.Box3();

const PARTITION_FRUSTUM_SCRATCH = new THREE.Frustum();
const PARTITION_FRUSTUM_MATRIX_SCRATCH = new THREE.Matrix4();
/** The partition frustum's screen-space pad (``config.lod.partitionFrustumMargin``), reused. */
const PARTITION_FRUSTUM_SCALE = new THREE.Matrix4();

/**
 * Projection × view padded by ``config.lod.partitionFrustumMargin`` on x/y:
 * cold parts have no loaded footprint to union, so the pad preloads them
 * before entry and keeps entry and exit symmetric.
 */
function setPartitionFrustum(projView: THREE.Matrix4, camera: THREE.Camera): void {
  const pad = 1 / (1 + config.lod.partitionFrustumMargin);
  PARTITION_FRUSTUM_SCALE.makeScale(pad, pad, 1);
  PARTITION_FRUSTUM_MATRIX_SCRATCH.copy(projView).premultiply(PARTITION_FRUSTUM_SCALE);
  // The camera's clip convention (WebGPU depth range, reversed depth), as #2988.
  PARTITION_FRUSTUM_SCRATCH.setFromProjectionMatrix(
    PARTITION_FRUSTUM_MATRIX_SCRATCH,
    camera.coordinateSystem,
    camera.reversedDepth
  );
}

const FOOTPRINT_BOX3_SCRATCH = new THREE.Box3();
// Bit flags returned by evaluatePartitionEntry so one child scan reports both effects.
const PARTITION_VISIBILITY_CHANGED = 1;
const PARTITION_BECAME_VISIBLE = 2;
// Part paths that re-entered the frustum THIS frame, collected by
// ``evaluatePartitionEntry`` and merged into ``partitionResyncPending`` only
// on a rising edge (rare), so the steady-state per-frame path allocates nothing.
const RISING_PARTS_SCRATCH = new Set<string>();
// Partitions skipped this frame because their wrapper was hidden (re-checked
// after the LOD pass, which may reveal them). Reused: no per-frame allocation.
const HIDDEN_PARTITIONS_SCRATCH: PartitionGroupEntry[] = [];
/** Scratch for ``LODGroupRegistry.rankPartitionPartsForLoad`` (load time, not per frame). */
const LOAD_RANK_LOCAL_BOX: BoundingBox = { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };
const LOAD_RANK_OPTIONS: WorldBoxOptions = {
  worldBoxScratch: { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } },
};

/**
 * Whether a resync naming `targets` covers the part keyed `partKey` of the
 * partition at `wrapperPath`: the wrapper itself, the part, or an ancestor of
 * the part (a nested partition's outer part). Node paths are `/`-separated.
 */
function isPartTargeted(
  partKey: string,
  wrapperPath: string,
  targets: ReadonlySet<string>
): boolean {
  if (targets.has(wrapperPath) || targets.has(partKey)) return true;
  for (const target of targets) {
    if (partKey.startsWith(target.endsWith('/') ? target : `${target}/`)) return true;
  }
  return false;
}

/**
 * Record a part that just re-entered the frustum, by its registered node path
 * (``PartitionGroupChild.path`` — the loader-registry key for a leaf part and
 * the prefix of a nested ``kind=lod`` part's level loaders). A part with no
 * path cannot be targeted, so it is recorded as the WRAPPER path: resync the
 * whole partition rather than silently miss it.
 */
function noteRisingPart(sink: Set<string>, partPath: string, wrapperPath: string): void {
  sink.add(partPath || wrapperPath);
}

function unionPartitionFootprints(objects: readonly THREE.Object3D[], target: THREE.Box3): void {
  for (const object of objects) {
    FOOTPRINT_BOX3_SCRATCH.setFromObject(object);
    if (!FOOTPRINT_BOX3_SCRATCH.isEmpty()) target.union(FOOTPRINT_BOX3_SCRATCH);
  }
}

/**
 * Whether any object of a part has moved since its footprint box was captured.
 * ``cached`` holds one 16-element world-matrix block per object, in
 * ``objects`` order, sized by ``registerPartition``. A part that later gained an
 * object reads ``undefined`` past the end and so counts as moved, which
 * recaptures and grows the array; one that lost an object simply stops
 * comparing the trailing blocks.
 *
 * Only the part's own objects are compared. A transform on a DESCENDANT of one
 * of them is not detected — descendant transforms are applied at node-creation
 * time and never animated, and every geometry commit dirties the part
 * explicitly. Anything that starts moving a descendant later has to call
 * ``invalidatePartitionFootprint``.
 */
function partitionFootprintMatricesDiffer(
  objects: readonly THREE.Object3D[],
  cached: readonly number[]
): boolean {
  for (let index = 0; index < objects.length; index++) {
    const elements = objects[index].matrixWorld.elements;
    const offset = index * 16;
    for (let element = 0; element < 16; element++) {
      if (cached[offset + element] !== elements[element]) return true;
    }
  }
  return false;
}

/** Rebuild a part's cached footprint box and the world matrices it was taken at. */
function capturePartitionFootprint(
  objects: readonly THREE.Object3D[],
  target: THREE.Box3,
  cached: number[]
): void {
  target.makeEmpty();
  unionPartitionFootprints(objects, target);
  // Snapshot AFTER the union: ``Box3.expandByObject`` refreshes each object's
  // world matrix, so a snapshot taken first would lag by one recompute and make
  // every following frame look like a move.
  for (let index = 0; index < objects.length; index++) {
    objects[index].matrixWorld.toArray(cached, index * 16);
  }
}

/** `'/'` — the path segment separator ``LODGroupRegistry.invalidatePartitionFootprint`` splits on. */
const SLASH = 0x2f;

/**
 * One registered part, as the footprint-invalidation path index stores it: the
 * partition it belongs to and its index in that partition's children. A part's
 * path may be shared (duplicates) or empty (a pathless part, keyed by `''`, which
 * is a segment prefix of every absolute path — the same match the linear rule
 * `nodePath.startsWith(childPath + '/')` gave it).
 */
export interface PartitionPartRef {
  entryPath: string;
  index: number;
}

/**
 * A deferred part's activation state (B4). `requested` is set when the
 * per-frame gate asked the loader for a pass to activate it (a pending
 * resync), `running` while that pass's activation is in flight; both clear
 * when the pass declines it, so a part still wanted is asked for again. A
 * request no pass has reached within `config.lod.lazyActivationRequestTimeoutMs` (the resync was
 * rejected, or superseded by a pass targeting other parts) expires, and a
 * one-shot wake at expiry lets a parked camera ask again (#2944).
 */
export interface LazyPartState {
  requested: boolean;
  /** Registry clock (``frame.nowMs``) of the outstanding request. */
  requestedAtMs: number;
  /** The in-flight activation, if one is running. */
  running: Promise<void> | null;
  /**
   * The loader passes awaiting the running activation, each of which sweeps
   * the part's loaders itself once it settles (see
   * ``LODGroupRegistry.activatePartitionParts``); `true` is a claim that
   * cannot be withdrawn. A claim only counts while its pass is alive: a
   * superseded pass commits nothing, so an activation settling with no live
   * claim resyncs the part itself. Empty ⇒ started ahead of any pass
   * (`prefetchSlice`).
   */
  claims: Array<AbortSignal | true>;
  /**
   * The last activation rejected (its leaves recorded a retryable failure):
   * the per-frame gate does not ask again until a Retry re-arms it
   * (``LODGroupRegistry.retryLazyChildByNodePath``).
   */
  failed: boolean;
}

/** Per-part per-frame state of a registered partition (parallel to its children). */
export interface PartitionChildCache {
  source: PartitionGroupEntry;
  localBoxScratch: BoundingBox;
  worldBoxOptions: WorldBoxOptions;
  footprintBox: THREE.Box3;
  /**
   * World transforms ``footprintBox`` was captured at — one flattened
   * 16-element block per object of the part, in ``children[i].objects`` order.
   */
  footprintMatrixWorld: number[];
  footprintDirty: boolean;
  /** Last frustum test (`true` until the first evaluation). */
  inFrustum: boolean;
  /** Deferred part state (B4): see {@link PartitionGroupChild.activate}. */
  lazy: LazyPartState | null;
}

/** Whether a part can draw anything for `view` (always, when there is no view). */
function partitionChildInSlice(
  child: PartitionGroupChild,
  view: PartitionSliceView | undefined
): boolean {
  if (!view || child.sliceExempt === true) return true;
  return partBoundsIntersectSlice(child.positionBounds, view, child.extendDims);
}

/**
 * Stamp one part's objects. `partitionFrustumVisible` (what draws, and what
 * refinement / capture read) is `inFrustum && inSlice`; `partitionInFrustum`
 * is the frustum test alone, which a view pass reads with its OWN slice.
 * Returns the ``PARTITION_*`` flags; a rising edge is a FRUSTUM re-entry only —
 * a slice entry is committed by the pass that moved the slice, which swept it.
 */
function updatePartitionObjectVisibility(
  objects: readonly THREE.Object3D[],
  inFrustum: boolean,
  inSlice: boolean
): number {
  const visible = inFrustum && inSlice;
  // Any previously culled object makes the whole part a rising edge.
  let wasInFrustum = true;
  let changed = false;
  for (const object of objects) {
    if (isPartitionFrustumCulled(object)) wasInFrustum = false;
    object.userData.partitionInFrustum = inFrustum;
    object.userData.partitionFrustumVisible = visible;
    if (object.visible !== visible) {
      object.visible = visible;
      changed = true;
    }
  }
  return (
    (changed ? PARTITION_VISIBILITY_CHANGED : 0) |
    (inFrustum && !wasInFrustum ? PARTITION_BECAME_VISIBLE : 0)
  );
}

/**
 * Every registered ``kind=partition`` group of one registry, with its per-part
 * state; see the module doc.
 */
export class PartitionGate implements TickDemand {
  private readonly partitionEntries = new Map<string, PartitionGroupEntry>();
  private readonly partitionCaches = new Map<string, { children: PartitionChildCache[] }>();
  /**
   * Part path → every registered part carrying it (B9c): the index that makes
   * {@link invalidatePartitionFootprint} O(path depth) instead of a scan over
   * every part of every partition. Maintained by ``registerPartition`` /
   * ``unregister`` / ``clear``; ``partitionPartKeys`` remembers which keys each
   * partition added so it can take exactly those back out.
   */
  private readonly partitionPartsByPath = new Map<string, PartitionPartRef[]>();
  private readonly partitionPartKeys = new Map<string, string[]>();
  /**
   * Partition rising edges waiting for their wrapper to be visible and the
   * loader to be idle: wrapper path → the re-entering PART paths. A set that
   * contains the wrapper path itself means "resync the whole partition" (a
   * pathless part). Coalesced across frames; flushed as ONE
   * ``requestReprocess(paths)`` call.
   */
  private readonly partitionResyncPending = new Map<string, Set<string>>();
  /** Reused to feed ``transformBoundingBox``'s matrix arg. */
  private readonly matrixScratch: number[] = new Array(16).fill(0);
  /** This frame: a visible wrapper has a resync pending. */
  private visiblePendingResync = false;

  constructor(
    private readonly deps: LODGroupRegistryDeps,
    private readonly host: PartitionGateHost
  ) {}

  /** Number of registered partitions. */
  size(): number {
    return this.partitionEntries.size;
  }

  /** Whether ``path`` is a registered partition. */
  has(path: string): boolean {
    return this.partitionEntries.has(path);
  }

  /** The paths of every registered partition. */
  paths(): IterableIterator<string> {
    return this.partitionEntries.keys();
  }

  /** Drop the partition at ``path`` (restoring its parts' visibility), if any. */
  unregister(path: string): void {
    this.cancelRetryWakes(path);
    const partition = this.partitionEntries.get(path);
    if (partition) this.restorePartitionChildren(partition);
    this.partitionResyncPending.delete(path);
    this.partitionEntries.delete(path);
    this.partitionCaches.delete(path);
    this.forgetPartitionParts(path);
  }

  /** Drop every partition (restoring their parts' visibility). */
  clear(): void {
    for (const partition of this.partitionEntries.values()) {
      this.restorePartitionChildren(partition);
    }
    this.partitionEntries.clear();
    this.partitionCaches.clear();
    this.partitionPartsByPath.clear();
    this.partitionPartKeys.clear();
    this.partitionResyncPending.clear();
  }

  /**
   * First partition pass of an evaluated frame, BEFORE the LOD pass (so a
   * lod_group nested in a part sees this frame's cull): gate every visible
   * wrapper against the padded frustum of ``projView``. A wrapper hidden now is
   * skipped and remembered: a lod_group may reveal it this frame (the overview
   * recipe's fine level), and {@link endFrame} gates it before that frame
   * draws. Returns whether a part was culled or restored.
   */
  beginFrame(
    displayDims: readonly number[],
    projView: THREE.Matrix4,
    camera: THREE.Camera
  ): boolean {
    setPartitionFrustum(projView, camera);
    HIDDEN_PARTITIONS_SCRATCH.length = 0;
    this.visiblePendingResync = false;
    let cullChanged = false;
    for (const entry of this.partitionEntries.values()) {
      if (!isEffectivelyVisible(entry.groupObject)) {
        HIDDEN_PARTITIONS_SCRATCH.push(entry);
        continue;
      }
      if (this.gateVisible(entry, displayDims)) cullChanged = true;
    }
    return cullChanged;
  }

  /**
   * After the LOD pass: gate the wrappers it revealed, then hand the pending
   * rising edges to the loader once it is idle. Returns whether a part was
   * culled or restored.
   */
  endFrame(displayDims: readonly number[]): boolean {
    let cullChanged = false;
    for (const entry of HIDDEN_PARTITIONS_SCRATCH) {
      if (!isEffectivelyVisible(entry.groupObject)) continue;
      if (this.gateVisible(entry, displayDims)) cullChanged = true;
    }
    HIDDEN_PARTITIONS_SCRATCH.length = 0;
    if (this.partitionResyncPending.size > 0 && this.deps.isUpdateInProgress?.() !== true) {
      this.flushPartitionResyncs();
    }
    return cullChanged;
  }

  tickUntilMs(): number {
    // Nobody can run a resync without ``requestReprocess``: nothing to wait for.
    if (!this.deps.requestReprocess) return NO_TICK;
    if (this.visiblePendingResync && this.partitionResyncPending.size > 0) return UNTIL_RESOLVED;
    return NO_TICK;
  }

  /** {@link evaluatePartitionFrame} for one visible wrapper, noting a pending resync. */
  private gateVisible(entry: PartitionGroupEntry, displayDims: readonly number[]): boolean {
    const changed = this.evaluatePartitionFrame(entry, displayDims);
    if (this.partitionResyncPending.has(entry.path)) this.visiblePendingResync = true;
    return changed;
  }

  /** Cancel the retry wakes of the partition registered at ``path``'s deferred parts. */
  private cancelRetryWakes(path: string): void {
    for (const child of this.partitionCaches.get(path)?.children ?? []) {
      if (child.lazy) this.host.retryWakes.cancel(child.lazy);
    }
  }

  /** Register a partition whose children are independently frustum-gated. */
  registerPartition(entry: PartitionGroupEntry): void {
    this.cancelRetryWakes(entry.path);
    this.forgetPartitionParts(entry.path);
    this.partitionEntries.set(entry.path, entry);
    this.indexPartitionParts(entry);
    this.partitionCaches.set(entry.path, {
      children: entry.children.map((child) => ({
        source: { path: entry.path, groupObject: entry.groupObject, children: [child] },
        localBoxScratch: {
          min: { x: 0, y: 0, z: 0 },
          max: { x: 0, y: 0, z: 0 },
        },
        worldBoxOptions: {
          worldBoxScratch: {
            min: { x: 0, y: 0, z: 0 },
            max: { x: 0, y: 0, z: 0 },
          },
        },
        footprintBox: new THREE.Box3(),
        footprintMatrixWorld: new Array<number>(child.objects.length * 16).fill(0),
        footprintDirty: true,
        // Until the first frame evaluates it. A deferred part was deferred as
        // outside the padded frustum (or the slice), so a pass reaching
        // `activatePartitionParts` first must not activate it on a guess.
        inFrustum: !child.activate,
        lazy: child.activate
          ? { requested: false, requestedAtMs: 0, running: null, claims: [], failed: false }
          : null,
      })),
    });
    for (const child of entry.children) {
      // Each loader path resolves to its emitted object, so stamping them all
      // lets the loader gate use its normal ancestor walk for multi-object parts.
      for (const object of child.objects) object.userData.partitionFrustumVisible = true;
    }
  }

  /**
   * Mark the owning partition part's rendered footprint stale after a geometry commit.
   *
   * ``SceneLoader.updatePointsGeometry`` and the three ``commit*Geometry`` methods
   * are the complete geometry-attach funnels, including lazy LOD children and
   * additive rungs. They dirty the part before writing, so a commit that hands off
   * geometry and then throws cannot leave the previous footprint cached.
   * ``registerPartition`` starts every part dirty, which also covers a commit that
   * races registration. A path under a partition that matches no registered child
   * dirties the whole partition conservatively rather than allowing an
   * under-covering stale box. Footprints are cached in world space; the per-frame
   * gate separately detects transform changes.
   */
  invalidatePartitionFootprint(nodePath: string): void {
    if (this.partitionCaches.size === 0) return;
    // A partition or part is affected exactly when its path is ``nodePath`` or a
    // segment prefix of it, so enumerate those prefixes (O(depth)) and look them
    // up, instead of scanning every registered part — a commit per part on a
    // 2000-part partition made the scan O(parts²) per pass (B9c).
    const touched: string[] = [];
    const hits: PartitionPartRef[] = [];
    for (let end = nodePath.length; end >= 0; end--) {
      if (end !== nodePath.length && nodePath.charCodeAt(end) !== SLASH) continue;
      const prefix = nodePath.slice(0, end);
      if (this.partitionEntries.has(prefix)) touched.push(prefix);
      const parts = this.partitionPartsByPath.get(prefix);
      if (parts) hits.push(...parts);
    }
    for (const entryPath of touched) this.dirtyPartitionParts(entryPath, hits);
  }

  /**
   * Dirty the ``hits`` belonging to partition ``entryPath`` — or, when none
   * does (a commit under the partition matching no registered part), every part
   * of it, conservatively, rather than keep an under-covering stale box.
   */
  private dirtyPartitionParts(entryPath: string, hits: readonly PartitionPartRef[]): void {
    const cache = this.partitionCaches.get(entryPath);
    if (!cache) return;
    let matched = false;
    for (const hit of hits) {
      if (hit.entryPath !== entryPath) continue;
      cache.children[hit.index].footprintDirty = true;
      matched = true;
    }
    if (matched) return;
    for (const childCache of cache.children) childCache.footprintDirty = true;
  }

  /** Index a registered partition's parts by path for {@link invalidatePartitionFootprint}. */
  private indexPartitionParts(entry: PartitionGroupEntry): void {
    const keys = entry.children.map((child) => child.path);
    keys.forEach((key, index) => {
      let refs = this.partitionPartsByPath.get(key);
      if (!refs) {
        refs = [];
        this.partitionPartsByPath.set(key, refs);
      }
      refs.push({ entryPath: entry.path, index });
    });
    this.partitionPartKeys.set(entry.path, keys);
  }

  /** Drop a partition's parts from the path index. */
  private forgetPartitionParts(entryPath: string): void {
    const keys = this.partitionPartKeys.get(entryPath);
    if (!keys) return;
    this.partitionPartKeys.delete(entryPath);
    for (const key of new Set(keys)) {
      const refs = this.partitionPartsByPath.get(key);
      if (!refs) continue;
      const kept = refs.filter((ref) => ref.entryPath !== entryPath);
      if (kept.length > 0) this.partitionPartsByPath.set(key, kept);
      else this.partitionPartsByPath.delete(key);
    }
  }

  /**
   * Re-stamp every partition part against the committed view NOW (B4). The
   * owning loader calls this right after a pass commits, so a part that left
   * the slice is hidden, and one that entered it is shown and refinable, in
   * the same frame as the commit — not one evaluation later (refinement,
   * scheduled at the pass tail, reads these stamps before the next frame).
   * The frustum half is the last evaluation's; nothing here is a rising edge.
   */
  applyCommittedSlice(): void {
    const committed = this.deps.getCommittedViewState?.();
    for (const entry of this.partitionEntries.values()) {
      const cache = this.partitionCaches.get(entry.path);
      if (!cache) continue;
      for (let index = 0; index < entry.children.length; index++) {
        const child = entry.children[index];
        const inSlice = partitionChildInSlice(child, committed);
        const flags = updatePartitionObjectVisibility(
          child.objects,
          cache.children[index].inFrustum,
          inSlice
        );
        if ((flags & PARTITION_VISIBILITY_CHANGED) !== 0) this.host.markDrawn();
      }
    }
  }

  /**
   * Whether the loader at `path` can draw anything for `view` — `false` when
   * any partition part enclosing it is provably empty on a discrete hidden
   * dimension (see `partition-slice-gate.ts`). A view pass skips such loaders:
   * the part is hidden from the commit on ({@link applyCommittedSlice}), so its
   * previous geometry is never drawn for `view`. O(path depth).
   */
  isPathInPartitionSlice(path: string, view: PartitionSliceView): boolean {
    if (this.partitionPartsByPath.size === 0) return true;
    for (let end = path.length; end >= 0; end--) {
      if (end !== path.length && path.charCodeAt(end) !== SLASH) continue;
      const refs = this.partitionPartsByPath.get(path.slice(0, end));
      if (refs && !refs.every((ref) => this.partRefInSlice(ref, view))) return false;
    }
    return true;
  }

  private partRefInSlice(ref: PartitionPartRef, view: PartitionSliceView): boolean {
    const child = this.partitionEntries.get(ref.entryPath)?.children[ref.index];
    return child === undefined || partitionChildInSlice(child, view);
  }

  /**
   * Activate the deferred parts a loader pass for `view` needs (B4): every
   * deferred part in the padded frustum (last evaluation), under an effectively
   * visible wrapper and in `view`'s slice — restricted to `targets` (a targeted
   * resync) when given. An activation attaches and REGISTERS the part's
   * loaders without loading their data: resolves, once every activation it
   * started (or joined) has settled, with the keys of those parts, whose new
   * loaders the pass then sweeps itself and commits with everything else.
   * `claim` is the pass's abort signal (`true`: a claim that is never
   * withdrawn); `false` (`prefetchSlice`, activating ahead of the next slice)
   * claims nothing: that slice's pass finds the loaders registered. A deferred
   * part under `targets` that does not qualify has its request cleared, so the
   * per-frame gate asks again while it is still wanted.
   */
  activatePartitionParts(
    view: PartitionSliceView,
    targets?: ReadonlySet<string>,
    claim: AbortSignal | boolean = true
  ): Promise<string[]> {
    const runs: Promise<void>[] = [];
    const keys: string[] = [];
    for (const entry of this.partitionEntries.values()) {
      const cache = this.partitionCaches.get(entry.path);
      if (!cache) continue;
      for (let index = 0; index < entry.children.length; index++) {
        const run = this.activatePart(entry, index, cache.children[index], view, targets);
        if (!run) continue;
        runs.push(run.promise);
        keys.push(run.key);
        if (claim !== false) run.lazy.claims.push(claim);
      }
    }
    return runs.length === 0 ? Promise.resolve(keys) : Promise.all(runs).then(() => keys);
  }

  private activatePart(
    entry: PartitionGroupEntry,
    index: number,
    childCache: PartitionChildCache,
    view: PartitionSliceView,
    targets: ReadonlySet<string> | undefined
  ): { promise: Promise<void>; key: string; lazy: LazyPartState } | null {
    const child = entry.children[index];
    const lazy = childCache.lazy;
    if (!lazy || !child.activate || lazy.failed) return null;
    const partKey = child.path || entry.path;
    if (targets && !isPartTargeted(partKey, entry.path, targets)) return null;
    if (!lazy.running) {
      // A pass reached the part: its request is answered.
      lazy.requested = false;
      this.host.retryWakes.reset(lazy);
    }
    if (!this.partActivationWanted(entry, child, childCache, view)) return null;
    // Already loading (activated ahead of this view): the caller still waits for it.
    lazy.running ??= this.runActivation(entry, partKey, child, childCache, lazy);
    return { promise: lazy.running, key: partKey, lazy };
  }

  /** In the padded frustum (last evaluation), in `view`'s slice, under a visible wrapper. */
  private partActivationWanted(
    entry: PartitionGroupEntry,
    child: PartitionGroupChild,
    childCache: PartitionChildCache,
    view: PartitionSliceView
  ): boolean {
    return (
      childCache.inFrustum &&
      partitionChildInSlice(child, view) &&
      isEffectivelyVisible(entry.groupObject)
    );
  }

  private runActivation(
    entry: PartitionGroupEntry,
    partKey: string,
    child: PartitionGroupChild,
    childCache: PartitionChildCache,
    lazy: LazyPartState
  ): Promise<void> {
    const activate = child.activate as () => Promise<void>;
    return activate().then(
      () => {
        perfCounters.add(S_PARTS_ACTIVATED);
        this.settleActivation(entry, partKey, child, childCache, lazy);
      },
      (error: unknown) => {
        log.warning(
          Modules.SCENE_LOADER,
          `partition part ${child.path} failed to activate: ${String(error)}`
        );
        // Re-armable: its loaders recorded a retryable failure, and a Retry
        // clears `failed` (see `retryLazyChildByNodePath`).
        lazy.running = null;
        lazy.requested = false;
        this.host.retryWakes.cancel(lazy);
        lazy.claims = [];
        lazy.failed = true;
      }
    );
  }

  /**
   * A part's activation registered its loaders: never activate the slot again.
   * Nothing drawn changed — its placeholders are empty until a pass commits
   * them. With no LIVE claim (started ahead of any pass, or every pass that
   * awaited it was superseded) and the COMMITTED view showing it, nothing else
   * will sweep the new loaders for that view: resync it. A part of a
   * registration that has since been cleared or replaced (dataset switch) is
   * left alone.
   */
  private settleActivation(
    entry: PartitionGroupEntry,
    partKey: string,
    child: PartitionGroupChild,
    childCache: PartitionChildCache,
    lazy: LazyPartState
  ): void {
    child.activate = undefined;
    this.host.retryWakes.cancel(lazy);
    if (childCache.lazy === lazy) childCache.lazy = null;
    childCache.footprintDirty = true;
    if (this.partitionEntries.get(entry.path) !== entry) return;
    const claimLive = lazy.claims.some((claim) => claim === true || !claim.aborted);
    const committed = this.deps.getCommittedViewState?.();
    if (
      !claimLive &&
      childCache.inFrustum &&
      partitionChildInSlice(child, committed) &&
      isEffectivelyVisible(entry.groupObject)
    ) {
      this.notePartitionRisingEdge(entry.path, new Set([partKey]));
      this.host.keepTicking();
    }
  }

  /**
   * Load-time ranking of a partition's parts against the CURRENT camera (B4):
   * which intersect the padded partition frustum, and a nearest-first load
   * order. The scene manager frames the opening camera from the root metadata
   * before any node loads (`LoaderConfig.onSceneMetadata`), so this is the pose
   * the scene opens on. `null` when there is no usable view (no displayed dims
   * yet, an empty viewport), or when no part is in the frustum at all: a scene
   * framed after its load (an authored `target_node` names a node not built
   * yet) would otherwise defer everything its opening view is about to show.
   */
  rankPartitionPartsForLoad(
    groupObject: THREE.Object3D,
    bounds: readonly { min: readonly number[]; max: readonly number[] }[]
  ): { inFrustum: boolean[]; order: number[] } | null {
    const displayDims = this.deps.getDisplayDims();
    const view = this.host.view();
    if (displayDims.length < 2 || view.viewportCss === null) return null;
    setPartitionFrustum(view.projView, view.camera);
    groupObject.updateWorldMatrix(true, false);
    const inFrustum: boolean[] = [];
    const distance: number[] = [];
    for (const positionBounds of bounds) {
      const box = computeEntryWorldBox(
        { groupObject, children: [{ positionBounds }] },
        displayDims,
        LOAD_RANK_LOCAL_BOX,
        this.matrixScratch,
        LOAD_RANK_OPTIONS
      );
      if (!box) {
        inFrustum.push(true);
        distance.push(0);
        continue;
      }
      WORLD_BOX3_SCRATCH.min.set(box.min.x, box.min.y, box.min.z);
      WORLD_BOX3_SCRATCH.max.set(box.max.x, box.max.y, box.max.z);
      inFrustum.push(PARTITION_FRUSTUM_SCRATCH.intersectsBox(WORLD_BOX3_SCRATCH));
      distance.push(WORLD_BOX3_SCRATCH.distanceToPoint(view.cameraWorldPosition));
    }
    if (!inFrustum.some(Boolean)) return null;
    const order = bounds.map((_, i) => i).sort((a, b) => distance[a] - distance[b] || a - b);
    return { inFrustum, order };
  }

  restorePartitionChildren(entry: PartitionGroupEntry): void {
    this.host.markDrawn();
    for (const child of entry.children) {
      for (const object of child.objects) {
        object.visible = true;
        delete object.userData.partitionFrustumVisible;
        delete object.userData.partitionInFrustum;
      }
    }
  }

  /**
   * Frustum- and slice-gate one VISIBLE partition for this frame and record its
   * rising edges. Returns whether a part was culled or restored.
   */
  private evaluatePartitionFrame(
    entry: PartitionGroupEntry,
    displayDims: readonly number[]
  ): boolean {
    RISING_PARTS_SCRATCH.clear();
    const result = this.evaluatePartitionEntry(
      entry,
      displayDims,
      PARTITION_FRUSTUM_SCRATCH,
      RISING_PARTS_SCRATCH
    );
    // Only parts with something stale to resync are recorded (a settled
    // re-entry just re-shows), so an empty set means no resync at all — never
    // an empty-path request, which the loader would read as a FULL re-sweep.
    if (RISING_PARTS_SCRATCH.size > 0) {
      this.notePartitionRisingEdge(entry.path, RISING_PARTS_SCRATCH);
    }
    RISING_PARTS_SCRATCH.clear();
    if ((result & PARTITION_VISIBILITY_CHANGED) === 0) return false;
    this.host.markDrawn();
    return true;
  }

  /**
   * Frustum- and slice-gate one partition's parts. Returns the ``PARTITION_*``
   * bit flags; parts that re-entered the frustum this frame are added to
   * ``risingParts`` by node path (``child.path``), or as the WRAPPER path when a
   * part has none so the caller resyncs the whole partition rather than missing
   * it — unless the part is already settled for the current view
   * ({@link partitionPartSettled}). A deferred part that is in the frustum and
   * the committed slice is added too: the resync pass is what activates it.
   */
  private evaluatePartitionEntry(
    entry: PartitionGroupEntry,
    displayDims: readonly number[],
    frustum: THREE.Frustum,
    risingParts: Set<string>
  ): number {
    const cache = this.partitionCaches.get(entry.path);
    if (!cache) return 0;
    const committed = this.deps.getCommittedViewState?.();
    let result = 0;
    for (let index = 0; index < entry.children.length; index++) {
      const child = entry.children[index];
      const childCache = cache.children[index];
      childCache.inFrustum = this.partitionPartInFrustum(child, childCache, displayDims, frustum);
      const inSlice = partitionChildInSlice(child, committed);
      const flags = updatePartitionObjectVisibility(child.objects, childCache.inFrustum, inSlice);
      result |= flags;
      if (childCache.lazy) {
        const wanted = childCache.inFrustum && inSlice;
        this.requestLazyActivation(risingParts, entry.path, child, childCache.lazy, wanted);
      } else if ((flags & PARTITION_BECAME_VISIBLE) !== 0 && !this.partitionPartSettled(child)) {
        noteRisingPart(risingParts, child.path, entry.path);
      }
    }
    return result;
  }

  /** Frustum test of one part: its stored bounds united with its rendered footprint. */
  private partitionPartInFrustum(
    child: PartitionGroupChild,
    childCache: PartitionChildCache,
    displayDims: readonly number[],
    frustum: THREE.Frustum
  ): boolean {
    const worldBox = computeEntryWorldBox(
      childCache.source,
      displayDims,
      childCache.localBoxScratch,
      this.matrixScratch,
      childCache.worldBoxOptions
    );
    if (!worldBox) return true;
    WORLD_BOX3_SCRATCH.min.set(worldBox.min.x, worldBox.min.y, worldBox.min.z);
    WORLD_BOX3_SCRATCH.max.set(worldBox.max.x, worldBox.max.y, worldBox.max.z);
    for (const object of child.objects) object.updateWorldMatrix(false, false);
    if (
      childCache.footprintDirty ||
      partitionFootprintMatricesDiffer(child.objects, childCache.footprintMatrixWorld)
    ) {
      capturePartitionFootprint(
        child.objects,
        childCache.footprintBox,
        childCache.footprintMatrixWorld
      );
      childCache.footprintDirty = false;
    }
    if (!childCache.footprintBox.isEmpty()) {
      WORLD_BOX3_SCRATCH.union(childCache.footprintBox);
    }
    return frustum.intersectsBox(WORLD_BOX3_SCRATCH);
  }

  /**
   * Whether a part that just re-entered the frustum has nothing to resync
   * (B9c): every tracked leaf under it already holds a commit for the CURRENT
   * view version with a complete ladder, so the view did not move while it was
   * culled and a targeted re-sweep would only re-derive what is on screen.
   * Unknown (no view-version wiring, or a leaf never committed) is NOT settled,
   * so such a part resyncs exactly as before.
   */
  private partitionPartSettled(child: PartitionGroupChild): boolean {
    const version = this.deps.getViewVersion?.();
    if (version == null) return false;
    return child.objects.every((object) =>
      subtreeSweepSettled(object as unknown as SweepNode, version)
    );
  }

  /**
   * Ask for a pass that activates a deferred part — once, while it is `wanted`
   * (in the frustum and the committed slice) and no activation is requested or
   * running. The request rides the rising-edge resync: the targeted pass it
   * triggers calls {@link activatePartitionParts} for the part.
   */
  private requestLazyActivation(
    risingParts: Set<string>,
    entryPath: string,
    child: PartitionGroupChild,
    lazy: LazyPartState,
    wanted: boolean
  ): void {
    if (!wanted || lazy.running || lazy.failed) return;
    const wakes = this.host.retryWakes;
    const timeoutMs = wakes.delay(lazy, config.lod.lazyActivationRequestTimeoutMs);
    if (lazy.requested) {
      // Outstanding: one wake (scheduled with the request) asks again at expiry.
      if (this.host.frame.nowMs - lazy.requestedAtMs < timeoutMs) return;
      wakes.backOff(lazy);
    }
    lazy.requested = true;
    lazy.requestedAtMs = this.host.frame.nowMs;
    wakes.schedule(lazy, wakes.delay(lazy, config.lod.lazyActivationRequestTimeoutMs));
    noteRisingPart(risingParts, child.path, entryPath);
  }

  /**
   * Rising edge (rare): remember WHICH parts of ``wrapperPath`` came back so
   * the resync can be targeted at their loaders instead of re-sweeping the
   * whole scene. Coalesces with parts already pending for the same wrapper.
   */
  private notePartitionRisingEdge(wrapperPath: string, parts: ReadonlySet<string>): void {
    let pending = this.partitionResyncPending.get(wrapperPath);
    if (!pending) {
      pending = new Set<string>();
      this.partitionResyncPending.set(wrapperPath, pending);
    }
    for (const partPath of parts) pending.add(partPath);
  }

  /**
   * Hand every pending rising edge whose wrapper is visible to the loader as
   * ONE ``requestReprocess(paths)`` call; hidden wrappers stay pending, dropped
   * wrappers are forgotten. A set that contains its own wrapper path (a
   * pathless part) collapses to the wrapper path alone — it already covers
   * every part.
   */
  private flushPartitionResyncs(): void {
    const requestReprocess = this.deps.requestReprocess;
    if (!requestReprocess) {
      // Unwired (an embedder without a loader pass): nothing will ever run
      // these, and a pending set would keep the loop ticking every frame.
      this.partitionResyncPending.clear();
      return;
    }
    let resyncPaths: string[] | null = null;
    for (const [path, parts] of this.partitionResyncPending) {
      const entry = this.partitionEntries.get(path);
      if (!entry) {
        this.partitionResyncPending.delete(path);
        continue;
      }
      if (!isEffectivelyVisible(entry.groupObject)) continue;
      this.partitionResyncPending.delete(path);
      resyncPaths ??= [];
      if (parts.has(path)) resyncPaths.push(path);
      else resyncPaths.push(...parts);
    }
    if (resyncPaths) requestReprocess(resyncPaths);
  }

  /**
   * Re-arm a deferred partition part whose activation failed (B4): the
   * per-frame gate asks for a pass to activate it again while it is wanted.
   */
  rearmFailedPartitionPart(path: string): boolean {
    for (const [entryPath, entry] of this.partitionEntries) {
      const cache = this.partitionCaches.get(entryPath);
      const index = entry.children.findIndex((child) => child.path === path);
      const lazy = index < 0 ? null : cache?.children[index]?.lazy;
      if (!lazy?.failed) continue;
      lazy.failed = false;
      this.host.keepTicking();
      return true;
    }
    return false;
  }

  /**
   * Whether a visible partition has a rising-edge resync waiting for the
   * owning loader to become idle. Unlike
   * {@link pendingPartitionResyncsQuiescent}, this deliberately mirrors
   * {@link flushPartitionResyncs}'s wrapper-level visibility gate: every queued
   * part under a visible wrapper is dispatched, even if that part re-exits
   * before the flush. Pending work retained under a hidden wrapper is not
   * actionable and must not keep wide settledness false indefinitely; neither
   * can work when no resync dispatcher is wired.
   */
  hasVisiblePendingPartitionResync(): boolean {
    if (!this.deps.requestReprocess) return false;
    for (const path of this.partitionResyncPending.keys()) {
      const entry = this.partitionEntries.get(path);
      if (entry && isEffectivelyVisible(entry.groupObject)) return true;
    }
    return false;
  }

  /**
   * Whether visible partition parts have no pending resync or incomplete commit.
   * Unlike {@link hasVisiblePendingPartitionResync}, pending resyncs count here
   * only while their specific parts contribute pixels to the capture frame.
   */
  partitionsCaptureQuiescent(version: number | null): boolean {
    if (!this.pendingPartitionResyncsQuiescent()) return false;
    for (const entry of this.partitionEntries.values()) {
      if (!this.partitionEntryCaptureQuiescent(entry, version)) return false;
    }
    return true;
  }

  private pendingPartitionResyncsQuiescent(): boolean {
    if (!this.deps.requestReprocess) return true;
    for (const [path, parts] of this.partitionResyncPending) {
      const entry = this.partitionEntries.get(path);
      if (!entry) {
        continue;
      }
      if (this.pendingPartitionResyncContributes(entry, parts)) return false;
    }
    return true;
  }

  private pendingPartitionResyncContributes(
    entry: PartitionGroupEntry,
    parts: ReadonlySet<string>
  ): boolean {
    if (!isEffectivelyVisible(entry.groupObject)) return false;
    if (parts.has(entry.path)) return this.partitionHasVisiblePart(entry);
    return entry.children.some(
      (child) => parts.has(child.path) && this.partitionChildIsVisible(child)
    );
  }

  private partitionEntryCaptureQuiescent(
    entry: PartitionGroupEntry,
    version: number | null
  ): boolean {
    if (!isEffectivelyVisible(entry.groupObject)) return true;
    for (const child of entry.children) {
      if (!this.partitionChildCaptureQuiescent(child, version)) return false;
    }
    return true;
  }

  private partitionChildCaptureQuiescent(
    child: PartitionGroupChild,
    version: number | null
  ): boolean {
    if (!this.partitionChildIsVisible(child)) return true;
    for (const object of child.objects) {
      const progress = subtreeDisplayProgress(object as unknown as ProgressNode, version);
      if (progress && (!progress.fresh || !progress.complete)) return false;
    }
    return true;
  }

  /** Whether any registered partition part contributes pixels to this frame. */
  anyVisiblePartitionPart(): boolean {
    for (const entry of this.partitionEntries.values()) {
      if (!isEffectivelyVisible(entry.groupObject)) continue;
      if (this.partitionHasVisiblePart(entry)) return true;
    }
    return false;
  }

  private partitionHasVisiblePart(entry: PartitionGroupEntry): boolean {
    return entry.children.some((child) => this.partitionChildIsVisible(child));
  }

  private partitionChildIsVisible(child: PartitionGroupChild): boolean {
    return child.objects.some((object) => object.userData.partitionFrustumVisible !== false);
  }
}
