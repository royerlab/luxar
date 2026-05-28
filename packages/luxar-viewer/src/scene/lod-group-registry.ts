/**
 * Per-frame LOD-group selector.
 *
 * Tracks every `lod_group` scene-graph node currently loaded. For each
 * one, every frame:
 *
 *   1. Fold each child's nD ``positionBounds`` directly into a cached
 *      per-entry **local-space** :type:`BoundingBox`, using the current
 *      ``displayDims`` to map nD axes onto X/Y/Z. (No intermediate
 *      per-child boxes — the union is computed in place.)
 *   2. Transform the local box into world space via
 *      :func:`transformBoundingBox` and the lod_group's ``matrixWorld``.
 *   3. Project the 8 corners through the camera to NDC and back to
 *      pixel coordinates; the diagonal of the screen-space AABB is
 *      the selector metric.
 *   4. Pick the **finest** child whose ``min_pixel_size`` threshold is
 *      satisfied by that diagonal, with 10% asymmetric hysteresis on
 *      the downgrade direction to suppress threshold-edge flicker.
 *   5. If the desired child differs from the current active one, swap
 *      visibility atomically.
 *
 * The atomic-swap invariant on initial load is realized by
 * ``loadLodGroupNode`` (sequential awaits + ``visible=false`` after
 * attach + a single ``register()`` call at the end) — no per-child
 * ``ready`` gate is needed in the registry.
 *
 * The bbox infrastructure is shared with the scene-bounds cache and
 * camera framing — ``projectBoundsToDisplayDims`` /
 * ``transformBoundingBox`` live in
 * ``scene-manager/clipping/bounds-math.ts`` and are reused here.
 *
 * Wiring: the SceneLoader instantiates one registry per scene; the
 * pipeline hooks ``evaluatePerFrame`` into ``AnimationController``
 * alongside the dynamic-clipping callback. Manual override
 * (``setSelectorMode(path, { lockLevel: i })``) bypasses the auto
 * selector — driven by the layers-panel dropdown.
 *
 * @module scene/lod-group-registry
 */

import * as THREE from 'three';

import {
  type BoundingBox,
  transformBoundingBox,
} from './scene-manager/clipping/bounds-math';
import { log, Modules } from '../utils/log';
import type { LODGroupSelectorMode } from '../types/lod-group';

/** Asymmetric hysteresis on the "downgrade to coarser" direction. */
const HYSTERESIS_RATIO = 0.1;

/** One LOD-group child as tracked by the registry. */
export interface LODGroupChild {
  /** The leaf THREE node (gsplats / points / lines / group). */
  object: THREE.Object3D;
  /** Strictly monotonic increasing in coarsest→finest order. */
  minPixelSize: number;
  /**
   * Raw nD position bounds (from the child's ``position_bounds`` zarr
   * attribute). Stored unprojected because ``displayDims`` can change
   * at runtime (user picks different dimensions to display) — the
   * selector re-projects each frame.
   */
  positionBounds: { min: readonly number[]; max: readonly number[] };
}

/** One LOD-group entry tracked by the registry. */
export interface LODGroupEntry {
  /** Scene path (for diagnostics + UI lookup). */
  path: string;
  /** The lod_group's THREE container. World matrix lives here. */
  groupObject: THREE.Object3D;
  /**
   * Children in coarsest→finest order (== insertion order on disk,
   * == ascending ``minPixelSize``).
   */
  children: LODGroupChild[];
  /** Current selector mode (``'auto'`` or ``{ lockLevel: i }``). */
  selectorMode: LODGroupSelectorMode;
  /** Initial active level, used when nothing else has selected yet. */
  defaultLevel: number;
  /** Index into ``children`` of the currently-visible child. */
  activeChildIndex: number;
}

/**
 * Internal per-entry cache populated by ``register()``. Lets
 * ``evaluateEntry`` run without allocating on the hot path: the
 * threshold list is rebuilt once at registration and the scratch
 * boxes are reused every frame.
 *
 * Kept separate from the public ``LODGroupEntry`` interface so test
 * code (which constructs entries directly) doesn't have to populate
 * caches — ``register()`` does it for them.
 */
interface LODGroupEntryCache {
  thresholds: number[];
  localBoxScratch: BoundingBox;
}

/** Module-scope scratch for ``projectBoxDiagonalPx``. Single-threaded. */
const CORNER_SCRATCH = new THREE.Vector3();

/**
 * Injected view-state accessors. Lets the registry stay test-friendly
 * (mock the camera, the viewport, the slice) without coupling to the
 * SceneManager singleton.
 */
export interface LODGroupRegistryDeps {
  getCamera(): THREE.Camera;
  /** Viewport size in CSS pixels (matches the renderer canvas). */
  getViewportSize(): { width: number; height: number };
  /** Which dimensions of the data are being projected to screen. */
  getDisplayDims(): readonly number[];
}

/**
 * Project a world-space :type:`BoundingBox` through the camera and
 * return the diagonal of the screen-space AABB in pixels.
 *
 * Treats the bbox's 8 corners independently (works for both
 * perspective and orthographic projection without a closed-form
 * radius). NDC → pixels assumes the viewport size matches the
 * renderer canvas.
 *
 * Exported for unit testing.
 */
export function projectBoxDiagonalPx(
  box: BoundingBox,
  camera: THREE.Camera,
  viewport: { width: number; height: number }
): number {
  // Reuse a module-scope Vector3. ``evaluatePerFrame`` is invoked from
  // a single per-frame callback, so this is safe — no concurrent
  // entries into this function.
  const corner = CORNER_SCRATCH;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < 8; i++) {
    corner.set(
      i & 1 ? box.max.x : box.min.x,
      i & 2 ? box.max.y : box.min.y,
      i & 4 ? box.max.z : box.min.z
    );
    corner.project(camera);
    if (corner.x < minX) minX = corner.x;
    if (corner.x > maxX) maxX = corner.x;
    if (corner.y < minY) minY = corner.y;
    if (corner.y > maxY) maxY = corner.y;
  }
  const widthPx = (maxX - minX) * 0.5 * viewport.width;
  const heightPx = (maxY - minY) * 0.5 * viewport.height;
  return Math.hypot(widthPx, heightPx);
}

/**
 * Pick the desired child index given a screen-space diagonal and the
 * current active index. Applies 10% asymmetric hysteresis on the
 * downgrade direction.
 *
 * The "natural" pick is the finest child whose ``minPixelSize`` is
 * less than or equal to ``diagonalPx``. Hysteresis only resists
 * dropping back to a coarser level: when downgrading from index
 * ``currentIdx``, the metric must fall below
 * ``thresholds[currentIdx] * (1 - 0.1)``; otherwise we stay on the
 * current level even though the natural pick is coarser.
 *
 * Exported for unit testing.
 */
export function pickChildWithHysteresis(
  thresholds: readonly number[],
  currentIdx: number,
  diagonalPx: number,
  hysteresisRatio: number = HYSTERESIS_RATIO
): number {
  if (thresholds.length === 0) return -1;

  // Natural pick: finest child with threshold ≤ diagonalPx. Thresholds
  // are monotonic increasing in coarsest→finest order, so scan upward
  // until the threshold exceeds the metric.
  let natural = 0;
  for (let i = 0; i < thresholds.length; i++) {
    if (thresholds[i] <= diagonalPx) natural = i;
    else break;
  }

  if (natural === currentIdx) return currentIdx;
  if (natural > currentIdx) return natural; // upgrade: literal threshold wins

  // Downgrade: require metric to drop below current's threshold *
  // (1 - hysteresisRatio). Otherwise stay on the current finer level.
  const currentThreshold = thresholds[currentIdx];
  if (diagonalPx < currentThreshold * (1 - hysteresisRatio)) {
    return natural;
  }
  return currentIdx;
}

/**
 * Tracks all loaded ``lod_group`` nodes in a scene; evaluates per-frame
 * to pick which child renders.
 */
export class LODGroupRegistry {
  private entries: Map<string, LODGroupEntry> = new Map();
  /**
   * Per-entry register-time cache (parallel to ``entries`` by path).
   * Populated by :meth:`register`. Stored on the side so the public
   * ``LODGroupEntry`` interface stays test-friendly (callers don't
   * have to compute thresholds or allocate scratch boxes).
   */
  private caches: Map<string, LODGroupEntryCache> = new Map();
  /** Reused per frame to feed ``transformBoundingBox``'s matrix arg. */
  private readonly matrixScratch: number[] = new Array(16).fill(0);

  constructor(private deps: LODGroupRegistryDeps) {}

  /** Register a newly-loaded lod_group (called by the scene loader). */
  register(entry: LODGroupEntry): void {
    this.entries.set(entry.path, entry);
    this.caches.set(entry.path, {
      thresholds: entry.children.map((c) => c.minPixelSize),
      localBoxScratch: {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0, y: 0, z: 0 },
      },
    });
    // Apply initial visibility: only the active child is visible.
    for (let i = 0; i < entry.children.length; i++) {
      entry.children[i].object.visible = i === entry.activeChildIndex;
    }
  }

  /** Drop an lod_group from the registry (called on scene teardown). */
  unregister(path: string): void {
    this.entries.delete(path);
    this.caches.delete(path);
  }

  /** Clear all entries (called on full scene tear-down). */
  clear(): void {
    this.entries.clear();
    this.caches.clear();
  }

  /** Number of registered lod_groups (mainly for tests / diagnostics). */
  size(): number {
    return this.entries.size;
  }

  /** Lookup an entry by path (mainly for tests / UI). */
  get(path: string): LODGroupEntry | undefined {
    return this.entries.get(path);
  }

  /** All registered entries (mainly for the layers panel UI). */
  list(): LODGroupEntry[] {
    return Array.from(this.entries.values());
  }

  /**
   * Update an lod_group's selector mode. ``'auto'`` re-enables
   * view-driven selection; ``{ lockLevel: i }`` pins the lod_group to
   * child index ``i`` (0-based in coarsest→finest order). An
   * out-of-range ``lockLevel`` is **clamped** into ``[0, n-1]`` with a
   * warning — throwing here would force every UI caller to guard
   * against stale registry state.
   *
   * Visibility is *not* swapped synchronously — the next
   * ``evaluatePerFrame()`` call will pick the new desired child. (This
   * matches the per-frame contract for the auto path; a synchronous
   * swap would diverge.)
   */
  setSelectorMode(path: string, mode: LODGroupSelectorMode): void {
    const entry = this.entries.get(path);
    if (!entry) return;
    if (mode === 'auto') {
      entry.selectorMode = mode;
      return;
    }
    const n = entry.children.length;
    if (n === 0) {
      entry.selectorMode = mode;
      return;
    }
    const clamped = Math.max(0, Math.min(mode.lockLevel, n - 1));
    if (clamped !== mode.lockLevel) {
      log.warning(
        Modules.SCENE_LOADER,
        `lod_group ${path}: lockLevel ${mode.lockLevel} out of range ` +
          `[0, ${n - 1}], clamped to ${clamped}`
      );
    }
    entry.selectorMode = { lockLevel: clamped };
  }

  /**
   * Per-frame evaluation. Wired through
   * ``AnimationController.addPerFrameCallback`` by the app pipeline.
   */
  evaluatePerFrame(): void {
    if (this.entries.size === 0) return;
    const camera = this.deps.getCamera();
    const viewport = this.deps.getViewportSize();
    const displayDims = this.deps.getDisplayDims();
    if (viewport.width === 0 || viewport.height === 0) return;
    if (displayDims.length < 2) return;

    for (const entry of this.entries.values()) {
      this.evaluateEntry(entry, camera, viewport, displayDims);
    }
  }

  private evaluateEntry(
    entry: LODGroupEntry,
    camera: THREE.Camera,
    viewport: { width: number; height: number },
    displayDims: readonly number[]
  ): void {
    // Pick the desired child index.
    let desired: number;
    if (entry.selectorMode !== 'auto') {
      desired = entry.selectorMode.lockLevel;
    } else {
      const cache = this.caches.get(entry.path);
      if (!cache) return; // shouldn't happen — register() populates this.

      // Fold each child's nD bounds directly into the cached
      // local-space box. No intermediate per-child boxes; we walk
      // children once and unify in place. Children with bogus bounds
      // (mismatched min/max lengths, empty) are skipped.
      const local = cache.localBoxScratch;
      let any = false;
      for (let ci = 0; ci < entry.children.length; ci++) {
        const pb = entry.children[ci].positionBounds;
        if (
          pb.min.length === 0 ||
          pb.max.length === 0 ||
          pb.min.length !== pb.max.length
        ) {
          continue;
        }
        // Project to X/Y/Z. Unmapped axes default to 0 (matches
        // ``projectBoundsToDisplayDims``'s defensive fallback).
        let x0 = 0;
        let x1 = 0;
        let y0 = 0;
        let y1 = 0;
        let z0 = 0;
        let z1 = 0;
        if (displayDims.length > 0) {
          const d0 = displayDims[0];
          if (d0 < pb.min.length) {
            x0 = pb.min[d0];
            x1 = pb.max[d0];
          }
        }
        if (displayDims.length > 1) {
          const d1 = displayDims[1];
          if (d1 < pb.min.length) {
            y0 = pb.min[d1];
            y1 = pb.max[d1];
          }
        }
        if (displayDims.length > 2) {
          const d2 = displayDims[2];
          if (d2 < pb.min.length) {
            z0 = pb.min[d2];
            z1 = pb.max[d2];
          }
        }
        if (!any) {
          local.min.x = x0;
          local.max.x = x1;
          local.min.y = y0;
          local.max.y = y1;
          local.min.z = z0;
          local.max.z = z1;
          any = true;
        } else {
          if (x0 < local.min.x) local.min.x = x0;
          if (x1 > local.max.x) local.max.x = x1;
          if (y0 < local.min.y) local.min.y = y0;
          if (y1 > local.max.y) local.max.y = y1;
          if (z0 < local.min.z) local.min.z = z0;
          if (z1 > local.max.z) local.max.z = z1;
        }
      }
      if (!any) return;

      // Lift to world space. THREE updates matrix lazily; force a
      // refresh before reading — cheap and idempotent.
      entry.groupObject.updateWorldMatrix(true, false);
      // Copy matrixWorld.elements into a reusable array instead of
      // allocating one via ``.toArray()`` every frame.
      const elements = entry.groupObject.matrixWorld.elements;
      const m = this.matrixScratch;
      for (let i = 0; i < 16; i++) m[i] = elements[i];
      const worldBox = transformBoundingBox(local, m);

      const diagonalPx = projectBoxDiagonalPx(worldBox, camera, viewport);
      desired = pickChildWithHysteresis(
        cache.thresholds,
        entry.activeChildIndex,
        diagonalPx
      );
    }

    if (desired === entry.activeChildIndex) return;
    // Atomic swap: hide outgoing, show incoming. The atomic-swap
    // invariant on initial load is enforced by ``loadLodGroupNode``
    // (sequential awaits + ``visible=false`` after attach + a single
    // ``register()`` call at the end), so no per-child readiness gate
    // is needed here.
    entry.children[entry.activeChildIndex].object.visible = false;
    entry.children[desired].object.visible = true;
    entry.activeChildIndex = desired;
  }
}
