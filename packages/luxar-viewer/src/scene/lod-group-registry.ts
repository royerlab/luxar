/**
 * Per-frame LOD-group selector.
 *
 * Tracks every `lod_group` scene-graph node currently loaded. For each
 * one, every frame:
 *
 *   1. Project each child's nD ``positionBounds`` to a 3D
 *      :type:`BoundingBox` via :func:`projectBoundsToDisplayDims` (using
 *      the current ``displayDims`` from the view state).
 *   2. Union them via :func:`mergeBoundingBoxes`, then transform into
 *      world space via :func:`transformBoundingBox` (using the
 *      lod_group's ``matrixWorld``).
 *   3. Project the 8 corners through the camera to NDC and back to
 *      pixel coordinates; the diagonal of the screen-space AABB is
 *      the selector metric.
 *   4. Pick the **finest** child whose ``min_pixel_size`` threshold is
 *      satisfied by that diagonal, with 10% asymmetric hysteresis on
 *      the downgrade direction to suppress threshold-edge flicker.
 *   5. If the desired child differs from the current active one and
 *      the desired child has been marked ``ready``, swap visibility;
 *      otherwise keep the current active child visible until the new
 *      one's geometry has been committed by the loader.
 *
 * The bbox infrastructure is shared with the scene-bounds cache and
 * camera framing — ``projectBoundsToDisplayDims`` /
 * ``mergeBoundingBoxes`` / ``transformBoundingBox`` all live in
 * ``scene-manager/clipping/bounds-math.ts`` and are reused here rather
 * than duplicated.
 *
 * Wiring: the SceneLoader instantiates one registry per scene; the
 * pipeline hooks ``evaluatePerFrame`` into ``AnimationController``
 * alongside the dynamic-clipping callback. Manual override
 * (``setSelectorMode(path, { lockLevel: i })``) bypasses the auto
 * selector — driven by the layers-panel dropdown landing in PR 2c.
 *
 * @module scene/lod-group-registry
 */

import * as THREE from 'three';

import {
  type BoundingBox,
  mergeBoundingBoxes,
  projectBoundsToDisplayDims,
  transformBoundingBox,
} from './scene-manager/clipping/bounds-math';
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
  /**
   * Per-child readiness flag. The scene loader flips this to ``true``
   * once the child's first geometry has been committed; until then the
   * selector keeps the previous active child visible so the user never
   * stares at a blank lod_group during a level swap.
   */
  ready: boolean;
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
  const corner = new THREE.Vector3();
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

  constructor(private deps: LODGroupRegistryDeps) {}

  /** Register a newly-loaded lod_group (called by the scene loader). */
  register(entry: LODGroupEntry): void {
    this.entries.set(entry.path, entry);
    // Apply initial visibility: only the active child is visible.
    for (let i = 0; i < entry.children.length; i++) {
      entry.children[i].object.visible = i === entry.activeChildIndex;
    }
  }

  /** Drop an lod_group from the registry (called on scene teardown). */
  unregister(path: string): void {
    this.entries.delete(path);
  }

  /** Clear all entries (called on full scene tear-down). */
  clear(): void {
    this.entries.clear();
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
   * child index ``i`` (0-based in coarsest→finest order).
   *
   * Visibility is *not* swapped synchronously — the next
   * ``evaluatePerFrame()`` call will pick the new desired child. (This
   * matches the per-frame contract for the auto path; a synchronous
   * swap would diverge.)
   */
  setSelectorMode(path: string, mode: LODGroupSelectorMode): void {
    const entry = this.entries.get(path);
    if (!entry) return;
    if (mode !== 'auto') {
      const idx = mode.lockLevel;
      if (idx < 0 || idx >= entry.children.length) {
        throw new RangeError(
          `lockLevel ${idx} out of range for lod_group ${path} ` +
            `(${entry.children.length} children)`
        );
      }
    }
    entry.selectorMode = mode;
  }

  /**
   * Mark a child as ready. The scene loader calls this once the
   * child's geometry has data committed; the registry only swaps to
   * a child once it's been marked ready.
   */
  markChildReady(path: string, childIndex: number): void {
    const entry = this.entries.get(path);
    if (!entry || childIndex < 0 || childIndex >= entry.children.length) return;
    entry.children[childIndex].ready = true;
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
      // Project each child's nD bounds to a 3D box, then union; this
      // is the lod_group's effective bbox in **local** space.
      const perChildBoxes: BoundingBox[] = [];
      for (const child of entry.children) {
        const pb = child.positionBounds;
        if (
          pb.min.length === 0 ||
          pb.max.length === 0 ||
          pb.min.length !== pb.max.length
        ) {
          continue;
        }
        perChildBoxes.push(
          projectBoundsToDisplayDims(pb.min, pb.max, displayDims)
        );
      }
      if (perChildBoxes.length === 0) return;
      const localBox = mergeBoundingBoxes(perChildBoxes);

      // Lift to world space. THREE updates matrix lazily; force a
      // refresh before reading — cheap and idempotent.
      entry.groupObject.updateWorldMatrix(true, false);
      const worldBox = transformBoundingBox(
        localBox,
        entry.groupObject.matrixWorld.toArray()
      );

      const diagonalPx = projectBoxDiagonalPx(worldBox, camera, viewport);
      const thresholds = entry.children.map((c) => c.minPixelSize);
      desired = pickChildWithHysteresis(
        thresholds,
        entry.activeChildIndex,
        diagonalPx
      );
    }

    if (desired === entry.activeChildIndex) return;
    // Keep the current active child visible until the desired one is
    // ready. This avoids a blank flash when the camera leaps to a
    // detail level whose geometry hasn't streamed in yet.
    if (!entry.children[desired].ready) return;

    entry.children[entry.activeChildIndex].object.visible = false;
    entry.children[desired].object.visible = true;
    entry.activeChildIndex = desired;
  }
}
