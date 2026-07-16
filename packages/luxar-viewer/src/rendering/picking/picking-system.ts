/**
 * GPU Picking System for Luxar.
 *
 * Orchestrates the picking pipeline:
 * 1. Mousemove records the cursor position and schedules a settle
 *    check on the next animation frame.
 * 2. A pick fires only when BOTH the mouse and the pick-buffer
 *    (camera/geometry) have been stable for `HOVER_SETTLE_MS`. This
 *    "two-axis settle" mirrors the drag-suppression UX: tooltips
 *    appear when the world is quiet and hide whenever anything is
 *    moving. Moving the cursor over a static scene costs nothing; a
 *    stationary cursor during animation gets no stale tooltip.
 * 3. If the buffer is dirty when the pick fires, ALL registered pick
 *    nodes are rendered to a cached RGBA32F target at half-res.
 * 4. Ray-AABB culling (using a per-node cached world AABB) skips the
 *    readback when the cursor is over empty space.
 * 5. Readback of a 5×5 region followed by brightness-weighted
 *    majority voting picks the winning (nodeId, elementId).
 *
 * The pick buffer encodes: R=nodeId, G=elementId, B=brightness, A=1.0
 * Brightness-as-depth (gl_FragDepth = 1 - brightness) ensures the
 * brightest element at each pixel wins the depth test. Exception:
 * gsplat nodes in surface ('normal') blending mode write real projected
 * depth instead (front-most wins) — renderPickBuffer syncs the
 * convention from each main material's blendingMode per render.
 *
 * Caching: the pick buffer is only re-rendered when dirty (camera move,
 * geometry update, window resize). World AABBs are cached per node and
 * survive camera motion — only register/unregister or geometry commit
 * invalidates them.
 */

import * as THREE from 'three';
import type { PostProcessingManager } from '../post-processing/post-processing-manager';
import { isCameraAwareMaterial } from '../materials/_shared/camera-aware-material';
import { isNormalMode } from '../blending-state';
import type { BlendingMode } from '../material-manager';
import {
  disposePickMaterial,
  unregisterAllPickMaterials,
  type PickNodeEntry,
} from './picking-system/registration';
import { rayHitsAnyNode, invalidateBoxCache } from './picking-system/ray-aabb';
import { voteWinner, type VoteEntry } from './picking-system/pick-render';
import { SettleScheduler } from './picking-system/settle-scheduler';
import { applyLensDistortion } from './picking-system/lens-distortion';
import type { Renderer, RendererCapabilities } from '../renderer-capabilities';
import { readPixelsCompactAsync } from '../post-processing/hdr/pixel-utils';
import {
  getCameraFovRadians,
  isOrthographicCamera,
  getOrthoFrustumHeight,
} from '../../utils/camera-utils';
import type { LuxarCamera } from '../../utils/camera-utils';
import { log, Modules } from '../../utils/log';
import { clamp } from '../../utils/clamp';

/** Result of a successful pick operation. */
export interface PickResult {
  /** Assigned pick ID of the node */
  nodeId: number;
  /** Element index within the node (point index, segment instance, splat instance) */
  elementId: number;
  /** Brightness weight of the winning vote */
  brightness: number;
  /** Reference to the main scene object */
  mainNode: THREE.Object3D;
}

/** Size of the pick buffer in pixels (5x5 = 25 pixels). */
const PICK_SIZE = 5;

/**
 * Hard cap on either dimension of the pick render target. Pick IDs do not
 * need pixel-perfect resolution, so on 4K/5K screens we stop scaling up to
 * keep per-pick GPU cost bounded.
 */
export const MAX_PICK_BUFFER_DIM = 1024;

/**
 * Compute the pick render target size from the renderer's drawing-buffer
 * size: half resolution, capped at MAX_PICK_BUFFER_DIM on the larger
 * axis, never smaller than 1 pixel.
 *
 * The cap is applied as a SINGLE uniform scale on both axes so the pick
 * buffer always preserves the drawing-buffer (= camera) aspect ratio.
 * This is load-bearing for gsplat picking: the gsplat pick shader maps
 * view space to pick-buffer pixels manually with uFx == uFy (a
 * square-pixel assumption), so a pick buffer with a different aspect
 * than the camera displaces gsplat picks horizontally away from screen
 * center. Points/lines pick through the aspect-aware projectionMatrix
 * and tolerate any aspect — but only aspect-preserving sizing keeps all
 * three geometry types consistent.
 */
export function computePickBufferSize(drawW: number, drawH: number): { w: number; h: number } {
  const halfW = Math.max(drawW / 2, 1);
  const halfH = Math.max(drawH / 2, 1);
  const scale = Math.min(1, MAX_PICK_BUFFER_DIM / halfW, MAX_PICK_BUFFER_DIM / halfH);
  return {
    w: Math.max(1, Math.floor(halfW * scale)),
    h: Math.max(1, Math.floor(halfH * scale)),
  };
}

export class PickingSystem {
  private pickScene: THREE.Scene;
  private pickTarget: THREE.WebGLRenderTarget;
  private nodeMap: Map<number, PickNodeEntry> = new Map();
  private nextPickId = 1;
  private raycaster: THREE.Raycaster;
  private ndcCoord: THREE.Vector2;

  // Reusable objects to avoid per-pick allocations.
  private _lastReadX = 0;
  private _lastReadY = 0;
  private _savedClearColor = new THREE.Color();
  private _savedClearAlpha = 0;
  private _lensUV = { x: 0, y: 0 };
  private _pickResolution = new THREE.Vector2();

  // Pick-buffer cache: re-render the offscreen target only when the
  // view changes (camera/geometry/resize).
  private _dirty = true;
  private _drawBufSize = new THREE.Vector2();

  // Monotonic pick sequence. `performPick` is async (the GPU readback
  // resolves a frame or more later), so without a guard a slow readback
  // from an older pick can resolve AFTER a newer pick/move and clobber it
  // with a stale tooltip. Every supersede — a fresh pick, a mousemove, a
  // dirty, or a mouseleave — bumps this counter; `performPick` captures it
  // at fire time and drops its post-readback emit if it's no longer the
  // latest. The pre-readback (cull-miss / fade) emits are synchronous and
  // already in fire order, so only the post-`await` emit needs the guard.
  private _pickSeq = 0;

  // Last pick-buffer dimensions actually pushed to `pickTarget.setSize`.
  // Used as an explicit guard against rapid-resize churn: on every pick
  // we recompute (pickW, pickH) from the current drawing-buffer size and
  // only call setSize when the values actually changed since last frame.
  // Without this, an orbit + window-resize loop can call setSize many
  // times per second, each reallocating the underlying GPU buffer.
  // -1 sentinel ensures the first pick always triggers an explicit setSize.
  private _lastPickW = -1;
  private _lastPickH = -1;

  // Cached canvas rect (invalidated on resize via markDirty).
  private _canvasRect: DOMRect | null = null;

  // Settle scheduler: owns the rAF lifecycle and the mouse/dirty
  // timestamps. The orchestrator forwards events (onMouseMove,
  // markDirty, suppress) and exposes the scheduler's bookkeeping via
  // `getDiagnostics()`.
  private scheduler: SettleScheduler;

  // World-AABB cache: avoids re-applying matrixWorld per node per pick.
  // Invalidated only on register/unregister/geometry-commit — camera motion
  // does NOT invalidate.
  private _worldBoxCache: Map<number, THREE.Box3> = new Map();

  // Pre-allocated readback scratch (eliminates per-pick allocations).
  private _readDst: Float32Array;
  private _readFlipped: Float32Array;

  // Reused vote map (cleared per readback instead of `new Map()`).
  private _votes: Map<number, VoteEntry> = new Map();

  /**
   * Optional predicate gating whether picks should fire. Defaults to
   * always-true. App wires this to overlayManager visibility so we
   * skip the pick entirely when no hover tooltip would display the
   * result.
   */
  private _shouldPick: () => boolean = () => true;

  /** Optional post-processing reference for lens distortion correction. */
  private postProcessing: PostProcessingManager | null = null;

  constructor(
    private renderer: Renderer,
    private capabilities: RendererCapabilities,
    private camera: THREE.Camera,
    private onPickResult: (result: PickResult | null) => void
  ) {
    this.pickScene = new THREE.Scene();
    // No background — pick buffer clears to (0,0,0,0) which means "no hit"

    // Cached pick buffer at half viewport resolution. Resized dynamically in performPick.
    // Rendered once per view change; hover events only readback from the cached buffer.
    this.pickTarget = new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.FloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true,
      stencilBuffer: false,
    });

    this._readDst = new Float32Array(PICK_SIZE * PICK_SIZE * 4);
    this._readFlipped = new Float32Array(PICK_SIZE * PICK_SIZE * 4);
    this.raycaster = new THREE.Raycaster();
    this.ndcCoord = new THREE.Vector2();

    this.scheduler = new SettleScheduler({
      now: () => performance.now(),
      shouldFire: () => this._shouldPick(),
      firePick: (x, y) => {
        void this.performPick(x, y);
      },
    });

    log.info(Modules.RENDERER, 'PickingSystem initialized (5x5 RGBA32F)');
  }

  /** Allocate the next pick ID (incrementing counter, starts at 1). 0 = background. */
  allocatePickId(): number {
    return this.nextPickId++;
  }

  /** Register a main scene node and its picking shadow node. */
  registerNode(mainNode: THREE.Object3D, pickNode: THREE.Object3D, pickId: number): void {
    // Disable automatic matrix updates on pick nodes. Their matrixWorld is
    // manually synced from the main node in renderPickBuffer(). Without this,
    // renderer.render() calls scene.updateMatrixWorld() which recomputes
    // matrixWorld from the pick node's local transform (identity) and parent
    // (pickScene = identity), overwriting the correct synced transform.
    pickNode.matrixAutoUpdate = false;
    pickNode.matrixWorldAutoUpdate = false;

    // forward link main → pick so commit helpers (e.g.
    // `syncPointMaterialWithGeometry`) can reach the picking material
    // without a reverse map lookup. Keeps lifecycle simple — when the
    // main node disposes, picking-system.unregisterNode also clears
    // this via the nodeMap removal.
    mainNode.userData.pickNode = pickNode;

    this.nodeMap.set(pickId, { main: mainNode, pick: pickNode });
    this._worldBoxCache.delete(pickId);
  }

  /** Unregister a node by its pick ID and dispose its pick material. */
  unregisterNode(pickId: number): void {
    const entry = this.nodeMap.get(pickId);
    if (entry) {
      this._disposePickMaterial(entry.pick as THREE.Mesh);
    }
    this.nodeMap.delete(pickId);
    this._worldBoxCache.delete(pickId);
  }

  /**
   * Invalidate cached world-space AABBs. Call after geometry or
   * matrixWorld changes; camera-only motion does NOT need this and
   * should use {@link markDirty} alone.
   *
   * @param pickId - Specific node to invalidate, or omit to drop all.
   */
  invalidateBoxes(pickId?: number): void {
    invalidateBoxCache(this._worldBoxCache, pickId);
  }

  /**
   * Set the predicate gating whether picks fire. Used by app.ts to
   * skip picking when no hover overlay is visible (no consumer for
   * the result).
   */
  setShouldPick(predicate: () => boolean): void {
    this._shouldPick = predicate;
  }

  /**
   * Dispose the material(s) attached to a pick mesh. Handles the rare
   * `material: array` case so a future custom-multi-material pick node
   * doesn't leak shaders.
   */
  private _disposePickMaterial(mesh: THREE.Mesh): void {
    disposePickMaterial(mesh);
  }

  /** Number of registered pick nodes. */
  get registeredNodeCount(): number {
    return this.nodeMap.size;
  }

  /**
   * Read-only snapshot of settle-scheduler timestamps and registration
   * count. Exposed for E2E tests (the hover-tooltip spec polls
   * `lastPickFiredTime` to verify a pick fired without reaching into
   * private fields). Timestamps are `performance.now()` values; 0
   * means "never". Not part of the production API surface — treat as
   * an observability hook, not an interaction point.
   */
  getDiagnostics(): {
    lastPickFiredTime: number;
    lastMouseMoveTime: number;
    lastDirtyTime: number;
    registeredNodeCount: number;
    suppressed: boolean;
  } {
    return {
      lastPickFiredTime: this.scheduler.lastPickFiredTime,
      lastMouseMoveTime: this.scheduler.lastMouseMoveTime,
      lastDirtyTime: this.scheduler.lastDirtyTime,
      registeredNodeCount: this.nodeMap.size,
      suppressed: this.scheduler.isSuppressed,
    };
  }

  /**
   * Drop all node registrations *without* disposing the pick
   * materials. Used after a WebGL context-loss event: the pick
   * materials' shader programs are already invalid (the context they
   * were compiled against is gone), and calling `dispose()` on them
   * would throw on some drivers. The caller (`NodeFactory.rebuildAfterContextRestore`)
   * is responsible for re-registering every scene node afterward,
   * which produces fresh pick materials against the new context.
   *
   * Pick materials are registered with `materialManager.register(...)`
   * at construction (see `node-factory.ts`). Without unregistering
   * them here, repeated context-restore cycles accumulate stale
   * references in the materialManager registry — camera-uniform
   * updates would target dead materials and the
   * `getStats().totalRegistered` count grows unboundedly. We
   * unregister WITHOUT disposing (calls
   * `materialManager.unregister(material)` not `dispose(material)`),
   * matching the "no-dispose during context loss" contract for
   * visible materials.
   *
   * Distinct from `unregisterNode(id)` which intentionally disposes
   * the pick material when removing a single live node.
   */
  clearRegistrationsForRebuild(): void {
    unregisterAllPickMaterials(this.nodeMap);
    this.nodeMap.clear();
    this._worldBoxCache.clear();
    while (this.pickScene.children.length > 0) {
      this.pickScene.remove(this.pickScene.children[0]);
    }
    this._dirty = true;
    this.scheduler.markDirty();
  }

  /** Update camera reference (e.g., after perspective ↔ orthographic swap). */
  setCamera(camera: THREE.Camera): void {
    this.camera = camera;
    this._dirty = true; // Must re-render pick buffer with new projection
    this.scheduler.markDirty();
  }

  /** Set post-processing reference for lens distortion correction. */
  setPostProcessing(pp: PostProcessingManager | null): void {
    this.postProcessing = pp;
  }

  /**
   * Invalidate the cached pick buffer. Call when camera, geometry, or
   * viewport changes. Fades the current hover overlay (matches the
   * drag-suppression UX: while the camera is moving, tooltips hide).
   * The rAF scheduler will fire a fresh pick once everything has been
   * still for HOVER_SETTLE_MS.
   */
  markDirty(): void {
    this._dirty = true;
    this._canvasRect = null;
    // Supersede any in-flight readback so its late result can't override
    // this fade.
    this._pickSeq++;
    // Fade overlay (OverlayManager dedupes against the last state, so
    // repeated calls do no DOM work).
    this.onPickResult(null);
    this.scheduler.markDirty();
  }

  /**
   * Invalidate ONLY the cached canvas rect. Call on page scroll / layout
   * shifts that move the canvas without changing the 3D view: the rect
   * (from `getBoundingClientRect()`) maps `event.clientX/Y` into
   * canvas-local pick coordinates, so a stale rect after a scroll would
   * offset every pick. Unlike {@link markDirty} this does NOT re-render
   * the pick buffer or fade the tooltip — the view is unchanged, only the
   * canvas's screen position moved. The next mousemove lazily recomputes
   * the rect.
   *
   * We also drop any pending settle pick: its stored coordinate is
   * already canvas-local (converted at mousemove time against the
   * now-stale rect), so firing it after the scroll would pick the wrong
   * spot. Cancelling lets the next mousemove re-arm with a fresh,
   * correctly-mapped coordinate.
   */
  invalidateCanvasRect(): void {
    this._canvasRect = null;
    this.scheduler.cancelPending();
  }

  /**
   * Suppress or resume picking (e.g. during orbit/pan/zoom interactions).
   * `true` cancels any pending rAF; `false` re-enables the scheduler
   * and re-arms it when there's a pending cursor position. The re-arm
   * matters for "orbit-and-release without moving the mouse": the
   * camera dirtied during the suppressed window, so once it settles
   * for HOVER_SETTLE_MS the rAF tick fires the camera-settle re-pick
   * naturally — no mouse wiggle required.
   */
  suppress(value: boolean): void {
    this.scheduler.setSuppressed(value);
  }

  /**
   * Handle mouse move. Records the cursor position, fades any visible
   * tooltip, and arms the settle scheduler. Performs zero picking
   * work directly — the actual pick fires from the rAF loop once both
   * the mouse and the pick buffer have been still for HOVER_SETTLE_MS.
   *
   * Note: we do NOT early-return while suppressed (orbit/pan/zoom). The
   * scheduler still tracks the latest cursor position during suppression
   * (without scheduling a pick) so the re-pick on release uses where the
   * cursor actually is, not a stale pre-orbit position.
   */
  onMouseMove(event: MouseEvent): void {
    if (!this._canvasRect) {
      this._canvasRect = this.renderer.domElement.getBoundingClientRect();
    }
    const x = event.clientX - this._canvasRect.left;
    const y = event.clientY - this._canvasRect.top;
    // Supersede any in-flight readback — the cursor moved, so an older
    // pick's late result is now stale.
    this._pickSeq++;
    // Fade existing overlay while moving (dedupe-safe).
    this.onPickResult(null);
    this.scheduler.recordMouseMove(x, y);
  }

  /**
   * Cursor left the canvas. Drop the pending position so the
   * camera-settle re-pick path doesn't fire a stale pick when the
   * cursor isn't even over the viewer, and cancel any pending rAF.
   */
  onMouseLeave(): void {
    // Supersede any in-flight readback so it can't re-show a tooltip after
    // the cursor has already left the canvas.
    this._pickSeq++;
    this.scheduler.recordMouseLeave();
    this.onPickResult(null);
  }

  /** Clean up all resources — render target, pick materials, scene. */
  dispose(): void {
    this.scheduler.dispose();

    // Dispose all pick materials (unregisters from materialManager automatically)
    for (const entry of this.nodeMap.values()) {
      this._disposePickMaterial(entry.pick as THREE.Mesh);
    }

    // Clean up pick scene children (paranoia — should be empty between renders)
    while (this.pickScene.children.length > 0) {
      this.pickScene.remove(this.pickScene.children[0]);
    }

    this.pickTarget.dispose();
    this.nodeMap.clear();
    this._worldBoxCache.clear();
    this._votes.clear();
    this.postProcessing = null;
    log.info(Modules.RENDERER, 'PickingSystem disposed');
  }

  // ---------------------------------------------------------------------------
  // Private implementation
  // ---------------------------------------------------------------------------

  /**
   * Perform a pick at the given screen coordinates.
   *
   * If the pick buffer is dirty (camera/geometry/resize changed), re-renders
   * ALL registered nodes to the cached buffer first. Otherwise just reads
   * from the cached buffer — zero GPU cost on hover.
   */
  private async performPick(screenX: number, screenY: number): Promise<void> {
    // Claim this pick's slot. Any later pick/move/dirty/leave bumps
    // `_pickSeq`, marking this readback stale (see the post-`await` guard).
    const pickSeq = ++this._pickSeq;

    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    if (width === 0 || height === 0) return;

    // Pick buffer at half resolution (IDs don't need full res, 4× fewer pixels)
    // and additionally capped at MAX_PICK_BUFFER_DIM per axis so 4K/5K screens
    // do not pay full cost. Cursor coordinates rescale automatically below
    // because they are computed from pickW/width and pickH/height.
    //
    // Coupling note: getDrawingBufferSize is CSS size × the ACTIVE pixel
    // ratio, so every adaptive-DPR step changes pickW/pickH → target
    // reallocation + full pick re-render on the next hover. Cost-only
    // (never correctness), lazy, and the DPR manager's probe backoff +
    // ceiling keep step frequency low; if profiling ever flags this,
    // the lever is quantizing the pick size or sizing from CSS px.
    const drawBuf = this.renderer.getDrawingBufferSize(this._drawBufSize);
    const { w: pickW, h: pickH } = computePickBufferSize(drawBuf.x, drawBuf.y);

    // Check if pick target needs resize (also triggers re-render).
    // Compare against `_lastPickW/_lastPickH` (explicit cache) instead
    // of reading `this.pickTarget.width/.height` — this guards against
    // a setSize-then-readback storm on rapid resize loops, since
    // `setSize` itself reallocates the underlying GPU buffer.
    const sizeChanged = this._lastPickW !== pickW || this._lastPickH !== pickH;
    if (sizeChanged) {
      this.pickTarget.setSize(pickW, pickH);
      this._lastPickW = pickW;
      this._lastPickH = pickH;
      this._dirty = true;
    }

    // Re-render pick buffer if dirty (camera moved, geometry changed, resized)
    if (this._dirty) {
      this.renderPickBuffer();
      this._dirty = false;
    }

    // Apply lens distortion correction if post-processing distortion is active.
    // The post-processing shader maps each output pixel to a source pixel via
    // Brown-Conrady distortion. We apply the same forward map to the mouse coords
    // so they index into the undistorted pick buffer correctly.
    let correctedX = screenX;
    let correctedY = screenY;
    const lensParams = this.postProcessing?.getLensDistortionParams();
    if (lensParams) {
      const uv = applyLensDistortion(screenX / width, screenY / height, lensParams, this._lensUV);
      correctedX = uv.x * width;
      correctedY = uv.y * height;
    }

    // Compute cursor position in pick buffer pixels (half res). The
    // readback primitive returns canonical top-down rows on both
    // backends (see hdr/pixel-utils.ts), so the cursor maps to the
    // pick buffer with no Y inversion — matches screen Y growing
    // downward.
    const scaleX = pickW / width;
    const scaleY = pickH / height;
    // `correctedX/correctedY` can fall slightly outside [0, width/height]
    // under extreme lens distortion at canvas corners. `Math.floor` on a
    // negative value yields a negative integer; the downstream `clamp(...,
    // 0, ...)` corrects that into a valid pick-buffer index. Keep the
    // multiply unclamped — the clamp is the single source of truth for
    // the legal index range.
    const cursorX = Math.floor(correctedX * scaleX);
    const cursorY = Math.floor(correctedY * scaleY);

    const half = Math.floor(PICK_SIZE / 2);
    this._lastReadX = clamp(cursorX - half, 0, pickW - PICK_SIZE);
    this._lastReadY = clamp(cursorY - half, 0, pickH - PICK_SIZE);

    // Ray-BBox culling: quick check if cursor is near any node at all
    // Use corrected coordinates so the ray matches the undistorted pick buffer
    this.ndcCoord.set((correctedX / width) * 2 - 1, -(correctedY / height) * 2 + 1);
    this.raycaster.setFromCamera(this.ndcCoord, this.camera);
    const ray = this.raycaster.ray;

    if (!rayHitsAnyNode(ray, this.nodeMap, this._worldBoxCache)) {
      this.onPickResult(null);
      return;
    }

    // Readback 5×5 pixels at cursor from cached buffer and vote.
    // The async readback path uses `readRenderTargetPixelsAsync`,
    // available on both WebGLRenderer and WebGPURenderer in r184 —
    // a uniform API that works on both backends. See
    // PICKING_DESIGN.md for the 1-frame-latency rationale.
    const result = await this.readbackAndVote();
    // Drop the result if a newer pick/move/dirty/leave superseded us while
    // the readback was in flight — emitting it would clobber fresher state
    // with a stale tooltip.
    if (pickSeq !== this._pickSeq) return;
    this.onPickResult(result);
  }

  /**
   * Render ALL registered pick nodes to the cached pick buffer.
   * Called only when the buffer is dirty (camera/geometry/resize changed).
   * Renders at half resolution for performance — pick IDs don't need full res.
   */
  private renderPickBuffer(): void {
    const renderer = this.renderer;

    // Save full renderer state (render target, scissor, clear color).
    // `getClearColor` on the union expects `Color4` (with alpha);
    // WebGLRenderer's `Color` is read-compatible at runtime — the
    // cast suppresses the TS-only mismatch.
    const savedRenderTarget = renderer.getRenderTarget();
    const savedScissorTest = renderer.getScissorTest();
    renderer.getClearColor(this._savedClearColor as unknown as THREE.Color & { a: number });
    this._savedClearAlpha = renderer.getClearAlpha();

    // Compute pick-buffer resolution and camera params for material updates.
    // Pick materials (especially GSplats) use uResolution for screen-space positioning.
    // They must be updated to match the half-res pick buffer — otherwise
    // vCenterScreen (full-res) vs gl_FragCoord (half-res) will mismatch.
    const pickRes = this._pickResolution;
    pickRes.set(this.pickTarget.width, this.pickTarget.height);
    const cam = this.camera as LuxarCamera;
    const isOrtho = isOrthographicCamera(cam);
    const fov = isOrtho ? getOrthoFrustumHeight(cam) : getCameraFovRadians(cam);

    // Sync and add ALL registered nodes to pick scene
    for (const entry of this.nodeMap.values()) {
      // Sync geometry (main node's geometry may have been replaced by view updates)
      const mainGeom = (entry.main as THREE.Mesh).geometry;
      if (mainGeom) {
        (entry.pick as THREE.Mesh).geometry = mainGeom;
      }
      // Sync world transform
      entry.pick.matrixWorld.copy(entry.main.matrixWorld);

      // Update pick material camera params to match half-res pick buffer
      const mat = (entry.pick as THREE.Mesh).material;
      if (isCameraAwareMaterial(mat)) {
        mat.updateCameraParams(fov, pickRes, isOrtho);
      }

      // Pick-depth convention sync: under depth-sorted alpha-over
      // ('normal') the user sees an occluding surface, so the pick
      // depth must be the real projected depth (front-most wins)
      // instead of brightness-as-depth (brightest wins — right for the
      // commutative modes, but it could pick a brighter splat BEHIND
      // the visible surface). Only gsplat pick materials implement
      // setSurfacePickDepth; points/lines are unaffected.
      const surfaceAware = mat as { setSurfacePickDepth?: (on: boolean) => void };
      if (typeof surfaceAware.setSurfacePickDepth === 'function') {
        // entry.main is typed Object3D — non-mesh mains have no material.
        const mainMat = (entry.main as THREE.Mesh).material as
          | THREE.Material
          | THREE.Material[]
          | undefined;
        const single = Array.isArray(mainMat) ? mainMat[0] : mainMat;
        const mode = (single?.userData.blendingMode ?? 'additive') as BlendingMode;
        surfaceAware.setSurfacePickDepth(isNormalMode(mode));
      }

      this.pickScene.add(entry.pick);
    }

    // Render to pick target at half resolution (pick IDs don't need full res)
    renderer.setRenderTarget(this.pickTarget);
    renderer.setScissorTest(false);
    renderer.setClearColor(0x000000, 0);
    renderer.clear(true, true, false);
    renderer.render(this.pickScene, this.camera);

    // Remove all nodes from pick scene
    for (const entry of this.nodeMap.values()) {
      this.pickScene.remove(entry.pick);
    }

    // Restore full renderer state. `setRenderTarget` cast: see the
    // post-processing-manager rationale (round-tripping
    // `getRenderTarget` → `setRenderTarget` is safe on both
    // backends at runtime; TypeScript's union intersection is
    // stricter than either backend alone).
    renderer.setRenderTarget(savedRenderTarget as THREE.WebGLRenderTarget | null);
    renderer.setScissorTest(savedScissorTest);
    renderer.setClearColor(this._savedClearColor, this._savedClearAlpha);
  }

  /**
   * Read back the 5x5 pick buffer and perform brightness-weighted majority voting.
   * Returns the winning PickResult or null if all pixels are background.
   *
   * Async readback (`readRenderTargetPixelsAsync`) works on both
   * WebGLRenderer and WebGPURenderer in r184. The 1-frame latency
   * on hover is documented in `PICKING_DESIGN.md`.
   */
  private async readbackAndVote(): Promise<PickResult | null> {
    // Unified readback. The primitive hides backend signature dispatch,
    // WebGPU row-padding compaction, and Y-orientation flipping —
    // accepting `(x, y)` in canonical **top-down** pick-buffer space and
    // returning rows in the same top-down order. `_lastReadX/Y` are
    // computed in top-down coordinates in `performPick`, so we pass
    // them through unchanged; the primitive translates `y` to the
    // backend's framebuffer convention internally. Pre-allocated
    // `_readDst`/`_readFlipped` keep the hot path allocation-free.
    const { pixels } = await readPixelsCompactAsync(this.renderer, this.capabilities, {
      target: this.pickTarget,
      x: this._lastReadX,
      y: this._lastReadY,
      width: PICK_SIZE,
      height: PICK_SIZE,
      kind: 'rgba32f',
      out: this._readDst,
      flipOut: this._readFlipped,
    });

    const winner = voteWinner(pixels, PICK_SIZE, this._votes);
    if (!winner) return null;
    const nodeEntry = this.nodeMap.get(winner.nodeId);
    if (!nodeEntry) return null;
    return {
      nodeId: winner.nodeId,
      elementId: winner.elementId,
      brightness: winner.weight,
      mainNode: nodeEntry.main,
    };
  }
}
