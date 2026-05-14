/**
 * GPU Picking System for Luxar.
 *
 * Orchestrates the picking pipeline:
 * 1. Debounced mousemove triggers a pick
 * 2. If dirty: re-render ALL registered nodes to cached RGBA32F pick buffer (half-res)
 * 3. Ray-BBox culling to skip readback if cursor is in empty space
 * 4. Readback 5×5 pixels at cursor + brightness-weighted majority voting
 * 5. Callback with winning (nodeId, elementId) or null
 *
 * The pick buffer encodes: R=nodeId, G=elementId, B=brightness, A=1.0
 * Brightness-as-depth (gl_FragDepth = 1 - brightness) ensures the
 * brightest element at each pixel wins the depth test.
 *
 * Caching: the pick buffer is only re-rendered when dirty (camera move,
 * geometry update, window resize). Hover events just readback from the
 * cached buffer — zero GPU cost per hover.
 */

import * as THREE from 'three';
import type { PostProcessingManager } from '../post-processing/post-processing-manager';
import { isCameraAwareMaterial } from '../camera-aware-material';
import { materialManager } from '../material-manager';
import type { Renderer } from '../renderer-capabilities';
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

/** Internal tracking of a registered node pair. */
interface PickNodeEntry {
  main: THREE.Object3D;
  pick: THREE.Object3D;
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
 * size: half resolution per axis, capped at MAX_PICK_BUFFER_DIM, never
 * smaller than 1 pixel.
 */
export function computePickBufferSize(drawW: number, drawH: number): { w: number; h: number } {
  const w = clamp(Math.floor(drawW / 2), 1, MAX_PICK_BUFFER_DIM);
  const h = clamp(Math.floor(drawH / 2), 1, MAX_PICK_BUFFER_DIM);
  return { w, h };
}

/** Debounce delay in milliseconds before triggering a pick re-render. */
const DEBOUNCE_MS = 10;

export class PickingSystem {
  private pickScene: THREE.Scene;
  private pickTarget: THREE.WebGLRenderTarget;
  private readBuffer: Float32Array;
  private nodeMap: Map<number, PickNodeEntry> = new Map();
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingMouse: { x: number; y: number } | null = null;
  private nextPickId = 1;
  private raycaster: THREE.Raycaster;
  private ndcCoord: THREE.Vector2;

  // Reusable objects to avoid per-pick allocations
  private _box = new THREE.Box3();
  private _lastReadX = 0;
  private _lastReadY = 0;
  private _savedClearColor = new THREE.Color();
  private _savedClearAlpha = 0;
  private _lensUV = { x: 0, y: 0 }; // reusable return for applyLensDistortion
  private _pickResolution = new THREE.Vector2(); // reusable for pick buffer resolution

  // Cache: only re-render when the view changes
  private _dirty = true;
  private _drawBufSize = new THREE.Vector2();

  // Cached canvas rect (invalidated on resize via markDirty)
  private _canvasRect: DOMRect | null = null;

  // Throttle clean-buffer picks to max ~60Hz (one per rAF)
  private _lastPickTime = 0;

  /** When true, picking is suppressed (e.g. during orbit/pan/zoom). */
  private _suppressed = false;

  /** Optional post-processing reference for lens distortion correction. */
  private postProcessing: PostProcessingManager | null = null;

  constructor(
    private renderer: Renderer,
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

    this.readBuffer = new Float32Array(PICK_SIZE * PICK_SIZE * 4);
    this.raycaster = new THREE.Raycaster();
    this.ndcCoord = new THREE.Vector2();

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
  }

  /** Unregister a node by its pick ID and dispose its pick material. */
  unregisterNode(pickId: number): void {
    const entry = this.nodeMap.get(pickId);
    if (entry) {
      this._disposePickMaterial(entry.pick as THREE.Mesh);
    }
    this.nodeMap.delete(pickId);
  }

  /**
   * Dispose the material(s) attached to a pick mesh. Handles the rare
   * `material: array` case so a future custom-multi-material pick node
   * doesn't leak shaders.
   */
  private _disposePickMaterial(mesh: THREE.Mesh): void {
    const material = mesh.material;
    if (Array.isArray(material)) {
      for (const m of material) {
        m?.dispose?.();
      }
    } else {
      material?.dispose?.();
    }
  }

  /** Number of registered pick nodes. */
  get registeredNodeCount(): number {
    return this.nodeMap.size;
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
    for (const entry of this.nodeMap.values()) {
      const material = (entry.pick as THREE.Mesh).material;
      const list = Array.isArray(material) ? material : [material];
      for (const m of list) {
        // Pick materials are constructed in NodeFactory and ALWAYS
        // implement CameraAwareMaterial (Point/Line/GSplatPickingMaterial
        // each declare `implements CameraAwareMaterial`), so the cast
        // is safe. `isCameraAwareMaterial(m)` is the runtime guard.
        if (m && isCameraAwareMaterial(m)) {
          materialManager.unregister(m);
        }
      }
    }
    this.nodeMap.clear();
    while (this.pickScene.children.length > 0) {
      this.pickScene.remove(this.pickScene.children[0]);
    }
    this._dirty = true;
  }

  /** Update camera reference (e.g., after perspective ↔ orthographic swap). */
  setCamera(camera: THREE.Camera): void {
    this.camera = camera;
    this._dirty = true; // Must re-render pick buffer with new projection
  }

  /** Set post-processing reference for lens distortion correction. */
  setPostProcessing(pp: PostProcessingManager | null): void {
    this.postProcessing = pp;
  }

  /** Invalidate the cached pick buffer. Call when camera, geometry, or viewport changes. */
  markDirty(): void {
    this._dirty = true;
    this._canvasRect = null; // Invalidate cached rect (may have resized)
  }

  /** Suppress or resume picking (e.g. during orbit/pan/zoom interactions). */
  suppress(value: boolean): void {
    this._suppressed = value;
    if (value && this.debounceTimer !== null) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
  }

  /**
   * Handle mouse move: pick immediately from cache, debounce re-renders.
   *
   * When the pick buffer is clean (cached), readback is ~0.1ms so we fire
   * immediately for instant tooltip response. When dirty (camera/geometry
   * changed), we debounce to avoid re-rendering on every frame during
   * orbit/pan/zoom — the render only fires once the mouse settles.
   */
  onMouseMove(event: MouseEvent): void {
    if (this._suppressed) return;

    // Cache getBoundingClientRect to avoid forced reflow on every mousemove
    if (!this._canvasRect) {
      this._canvasRect = this.renderer.domElement.getBoundingClientRect();
    }
    this.pendingMouse = {
      x: event.clientX - this._canvasRect.left,
      y: event.clientY - this._canvasRect.top,
    };

    if (this.debounceTimer !== null) {
      clearTimeout(this.debounceTimer);
    }

    if (!this._dirty) {
      // Buffer is cached — readback is instant, but throttle to ~60Hz
      // to avoid redundant picks when mouse fires faster than display refresh
      const now = performance.now();
      if (now - this._lastPickTime < 16) return;
      this._lastPickTime = now;
      this.performPick(this.pendingMouse.x, this.pendingMouse.y);
    } else {
      // Buffer needs re-render — debounce to avoid rendering during active interaction
      this.debounceTimer = setTimeout(() => {
        this.debounceTimer = null;
        if (this.pendingMouse) {
          this.performPick(this.pendingMouse.x, this.pendingMouse.y);
        }
      }, DEBOUNCE_MS);
    }
  }

  /** Clean up all resources — render target, pick materials, scene. */
  dispose(): void {
    if (this.debounceTimer !== null) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }

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
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    if (width === 0 || height === 0) return;

    // Pick buffer at half resolution (IDs don't need full res, 4× fewer pixels)
    // and additionally capped at MAX_PICK_BUFFER_DIM per axis so 4K/5K screens
    // do not pay full cost. Cursor coordinates rescale automatically below
    // because they are computed from pickW/width and pickH/height.
    const drawBuf = this.renderer.getDrawingBufferSize(this._drawBufSize);
    const { w: pickW, h: pickH } = computePickBufferSize(drawBuf.x, drawBuf.y);

    // Check if pick target needs resize (also triggers re-render)
    const sizeChanged = this.pickTarget.width !== pickW || this.pickTarget.height !== pickH;
    if (sizeChanged) {
      this.pickTarget.setSize(pickW, pickH);
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
      const uv = this.applyLensDistortion(screenX / width, screenY / height, lensParams);
      correctedX = uv.x * width;
      correctedY = uv.y * height;
    }

    // Compute cursor position in pick buffer pixels (half res), Y-flipped for WebGL
    const scaleX = pickW / width;
    const scaleY = pickH / height;
    const cursorX = Math.floor(correctedX * scaleX);
    const cursorY = pickH - Math.floor(correctedY * scaleY);

    const half = Math.floor(PICK_SIZE / 2);
    this._lastReadX = clamp(cursorX - half, 0, pickW - PICK_SIZE);
    this._lastReadY = clamp(cursorY - half, 0, pickH - PICK_SIZE);

    // Ray-BBox culling: quick check if cursor is near any node at all
    // Use corrected coordinates so the ray matches the undistorted pick buffer
    this.ndcCoord.set((correctedX / width) * 2 - 1, -(correctedY / height) * 2 + 1);
    this.raycaster.setFromCamera(this.ndcCoord, this.camera);
    const ray = this.raycaster.ray;

    let nearAnyNode = false;
    for (const entry of this.nodeMap.values()) {
      const geom = (entry.main as THREE.Mesh).geometry ?? (entry.main as THREE.Points).geometry;
      if (!geom || !geom.boundingBox) continue;
      this._box.copy(geom.boundingBox);
      this._box.applyMatrix4(entry.main.matrixWorld);
      if (ray.intersectsBox(this._box)) {
        nearAnyNode = true;
        break;
      }
    }

    if (!nearAnyNode) {
      this.onPickResult(null);
      return;
    }

    // Readback 5×5 pixels at cursor from cached buffer and vote.
    // The async readback path uses `readRenderTargetPixelsAsync`,
    // available on both WebGLRenderer and WebGPURenderer in r184 —
    // a uniform API that works on both backends. See
    // PICKING_DESIGN.md for the 1-frame-latency rationale.
    const result = await this.readbackAndVote();
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
      const mainGeom = (entry.main as THREE.Mesh).geometry ?? (entry.main as THREE.Points).geometry;
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
   * Apply Brown-Conrady lens distortion to UV coordinates.
   * TypeScript port of the GLSL `applyDistortion` in the mega-shader
   * fragment (`rendering/post-processing/mega-shader.glsl.ts`). Uses
   * the green-channel distortion (reference, no chromatic offset).
   *
   * This maps from distorted screen space to undistorted source space — exactly
   * what we need to convert mouse coords on the distorted display to pick buffer coords.
   * Writes result in-place to this._lensUV to avoid per-call allocation.
   */
  private applyLensDistortion(
    u: number,
    v: number,
    params: {
      distortion: THREE.Vector2;
      principalPoint: THREE.Vector2;
      focalLength: THREE.Vector2;
      skew: number;
    }
  ): { x: number; y: number } {
    // UV [0,1] → normalized [-1,1]
    const xn = 2.0 * (u - 0.5);
    const yn = 2.0 * (v - 0.5);

    // Brown-Conrady radial distortion: r' = r * (1 + k * r²)
    const r2 = xn * xn + yn * yn;
    const xd = (1.0 + params.distortion.x * r2) * xn;
    const yd = (1.0 + params.distortion.y * r2) * yn;

    // Camera intrinsic matrix K × distorted point → back to [0,1] UV
    const fx = params.focalLength.x;
    const fy = params.focalLength.y;

    this._lensUV.x = (fx * xd + params.skew * fx * yd + params.principalPoint.x) * 0.5 + 0.5;
    this._lensUV.y = (fy * yd + params.principalPoint.y) * 0.5 + 0.5;
    return this._lensUV;
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
    // Cast to WebGLRenderer resolves the union-signature clash —
    // WebGPURenderer's `readRenderTargetPixelsAsync` omits the
    // destBuffer arg in its TS signature but accepts the same call
    // shape at runtime (it ignores the extra arg and uses its own
    // internal buffer, returning it via the Promise; the existing
    // `this.readBuffer` is then filled by the WebGL path or stays
    // untouched on the WebGPU path — only the WebGL path is
    // currently exercised under the default renderer with
    // `forceWebGL` dropped because Playwright's chromium falls back
    // to the WebGL2 backend). Production code on a real WebGPU
    // adapter would need the Promise's return value, see TODO
    // below.
    // TODO: under real WebGPU dispatch, use the Promise return value
    // (a Uint8Array) instead of relying on `readBuffer` being filled.
    await (this.renderer as THREE.WebGLRenderer).readRenderTargetPixelsAsync(
      this.pickTarget,
      this._lastReadX,
      this._lastReadY,
      PICK_SIZE,
      PICK_SIZE,
      this.readBuffer
    );

    // Brightness-weighted majority voting
    // Use numeric key (nodeId * 2^24 + elementId) to avoid string allocation per pixel.
    // Both nodeId and elementId fit in 24 bits (float32 mantissa), so this is lossless.
    const votes = new Map<number, { nodeId: number; elementId: number; weight: number }>();

    for (let i = 0; i < PICK_SIZE * PICK_SIZE; i++) {
      const r = this.readBuffer[i * 4]; // nodeId
      const g = this.readBuffer[i * 4 + 1]; // elementId
      const b = this.readBuffer[i * 4 + 2]; // brightness

      // Skip background pixels (nodeId = 0 means no hit)
      if (r < 0.5) continue;

      const nodeId = Math.round(r);
      const elementId = Math.round(g);
      const key = nodeId * 16777216 + elementId; // nodeId << 24 | elementId (safe for 24-bit ints)

      const existing = votes.get(key);
      if (existing) {
        existing.weight += b;
      } else {
        votes.set(key, { nodeId, elementId, weight: b });
      }
    }

    // Find the winner (highest total brightness weight)
    let winner: { nodeId: number; elementId: number; weight: number } | null = null;
    for (const entry of votes.values()) {
      if (!winner || entry.weight > winner.weight) {
        winner = entry;
      }
    }

    if (!winner) return null;

    // Look up the main node
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
