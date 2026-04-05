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
import { log, Modules } from '../../utils/log';

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

/** Debounce delay in milliseconds before triggering a pick. */
const DEBOUNCE_MS = 100;

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

  // Cache: only re-render when the view changes
  private _dirty = true;
  private _drawBufSize = new THREE.Vector2();

  constructor(
    private renderer: THREE.WebGLRenderer,
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
    this.nodeMap.set(pickId, { main: mainNode, pick: pickNode });
  }

  /** Unregister a node by its pick ID. */
  unregisterNode(pickId: number): void {
    this.nodeMap.delete(pickId);
  }

  /** Number of registered pick nodes. */
  get registeredNodeCount(): number {
    return this.nodeMap.size;
  }

  /** Invalidate the cached pick buffer. Call when camera, geometry, or viewport changes. */
  markDirty(): void {
    this._dirty = true;
  }

  /**
   * Handle mouse move: debounce and schedule a pick.
   * Called directly from the canvas mousemove listener.
   */
  onMouseMove(event: MouseEvent): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pendingMouse = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };

    if (this.debounceTimer !== null) {
      clearTimeout(this.debounceTimer);
    }
    this.debounceTimer = setTimeout(() => {
      this.debounceTimer = null;
      if (this.pendingMouse) {
        this.performPick(this.pendingMouse.x, this.pendingMouse.y);
      }
    }, DEBOUNCE_MS);
  }

  /** Clean up all resources. */
  dispose(): void {
    if (this.debounceTimer !== null) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    this.pickTarget.dispose();
    this.nodeMap.clear();
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
  private performPick(screenX: number, screenY: number): void {
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    if (width === 0 || height === 0) return;

    // Pick buffer at half resolution (IDs don't need full res, 4× fewer pixels)
    const drawBuf = this.renderer.getDrawingBufferSize(this._drawBufSize);
    const pickW = Math.max(1, Math.floor(drawBuf.x / 2));
    const pickH = Math.max(1, Math.floor(drawBuf.y / 2));

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

    // Compute cursor position in pick buffer pixels (half res), Y-flipped for WebGL
    const scaleX = pickW / width;
    const scaleY = pickH / height;
    const cursorX = Math.floor(screenX * scaleX);
    const cursorY = pickH - Math.floor(screenY * scaleY);

    const half = Math.floor(PICK_SIZE / 2);
    this._lastReadX = Math.max(0, Math.min(cursorX - half, pickW - PICK_SIZE));
    this._lastReadY = Math.max(0, Math.min(cursorY - half, pickH - PICK_SIZE));

    // Ray-BBox culling: quick check if cursor is near any node at all
    this.ndcCoord.set((screenX / width) * 2 - 1, -(screenY / height) * 2 + 1);
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

    // Readback 5×5 pixels at cursor from cached buffer and vote
    const result = this.readbackAndVote();
    this.onPickResult(result);
  }

  /**
   * Render ALL registered pick nodes to the cached pick buffer.
   * Called only when the buffer is dirty (camera/geometry/resize changed).
   * Renders at half resolution for performance — pick IDs don't need full res.
   */
  private renderPickBuffer(): void {
    const renderer = this.renderer;

    // Save renderer state
    const savedRenderTarget = renderer.getRenderTarget();
    const savedScissorTest = renderer.getScissorTest();

    // Sync and add ALL registered nodes to pick scene
    for (const entry of this.nodeMap.values()) {
      // Sync geometry (main node's geometry may have been replaced by view updates)
      const mainGeom = (entry.main as THREE.Mesh).geometry ?? (entry.main as THREE.Points).geometry;
      if (mainGeom) {
        (entry.pick as THREE.Mesh).geometry = mainGeom;
      }
      // Sync world transform
      entry.pick.matrixWorld.copy(entry.main.matrixWorld);
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

    // Restore renderer state
    renderer.setRenderTarget(savedRenderTarget);
    renderer.setScissorTest(savedScissorTest);
  }

  /**
   * Read back the 5x5 pick buffer and perform brightness-weighted majority voting.
   * Returns the winning PickResult or null if all pixels are background.
   */
  private readbackAndVote(): PickResult | null {
    this.renderer.readRenderTargetPixels(
      this.pickTarget,
      this._lastReadX,
      this._lastReadY,
      PICK_SIZE,
      PICK_SIZE,
      this.readBuffer
    );

    // Brightness-weighted majority voting
    const votes = new Map<string, { nodeId: number; elementId: number; weight: number }>();

    for (let i = 0; i < PICK_SIZE * PICK_SIZE; i++) {
      const r = this.readBuffer[i * 4]; // nodeId
      const g = this.readBuffer[i * 4 + 1]; // elementId
      const b = this.readBuffer[i * 4 + 2]; // brightness

      // Skip background pixels (nodeId = 0 means no hit)
      if (r < 0.5) continue;

      const nodeId = Math.round(r);
      const elementId = Math.round(g);
      const key = `${nodeId}:${elementId}`;

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
