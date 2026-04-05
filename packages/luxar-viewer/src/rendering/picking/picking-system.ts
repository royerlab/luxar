/**
 * GPU Picking System for Luxar.
 *
 * Orchestrates the picking pipeline:
 * 1. Debounced mousemove triggers a pick
 * 2. Ray-BBox culling filters candidate nodes
 * 3. 5x5 scissored render to RGBA32F pick buffer
 * 4. Readback + brightness-weighted majority voting
 * 5. Callback with winning (nodeId, elementId) or null
 *
 * The pick buffer encodes: R=nodeId, G=elementId, B=brightness, A=1.0
 * Brightness-as-depth (gl_FragDepth = 1 - brightness) ensures the
 * brightest element at each pixel wins the depth test.
 *
 * Zero impact on the main render loop — the pick pass only fires when
 * the mouse has been stationary for ~100ms, and renders at most 25 pixels.
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

  constructor(
    private renderer: THREE.WebGLRenderer,
    private camera: THREE.Camera,
    private onPickResult: (result: PickResult | null) => void
  ) {
    this.pickScene = new THREE.Scene();
    // No background — pick buffer clears to (0,0,0,0) which means "no hit"

    this.pickTarget = new THREE.WebGLRenderTarget(PICK_SIZE, PICK_SIZE, {
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
   * Perform a full pick at the given screen coordinates.
   */
  private performPick(screenX: number, screenY: number): void {
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    if (width === 0 || height === 0) return;

    // Convert screen coords to NDC [-1, +1]
    this.ndcCoord.set((screenX / width) * 2 - 1, -(screenY / height) * 2 + 1);

    // Build ray from camera through cursor
    this.raycaster.setFromCamera(this.ndcCoord, this.camera);
    const ray = this.raycaster.ray;

    // Ray-BBox culling: find candidate nodes
    const candidates: PickNodeEntry[] = [];
    for (const entry of this.nodeMap.values()) {
      const pickObj = entry.pick;

      // Get the bounding box from the geometry
      const geom = (pickObj as THREE.Mesh).geometry ?? (pickObj as THREE.Points).geometry;
      if (!geom || !geom.boundingBox) continue;

      // Transform bounding box to world space
      this._box.copy(geom.boundingBox);

      // Sync the pick node's world matrix from the main node
      entry.pick.matrixWorld.copy(entry.main.matrixWorld);
      this._box.applyMatrix4(entry.pick.matrixWorld);

      if (ray.intersectsBox(this._box)) {
        candidates.push(entry);
      }
    }

    if (candidates.length === 0) {
      this.onPickResult(null);
      return;
    }

    // Render candidates to pick buffer
    this.renderPickBuffer(candidates);

    // Read back and vote
    const result = this.readbackAndVote();
    this.onPickResult(result);
  }

  /**
   * Render candidate pick nodes to the 5x5 pick buffer.
   * Temporarily adds candidates to the pick scene, renders, then removes them.
   */
  private renderPickBuffer(candidates: PickNodeEntry[]): void {
    const renderer = this.renderer;

    // Save renderer state
    const savedRenderTarget = renderer.getRenderTarget();
    const savedScissorTest = renderer.getScissorTest();

    // Add candidates to pick scene
    for (const entry of candidates) {
      this.pickScene.add(entry.pick);
    }

    // Render to pick target (full 5x5).
    // TODO: implement sub-frustum projection to resolve per-element picking.
    // Currently the 5x5 buffer represents the entire viewport, so picking
    // identifies the correct node but may not distinguish individual elements
    // when multiple overlap.
    renderer.setRenderTarget(this.pickTarget);
    renderer.setClearColor(0x000000, 0);
    renderer.clear(true, true, false);
    renderer.render(this.pickScene, this.camera);

    // Remove candidates from pick scene
    for (const entry of candidates) {
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
      0,
      0,
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
