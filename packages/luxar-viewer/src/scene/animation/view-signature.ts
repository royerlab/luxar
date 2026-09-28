/**
 * The camera half of the render-on-change decision: has the view that the
 * next render would draw changed since the last frame that was actually
 * rendered?
 *
 * A view is the camera object itself (the ortho toggle swaps it), its world
 * matrix, its projection matrix and the drawing-buffer size (a resize or a
 * DPR step clears the canvas and changes every pixel). Those 34 numbers are
 * compared against the copy taken at the last render, in a preallocated
 * `Float64Array` — the check runs on every loop tick, so it allocates
 * nothing.
 *
 * `camera.updateMatrixWorld()` runs before the read. The render calls it
 * anyway (and the per-frame depth-sort pass already does), so it changes no
 * pixel; without it a camera moved through `position` / `quaternion` since
 * the last render would still report the stale world matrix of that render.
 *
 * @module scene/animation/view-signature
 */

import type * as THREE from 'three';

/** Where the signature reads the live view from. Every accessor is called per tick. */
export interface ViewSignatureSource {
  /** The camera the next render draws with (null before the renderer exists). */
  getCamera(): THREE.Camera | null;
  /** The canvas whose drawing buffer the render fills (null when absent). */
  getDrawingBuffer(): { width: number; height: number } | null;
}

/** Number of values in a signature: 16 world + 16 projection + width + height. */
const SIGNATURE_LENGTH = 34;

export class ViewSignature {
  private readonly current = new Float64Array(SIGNATURE_LENGTH);
  private readonly rendered = new Float64Array(SIGNATURE_LENGTH);
  private currentCamera: THREE.Camera | null = null;
  private renderedCamera: THREE.Camera | null = null;
  /** False until a render was committed (and after {@link invalidate}). */
  private hasRendered = false;

  constructor(private readonly source: ViewSignatureSource) {}

  /**
   * Read the live view and report whether it differs from the last committed
   * (rendered) one. Always true before the first commit.
   */
  changed(): boolean {
    this.read();
    if (!this.hasRendered || this.currentCamera !== this.renderedCamera) return true;
    const a = this.current;
    const b = this.rendered;
    for (let i = 0; i < SIGNATURE_LENGTH; i++) {
      if (a[i] !== b[i]) return true;
    }
    return false;
  }

  /**
   * Record the live view as the one just rendered. Reads it afresh, so it is
   * correct whether or not {@link changed} ran this tick.
   */
  commit(): void {
    this.read();
    this.rendered.set(this.current);
    this.renderedCamera = this.currentCamera;
    this.hasRendered = true;
  }

  /** Forget the rendered view: the next {@link changed} reports true. */
  invalidate(): void {
    this.hasRendered = false;
  }

  private read(): void {
    const camera = this.source.getCamera();
    const buffer = this.source.getDrawingBuffer();
    this.currentCamera = camera;
    const out = this.current;
    if (camera) {
      camera.updateMatrixWorld();
      out.set(camera.matrixWorld.elements, 0);
      out.set(camera.projectionMatrix.elements, 16);
    } else {
      out.fill(0, 0, 32);
    }
    out[32] = buffer?.width ?? 0;
    out[33] = buffer?.height ?? 0;
  }
}
