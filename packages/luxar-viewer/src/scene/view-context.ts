/**
 * The frame's view: one snapshot of the camera that the view-dependent
 * per-frame work (LOD selection, projected density) reads instead
 * of each re-deriving it from the live camera.
 *
 * Built lazily, on the first `get()` after an `invalidate()`. The
 * `dynamic-clipping` callback is the first `view`-phase callback and the only
 * camera write in that phase; it invalidates every frame before updating
 * near/far. Later view callbacks therefore build from the current projection.
 * Reads in the earlier `camera` phase or outside the frame loop may still see
 * the previous snapshot (depth sort deliberately reads the camera directly).
 * The returned object and its matrices are reused, so consumers must not keep
 * references across frames.
 *
 * The view matrix is `inverse(camera.matrixWorld)`, derived here rather than
 * read from `camera.matrixWorldInverse`, which three builds with the scale
 * removed; depth sort and projected density have always used the plain
 * inverse, and keeping it makes their switch to this snapshot bit-identical.
 */

import * as THREE from 'three';

/** Width and height, in the unit named by the field that holds it. */
export interface ViewSize {
  width: number;
  height: number;
}

/** One frame's camera snapshot. Read-only for consumers. */
export interface ViewContext {
  /** The camera the snapshot was taken from (replaced on an ortho/perspective swap). */
  readonly camera: THREE.Camera;
  /** Camera position in world space. */
  readonly cameraWorldPosition: THREE.Vector3;
  /** Unit view direction in world space (the camera's -Z). */
  readonly viewDirection: THREE.Vector3;
  /** World → view: `inverse(camera.matrixWorld)`. */
  readonly viewMatrix: THREE.Matrix4;
  /** The camera's projection matrix, as the frame renders with it. */
  readonly projectionMatrix: THREE.Matrix4;
  /** `projectionMatrix × viewMatrix`: world → clip. */
  readonly projView: THREE.Matrix4;
  /** Frustum of `projView`. */
  readonly frustum: THREE.Frustum;
  /** True for a parallel projection (`P[3][3] === 1`). */
  readonly isOrtho: boolean;
  /** Canvas size in CSS pixels, or null when the canvas has no layout size. */
  readonly viewportCss: ViewSize | null;
  /** Drawing-buffer size in physical pixels, or null when it is empty. */
  readonly drawingBuffer: ViewSize | null;
}

/** Where a {@link ViewContextProvider} reads the live view from. */
export interface ViewContextDeps {
  getCamera(): THREE.Camera;
  /** CSS-pixel canvas size; null (or a zero dimension) when collapsed. */
  getViewportCss(): ViewSize | null;
  /** Physical-pixel drawing-buffer size; null (or a zero dimension) when empty. */
  getDrawingBuffer(): ViewSize | null;
}

/** The provider's writable view of a {@link ViewContext} (consumers get the read-only one). */
export interface MutableViewContext {
  camera: THREE.Camera;
  cameraWorldPosition: THREE.Vector3;
  viewDirection: THREE.Vector3;
  viewMatrix: THREE.Matrix4;
  projectionMatrix: THREE.Matrix4;
  projView: THREE.Matrix4;
  frustum: THREE.Frustum;
  isOrtho: boolean;
  viewportCss: ViewSize | null;
  drawingBuffer: ViewSize | null;
}

function usableSize(size: ViewSize | null, into: ViewSize): ViewSize | null {
  if (!size || !(size.width > 0) || !(size.height > 0)) return null;
  into.width = size.width;
  into.height = size.height;
  return into;
}

/** Builds and caches the frame's {@link ViewContext}. */
export class ViewContextProvider {
  private readonly ctx: MutableViewContext;
  private readonly cssScratch: ViewSize = { width: 0, height: 0 };
  private readonly bufferScratch: ViewSize = { width: 0, height: 0 };
  private valid = false;

  constructor(private readonly deps: ViewContextDeps) {
    this.ctx = {
      camera: deps.getCamera(),
      cameraWorldPosition: new THREE.Vector3(),
      viewDirection: new THREE.Vector3(),
      viewMatrix: new THREE.Matrix4(),
      projectionMatrix: new THREE.Matrix4(),
      projView: new THREE.Matrix4(),
      frustum: new THREE.Frustum(),
      isOrtho: false,
      viewportCss: null,
      drawingBuffer: null,
    };
  }

  /** Mark the snapshot stale; the next `get()` rebuilds it. */
  invalidate(): void {
    this.valid = false;
  }

  /** The current frame's snapshot, built now if stale. */
  get(): ViewContext {
    if (!this.valid) this.build();
    return this.ctx;
  }

  private build(): void {
    const ctx = this.ctx;
    const camera = this.deps.getCamera();
    ctx.camera = camera;
    camera.updateMatrixWorld();
    ctx.viewMatrix.copy(camera.matrixWorld).invert();
    ctx.cameraWorldPosition.setFromMatrixPosition(camera.matrixWorld);
    const e = camera.matrixWorld.elements;
    ctx.viewDirection.set(-e[8], -e[9], -e[10]).normalize();
    ctx.projectionMatrix.copy(camera.projectionMatrix);
    ctx.projView.multiplyMatrices(ctx.projectionMatrix, ctx.viewMatrix);
    ctx.frustum.setFromProjectionMatrix(ctx.projView);
    ctx.isOrtho = ctx.projectionMatrix.elements[15] === 1;
    ctx.viewportCss = usableSize(this.deps.getViewportCss(), this.cssScratch);
    ctx.drawingBuffer = usableSize(this.deps.getDrawingBuffer(), this.bufferScratch);
    this.valid = true;
  }
}
