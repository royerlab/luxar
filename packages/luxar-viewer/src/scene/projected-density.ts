/**
 * Projected-density tracker — elements per drawing-buffer pixel, per node.
 *
 * Frame cost on element-dense views tracks ELEMENTS PER PIXEL, not pixels:
 * the 2026-09 audit measured a 1.5 M-point example framed into ~1 600 px at
 * 41.7 ms per frame at DPR 1 and 83 ms at DPR 0.5 (the same fragments
 * concentrated into fewer tiles), while the same points dollied 4× closer
 * rendered at 120 fps. Nothing in the viewer measured that density, so the
 * adaptive-DPR controller stepped resolution DOWN on exactly those scenes.
 *
 * This module measures it once per frame for every committed emissive data mesh
 * (points, lines, and gsplats; shaded triangle meshes are deliberately excluded):
 * the node's `geometry.boundingSphere` is projected to a screen ellipse in
 * drawing-buffer pixels (the buffer, not CSS pixels — otherwise the guard
 * would fight the DPR controller), and divided into the node's visible
 * element count. Consumers: the shader keep-fraction ladder (thinning),
 * the refinement rung cap, and `__luxarDebug.getPerf().density`.
 *
 * Hot-path invariants, same as `lod-group-registry.ts`: no per-frame
 * allocation (module scratches, a bound visitor, per-path records mutated
 * in place) and no scene mutation.
 *
 * @module scene/projected-density
 */

import * as THREE from 'three';
import { readVisibleElementCount } from '../data/scene-loader/monitor/visible-counts';
import { hasCommittedData } from '../types/committed-data';
import { isEffectivelyVisible } from '../utils/object-visibility';

/** One node's projected footprint for the last evaluated frame. */
export interface NodeDensity {
  path: string;
  /** Projected bounding-sphere area, drawing-buffer px, clipped to the buffer; 0 when off-screen. */
  areaPx: number;
  /** Visible (committed) element count. */
  elements: number;
  /** `elements / max(areaPx, 1)`; 0 when off-screen. */
  elementsPerPixel: number;
  onScreen: boolean;
  /** Frame stamp of the last evaluation that saw this node. */
  frame: number;
  /**
   * Keep fraction the density guard currently applies to this node (1 = no
   * thinning). Written by `scene/density-guard.ts`; 1 until it runs.
   */
  keep: number;
  /**
   * Whether the node's blend mode sums energy (additive / luminous /
   * volumetric). Written by the density guard; true until it runs, since the
   * blendable cap is the more permissive one.
   */
  blendable: boolean;
}

export interface ProjectedDensityDeps {
  enabled(): boolean;
  getRoot(): THREE.Object3D | null;
  getCamera(): THREE.Camera | null;
  /** Drawing-buffer size in physical pixels (`renderer.domElement.width/height`). */
  getDrawingBufferSize(): { width: number; height: number } | null;
  /**
   * Called for every committed data mesh after its record is refreshed (on-
   * or off-screen). The density guard hooks here so it sees the mesh itself,
   * which the path-keyed records deliberately do not retain.
   */
  onVisit?(mesh: THREE.Mesh, record: NodeDensity): void;
}

const VIEW_SCRATCH = new THREE.Matrix4();
const CENTER_SCRATCH = new THREE.Vector3();
const SCALE_SCRATCH = new THREE.Vector3();

/** Whether the object is a committed emissive geometry node tracked by the guard. */
function isTrackedDataMesh(obj: THREE.Object3D): obj is THREE.Mesh {
  const mesh = obj as THREE.Mesh;
  return Boolean(
    mesh.isMesh && mesh.userData.nodeType !== 'mesh' && mesh.name && hasCommittedData(mesh)
  );
}

/**
 * Project a view-space sphere to its screen ellipse area in buffer pixels.
 *
 * Perspective: radii scale by `P[0]/depth` and `P[5]/depth` (the projection
 * matrix's focal terms); a camera inside or in front of the sphere
 * (`depth <= radius`) counts as full-buffer. Orthographic: the focal terms
 * apply without the depth division. Off-screen (the ellipse does not touch
 * the NDC square) returns 0; otherwise the ellipse area, clipped to the
 * buffer. The clip is a coarse min, not an exact intersection — the guard
 * only needs the order of magnitude.
 */
export function projectSphereAreaPx(
  centerView: { x: number; y: number; z: number },
  radius: number,
  camera: THREE.Camera,
  width: number,
  height: number
): { areaPx: number; onScreen: boolean } {
  const p = camera.projectionMatrix.elements;
  const full = width * height;
  let rx: number;
  let ry: number;
  let cx: number;
  let cy: number;
  if ((camera as THREE.PerspectiveCamera).isPerspectiveCamera) {
    const depth = -centerView.z;
    if (depth <= radius) return { areaPx: full, onScreen: true };
    rx = (radius * p[0]) / depth;
    ry = (radius * p[5]) / depth;
    cx = (centerView.x * p[0]) / depth;
    cy = (centerView.y * p[5]) / depth;
  } else {
    rx = radius * p[0];
    ry = radius * p[5];
    cx = centerView.x * p[0] + p[12];
    cy = centerView.y * p[5] + p[13];
  }
  if (Math.abs(cx) > 1 + rx || Math.abs(cy) > 1 + ry) return { areaPx: 0, onScreen: false };
  const areaPx = Math.min(full, Math.PI * (rx * width * 0.5) * (ry * height * 0.5));
  return { areaPx, onScreen: true };
}

export class ProjectedDensityTracker {
  private deps: ProjectedDensityDeps | null = null;
  private readonly byPath = new Map<string, NodeDensity>();
  private frame = 0;
  private width = 0;
  private height = 0;
  private camera: THREE.Camera | null = null;
  private readonly visit = (obj: THREE.Object3D): void => this.visitObject(obj);

  configure(deps: ProjectedDensityDeps): void {
    this.deps = deps;
  }

  /**
   * Evaluate every committed data mesh under the root. Returns true when an
   * evaluation ran (guard enabled, root/camera/buffer available).
   */
  evaluate(): boolean {
    const deps = this.deps;
    if (!deps || !deps.enabled()) return false;
    const root = deps.getRoot();
    const camera = deps.getCamera();
    const size = deps.getDrawingBufferSize();
    if (!root || !camera || !size || size.width <= 0 || size.height <= 0) return false;
    this.frame += 1;
    this.width = size.width;
    this.height = size.height;
    this.camera = camera;
    camera.updateMatrixWorld();
    VIEW_SCRATCH.copy(camera.matrixWorld).invert();
    root.traverse(this.visit);
    // Drop records for nodes that no longer exist (disposed / dataset switch).
    for (const [path, rec] of this.byPath) {
      if (rec.frame !== this.frame) this.byPath.delete(path);
    }
    return true;
  }

  private visitObject(obj: THREE.Object3D): void {
    if (!isTrackedDataMesh(obj)) return;
    const mesh = obj;
    const bs = (mesh.geometry as THREE.BufferGeometry | undefined)?.boundingSphere;
    if (!bs || !this.camera) return;
    const rec = this.recordFor(mesh.name);
    rec.frame = this.frame;
    rec.elements = readVisibleElementCount(mesh.userData) ?? 0;
    this.measureIfVisible(mesh, bs, this.camera, rec);
    this.deps?.onVisit?.(mesh, rec);
  }

  private measureIfVisible(
    mesh: THREE.Mesh,
    bs: THREE.Sphere,
    camera: THREE.Camera,
    rec: NodeDensity
  ): void {
    if (isEffectivelyVisible(mesh)) {
      this.measure(mesh, bs, camera, rec);
      return;
    }
    rec.areaPx = 0;
    rec.elementsPerPixel = 0;
    rec.onScreen = false;
  }

  private recordFor(path: string): NodeDensity {
    let rec = this.byPath.get(path);
    if (!rec) {
      rec = {
        path,
        areaPx: 0,
        elements: 0,
        elementsPerPixel: 0,
        onScreen: false,
        frame: 0,
        keep: 1,
        blendable: true,
      };
      this.byPath.set(path, rec);
    }
    return rec;
  }

  private measure(
    mesh: THREE.Mesh,
    bs: THREE.Sphere,
    camera: THREE.Camera,
    rec: NodeDensity
  ): void {
    CENTER_SCRATCH.copy(bs.center).applyMatrix4(mesh.matrixWorld).applyMatrix4(VIEW_SCRATCH);
    SCALE_SCRATCH.setFromMatrixScale(mesh.matrixWorld);
    const radius = bs.radius * Math.max(SCALE_SCRATCH.x, SCALE_SCRATCH.y, SCALE_SCRATCH.z);
    const { areaPx, onScreen } = projectSphereAreaPx(
      CENTER_SCRATCH,
      radius,
      camera,
      this.width,
      this.height
    );
    rec.areaPx = areaPx;
    rec.onScreen = onScreen;
    rec.elementsPerPixel = onScreen ? rec.elements / Math.max(areaPx, 1) : 0;
  }

  /** Last record for a node path, or undefined when never seen / pruned. */
  get(path: string): NodeDensity | undefined {
    return this.byPath.get(path);
  }

  /** Copy of every record, for `getPerf().density`. */
  snapshot(): Record<string, NodeDensity> {
    const out: Record<string, NodeDensity> = {};
    for (const [path, rec] of this.byPath) out[path] = { ...rec };
    return out;
  }

  /** Forget every record (tests, dataset teardown). */
  reset(): void {
    this.byPath.clear();
    this.frame = 0;
  }
}

/** Config master switch AND the per-session option (`?no-density-guard` → false). */
export function resolveDensityGuardEnabled(
  configEnabled: boolean,
  option: boolean | undefined
): boolean {
  return configEnabled && (option ?? true);
}

const tracker = new ProjectedDensityTracker();

/** The app-wide tracker; configured once by the init pipeline. */
export function getProjectedDensityTracker(): ProjectedDensityTracker {
  return tracker;
}

/** `getPerf().density` provider. */
export function snapshotProjectedDensity(): Record<string, NodeDensity> {
  return tracker.snapshot();
}
