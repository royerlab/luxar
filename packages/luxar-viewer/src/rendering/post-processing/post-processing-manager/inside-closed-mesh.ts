/**
 * Whether a world-space point lies inside a closed mesh — the test the refraction split
 * uses to tell that the camera is inside a refracting shell.
 *
 * Bounding-sphere and box rejection first (the common case: the camera is outside every
 * glass), then RAY PARITY against the drawn triangles: a ray from the point crosses
 * a closed surface an odd number of times when the point is inside. This works for
 * non-convex shells too — unlike a bounding-volume test, which would call a
 * camera inside the bounding sphere of a lens or a torus "inside" while it can still see
 * the front faces. The ray direction is deliberately irrational so it does not graze a
 * shared edge or vertex of an axis-aligned or icosahedral mesh, where a hit could be
 * counted twice. Only closed surfaces are eligible: every edge of the drawn triangles
 * must have two incident faces, with seam vertices welded by rounded position.
 * Closedness is cached by geometry attribute versions and draw range; parity is cached
 * per mesh, camera position and world matrix. Above 20,000 triangles the glass stays
 * double-sided rather than doing a large topology scan in a render frame.
 *
 * @module rendering/post-processing/post-processing-manager/inside-closed-mesh
 */

import * as THREE from 'three';

const _sphere = new THREE.Sphere();
const _box = new THREE.Box3();
const _inverse = new THREE.Matrix4();
const _ray = new THREE.Ray(
  new THREE.Vector3(),
  new THREE.Vector3(0.8017, 0.4954, 0.3343).normalize()
);
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c = new THREE.Vector3();
const _hit = new THREE.Vector3();
const MAX_TOPOLOGY_TRIANGLES = 20_000;

interface GeometryState {
  position: THREE.BufferAttribute;
  positionVersion: number;
  index: THREE.BufferAttribute | null;
  indexVersion: number;
  start: number;
  end: number;
  closed: boolean;
}

interface PointState {
  geometry: GeometryState;
  eye: THREE.Vector3;
  worldMatrix: THREE.Matrix4;
  inside: boolean;
}

const geometryStates = new WeakMap<THREE.BufferGeometry, GeometryState>();
const pointStates = new WeakMap<THREE.Mesh, PointState>();

function drawnRange(geometry: THREE.BufferGeometry, count: number): [number, number] {
  const start = Math.min(count, Math.max(0, Math.ceil(geometry.drawRange.start)));
  const end = Math.min(count, start + geometry.drawRange.count);
  return [start, start + Math.floor((end - start) / 3) * 3];
}

function weldedVertex(
  position: THREE.BufferAttribute,
  index: number,
  tolerance: number,
  vertices: Map<string, number>
): number {
  const x = position.getX(index);
  const y = position.getY(index);
  const z = position.getZ(index);
  if (!Number.isFinite(x + y + z)) return -1;
  const key = `${Math.round(x / tolerance)},${Math.round(y / tolerance)},${Math.round(z / tolerance)}`;
  let id = vertices.get(key);
  if (id !== undefined) return id;
  id = vertices.size;
  vertices.set(key, id);
  return id;
}

function addEdges(edges: Map<string, number>, a: number, b: number, c: number): boolean {
  for (const [u, v] of [
    [a, b],
    [b, c],
    [c, a],
  ]) {
    const key = u < v ? `${u},${v}` : `${v},${u}`;
    const count = (edges.get(key) ?? 0) + 1;
    if (count > 2) return false;
    edges.set(key, count);
  }
  return true;
}

function isClosedSurface(
  position: THREE.BufferAttribute,
  index: THREE.BufferAttribute | null,
  start: number,
  end: number,
  box: THREE.Box3
): boolean {
  if (end - start < 12 || (end - start) / 3 > MAX_TOPOLOGY_TRIANGLES) return false;
  const extent = box.getSize(new THREE.Vector3()).length();
  const tolerance = Math.max(extent * 1e-6, 1e-12);
  const welded = new Map<string, number>();
  const vertices = new Map<number, number>();
  const edges = new Map<string, number>();
  const corner = (k: number): number => (index ? index.getX(k) : k);
  const vertex = (k: number): number => {
    const i = corner(k);
    let id = vertices.get(i);
    if (id === undefined) {
      id = weldedVertex(position, i, tolerance, welded);
      vertices.set(i, id);
    }
    return id;
  };
  for (let k = start; k < end; k += 3) {
    const a = vertex(k);
    const b = vertex(k + 1);
    const c = vertex(k + 2);
    if (a < 0 || a === b || b === c || c === a) return false;
    if (!addEdges(edges, a, b, c)) return false;
  }
  return [...edges.values()].every((count) => count === 2);
}

function sameGeometryState(
  prior: GeometryState | undefined,
  position: THREE.BufferAttribute,
  index: THREE.BufferAttribute | null,
  start: number,
  end: number
): prior is GeometryState {
  return (
    prior !== undefined &&
    prior.position === position &&
    prior.positionVersion === position.version &&
    prior.index === index &&
    prior.indexVersion === (index?.version ?? 0) &&
    prior.start === start &&
    prior.end === end
  );
}

function geometryState(
  geometry: THREE.BufferGeometry,
  position: THREE.BufferAttribute
): GeometryState {
  const index = geometry.getIndex();
  const [start, end] = drawnRange(geometry, index ? index.count : position.count);
  const prior = geometryStates.get(geometry);
  if (sameGeometryState(prior, position, index, start, end)) return prior;
  const state = {
    position,
    positionVersion: position.version,
    index,
    indexVersion: index?.version ?? 0,
    start,
    end,
    closed: isClosedSurface(position, index, start, end, geometry.boundingBox!),
  };
  geometryStates.set(geometry, state);
  return state;
}

function withinBounds(mesh: THREE.Mesh, worldPoint: THREE.Vector3): boolean {
  const geometry = mesh.geometry;
  if (!geometry.boundingSphere) geometry.computeBoundingSphere();
  if (!geometry.boundingSphere) return false;
  _sphere.copy(geometry.boundingSphere).applyMatrix4(mesh.matrixWorld);
  if (!_sphere.containsPoint(worldPoint)) return false;
  if (!geometry.boundingBox) geometry.computeBoundingBox();
  if (!geometry.boundingBox) return false;
  _box.copy(geometry.boundingBox).applyMatrix4(mesh.matrixWorld);
  return _box.containsPoint(worldPoint);
}

/**
 * @param mesh - A mesh whose world matrix is current.
 * @param worldPoint - The point to classify (the camera position).
 * @returns True when the point is strictly inside the closed surface.
 */
export function isPointInsideClosedMesh(mesh: THREE.Mesh, worldPoint: THREE.Vector3): boolean {
  const geometry = mesh.geometry;
  const position = geometry.getAttribute('position') as THREE.BufferAttribute | undefined;
  if (!position || position.count < 3) return false;
  if (!withinBounds(mesh, worldPoint)) return false;

  const state = geometryState(geometry, position);
  if (!state.closed) return false;
  const prior = pointStates.get(mesh);
  if (
    prior?.geometry === state &&
    prior.eye.equals(worldPoint) &&
    prior.worldMatrix.equals(mesh.matrixWorld)
  )
    return prior.inside;

  _inverse.copy(mesh.matrixWorld).invert();
  _ray.origin.copy(worldPoint).applyMatrix4(_inverse);
  const inside = countCrossings(position, state.index, state.start, state.end) % 2 === 1;
  pointStates.set(mesh, {
    geometry: state,
    eye: worldPoint.clone(),
    worldMatrix: mesh.matrixWorld.clone(),
    inside,
  });
  return inside;
}

/** How many of the geometry's triangles `_ray` (local space) passes through. */
function countCrossings(
  position: THREE.BufferAttribute,
  index: THREE.BufferAttribute | null,
  start: number,
  end: number
): number {
  const corner = (k: number): number => (index ? index.getX(k) : k);
  let crossings = 0;
  for (let k = start; k < end; k += 3) {
    _a.fromBufferAttribute(position, corner(k));
    _b.fromBufferAttribute(position, corner(k + 1));
    _c.fromBufferAttribute(position, corner(k + 2));
    if (_ray.intersectTriangle(_a, _b, _c, false, _hit)) crossings++;
  }
  return crossings;
}
