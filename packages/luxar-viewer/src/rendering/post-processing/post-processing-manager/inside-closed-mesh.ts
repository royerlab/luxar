/**
 * Whether a world-space point lies inside a closed mesh — the test the refraction split
 * uses to tell that the camera is inside a refracting shell.
 *
 * A cheap bounding-sphere rejection first (the common case: the camera is outside every
 * glass), then RAY PARITY against the mesh's own triangles: a ray from the point crosses
 * a closed surface an odd number of times exactly when the point is inside. That is exact
 * for any closed mesh, convex or not — unlike a bounding-volume test, which would call a
 * camera inside the bounding sphere of a lens or a torus "inside" while it can still see
 * the front faces. The ray direction is deliberately irrational so it does not graze a
 * shared edge or vertex of an axis-aligned or icosahedral mesh, where a hit could be
 * counted twice. An open mesh has no inside; parity then answers something arbitrary,
 * which is why the caller only asks about meshes it treats as closed shells.
 *
 * @module rendering/post-processing/post-processing-manager/inside-closed-mesh
 */

import * as THREE from 'three';

const _sphere = new THREE.Sphere();
const _inverse = new THREE.Matrix4();
const _ray = new THREE.Ray(
  new THREE.Vector3(),
  new THREE.Vector3(0.8017, 0.4954, 0.3343).normalize()
);
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c = new THREE.Vector3();
const _hit = new THREE.Vector3();

/**
 * @param mesh - A mesh whose world matrix is current.
 * @param worldPoint - The point to classify (the camera position).
 * @returns True when the point is strictly inside the closed surface.
 */
export function isPointInsideClosedMesh(mesh: THREE.Mesh, worldPoint: THREE.Vector3): boolean {
  const geometry = mesh.geometry;
  const position = geometry.getAttribute('position') as THREE.BufferAttribute | undefined;
  if (!position || position.count < 3) return false;
  if (!geometry.boundingSphere) geometry.computeBoundingSphere();
  if (!geometry.boundingSphere) return false;
  _sphere.copy(geometry.boundingSphere).applyMatrix4(mesh.matrixWorld);
  if (!_sphere.containsPoint(worldPoint)) return false;

  _inverse.copy(mesh.matrixWorld).invert();
  _ray.origin.copy(worldPoint).applyMatrix4(_inverse);
  return countCrossings(position, geometry.getIndex()) % 2 === 1;
}

/** How many of the geometry's triangles `_ray` (local space) passes through. */
function countCrossings(
  position: THREE.BufferAttribute,
  index: THREE.BufferAttribute | null
): number {
  const corner = (k: number): number => (index ? index.getX(k) : k);
  const triangles = Math.floor((index ? index.count : position.count) / 3);
  let crossings = 0;
  for (let t = 0; t < triangles; t++) {
    _a.fromBufferAttribute(position, corner(3 * t));
    _b.fromBufferAttribute(position, corner(3 * t + 1));
    _c.fromBufferAttribute(position, corner(3 * t + 2));
    if (_ray.intersectTriangle(_a, _b, _c, false, _hit)) crossings++;
  }
  return crossings;
}
