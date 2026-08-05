/**
 * Construct the `THREE.Mesh` for a Mesh node.
 *
 * ## The material here is a deliberate placeholder
 *
 * `THREE.MeshBasicMaterial` with `vertexColors`, which renders the surface
 * **unshaded** — flat per-vertex colour, no lighting. That is the stated outcome of
 * this phase (`docs/specs/MESH_NODE_SPEC.md` §11): prove the data path end to end
 * before taking on the shading model.
 *
 * Mesh is the first geometry type that needs lighting at all — the other three are
 * purely emissive — so the GLSL/TSL material pair, the `gl_FrontFacing` two-sided
 * normal flip, the derivative flat-normal fallback and the per-blend-mode variants
 * are the deepest single piece of work in the whole vertical, and they get their own
 * phase. Until then a mesh reads as a flat silhouette, which is honest about what
 * has been built rather than a half-working shader.
 *
 * Consequences worth stating plainly, because they look like bugs otherwise: a mesh
 * in this phase ignores `blending_mode`, `opacity`, `intensity`, `gamma` and
 * `offset`, and does not appear with appearance controls in the Layers panel. Those
 * arrive with the real material.
 *
 * @module rendering/node-factory/create-mesh-node
 */

import * as THREE from 'three';
import { buildMeshGeometry } from '../mesh-geometry';
import { applyTransform } from './transforms';
import type { MeshSide } from '../../data/mesh/projection';
import type { MeshDataLoader, MeshMetadata, MeshUserData } from '../../types/mesh';

/** Map the projection's epoch decision onto a three.js side constant. */
function threeSide(side: MeshSide): THREE.Side {
  return side === 'double' ? THREE.DoubleSide : THREE.FrontSide;
}

/**
 * Apply the epoch's material side to a mesh object.
 *
 * Separate from construction because `side` is a property of the current
 * `displayDims` epoch, not of the node: an odd-parity reflection stays
 * single-sided (the index post-pass restored winding) while an undecidable frame
 * forces double-sided regardless of the authored `double_sided`. So the commit path
 * re-applies it on every rebuild.
 *
 * Guarded on change: assigning `material.side` unconditionally would set
 * `needsUpdate` semantics in motion on every slice move, and on the WebGPU backend a
 * side change forces a pipeline rebuild.
 */
export function applyMeshSide(object: THREE.Mesh, side: MeshSide): void {
  const material = object.material as THREE.Material;
  const wanted = threeSide(side);
  if (material.side !== wanted) {
    material.side = wanted;
    material.needsUpdate = true;
  }
}

/**
 * Build the placeholder material for a mesh node.
 *
 * `vertexColors: true` is what makes the always-bound `color` attribute
 * (`mesh-geometry.ts`) reach the fragment. `transparent: false` keeps the surface on
 * the opaque pass, which is the correct default for a mesh and the reason it gets
 * real depth-buffer occlusion without any of the sorting machinery the emissive
 * types need.
 */
function createMeshPlaceholderMaterial(attrs: MeshMetadata): THREE.MeshBasicMaterial {
  return new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: attrs.double_sided ? THREE.DoubleSide : THREE.FrontSide,
  });
}

/**
 * Create an empty mesh node — a `THREE.Mesh` with a one-vertex degenerate geometry,
 * attached before any data is fetched.
 *
 * Same contract as the sibling `createEmptyXNode` factories: the commit path finds
 * this object by name and populates it, so an initial-load failure leaves a
 * recoverable scene state rather than a hole, and a later retry has somewhere to
 * write. The geometry is replaced in place on first commit.
 */
export function createEmptyMeshNode(
  path: string,
  attrs: MeshMetadata,
  loader: MeshDataLoader
): THREE.Mesh {
  const geometry = buildMeshGeometry({
    // One vertex and no indices: a valid, drawable-but-empty geometry. A truly
    // zero-vertex buffer makes `computeBoundingSphere` produce NaN bounds, which
    // the depth-sort coordinator and the raycaster both then refuse to use.
    position: new Float32Array(3),
    // The placeholder's buffer is brand new, so it is trivially "changed".
    positionChanged: true,
    indices: new Uint32Array(0),
    colors: null,
    vertexCount: 1,
    faceCount: 0,
  });

  const mesh = new THREE.Mesh(geometry, createMeshPlaceholderMaterial(attrs));
  mesh.name = path;
  // Safe for the same reason the other three are: every commit path refreshes the
  // bounds whenever `position` is replaced (see `updateMeshGeometry`), so the
  // sphere the frustum test reads never goes stale. Unlike the emissive types
  // there is no footprint expansion to apply — a triangle's extent IS its
  // vertices, so the vertex-spanning box is already the rendered footprint.
  mesh.frustumCulled = true;

  const userData: MeshUserData = {
    nodeType: 'mesh',
    loader,
    attrs,
    path,
    visibleTriangleCount: 0,
    visibleVertexCount: 0,
  };
  mesh.userData = userData;

  if (attrs.transform) applyTransform(mesh, attrs.transform);

  return mesh;
}
