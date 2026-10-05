/**
 * One commit per geometry type, through `SceneLoader`'s own commit methods.
 *
 * Several geometry-behaviour probes (`tests/_conformance/geometry-behaviours.ts`)
 * need the same thing: a scene with one node of a given type, and a real commit
 * of `count` elements into it, so they can observe what the commit did — the
 * render request, the depth-sort registration, the visible count the monitor
 * reads. Each type reaches the commit stage with a different staged payload;
 * this module is the one place those payloads are built.
 *
 * Points, Lines and GSplats nodes are bare `THREE.Mesh` placeholders: their
 * commits build the geometry themselves. A mesh commit writes INTO the node's
 * existing indexed geometry, so the mesh node is a real `createEmptyMeshNode`
 * and its payload a real `processMeshData` projection.
 *
 * @module tests/helpers/geometry-commits
 */

import * as THREE from 'three';

import type { SceneLoader } from '../../data/scene-loader';
import type { LoadedPointsData } from '../../data/data-loader-types';
import type { StagedLinesCommit } from '../../data/scene-loader/process/data-processor-lines';
import type { StagedGSplatsCommit } from '../../data/scene-loader/process/data-processor-gsplats';
import {
  processMeshData,
  type StagedMeshCommit,
} from '../../data/scene-loader/process/data-processor-mesh';
import { createEmptyMeshNode } from '../../rendering/node-factory/create-mesh-node';
import { getCommittedData } from '../../types/committed-data';
import { setPrefixParent } from '../../types/prefix-lineage';
import type { GeometryTypeName } from '../../types/format-contract';
import type { LoadedMeshData, MeshDataLoader, MeshMetadata, MeshViewState } from '../../types/mesh';

/** Reach-in surface for `SceneLoader`'s private commit methods. */
export interface SceneLoaderCommitInternals {
  rootGroup: THREE.Group | null;
  lodGroupRegistry: { invalidatePartitionFootprint(path: string): void } | null;
  _gpuBufferPool: unknown;
  updatePointsGeometry(path: string, data: LoadedPointsData): void;
  commitLinesGeometry(staged: StagedLinesCommit): void;
  commitGSplatsGeometry(staged: StagedGSplatsCommit): void;
  commitMeshGeometry(staged: StagedMeshCommit): void;
}

/** `loader`, typed for its private commit methods. */
export const commitInternals = (loader: SceneLoader): SceneLoaderCommitInternals =>
  loader as unknown as SceneLoaderCommitInternals;

const UNIT_BOX = new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));

/** A display-ready points payload of `pointCount` points. */
export function makePointsData(pointCount: number): LoadedPointsData {
  return {
    positions: new Float32Array(pointCount * 3),
    colors: new Uint8Array(pointCount * 3),
    radii: undefined,
    sharpness: undefined,
    pointCount,
    metadata: { bounds: UNIT_BOX.clone() },
  } as unknown as LoadedPointsData;
}

/** A staged lines commit of `segmentCount` segments at `path`. */
export function makeStagedLines(segmentCount = 2, path = '/lines'): StagedLinesCommit {
  return {
    path,
    sourceData: {
      positions: new Float32Array(segmentCount * 2 * 3),
      segments: new Uint32Array(segmentCount * 2),
      widths: new Float32Array(segmentCount * 2),
      colors: null,
      sharpness: null,
      segmentCount,
      vertexCount: segmentCount * 2,
      ndim: 3,
    },
    processed: {
      startPositions: new Float32Array(segmentCount * 3),
      endPositions: new Float32Array(segmentCount * 3),
      startColors: new Float32Array(segmentCount * 3),
      endColors: new Float32Array(segmentCount * 3),
      startWidths: new Float32Array(segmentCount),
      endWidths: new Float32Array(segmentCount),
      startSharpness: new Float32Array(segmentCount),
      endSharpness: new Float32Array(segmentCount),
      segmentLengths: new Float32Array(segmentCount),
      startJointCode: new Float32Array(segmentCount),
      endJointCode: new Float32Array(segmentCount),
      segmentCount,
    },
  } as StagedLinesCommit;
}

/** A staged gsplats commit of `splatCount` splats at `path`. */
export function makeStagedGSplats(splatCount = 2, path = '/g'): StagedGSplatsCommit {
  return {
    path,
    sourceData: {
      positions: new Float32Array(splatCount * 3),
      amplitudes: new Float32Array(splatCount),
      choleskyFactors: new Float32Array(splatCount * 6),
      colors: null,
      splatCount,
      ndim: 3,
    },
    processed: {
      centers3D: new Float32Array(splatCount * 3),
      choleskyFactors3D: new Float32Array(splatCount * 6),
      amplitudes: new Float32Array(splatCount),
      colors: new Float32Array(splatCount * 3),
      splatCount,
    },
  } as StagedGSplatsCommit;
}

/** Attrs of a plain, untextured, single-sided 3D mesh of `faceCount` triangles. */
function meshAttrs(faceCount: number): MeshMetadata {
  return {
    type: 'mesh',
    n_vertices: faceCount * 3,
    n_faces: faceCount,
    ndim: 3,
    has_normals: false,
    has_colors: false,
    has_scalars: false,
    has_uvs: false,
    has_texture: false,
    shading: 'flat',
    double_sided: true,
    ordering: 'none',
  };
}

/** `faceCount` separate triangles in the z = 0 plane, three vertices each. */
function meshData(faceCount: number): LoadedMeshData {
  const vertices = new Float32Array(faceCount * 9);
  const faces = new Uint32Array(faceCount * 3);
  for (let f = 0; f < faceCount; f++) {
    vertices.set([f, 0, 0, f + 1, 0, 0, f, 1, 0], f * 9);
    faces.set([f * 3, f * 3 + 1, f * 3 + 2], f * 3);
  }
  return {
    vertices,
    faces,
    normals: null,
    colors: null,
    scalars: undefined,
    vertexCount: faceCount * 3,
    faceCount,
    ndim: 3,
  };
}

const MESH_VIEW = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0],
  tolerance: [1e10, 1e10, 1e10],
} as MeshViewState;

/** A staged mesh commit of `faceCount` triangles at `path`, through the real projection. */
export function makeStagedMesh(faceCount: number, path: string): Promise<StagedMeshCommit> {
  return processMeshData(path, meshData(faceCount), MESH_VIEW, {
    normal_dims: undefined,
    double_sided: true,
  });
}

/** Per-type node placeholder + commit, both through real code. */
export interface GeometryCommitAdapter {
  /** The scene-graph path this type's node lives at in {@link makeCommitScene}. */
  readonly path: string;
  /** A node of this type, shaped the way its node loader leaves it before the first commit. */
  makeNode(capacity: number): THREE.Mesh;
  /**
   * Commit `count` elements to the node at {@link path}. With `extending`, the
   * payload carries prefix lineage to the node's last commit — the proof a
   * progressive ladder's next rung carries that it EXTENDS what is drawn
   * (`types/prefix-lineage.ts`), which the instanced commits' append path keys on.
   * Mesh ignores `extending`: its loader stamps no lineage and its commit
   * rewrites the whole index buffer.
   */
  commit(loader: SceneLoader, count: number, extending?: boolean): Promise<void>;
}

/** Stamp `payload` as an extension of what `path`'s node last committed. */
function extendsCommitted<T extends object>(
  loader: SceneLoader,
  path: string,
  payload: T,
  extending: boolean | undefined
): T {
  if (!extending) return payload;
  const node = commitInternals(loader).rootGroup?.getObjectByName(path);
  const committed = node ? getCommittedData(node as THREE.Mesh) : undefined;
  if (committed === undefined) throw new Error(`${path} has no commit to extend`);
  setPrefixParent(payload, committed as object);
  return payload;
}

/** A bare placeholder: the instanced commits build their own geometry. */
function placeholder(name: string, nodeType: GeometryTypeName): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.name = name;
  mesh.userData = {
    nodeType,
    attrs: {},
    visiblePointCount: 0,
    visibleSegmentCount: 0,
    visibleSplatCount: 0,
  };
  return mesh;
}

/** One adapter per geometry type. */
export const GEOMETRY_COMMITS: Readonly<Record<GeometryTypeName, GeometryCommitAdapter>> = {
  points: {
    path: '/p',
    makeNode: () => placeholder('/p', 'points'),
    commit: (loader, count, extending) => {
      const data = extendsCommitted(loader, '/p', makePointsData(count), extending);
      commitInternals(loader).updatePointsGeometry('/p', data);
      return Promise.resolve();
    },
  },
  lines: {
    path: '/lines',
    makeNode: () => placeholder('/lines', 'lines'),
    commit: (loader, count, extending) => {
      const staged = makeStagedLines(count, '/lines');
      extendsCommitted(loader, '/lines', staged.sourceData, extending);
      commitInternals(loader).commitLinesGeometry(staged);
      return Promise.resolve();
    },
  },
  gsplats: {
    path: '/g',
    makeNode: () => placeholder('/g', 'gsplats'),
    commit: (loader, count, extending) => {
      const staged = makeStagedGSplats(count, '/g');
      extendsCommitted(loader, '/g', staged.sourceData, extending);
      commitInternals(loader).commitGSplatsGeometry(staged);
      return Promise.resolve();
    },
  },
  mesh: {
    path: '/mesh',
    makeNode: (capacity) =>
      createEmptyMeshNode('/mesh', meshAttrs(capacity), {} as MeshDataLoader, null),
    // A mesh commit reads no prefix lineage: every commit rewrites the index whole.
    commit: async (loader, count) => {
      commitInternals(loader).commitMeshGeometry(await makeStagedMesh(count, '/mesh'));
    },
  },
};

/**
 * A root group holding one node of every geometry type, each sized for up to
 * `capacity` elements, installed as `loader`'s scene.
 */
export function makeCommitScene(loader: SceneLoader, capacity = 8): THREE.Group {
  const root = new THREE.Group();
  for (const adapter of Object.values(GEOMETRY_COMMITS)) root.add(adapter.makeNode(capacity));
  commitInternals(loader).rootGroup = root;
  return root;
}
