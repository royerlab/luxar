/**
 * Points-node creation helpers for NodeFactory.
 *
 * `createPointsGeometry` builds the InstancedBufferGeometry (one
 * shared unit-quad base + the RGBA32F point texture / `aSortedIndex`
 * storage pair — the per-point data lives in the texture, 3
 * texels/point; layout documented in `../point-geometry.ts`);
 * `createPointsMaterial` resolves the material backend through
 * materialManager and applies the colormap clone path when scalars are
 * requested; `createPointsNode` assembles both into the mesh + optional
 * picking shadow node.
 *
 * @module rendering/node-factory/create-points-node
 */

import * as THREE from 'three';
import { materialManager, type BlendingMode, type LuxarPointMaterial } from '../material-manager';
import { getColormapTexture } from '../colormap-textures';
import { supportsScalarColormap } from '../material-colormap-helpers';
import {
  createPointQuadGeometry,
  attachPointStorage,
  pointsNormalizationDivisor,
  writePointTexels,
  type PointTexelSource,
} from '../point-geometry';
import { writeSortedIndexIdentity } from '../element-storage';
import { clampPointCapacity } from '../element-texture-layout';
import { widenToFloat32 } from '../interleaved-attributes';
import type { LoadedPointsData, DataLoader } from '../../data/data-loader-types';
import type { PointsMetadata, PointsUserData } from '../../types/points';
import { log, Modules } from '../../utils/log';
import type { PickingSystem } from '../picking/picking-system';
import { applyTransform } from './transforms';
import { validateLoadedPointsData, validateColorMode } from './validation';

/**
 * Build a Points InstancedBufferGeometry from loaded data.
 *
 * Non-pool path (mirrors `createInstancedGSplatsMesh`'s geometry
 * portion): quad base + EXACT-SIZE point texture / `aSortedIndex`
 * storage pair, one fused texel write, identity ordering. The commit
 * fallback (commit-points-geometry.ts) disposes and re-creates the
 * geometry on every commit; `attachPointStorage`'s dispose listener
 * frees the texture with it.
 */
export function createPointsGeometry(
  data: LoadedPointsData,
  maxRadius: number = 1.0,
  isPlaceholder: boolean = false
): THREE.BufferGeometry {
  const geometry = createPointQuadGeometry();

  validateLoadedPointsData(data, isPlaceholder);

  // SEMANTIC clamp (mirrors createInstancedGSplatsMesh): every consumer
  // below (storage size, texel/ordering writes, instanceCount, userData
  // stamp) uses the same clamped count, so a request above the per-node
  // texture bound stays self-consistent instead of drawing instances
  // without texels.
  const pointCount = clampPointCapacity(data.positions.length / 3);

  // Geometry-owned point texture + identity ordering (exact-size — the
  // non-pool fallback carries no capacity headroom).
  const texture = attachPointStorage(geometry, pointCount);

  // Widen every field EXACTLY as the pool adapter does (same
  // widenToFloat32 calls, same normalization divisors, same fallback
  // fills) so texel values are identical on both commit paths.
  const positionsF32 =
    data.positions instanceof Float32Array
      ? data.positions
      : widenToFloat32(data.positions as ArrayLike<number>);

  let colorsF32: Float32Array;
  if (data.colors) {
    validateColorMode(data.colors, data.metadata);
    colorsF32 = widenToFloat32(
      data.colors.subarray(0, pointCount * 3) as ArrayLike<number>,
      pointsNormalizationDivisor(data.colors, /*normalized=*/ true)
    );
  } else {
    colorsF32 = new Float32Array(pointCount * 3);
    colorsF32.fill(1.0); // white default
  }

  // Radii (dtype-aware normalization).
  let radiusScale = 1.0;
  // World-space maximum radius, used below to expand boundingBox to the
  // rendered footprint (the shared three-geometry invariant — see the
  // boundingBox block). Distinct from radiusScale, which is only a shader
  // normalization factor (1.0 for Float32/Float16 world-unit radii,
  // maxRadius for Uint8 normalized radii — the texel holds the [0, 1]
  // widened value in that case); footprintRadius is always the real max
  // radius in world units regardless of dtype.
  let footprintRadius = 0.5; // matches the no-radii fill default below
  let radiiF32: Float32Array;
  if (data.radii) {
    footprintRadius = maxRadius;
    radiiF32 = widenToFloat32(
      data.radii.subarray(0, pointCount) as ArrayLike<number>,
      pointsNormalizationDivisor(data.radii, /*normalized=*/ true)
    );
    radiusScale = data.radii instanceof Uint8Array ? maxRadius : 1.0;
  } else {
    radiiF32 = new Float32Array(pointCount);
    radiiF32.fill(0.5);
  }

  // Sharpness (same dtype rules). Sharpness is authored in [0, 1]:
  // Float16/Float32 widen as-is, Uint8 normalizes ÷255. No scale needed.
  let sharpnessF32: Float32Array;
  if (data.sharpness) {
    sharpnessF32 = widenToFloat32(
      data.sharpness.subarray(0, pointCount) as ArrayLike<number>,
      pointsNormalizationDivisor(data.sharpness, /*normalized=*/ true)
    );
  } else {
    sharpnessF32 = new Float32Array(pointCount);
    sharpnessF32.fill(0.5); // default sharpness knob -> beta=2 (Gaussian)
  }

  // Scalars (USE_COLORMAP only). Absent ⇒ omitted from the source; the
  // writer stamps the 0.0 identity into texel2.x unconditionally.
  const scalarsF32 = data.scalars
    ? widenToFloat32(
        data.scalars.subarray(0, pointCount) as ArrayLike<number>,
        pointsNormalizationDivisor(data.scalars, /*normalized=*/ true)
      )
    : undefined;

  const texelSrc: PointTexelSource = {
    positions: positionsF32,
    colors: colorsF32,
    radii: radiiF32,
    sharpness: sharpnessF32,
    scalars: scalarsF32,
  };
  writePointTexels(texture, texelSrc, pointCount);
  writeSortedIndexIdentity(geometry, pointCount);

  // WebGLRenderer only issues an instanced draw when instanceCount is set.
  geometry.instanceCount = pointCount;
  geometry.setDrawRange(0, 6);

  // Bounding box/sphere of the per-instance positions, expanded by the
  // rendered footprint (the per-instance disc radius). This is the shared
  // three-geometry invariant: lines (`line-geometry.ts`) expand by max
  // half-width and gsplats (`gsplat-geometry.ts`) by maxRowNorm × truncation,
  // so frustum culling, the pick cull (`ray-aabb.ts`), and camera framing
  // treat all three identically off `boundingBox` alone — no per-geometry
  // special-casing. (The base-quad bounds never participate: mesh-level
  // culling tests these explicit instance-spanning bounds.)
  geometry.boundingBox = data.metadata.bounds.clone();
  if (footprintRadius > 0) geometry.boundingBox.expandByScalar(footprintRadius);
  geometry.boundingSphere = new THREE.Sphere();
  geometry.boundingBox.getBoundingSphere(geometry.boundingSphere);

  if (!geometry.userData) geometry.userData = {};
  geometry.userData.radiusScale = radiusScale;
  geometry.userData.pointCount = pointCount;

  return geometry;
}

/**
 * Resolve a Points material from `materialManager`, then apply the
 * colormap clone path when scalars + a non-null colormap are requested.
 */
export function createPointsMaterial(
  attrs: Partial<PointsMetadata>,
  radiusScale: number = 1.0,
  geometry?: THREE.BufferGeometry,
  path?: string
): LuxarPointMaterial {
  let material = materialManager.getPointMaterial({
    opacity: attrs.opacity ?? 1.0,
    gamma: attrs.gamma ?? 1.0,
    intensity: attrs.intensity ?? 1.0,
    offset: attrs.offset ?? 0.0,
    blendingMode: (attrs.blending_mode as BlendingMode) ?? 'additive',
    radiusScale,
  });

  const ptColormapName = attrs.colormap;
  const ptHasScalars = !!attrs.has_scalars;
  if (ptColormapName && ptHasScalars) {
    // USE_COLORMAP requires a `scalar` attribute. When `geometry` is
    // provided check the actual binding; when absent (tests calling
    // createPointsMaterial directly), trust the caller.
    const guardOK = !geometry || supportsScalarColormap('points', geometry);
    if (!guardOK) {
      log.warning(
        Modules.SCENE_LOADER,
        `[${path ?? '<points>'}] Scalar colormap requested but 'scalar' attribute is not bound on geometry. Colormap suppressed; rendering with vertex colors.`
      );
    } else {
      const ptLutBytes = (attrs as { customLutBytes?: Uint8Array }).customLutBytes;
      const ptColormapTex = getColormapTexture(ptColormapName, ptLutBytes);
      if (ptColormapTex) {
        // Detach pooled material from global updates before cloning so
        // disposeAll doesn't dispose the cache entry serving other
        // callers. The clone takes the global-update slot.
        materialManager.detachFromGlobalUpdates(material);
        material = material.clone() as typeof material;
        materialManager.register(material);
        material.updateColormapTexture(ptColormapTex);
        const ptScalarRange = attrs.scalar_data_range ?? [0, 1];
        material.updateScalarRange(ptScalarRange[0], ptScalarRange[1]);
      }
    }
  }

  return material;
}

/**
 * Build a Points mesh + optional picking shadow node.
 *
 * Each point is rendered as an instanced quad sprite, matching the
 * line + gsplat geometry pattern. The mesh's geometry is built by
 * `createPointsGeometry`, which attaches the point-texture storage
 * pair (RGBA32F texture + `aSortedIndex`) to a shared unit-quad base.
 *
 * Handles geometry creation, material selection, userData, and
 * transforms.
 */
export function createPointsNode(
  path: string,
  attrs: PointsMetadata,
  data: LoadedPointsData,
  loader: DataLoader,
  pickingSystem: PickingSystem | null,
  isPlaceholder: boolean = false
): THREE.Mesh {
  const maxRadius = attrs.max_radius ?? 1.0;
  const geometry = createPointsGeometry(data, maxRadius, isPlaceholder);

  const radiusScale = geometry.userData.radiusScale ?? 1.0;
  const material = createPointsMaterial(attrs, radiusScale, geometry, path);

  const points = new THREE.Mesh(geometry, material);
  points.name = path;
  // Frustum culling is safe because the geometry's bounds are
  // instance-spanning AND footprint-expanded (see the boundingBox block
  // in `createPointsGeometry`) — the shared three-geometry invariant
  // that already lets lines (`line-geometry.ts`) and gsplats
  // (`gsplat-geometry.ts`) cull with `frustumCulled = true`. Every
  // commit path refreshes the bounds (pool adapter + commit helper, or
  // full geometry recreation), so the sphere never goes stale.
  points.frustumCulled = true;

  points.userData = {
    nodeType: 'points',
    loader,
    attrs,
    maxRadius: attrs.max_radius ?? 1.0,
    visiblePointCount: data.pointCount,
  } as PointsUserData;

  if (attrs.transform) applyTransform(points, attrs.transform);

  if (pickingSystem) {
    const pickId = pickingSystem.allocatePickId();
    points.userData.pickId = pickId;
    const pickMaterial = materialManager.createPointPickingMaterial({
      nodeId: pickId,
      radiusScale,
    });
    materialManager.register(pickMaterial);
    // Share the same InstancedBufferGeometry — only material differs.
    // The shared instance-spanning bounds make the pick node cullable
    // too (default `frustumCulled = true`, matching lines/gsplats).
    const pickNode = new THREE.Mesh(geometry, pickMaterial);
    pickNode.matrixWorld.copy(points.matrixWorld);
    pickingSystem.registerNode(points, pickNode, pickId);
  }

  return points;
}

/**
 * Empty-buffer placeholder for the points node (pre-fetch placeholder).
 *
 * Constructs a minimal {@link LoadedPointsData} inline rather than
 * routing through `createEmptyPointsData()` (which needs a full
 * `ProjectionContext` with `chunkIndex`); the values that distinguish
 * the two paths (`ndim`, `dtypes`) are overwritten on the first
 * successful commit.
 */
export function createEmptyPointsNode(
  path: string,
  attrs: PointsMetadata,
  loader: DataLoader,
  pickingSystem: PickingSystem | null
): THREE.Mesh {
  const emptyData: LoadedPointsData = {
    positions: new Float32Array(0) as LoadedPointsData['positions'],
    pointCount: 0,
    ndim: 3,
    metadata: {
      totalPoints: attrs.n_points ?? 0,
      loadedPoints: 0,
      bounds: new THREE.Box3(),
      usedSpatialIndex: true,
      dtypes: {},
    },
  };
  // When the node carries a scalar field + colormap, stamp an empty
  // scalars field on the placeholder data. In the interleaved era this
  // bound an empty `aScalar` attribute so the fail-closed colormap
  // guard (`supportsScalarColormap`) passed at material-creation time;
  // with texture storage the scalar always rides texel2.x and there is
  // no `aScalar` attribute, so the geometry-attribute-based guard fails
  // closed regardless — the points-material Stage 3 migration re-points
  // the guard to the new storage. Kept so the placeholder and real data
  // declare the same field presence.
  if (attrs.has_scalars && attrs.colormap) {
    emptyData.scalars = new Float32Array(0) as LoadedPointsData['scalars'];
  }
  return createPointsNode(path, attrs, emptyData, loader, pickingSystem, /* isPlaceholder */ true);
}
