/**
 * Points-node creation helpers for NodeFactory.
 *
 * Two free functions: `createPointsGeometry` builds the
 * InstancedBufferGeometry (one shared unit-quad base + per-instance
 * attributes for center / color / radius / sharpness / scalar);
 * `createPointsMaterial` resolves the material backend through
 * materialManager and applies the colormap clone path when scalars
 * are requested.
 *
 * @module rendering/node-factory/create-points-node
 */

import * as THREE from 'three';
import { materialManager, type BlendingMode, type LuxarPointMaterial } from '../material-manager';
import { getColormapTexture } from '../colormap-textures';
import { supportsScalarColormap } from '../material-colormap-helpers';
import { createPointQuadGeometry } from '../point-geometry';
import type { LoadedPointsData } from '../../data/data-loader-types';
import type { PointsMetadata } from '../../types/points';
import { log, Modules } from '../../utils/log';
import { validateLoadedPointsData, validateColorMode } from './validation';

/** Build a Points InstancedBufferGeometry from loaded data. */
export function createPointsGeometry(
  data: LoadedPointsData,
  maxRadius: number = 1.0,
  isPlaceholder: boolean = false
): THREE.BufferGeometry {
  const geometry = createPointQuadGeometry();

  validateLoadedPointsData(data, isPlaceholder);

  const pointCount = data.positions.length / 3;

  // Per-instance centre positions. Float16 widens to Float32 because
  // InstancedBufferAttribute doesn't accept Float16 directly.
  let centersTyped: Float32Array;
  if (
    typeof globalThis.Float16Array !== 'undefined' &&
    data.positions instanceof globalThis.Float16Array
  ) {
    centersTyped = new Float32Array(data.positions);
  } else {
    centersTyped = data.positions as Float32Array;
  }
  geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(centersTyped, 3));

  // Per-instance colors. If absent, fill with white.
  if (data.colors) {
    validateColorMode(data.colors, data.metadata);
    const needsNormalization =
      data.colors instanceof Uint8Array || data.colors instanceof Uint16Array;
    geometry.setAttribute(
      'aColor',
      new THREE.InstancedBufferAttribute(data.colors, 3, needsNormalization)
    );
  } else {
    const defaultColors = new Float32Array(pointCount * 3).fill(1.0);
    geometry.setAttribute('aColor', new THREE.InstancedBufferAttribute(defaultColors, 3));
  }

  // Per-instance radii (with dtype-aware normalization).
  let radiusScale = 1.0;
  // World-space maximum radius, used below to expand boundingBox to the
  // rendered footprint (the shared three-geometry invariant — see the
  // boundingBox block). Distinct from radiusScale, which is only a shader
  // normalization factor (1.0 for Float32/Float16 world-unit radii,
  // maxRadius for Uint8 normalized radii); footprintRadius is always the
  // real max radius in world units regardless of dtype.
  let footprintRadius = 0.5; // matches the no-radii fill default below
  if (data.radii) {
    footprintRadius = maxRadius;
    if (
      typeof globalThis.Float16Array !== 'undefined' &&
      data.radii instanceof globalThis.Float16Array
    ) {
      geometry.setAttribute(
        'aRadius',
        new THREE.InstancedBufferAttribute(new Float32Array(data.radii), 1)
      );
      radiusScale = 1.0;
    } else if (data.radii instanceof Uint8Array) {
      geometry.setAttribute('aRadius', new THREE.InstancedBufferAttribute(data.radii, 1, true));
      radiusScale = maxRadius;
    } else {
      geometry.setAttribute(
        'aRadius',
        new THREE.InstancedBufferAttribute(data.radii as Float32Array, 1, false)
      );
      radiusScale = 1.0;
    }
  } else {
    geometry.setAttribute(
      'aRadius',
      new THREE.InstancedBufferAttribute(new Float32Array(pointCount).fill(0.5), 1)
    );
  }

  // Per-instance sharpness (same dtype rules). Sharpness is authored in
  // [0, 1]: Float16/Float32 are stored directly, Uint8 normalizes via the
  // buffer's `normalized:true` flag (uint8/255 → [0, 1]). No scale needed.
  if (data.sharpness) {
    if (
      typeof globalThis.Float16Array !== 'undefined' &&
      data.sharpness instanceof globalThis.Float16Array
    ) {
      geometry.setAttribute(
        'aSharpness',
        new THREE.InstancedBufferAttribute(new Float32Array(data.sharpness), 1)
      );
    } else if (data.sharpness instanceof Uint8Array) {
      geometry.setAttribute(
        'aSharpness',
        new THREE.InstancedBufferAttribute(data.sharpness, 1, true)
      );
    } else {
      geometry.setAttribute(
        'aSharpness',
        new THREE.InstancedBufferAttribute(data.sharpness as Float32Array, 1, false)
      );
    }
  } else {
    geometry.setAttribute(
      'aSharpness',
      new THREE.InstancedBufferAttribute(new Float32Array(pointCount).fill(0.5), 1)
    );
  }

  // Per-instance scalar (USE_COLORMAP only).
  if (data.scalars) {
    const scalarsTyped = data.scalars;
    if (
      typeof globalThis.Float16Array !== 'undefined' &&
      scalarsTyped instanceof globalThis.Float16Array
    ) {
      geometry.setAttribute(
        'aScalar',
        new THREE.InstancedBufferAttribute(new Float32Array(scalarsTyped), 1)
      );
    } else if (scalarsTyped instanceof Uint8Array) {
      geometry.setAttribute('aScalar', new THREE.InstancedBufferAttribute(scalarsTyped, 1, true));
    } else {
      geometry.setAttribute(
        'aScalar',
        new THREE.InstancedBufferAttribute(scalarsTyped as Float32Array, 1, false)
      );
    }
  }

  // WebGLRenderer only issues an instanced draw when instanceCount is set.
  geometry.instanceCount = pointCount;
  geometry.setDrawRange(0, 6);

  // Bounding box/sphere of the per-instance positions, expanded by the
  // rendered footprint (the per-instance disc radius). This is the shared
  // three-geometry invariant: lines (`line-geometry.ts`) expand by max
  // half-width and gsplats (`gsplat-geometry.ts`) by maxRowNorm × truncation,
  // so the pick cull (`ray-aabb.ts`) and camera framing treat all three
  // identically off `boundingBox` alone — no per-geometry special-casing.
  // (The base-quad bounds are irrelevant; frustum culling is disabled here.)
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
