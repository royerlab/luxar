/**
 * Lines-node creation helpers for NodeFactory.
 *
 * `createLinesNode` resolves the line material backend through
 * materialManager, applies the colormap clone path when scalars +
 * a non-null colormap are requested, detects the sharpness=2 fast
 * path, then builds the InstancedLinesMesh + optional picking
 * shadow node.
 *
 * @module rendering/node-factory/create-lines-node
 */

import * as THREE from 'three';
import { materialManager, type BlendingMode, type LuxarLineMaterial } from '../material-manager';
import { getColormapTexture } from '../colormap-textures';
import {
  createInstancedLinesMesh,
  isAllSharpnessTwo,
  type InstancedLinesMeshConfig,
} from '../line-geometry';
import type { LinesMetadata, LinesUserData, LinesDataLoader } from '../../types/lines';
import { log, Modules } from '../../utils/log';
import type { PickingSystem } from '../picking/picking-system';
import { applyTransform } from './transforms';

/** Build a Lines mesh + optional picking shadow node. */
export function createLinesNode(
  path: string,
  nodeAttrs: Record<string, unknown>,
  attrs: LinesMetadata,
  processed: InstancedLinesMeshConfig,
  loader: LinesDataLoader,
  pickingSystem: PickingSystem | null
): THREE.Mesh {
  let material: LuxarLineMaterial = materialManager.getLineMaterial({
    opacity: (attrs.opacity as number | undefined) ?? 1.0,
    gamma: (attrs.gamma as number | undefined) ?? 1.0,
    intensity: (attrs.intensity as number | undefined) ?? 1.0,
    offset: (attrs.offset as number | undefined) ?? 0.0,
    blendingMode: (attrs.blending_mode as string | undefined as BlendingMode) ?? 'additive',
  });

  // Apply colormap if specified and scalar data exists.
  const lnColormapName = nodeAttrs.colormap as string | undefined;
  const lnHasScalars = !!nodeAttrs.has_scalars;
  const linesScalarsReady = 'startScalars' in processed && 'endScalars' in processed;
  if (lnColormapName && lnHasScalars) {
    if (!linesScalarsReady) {
      log.warning(
        Modules.SCENE_LOADER,
        `[${path}] Line scalar colormap requested but scalar attributes are not bound. Colormap suppressed.`
      );
    } else {
      const lnLutBytes = nodeAttrs.customLutBytes as Uint8Array | undefined;
      const lnColormapTex = getColormapTexture(lnColormapName, lnLutBytes);
      if (lnColormapTex) {
        // Detach pooled material from global updates before cloning so
        // disposeAll doesn't dispose the cache entry serving other callers.
        materialManager.detachFromGlobalUpdates(material);
        material = material.clone() as typeof material;
        materialManager.register(material);
        material.updateColormapTexture(lnColormapTex);
        const lnScalarRange = (nodeAttrs.scalar_data_range as [number, number]) ?? [0, 1];
        material.updateScalarRange(lnScalarRange[0], lnScalarRange[1]);
      }
    }
  }

  // Sharpness fast path: when every per-vertex sharpness is 2.0, the
  // wrapper toggles `LUXAR_SHARPNESS_TWO` so the fragment shader
  // replaces its `pow(x, vSharpness)` with `x*x`.
  const sharpnessFastPath = isAllSharpnessTwo(processed);
  material.setSharpnessAllTwo(sharpnessFastPath);

  const mesh = createInstancedLinesMesh(processed, material);
  mesh.name = path;
  mesh.userData = {
    nodeType: 'lines',
    loader,
    attrs,
    maxWidth: attrs.max_width ?? 1.0,
    visibleSegmentCount: processed.segmentCount,
  } as LinesUserData;

  if (attrs.transform) applyTransform(mesh, attrs.transform);

  if (pickingSystem) {
    const pickId = pickingSystem.allocatePickId();
    mesh.userData.pickId = pickId;
    const pickMaterial = materialManager.createLinePickingMaterial({ nodeId: pickId });
    // Mirror the visual material's sharpness fast path on the picking
    // material so the pick shader skips its pow(...) too.
    pickMaterial.setSharpnessAllTwo(sharpnessFastPath);
    materialManager.register(pickMaterial);
    // Share the same InstancedBufferGeometry — only material differs.
    const pickNode = new THREE.Mesh(mesh.geometry, pickMaterial);
    pickNode.matrixWorld.copy(mesh.matrixWorld);
    pickingSystem.registerNode(mesh, pickNode, pickId);
  }

  return mesh;
}

/** Empty-buffer placeholder for the lines node (pre-fetch placeholder). */
export function createEmptyLinesNode(
  path: string,
  nodeAttrs: Record<string, unknown>,
  attrs: LinesMetadata,
  loader: LinesDataLoader,
  pickingSystem: PickingSystem | null
): THREE.Mesh {
  const emptyConfig: InstancedLinesMeshConfig = {
    startPositions: new Float32Array(0),
    endPositions: new Float32Array(0),
    startColors: new Float32Array(0),
    endColors: new Float32Array(0),
    startWidths: new Float32Array(0),
    endWidths: new Float32Array(0),
    startSharpness: new Float32Array(0),
    endSharpness: new Float32Array(0),
    segmentLengths: new Float32Array(0),
    startClipped: new Uint8Array(0),
    endClipped: new Uint8Array(0),
    segmentCount: 0,
  };
  // When the node carries a scalar field + colormap, bind empty scalar
  // arrays on the placeholder so the colormap clone path in
  // `createLinesNode` (gated on `'startScalars' in processed`) fires at
  // material-creation time. Without them the guard sees no scalars, logs
  // "Colormap suppressed", and nothing ever re-enables the LUT once real
  // scalars stream in (the commit writes into the existing placeholder
  // geometry). Gate on the SAME `nodeAttrs` fields the colormap-application
  // path above reads, so the placeholder matches exactly when colormap will
  // apply. Mirrors `node-factory.ts::createEmptyPointsNode`.
  if (nodeAttrs.colormap && nodeAttrs.has_scalars) {
    emptyConfig.startScalars = new Float32Array(0);
    emptyConfig.endScalars = new Float32Array(0);
  }
  return createLinesNode(path, nodeAttrs, attrs, emptyConfig, loader, pickingSystem);
}
