/**
 * GSplats-node creation helpers extracted from NodeFactory in P6/step 6.5.
 *
 * @module rendering/node-factory/create-gsplats-node
 */

import * as THREE from 'three';
import {
  materialManager,
  type BlendingMode,
  type LuxarGSplatMaterial,
} from '../material-manager';
import { getColormapTexture } from '../colormap-textures';
import {
  createInstancedGSplatsMesh,
  type InstancedGSplatsMeshConfig,
} from '../gsplat-geometry';
import type { GSplatsMetadata, GSplatsUserData, GSplatsDataLoader } from '../../types/gsplats';
import type { PickingSystem } from '../picking/picking-system';
import { applyTransform } from './transforms';

/** Build a GSplats mesh + optional picking shadow node. */
export function createGSplatsNode(
  path: string,
  nodeAttrs: Record<string, unknown>,
  attrs: GSplatsMetadata,
  meshConfig: InstancedGSplatsMeshConfig,
  loader: GSplatsDataLoader,
  pickingSystem: PickingSystem | null
): THREE.Mesh {
  let material: LuxarGSplatMaterial = materialManager.getGSplatMaterial({
    opacity: (attrs.opacity as number | undefined) ?? 1.0,
    gamma: (attrs.gamma as number | undefined) ?? 1.0,
    intensity: (attrs.intensity as number | undefined) ?? 1.0,
    offset: (attrs.offset as number | undefined) ?? 0.0,
    blendingMode: (attrs.blending_mode as string | undefined as BlendingMode) ?? 'additive',
    truncationRadius: (attrs.truncation_radius as number | undefined) ?? 3.0,
  });

  // Apply colormap if specified. With colormap='custom', the scene
  // loader has stashed bytes as `nodeAttrs.customLutBytes`; the
  // texture helper falls back to viridis on missing/invalid bytes.
  const gsColormapName = nodeAttrs.colormap as string | undefined;
  let gsplatMaterialCloned = false;
  if (gsColormapName) {
    const gsLutBytes = nodeAttrs.customLutBytes as Uint8Array | undefined;
    const gsColormapTex = getColormapTexture(gsColormapName, gsLutBytes);
    if (gsColormapTex) {
      materialManager.detachFromGlobalUpdates(material);
      // Both clones (GSplatMaterial.clone() / GSplatTSLMaterial.clone())
      // satisfy LuxarGSplatMaterial; `as typeof material` keeps the
      // backend-agnostic type and avoids narrowing to the WebGL2 class.
      material = material.clone() as typeof material;
      materialManager.register(material);
      gsplatMaterialCloned = true;
      material.updateColormapTexture(gsColormapTex);
      const ampRange = nodeAttrs.amplitude_data_range as [number, number] | undefined;
      const gsScalarRange = ampRange ?? [0, 1];
      material.updateScalarRange(gsScalarRange[0], gsScalarRange[1]);
    }
  }

  const mesh = createInstancedGSplatsMesh(meshConfig, material);
  mesh.name = path;
  mesh.userData = {
    nodeType: 'gsplats',
    loader,
    attrs,
    visibleSplatCount: meshConfig.splatCount,
    _layerMaterialCloned: gsplatMaterialCloned,
  } as GSplatsUserData;

  if (attrs.transform) applyTransform(mesh, attrs.transform);

  if (pickingSystem) {
    const pickId = pickingSystem.allocatePickId();
    mesh.userData.pickId = pickId;
    const pickMaterial = materialManager.createGSplatPickingMaterial({ nodeId: pickId });
    materialManager.register(pickMaterial);
    // Share the same InstancedBufferGeometry — only material differs.
    const pickNode = new THREE.Mesh(mesh.geometry, pickMaterial);
    pickNode.matrixWorld.copy(mesh.matrixWorld);
    pickingSystem.registerNode(mesh, pickNode, pickId);
  }

  return mesh;
}

/** Empty-buffer placeholder for the gsplats node. */
export function createEmptyGSplatsNode(
  path: string,
  nodeAttrs: Record<string, unknown>,
  attrs: GSplatsMetadata,
  loader: GSplatsDataLoader,
  pickingSystem: PickingSystem | null
): THREE.Mesh {
  const emptyConfig: InstancedGSplatsMeshConfig = {
    centers: new Float32Array(0),
    cholesky01: new Float32Array(0),
    cholesky23: new Float32Array(0),
    cholesky45: new Float32Array(0),
    amplitudes: new Float32Array(0),
    colors: new Float32Array(0),
    splatCount: 0,
  };
  return createGSplatsNode(path, nodeAttrs, attrs, emptyConfig, loader, pickingSystem);
}
