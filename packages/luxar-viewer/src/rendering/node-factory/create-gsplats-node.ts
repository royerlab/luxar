/**
 * GSplats-node creation helpers for NodeFactory.
 *
 * @module rendering/node-factory/create-gsplats-node
 */

import * as THREE from 'three';
import { materialManager, type BlendingMode, type LuxarGSplatMaterial } from '../material-manager';
import { getColormapTexture } from '../colormap-textures';
import { createInstancedGSplatsMesh, type InstancedGSplatsMeshConfig } from '../gsplat-geometry';
import { clampSplatCapacity } from '../element-texture-layout';
import { syncGSplatMaterialWithGeometry } from '../material-sync-helpers';
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
  // Rendering attrs come from `nodeAttrs` — the COMPOSED effective attrs
  // the loader passes in (`ctx.applyEffectiveAttrs(node)`) — so an
  // ancestor-authored opacity/absorption/blending_mode reaches the
  // material even in scenes with no `layer=true` node (the layers-panel
  // recompose path only exists for layers). Mirrors the points/lines
  // placeholders; reading the RAW `attrs` here silently dropped ancestor
  // values until the first panel interaction, if ever.
  // `truncation_radius` stays on `attrs`: it is a per-leaf geometry
  // property, deliberately NOT composited (see COMPOSITING_ATTRS).
  const material: LuxarGSplatMaterial = materialManager.getGSplatMaterial({
    opacity: (nodeAttrs.opacity as number | undefined) ?? 1.0,
    absorption: (nodeAttrs.absorption as number | undefined) ?? 1.0,
    gamma: (nodeAttrs.gamma as number | undefined) ?? 1.0,
    intensity: (nodeAttrs.intensity as number | undefined) ?? 1.0,
    offset: (nodeAttrs.offset as number | undefined) ?? 0.0,
    blendingMode: (nodeAttrs.blending_mode as string | undefined as BlendingMode) ?? 'additive',
    truncationRadius: (attrs.truncation_radius as number | undefined) ?? 3.0,
  });

  // Apply colormap if specified. With colormap='custom', the scene
  // loader has stashed bytes as `nodeAttrs.customLutBytes`; the
  // texture helper falls back to viridis on missing/invalid bytes.
  // GSplat materials are PER NODE (each carries the node's own
  // `uSplatTex`), so the colormap applies directly to the node-owned
  // material — the historical clone-on-divergence dance is gone.
  const gsColormapName = nodeAttrs.colormap as string | undefined;
  if (gsColormapName) {
    const gsLutBytes = nodeAttrs.customLutBytes as Uint8Array | undefined;
    const gsColormapTex = getColormapTexture(gsColormapName, gsLutBytes);
    if (gsColormapTex) {
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
    // Clamped like the commit path's stamp — the mesh draws at most the
    // per-node texture bound, and debug/UI counts must agree (mirrors
    // create-points-node).
    visibleSplatCount: clampSplatCapacity(meshConfig.splatCount),
    // Per-node material from creation: LayersPanel and the LOD
    // cross-fade honor this marker and mutate the material directly
    // instead of clone-on-first-use.
    _layerMaterialCloned: true,
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

  // Bind the mesh-owned splat texture + presence flags on the render
  // material (and, when picking is on, the just-created pick material)
  // UNCONDITIONALLY — points/lines sync here regardless of picking, and
  // a node created WITH RGBA initial data must reach the volumetric
  // w(a) gate before its first commit (the picking-gated sync left
  // uHasElementAlpha at 0 in that window when picking was disabled).
  syncGSplatMaterialWithGeometry(mesh);

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
