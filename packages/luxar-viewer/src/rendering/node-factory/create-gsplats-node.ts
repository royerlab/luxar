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
import { resolveColormapWindow } from '../display-range';
import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../config/constants';
import { clampTruncationRadius } from '../materials/gsplat/math';

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
  //
  // Resolve the colormap texture UP FRONT so the color-GOG decision below
  // can mirror the panel: on a colormapped node the authored
  // intensity/offset define the scalar WINDOW (value→LUT mapping), NOT a
  // post-LUT color gain — pushing them as both would double-apply (#936).
  // With colormap='custom', the scene loader has stashed bytes as
  // `nodeAttrs.customLutBytes`; the texture helper falls back to viridis on
  // missing/invalid bytes. A `colormap` attr can still fail the texture
  // lookup, so key the decision on the actual texture (the same condition
  // that gates `updateColormapTexture`), not just the attr's presence.
  const composedIntensity = (nodeAttrs.intensity as number | undefined) ?? 1.0;
  const composedOffset = (nodeAttrs.offset as number | undefined) ?? 0.0;
  const gsColormapName = nodeAttrs.colormap as string | undefined;
  const gsColormapTex = gsColormapName
    ? getColormapTexture(gsColormapName, nodeAttrs.customLutBytes as Uint8Array | undefined)
    : null;
  const hasColormap = !!gsColormapTex;

  const material: LuxarGSplatMaterial = materialManager.getGSplatMaterial({
    opacity: (nodeAttrs.opacity as number | undefined) ?? 1.0,
    absorption: (nodeAttrs.absorption as number | undefined) ?? 1.0,
    gamma: (nodeAttrs.gamma as number | undefined) ?? 1.0,
    // Identity color GOG when a colormap will be applied (the window is
    // pushed via `updateScalarRange` below); the authored gain only tints
    // direct-color nodes.
    intensity: hasColormap ? 1.0 : composedIntensity,
    offset: hasColormap ? 0.0 : composedOffset,
    blendingMode: (nodeAttrs.blending_mode as string | undefined as BlendingMode) ?? 'additive',
    truncationRadius: clampTruncationRadius(
      (attrs.truncation_radius as number | undefined) ?? GSPLAT_DEFAULT_TRUNCATION_RADIUS
    ),
  });

  // Apply the colormap. GSplat materials are PER NODE (each carries the
  // node's own `uSplatTex`), so the colormap applies directly to the
  // node-owned material — the historical clone-on-divergence dance is gone.
  if (gsColormapTex) {
    material.updateColormapTexture(gsColormapTex);
    // Scalar window == the display window the panel would recover from the
    // authored gain/offset (see `resolveColormapWindow` for the
    // leaf-vs-composed rule, shared with the points/lines factories).
    const leafRaw = attrs as unknown as Record<string, unknown>;
    const gsScalarRange = resolveColormapWindow(
      (nodeAttrs.amplitude_data_range as [number, number] | undefined) ?? [0, 1],
      {
        intensity: (leafRaw.intensity as number | undefined) ?? 1.0,
        offset: (leafRaw.offset as number | undefined) ?? 0.0,
      },
      { intensity: composedIntensity, offset: composedOffset }
    );
    material.updateScalarRange(gsScalarRange[0], gsScalarRange[1]);
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
    choleskyFactors: new Float32Array(0),
    amplitudes: new Float32Array(0),
    colors: new Float32Array(0),
    splatCount: 0,
  };
  return createGSplatsNode(path, nodeAttrs, attrs, emptyConfig, loader, pickingSystem);
}
