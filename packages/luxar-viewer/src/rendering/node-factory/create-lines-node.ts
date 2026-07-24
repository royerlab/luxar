/**
 * Lines-node creation helpers for NodeFactory.
 *
 * `createLinesNode` resolves the line material backend through
 * materialManager (PER NODE — each material carries the node's own
 * `uLineTex`), applies the colormap directly when scalars + a non-null
 * colormap are requested, then builds the InstancedLinesMesh +
 * optional picking shadow node.
 *
 * @module rendering/node-factory/create-lines-node
 */

import * as THREE from 'three';
import { materialManager, type BlendingMode } from '../material-manager';
import { getColormapTexture } from '../colormap-textures';
import { supportsScalarColormap } from '../material-colormap-helpers';
import { syncLineMaterialWithGeometry } from '../material-sync-helpers';
import { createInstancedLinesMesh, type InstancedLinesMeshConfig } from '../line-geometry';
import { clampLineCapacity } from '../element-texture-layout';
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
  // Rendering attrs come from `nodeAttrs` — the COMPOSED effective attrs
  // the loader passes in (`ctx.applyEffectiveAttrs(node)`) — so an
  // ancestor-authored opacity/intensity/blending_mode reaches the
  // material even in scenes with no `layer=true` node (the layers-panel
  // recompose path only exists for layers). Mirrors `createGSplatsNode`
  // and `createPointsMaterial` (points passes the composed attrs as its
  // sole attrs param); reading the RAW `attrs` here silently dropped
  // ancestor values until the first panel interaction, if ever.
  // Per-leaf geometry properties (`max_width`, `transform`) stay on
  // `attrs`: they are deliberately NOT composited (see COMPOSITING_ATTRS).
  const material = materialManager.getLineMaterial({
    opacity: (nodeAttrs.opacity as number | undefined) ?? 1.0,
    absorption: (nodeAttrs.absorption as number | undefined) ?? 1.0,
    gamma: (nodeAttrs.gamma as number | undefined) ?? 1.0,
    intensity: (nodeAttrs.intensity as number | undefined) ?? 1.0,
    offset: (nodeAttrs.offset as number | undefined) ?? 0.0,
    blendingMode: (nodeAttrs.blending_mode as string | undefined as BlendingMode) ?? 'additive',
  });

  const mesh = createInstancedLinesMesh(processed, material);
  mesh.name = path;

  // Apply colormap if specified and scalar data exists. Line materials
  // are PER NODE (each carries the node's own `uLineTex`), so the
  // colormap applies directly to the node-owned material — the
  // historical clone-on-divergence dance is gone (mirrors
  // `createPointsMaterial`). Presence rides the `userData.hasScalars`
  // stamp `createInstancedLinesMesh` just wrote (fail-closed).
  const lnColormapName = nodeAttrs.colormap as string | undefined;
  const lnHasScalars = !!nodeAttrs.has_scalars;
  if (lnColormapName && lnHasScalars) {
    if (!supportsScalarColormap('lines', mesh.geometry)) {
      log.warning(
        Modules.SCENE_LOADER,
        `[${path}] Line scalar colormap requested but no scalar data is bound in the line texture. Colormap suppressed; rendering with vertex colors.`
      );
    } else {
      const lnLutBytes = nodeAttrs.customLutBytes as Uint8Array | undefined;
      const lnColormapTex = getColormapTexture(lnColormapName, lnLutBytes);
      if (lnColormapTex) {
        material.updateColormapTexture(lnColormapTex);
        const lnScalarRange = (nodeAttrs.scalar_data_range as [number, number]) ?? [0, 1];
        material.updateScalarRange(lnScalarRange[0], lnScalarRange[1]);
      }
    }
  }

  mesh.userData = {
    nodeType: 'lines',
    loader,
    attrs,
    maxWidth: attrs.max_width ?? 1.0,
    // Clamped like the commit path's stamp — the geometry above wrote at
    // most the per-node texture bound, and debug/UI counts must agree
    // with drawn instances (mirrors createPointsNode / createGSplatsNode).
    visibleSegmentCount: clampLineCapacity(processed.segmentCount),
    // Per-node material from creation: LayersPanel and the LOD
    // cross-fade honor this marker and mutate the material directly
    // instead of clone-on-first-use (mirrors createPointsNode /
    // createGSplatsNode).
    _layerMaterialCloned: true,
  } as LinesUserData;

  // Bind the geometry-owned line texture on the render material right
  // away so a mesh created WITH data renders before any commit (node
  // factory initial data, tests) — mirrors createPointsNode.
  syncLineMaterialWithGeometry(mesh);

  if (attrs.transform) applyTransform(mesh, attrs.transform);

  if (pickingSystem) {
    const pickId = pickingSystem.allocatePickId();
    mesh.userData.pickId = pickId;
    const pickMaterial = materialManager.createLinePickingMaterial({ nodeId: pickId });
    materialManager.register(pickMaterial);
    // Share the same InstancedBufferGeometry — only material differs.
    const pickNode = new THREE.Mesh(mesh.geometry, pickMaterial);
    pickNode.matrixWorld.copy(mesh.matrixWorld);
    pickingSystem.registerNode(mesh, pickNode, pickId);
    // Bind the geometry-owned line texture on BOTH materials (the
    // render material was bound above; this covers the just-created
    // pick material so picking works before the first commit's sync).
    syncLineMaterialWithGeometry(mesh);
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
  // When the node carries a scalar field + colormap, declare empty
  // scalar arrays on the placeholder so `createInstancedLinesMesh`
  // stamps `userData.hasScalars = true` and the fail-closed colormap
  // guard in `createLinesNode` passes at material-creation time.
  // Without them the guard sees no scalars, logs "Colormap suppressed",
  // and nothing ever re-enables the LUT once real scalars stream in.
  // Gate on the SAME `nodeAttrs` fields the colormap-application path
  // above reads, so the placeholder matches exactly when colormap will
  // apply. Mirrors `create-points-node.ts::createEmptyPointsNode`.
  if (nodeAttrs.colormap && nodeAttrs.has_scalars) {
    emptyConfig.startScalars = new Float32Array(0);
    emptyConfig.endScalars = new Float32Array(0);
  }
  return createLinesNode(path, nodeAttrs, attrs, emptyConfig, loader, pickingSystem);
}
