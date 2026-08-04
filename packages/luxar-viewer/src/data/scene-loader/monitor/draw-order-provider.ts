/**
 * Live draw-order provider for the data-loading monitor's Scene Graph tab.
 *
 * Reads the cross-node draw order straight off the THREE root group each
 * tick: per data mesh the blending bucket + `depthWrite` (from the material)
 * and the resolved `renderOrder` the depth-sort coordinator assigned. Keyed
 * by scene-graph path (mesh `name`, stamped by the node factory), so the
 * monitor tree can show it next to the LOD chip.
 *
 * Pull (polled) rather than push because `renderOrder` is recomputed every
 * frame from the camera pose (`rendering/depth-sort-coordinator/render-order.ts`);
 * a per-tick read keeps the panel truthful as the view orbits. Mirrors
 * `visible-counts.ts` (the other THREE-reading, path-keyed monitor feed) and
 * is injected via `SceneLoaderMonitorPort.setDrawOrderProvider`.
 *
 * @module data/scene-loader/monitor/draw-order-provider
 */

import * as THREE from 'three';
import type { DrawOrderProvider, NodeDrawOrder } from '../../../types/data-monitor-types';
import { LOADER_TYPES, type LoaderTypeName } from '../../../types/format-contract';

/**
 * The viewer-drawable node types (the loader set: points / lines / gsplats).
 * Keyed on `LOADER_TYPES`, not `GEOMETRY_TYPES`, so drawability stays a single
 * capability — `computeDrawOrder` (debug-state) classifies identically, and a
 * future `mesh` loader is admitted in exactly one place.
 */
const DATA_NODE_TYPES: ReadonlySet<string> = new Set<LoaderTypeName>(LOADER_TYPES);

/**
 * Build a {@link DrawOrderProvider} over the live THREE root group. The group
 * is created fresh per scene load, so the provider closes over the one for
 * this scene (a reload installs a new provider — same lifecycle as the
 * LOD-progress provider).
 */
export function createDrawOrderProvider(rootGroup: THREE.Group | null): DrawOrderProvider {
  return {
    getDrawOrderStates(): Map<string, NodeDrawOrder> {
      const states = new Map<string, NodeDrawOrder>();
      if (!rootGroup) return states;

      // Manual recursion (not THREE's `traverse`, which descends into
      // `visible === false` subtrees). `renderOrder` is only ever assigned to
      // VISIBLE sorted meshes and never reset, so a hidden mesh (toggled-off
      // layer, inactive substitutive-LOD level) keeps a stale value — pruning
      // hidden subtrees keeps the panel honest. Mirrors `visible-counts.ts`.
      const visit = (object: THREE.Object3D): void => {
        if (!object.visible) return;
        if (object instanceof THREE.Mesh) {
          const nodeType = (object.userData as { nodeType?: string }).nodeType;
          if (nodeType && DATA_NODE_TYPES.has(nodeType) && object.name) {
            // Materials can be arrays; bucket + depthWrite are shared, so the
            // first material is representative.
            const material = Array.isArray(object.material) ? object.material[0] : object.material;
            states.set(object.name, {
              bucket: material?.transparent ? 'transparent' : 'opaque',
              depthWrite: !!material?.depthWrite,
              renderOrder: object.renderOrder,
            });
          }
        }
        for (const child of object.children) visit(child);
      };
      // The root group's own visibility shouldn't gate the whole scene;
      // descend straight into its children (matches visible-counts.ts).
      for (const child of rootGroup.children) visit(child);

      return states;
    },
  };
}
