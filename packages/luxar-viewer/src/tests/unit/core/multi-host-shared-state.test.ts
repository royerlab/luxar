/**
 * Page-global state on the multi-host path (a LuxarApp plus LuxarLayers, or
 * layers alone): what is per host now, and what is shared on purpose.
 *
 * - Materials: a host's node factory builds its nodes' materials in the HOST's
 *   `MaterialManager`, so another host's camera broadcast never reaches them.
 * - Data-worker pool: shared page-wide, released only by the last host.
 * - Blend warm-up: a commit is routed to the warm-up of the scene its node is in.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../rendering/node-factory';
import { MaterialManager, getPageMaterialManager } from '../../../rendering/material-manager';
import type { DataLoader } from '../../../data/data-loader-types';
import type { PointsMetadata } from '../../../types/points';
import { releaseWorkerPool, retainWorkerPool } from '../../../workers/worker-pool';
import {
  WebGLBlendWarmupManager,
  registerBlendWarmupManager,
  scheduleBlendModeProgramWarmupForObject,
  unregisterBlendWarmupManager,
} from '../../../rendering/webgl-blend-warmup';

function pointsPlaceholder(factory: NodeFactory): THREE.Mesh {
  const attrs = { n_points: 3 } as unknown as PointsMetadata;
  return factory.createEmptyPointsNode('/p', attrs, { dispose: vi.fn() } as unknown as DataLoader);
}

describe('materials are per host', () => {
  it("a layer factory's nodes take ONLY that layer's camera broadcast", () => {
    const layerMaterials = new MaterialManager();
    const layerFactory = new NodeFactory();
    layerFactory.setMaterialManager(layerMaterials);
    const layerNode = pointsPlaceholder(layerFactory);
    const appNode = pointsPlaceholder(new NodeFactory());

    const layerMaterial = layerNode.material as THREE.Material & {
      updateCameraParams: (...a: unknown[]) => void;
    };
    const appMaterial = appNode.material as THREE.Material & {
      updateCameraParams: (...a: unknown[]) => void;
    };
    const layerSpy = vi.spyOn(layerMaterial, 'updateCameraParams');
    const appSpy = vi.spyOn(appMaterial, 'updateCameraParams');

    // The app resizes: only the app's node hears it.
    getPageMaterialManager().updateCameraParams(new THREE.Vector2(1920, 1080), false, undefined, 2);
    expect(appSpy).toHaveBeenCalled();
    expect(layerSpy).not.toHaveBeenCalled();

    // The layer resizes: only the layer's node hears it.
    appSpy.mockClear();
    layerMaterials.updateCameraParams(new THREE.Vector2(400, 300), false, undefined, 1);
    expect(layerSpy).toHaveBeenCalled();
    expect(appSpy).not.toHaveBeenCalled();

    // …and the layer's teardown disposes its own materials, not the app's.
    const appDispose = vi.spyOn(appMaterial, 'dispose');
    layerMaterials.dispose();
    expect(appDispose).not.toHaveBeenCalled();
  });
});

describe('the data-worker pool is released by the last host', () => {
  it('keeps the pool while any host still holds it', () => {
    const app = {};
    const layer = {};
    retainWorkerPool(app);
    retainWorkerPool(layer);
    // A layer's dispose must not take the app's workers…
    expect(releaseWorkerPool(layer)).toBe(false);
    // …and the last host out terminates them.
    expect(releaseWorkerPool(app)).toBe(true);
  });

  it('still cleans up a pool nobody holds (a teardown after a failed init)', () => {
    expect(releaseWorkerPool(undefined)).toBe(true);
  });
});

describe('blend warm-up is routed to the scene a node lives in', () => {
  it("schedules a layer's node on the layer's warm-up, not the app's", () => {
    const layerScene = new THREE.Scene();
    const layerWarmup = new WebGLBlendWarmupManager();
    layerWarmup.configure({
      enabled: false,
      renderer: null,
      camera: null,
      targetScene: layerScene,
    });
    registerBlendWarmupManager(layerWarmup);
    const layerSchedule = vi.spyOn(layerWarmup, 'scheduleObject');
    const node = new THREE.Mesh();
    const group = new THREE.Group();
    group.add(node);
    layerScene.add(group);

    scheduleBlendModeProgramWarmupForObject(node);
    expect(layerSchedule).toHaveBeenCalledWith(node);

    // A node in no host's scene goes nowhere.
    layerSchedule.mockClear();
    scheduleBlendModeProgramWarmupForObject(new THREE.Mesh());
    expect(layerSchedule).not.toHaveBeenCalled();
    unregisterBlendWarmupManager(layerWarmup);
  });
});
