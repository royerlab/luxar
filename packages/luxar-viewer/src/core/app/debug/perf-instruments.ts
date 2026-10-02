/**
 * Debug-only perf instruments: the probes `?debug` sessions add on top of the
 * always-on perf counters.
 *
 * Installed from the standalone bootstrap's debug seed — BEFORE `init()` — so
 * a probe reading `getPerf()` during the first load (the window the perf gates
 * measure) sees them, and again from `installDebugInterface` for embedders
 * that turn on `debug` without the bootstrap. Every step is idempotent.
 *
 * @module core/app/debug/perf-instruments
 */

import * as THREE from 'three';
import { perfCounters } from '../../../profiling/perf-counters';
import { setLodLoadStatsEnabled } from '../../../data/scene-loader/lod-load-stats';
import { installRendererInfoSampler, type RendererInfoSource } from './renderer-info-sampler';

/** Marks `Object3D.prototype.getObjectByName` once it counts its calls. */
const GET_OBJECT_BY_NAME_COUNTED = Symbol.for('luxar.getObjectByNameCounted');

/**
 * Count every `Object3D.getObjectByName` call in the `scene.getObjectByName`
 * perf counter (a full subtree walk each — a hot-path smell worth gating on).
 * Patches a three prototype, so only debug sessions install it. Idempotent;
 * the wrapper forwards arguments and result unchanged.
 */
function installGetObjectByNameCounter(): void {
  const proto = THREE.Object3D.prototype as THREE.Object3D & Record<symbol, unknown>;
  if (proto[GET_OBJECT_BY_NAME_COUNTED]) return;
  Object.defineProperty(proto, GET_OBJECT_BY_NAME_COUNTED, { value: true });
  const slot = perfCounters.slot('scene.getObjectByName');
  const original = proto.getObjectByName;
  proto.getObjectByName = function countedGetObjectByName(this: THREE.Object3D, name: string) {
    perfCounters.add(slot);
    return original.call(this, name);
  };
}

/**
 * Turn on the debug perf instruments.
 *
 * - Lazy-LOD-load per-stage timing plus additive per-level ladder timing
 *   (`__luxarDebug.getLodLoadStats()`): the lazy `ensureLoaded` loads run
 *   outside any updateView cycle, so the UpdateProfiler never captures them.
 * - The `scene.getObjectByName` call counter (patches a three prototype).
 * - The `renderer.info` sampler behind `getPerf().rendererInfo`: between frames
 *   Three's autoReset leaves only the last post-processing pass in it. The
 *   renderer is read through `getRenderer` at every frame, so it may resolve
 *   to nothing until the app has built one. Re-installing replaces the prior
 *   subscription.
 */
export function installDebugPerfInstruments(
  getRenderer: () => RendererInfoSource | null | undefined
): void {
  setLodLoadStatsEnabled(true);
  installGetObjectByNameCounter();
  installRendererInfoSampler(getRenderer);
}
