/**
 * The LOD-group registry's dependencies, built in ONE place for both owners of
 * a render loop: the app pipeline (`pipeline.ts`) and the host-embedded layer
 * (`core/layer/luxar-layer.ts`). The two used to hand-copy this object, and the
 * layer's copy had drifted (no `getViewContext`). The layer still omits the two
 * optional deps only a playback-driving loop has — `getPlaybackPeriodMs` (it
 * has no dimension player: a host scrubbing `setDimensionValue` is never
 * "playing", so lazy levels keep the settle debounce) and `requestTick` (its
 * keep-alive calls fall back to `requestRender`, i.e. a host redraw).
 *
 * Everything that reads the LOADER is routed to the owning loader the factory
 * receives, never to `getSceneLoader('default')`, so a non-default loader's
 * registry consults its own pool, fault latch and view version.
 *
 * @module core/app/init/lod-registry-deps
 */

import type * as THREE from 'three';

import type { LODGroupRegistryOwner } from '../../../data/scene-loader';
import type { LODGroupRegistryDeps } from '../../../scene/lod-group-registry';
import type { ViewContext } from '../../../scene/view-context';
import { getGpuByteBudget } from '../../../rendering/gpu-byte-budget';

/** What the render-loop owner supplies; every accessor is read live. */
export interface LodRegistryWiring {
  /** The live camera (a getter: the ortho toggle / a host swaps the object). */
  getCamera(): THREE.Camera;
  /** Viewport size in CSS pixels. */
  getViewportSize(): { width: number; height: number };
  /** Displayed dimensions from this render-loop owner's scene. */
  getDisplayDims(): number[];
  /** Register fade clones with this owner's material manager. */
  registerMaterial(material: THREE.Material): void;
  /** The frame's shared camera snapshot; omitted ⇒ the registry builds its own. */
  getViewContext?(): ViewContext;
  /**
   * Playback period of the fastest playing dimension, or null: lazy LOD levels
   * reload every timepoint while playing, capped at the finest one whose
   * measured load fits the period. Omitted (no playback driver) ⇒ never capped.
   */
  getPlaybackPeriodMs?(): number | null;
  /** Ask for a redraw (a lazy fine level landed, a swap is due). */
  requestRender(): void;
  /**
   * Keep the loop ticking without a redraw while a lazy level loads. Omitted ⇒
   * the registry falls back to `requestRender` (a host that renders on wake).
   */
  requestTick?(): void;
  /** LOD cross-fade (ON by default; `?noLodFade` disables). Read once. */
  lodFade?: boolean;
  /** Streaming brightness compensation (ON by default; `?noLodEnergy`). Read once. */
  lodEnergyComp?: boolean;
  /** Force-finest capture override (`?lodFinest`). Read once. */
  lodFinest?: boolean;
  /** Area-unit threshold bias (`?lodBias`); the registry owns the neutral default. */
  lodBias?: number;
}

/** Build one registry's deps for `owner` (the loader whose registry it is). */
export function buildLodRegistryDeps(
  owner: LODGroupRegistryOwner,
  wiring: LodRegistryWiring
): LODGroupRegistryDeps {
  const lodFade = wiring.lodFade ?? true;
  const lodEnergyComp = wiring.lodEnergyComp ?? true;
  const lodFinest = wiring.lodFinest ?? false;
  const lodBias = wiring.lodBias;
  return {
    getCamera: () => wiring.getCamera(),
    getViewportSize: () => wiring.getViewportSize(),
    ...(wiring.getViewContext ? { getViewContext: wiring.getViewContext } : {}),
    // Empty, not [0, 1, 2], before dims resolve: the registry's
    // `displayDims.length < 2` early-return then skips evaluation, whereas the
    // plausible-looking default projects a 2D scene onto a phantom Z.
    getDisplayDims: () => wiring.getDisplayDims(),
    hasArchiveFault: () => owner.archiveFault !== null,
    hasNetworkFailureUnder: (path) => owner.hasNetworkFailureUnder(path),
    requestReprocess: (paths) => owner.requestReprocess(paths),
    // A view PASS in flight or queued — not a refinement hold, which the loader
    // parks a resync through (see `LODGroupRegistryOwner`).
    isUpdateInProgress: () => owner.isLoadPassInProgress(),
    // The view the drawn geometry was committed for: partition parts that miss
    // its hidden-dim slice are hidden and kept out of refinement (B4).
    getCommittedViewState: () => owner.committedViewState,
    // The single adaptive GPU-geometry budget shared with the buffer pool (one
    // VRAM authority), read live so context-loss backoff applies.
    getResidentByteBudget: () => getGpuByteBudget(),
    // Both halves of the budget are required: `lod-eviction` bails on
    // `!getResidentBytes`, so the budget alone would never demote a cold level.
    // A null pool (pre-construction / pooling off) reads as 0 bytes.
    getResidentBytes: () => owner.gpuBufferPool?.getResidentBytes() ?? 0,
    // Lets the registry tell a level's committed geometry is stale for the
    // current slice and show a coarser FRESH level until the re-slice commits.
    getViewVersion: () => owner.currentViewVersion,
    requestRender: () => wiring.requestRender(),
    ...(wiring.requestTick ? { requestTick: wiring.requestTick } : {}),
    getCrossFadeEnabled: () => lodFade,
    ...(wiring.getPlaybackPeriodMs ? { getPlaybackPeriodMs: wiring.getPlaybackPeriodMs } : {}),
    getEnergyCompEnabled: () => lodEnergyComp,
    getForceFinestLOD: () => lodFinest,
    getLodBias: () => lodBias,
    // A fade's clone-on-first-use material keeps receiving per-frame camera
    // uniforms (an unregistered gsplat clone would project with stale params).
    registerMaterial: (material) => wiring.registerMaterial(material),
  };
}
