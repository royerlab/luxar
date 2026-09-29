/**
 * Band preload: a `kind=lod` group parked near a level threshold makes the
 * neighbouring level resident in the background, without drawing it, so the
 * time-based dissolve can start the moment the camera crosses the threshold.
 *
 * The retired coverage cross-fade DREW both levels inside a band of
 * ±0.4 × (the smaller adjacent inter-threshold gap) around each threshold, so
 * the finer level was already loaded when the switch came. The time-based
 * dissolve drew one level there, and the finer level's load only began at the
 * crossing: the coarse level stayed on screen for that whole load.
 *
 * Geometry used throughout: a gsplats leaf with bounds [0,0,0]–[0.3,0.3,0.3]
 * under an identity camera projects to a legacy coverage metric of 0.5 on the
 * 800×600 viewport (fitted axis 600, FILL_FACTOR 0.5). Scaling the group
 * object scales the metric linearly.
 */

import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

import {
  LODGroupRegistry,
  type LODGroupChild,
  type LODGroupEntry,
} from '../../../scene/lod-group-registry';

interface FadeMatStub {
  userData: { blendingMode: string };
  _op: number;
  updateOpacity(v: number): void;
  getOpacity(): number;
  clone(): FadeMatStub;
}

function fadeMat(blendingMode = 'additive'): FadeMatStub {
  return {
    userData: { blendingMode },
    _op: 1,
    updateOpacity(v: number) {
      this._op = v;
    },
    getOpacity() {
      return this._op;
    },
    clone() {
      return fadeMat(blendingMode);
    },
  };
}

/** Commit what a successful load stamps on the leaf, and flip it ready. */
function land(child: LODGroupChild, version = 2): void {
  Object.assign(child.object.userData, {
    loadedViewVersion: version,
    visibleSplatCount: 100,
    committedLadderComplete: true,
  });
  child.ready = true;
  child.loading = false;
}

/** An eager (always-resident) level. */
function eagerChild(coverageFraction: number): LODGroupChild {
  const mesh = new THREE.Mesh();
  mesh.material = fadeMat() as unknown as THREE.Material;
  mesh.userData = { nodeType: 'gsplats' };
  const child: LODGroupChild = {
    object: mesh,
    coverageFraction,
    positionBounds: { min: [0, 0, 0], max: [0.3, 0.3, 0.3] },
  };
  land(child);
  return child;
}

/** A lazy level: nothing committed yet, with a load spy and a release thunk. */
function lazyChild(coverageFraction: number): {
  child: LODGroupChild;
  ensureLoaded: ReturnType<typeof vi.fn>;
} {
  const mesh = new THREE.Mesh();
  mesh.material = fadeMat() as unknown as THREE.Material;
  mesh.userData = { nodeType: 'gsplats' };
  mesh.visible = false;
  const ensureLoaded = vi.fn();
  const child: LODGroupChild = {
    object: mesh,
    coverageFraction,
    positionBounds: { min: [0, 0, 0], max: [0.3, 0.3, 0.3] },
    ready: false,
    ensureLoaded: ensureLoaded as () => void,
  };
  child.release = () => {
    child.ready = false;
    child.loading = false;
  };
  return { child, ensureLoaded };
}

function makeEntry(children: LODGroupChild[], active = 0): LODGroupEntry {
  return {
    path: '/g',
    groupObject: new THREE.Group(),
    children,
    selectorMode: 'auto',
    defaultLevel: active,
    activeChildIndex: active,
  };
}

function makeReg(
  opts: { crossFade?: boolean; playbackPeriodMs?: () => number | null; now?: () => number } = {}
): LODGroupRegistry {
  const camera = new THREE.Camera();
  camera.matrixWorldInverse.identity();
  camera.projectionMatrix.identity();
  let t = 0;
  return new LODGroupRegistry({
    getCamera: () => camera,
    getViewportSize: () => ({ width: 800, height: 600 }),
    getDisplayDims: () => [0, 1, 2],
    getViewVersion: () => 2,
    getCrossFadeEnabled: () => opts.crossFade ?? true,
    getPlaybackPeriodMs: opts.playbackPeriodMs,
    now: opts.now ?? (() => (t += 1000 / 60)),
  });
}

const opacity = (c: LODGroupChild): number =>
  ((c.object as THREE.Mesh).material as unknown as FadeMatStub).getOpacity();

describe('LODGroupRegistry — band preload of the neighbouring level', () => {
  it.fails('parked inside the band below a finer threshold: loads the finer level without drawing it', () => {
    // Thresholds [0, 0.7]: band half-width 0.4 × 0.7 = 0.28 → [0.42, 0.98].
    // Metric 0.5 sits inside it, below the threshold, so the coarse level is
    // the one selected and displayed.
    const reg = makeReg();
    const coarse = eagerChild(0);
    const { child: fine, ensureLoaded } = lazyChild(0.7);
    reg.register(makeEntry([coarse, fine]));

    reg.evaluatePerFrame();
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
    expect(fine.object.visible).toBe(false);
    expect(coarse.object.visible).toBe(true);
    expect(reg.get('/g')!.displayedChildIndex).toBe(0);

    // The load lands: the finer level is resident but still not drawn, and
    // its opacity is untouched (no blend).
    land(fine);
    for (let i = 0; i < 5; i++) reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(false);
    expect(opacity(fine)).toBe(1);
    expect(coarse.object.visible).toBe(true);
    expect(opacity(coarse)).toBe(1);
    expect(reg.get('/g')!.displayedChildIndex).toBe(0);
    expect(reg.isAnimating()).toBe(false);
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
  });

  it('outside the band: does not load the finer level', () => {
    // Thresholds [0, 0.9]: band [0.54, 1.26]; metric 0.5 is outside it.
    const reg = makeReg();
    const coarse = eagerChild(0);
    const { child: fine, ensureLoaded } = lazyChild(0.9);
    reg.register(makeEntry([coarse, fine]));
    for (let i = 0; i < 5; i++) reg.evaluatePerFrame();
    expect(ensureLoaded).not.toHaveBeenCalled();
    expect(fine.object.visible).toBe(false);
  });

  it('with the dissolve off (?noLodFade): does not preload', () => {
    const reg = makeReg({ crossFade: false });
    const coarse = eagerChild(0);
    const { child: fine, ensureLoaded } = lazyChild(0.7);
    reg.register(makeEntry([coarse, fine]));
    for (let i = 0; i < 5; i++) reg.evaluatePerFrame();
    expect(ensureLoaded).not.toHaveBeenCalled();
  });

  it('during playback: does not preload', () => {
    const reg = makeReg({ playbackPeriodMs: () => 100 });
    const coarse = eagerChild(0);
    const { child: fine, ensureLoaded } = lazyChild(0.7);
    reg.register(makeEntry([coarse, fine]));
    for (let i = 0; i < 5; i++) reg.evaluatePerFrame();
    expect(ensureLoaded).not.toHaveBeenCalled();
  });

  it.fails('crossing the threshold after the preload starts the dissolve on that frame', () => {
    const reg = makeReg();
    const coarse = eagerChild(0);
    const { child: fine, ensureLoaded } = lazyChild(0.7);
    const entry = makeEntry([coarse, fine]);
    reg.register(entry);
    reg.evaluatePerFrame();
    expect(ensureLoaded).toHaveBeenCalledTimes(1); // the preload, before any crossing
    land(fine);
    reg.evaluatePerFrame();

    // Zoom in past the threshold: metric 0.5 × 1.6 = 0.8 ≥ 0.7.
    entry.groupObject.scale.setScalar(1.6);
    reg.evaluatePerFrame();
    expect(reg.get('/g')!.displayedChildIndex).toBe(1);
    expect(fine.object.visible).toBe(true);
    expect(coarse.object.visible).toBe(true); // the outgoing half of the dissolve
    expect(opacity(fine)).toBeLessThan(1);
    expect(reg.isAnimating()).toBe(true);
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
  });

  it.fails('parked inside the band above a coarser threshold: loads the coarser level', () => {
    // Thresholds [0, 0.3, 0.45]; metric 0.5 selects the finest. Its boundary
    // 0.45 has a band of 0.4 × min(0.15, 0.15) = 0.06 → [0.39, 0.51].
    const reg = makeReg();
    const coarse = eagerChild(0);
    const { child: mid, ensureLoaded } = lazyChild(0.3);
    const fine = eagerChild(0.45);
    reg.register(makeEntry([coarse, mid, fine], 2));
    reg.evaluatePerFrame();
    expect(reg.get('/g')!.displayedChildIndex).toBe(2);
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
    expect(mid.object.visible).toBe(false);
  });

  it.fails('does not reload a preloaded level evicted while the group stays parked', () => {
    const reg = makeReg();
    const coarse = eagerChild(0);
    const { child: fine, ensureLoaded } = lazyChild(0.7);
    const entry = makeEntry([coarse, fine]);
    reg.register(entry);
    reg.evaluatePerFrame();
    land(fine);
    reg.evaluatePerFrame();

    // VRAM pressure releases the hidden level. Parked in the band, the
    // registry must not load it straight back (load → evict → load …).
    fine.release!();
    for (let i = 0; i < 10; i++) reg.evaluatePerFrame();
    expect(ensureLoaded).toHaveBeenCalledTimes(1);

    // Leaving the band (metric 0.25) and coming back is a new visit.
    entry.groupObject.scale.setScalar(0.5);
    reg.evaluatePerFrame();
    entry.groupObject.scale.setScalar(1);
    reg.evaluatePerFrame();
    expect(ensureLoaded).toHaveBeenCalledTimes(2);
  });

  it.fails('a level released while its layer was hidden is preloaded again once the layer is shown', () => {
    const reg = makeReg();
    const coarse = eagerChild(0);
    const { child: fine, ensureLoaded } = lazyChild(0.7);
    const entry = makeEntry([coarse, fine]);
    reg.register(entry);
    reg.evaluatePerFrame();
    land(fine);
    reg.evaluatePerFrame();

    // Hidden layer: its levels are the first to go under VRAM pressure.
    entry.groupObject.visible = false;
    reg.evaluatePerFrame();
    fine.release!();
    reg.evaluatePerFrame();
    expect(ensureLoaded).toHaveBeenCalledTimes(1);

    entry.groupObject.visible = true;
    reg.evaluatePerFrame();
    expect(ensureLoaded).toHaveBeenCalledTimes(2);
    expect(fine.object.visible).toBe(false);
  });

  it('does not preload while the selected level itself is still loading', () => {
    // Thresholds [0, 0.3, 0.6]; metric 0.5 selects the lazy middle level,
    // which is not ready yet, and sits in the 0.6 boundary's band [0.48, 0.72].
    // The selected level's load has the fetches to itself.
    const reg = makeReg();
    const coarse = eagerChild(0);
    const { child: mid, ensureLoaded: midLoad } = lazyChild(0.3);
    const { ensureLoaded: fineLoad, child: fine } = lazyChild(0.6);
    reg.register(makeEntry([coarse, mid, fine]));
    reg.evaluatePerFrame();
    expect(midLoad).toHaveBeenCalledTimes(1);
    expect(fineLoad).not.toHaveBeenCalled();
    expect(fine.object.visible).toBe(false);
  });
});
