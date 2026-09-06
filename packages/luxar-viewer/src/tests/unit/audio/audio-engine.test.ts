// @vitest-environment jsdom
/**
 * `AudioEngine` — the graph it builds, ducking, mute, the autoplay gate, the
 * listener re-parenting on a camera swap, state and events. The camera, dims
 * manager and scene graph are ports, so this is a plain node/jsdom test on the
 * fake AudioContext.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import { AudioEngine, type AudioEngineDeps } from '../../../audio/audio-engine';
import type { SoundSourceDescriptor } from '../../../types/audio';
import type { SimpleDims } from '../../../types/dims';
import { StorageKeys } from '../../../utils/storage-keys';
import {
  FakeAudioContext,
  FakeGainNode,
  installFakeAudioContext,
} from '../../mocks/fake-audio-context.mock';

function storyDims(step: number): SimpleDims {
  return {
    ndim: 4,
    currentStep: [step, 0, 0, 0],
    displayed: [1, 2, 3],
    metadata: [
      { name: 'story', unit: '', scale: 1, discrete: true, step: 1, range: [0, 2], display: false },
      { name: 'x', unit: 'um', scale: 1, display: true },
      { name: 'y', unit: 'um', scale: 1, display: true },
      { name: 'z', unit: 'um', scale: 1, display: true },
    ],
  };
}

function soundPlaceholder(
  path: string,
  raw: Record<string, unknown>,
  rows?: number[][]
): THREE.Group {
  const g = new THREE.Group();
  g.name = path;
  g.userData.nodeType = 'sound';
  const desc: SoundSourceDescriptor = {
    path,
    name: path.split('/').pop()!,
    rawAttrs: { audio_file: 'audio.mp3', ...raw },
    positions: rows ? Float32Array.from(rows.flat()) : null,
    nPositions: rows?.length ?? 0,
    ndim: 4,
    readClip: async () => new Uint8Array([1, 2, 3, 4]),
  };
  g.userData.sound = desc;
  return g;
}

interface Harness {
  ctx: FakeAudioContext;
  engine: AudioEngine;
  root: THREE.Group;
  camera: THREE.PerspectiveCamera;
  swapCamera(): THREE.PerspectiveCamera;
  dims: SimpleDims;
  setStep(step: number): void;
  events: Array<{ event: string; name: string }>;
  uiChanges: number;
}

function makeHarness(state: 'running' | 'suspended' = 'running'): Harness {
  const ctx = installFakeAudioContext(state);
  let camera = new THREE.PerspectiveCamera();
  const cameraListeners = new Set<() => void>();
  const dimsListeners = new Set<() => void>();
  const h = {
    ctx,
    root: new THREE.Group(),
    dims: storyDims(0),
    events: [] as Array<{ event: string; name: string }>,
    uiChanges: 0,
  } as Harness;
  const deps: AudioEngineDeps = {
    getCamera: () => camera,
    onCameraReplaced: (cb) => {
      cameraListeners.add(cb);
      return () => cameraListeners.delete(cb);
    },
    getDims: () => h.dims,
    onDimsChanged: (cb) => {
      dimsListeners.add(cb);
      return () => dimsListeners.delete(cb);
    },
    getSceneGraph: () => null,
    getSceneScale: () => 100,
    container: () => document.body,
    emit: (event, payload) => h.events.push({ event, name: payload.name }),
    notifyUiChanged: () => {
      h.uiChanges++;
    },
  };
  h.engine = new AudioEngine(deps);
  Object.defineProperty(h, 'camera', { get: () => camera });
  h.swapCamera = () => {
    camera = new THREE.PerspectiveCamera();
    for (const cb of cameraListeners) cb();
    return camera;
  };
  h.setStep = (step) => {
    h.dims = storyDims(step);
    for (const cb of dimsListeners) cb();
  };
  return h;
}

async function flush(): Promise<void> {
  // decodeAll awaits readClip + decodeAudioData per node.
  for (let i = 0; i < 6; i++) await Promise.resolve();
}

beforeEach(() => {
  localStorage.clear();
  document.body.innerHTML = '';
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('AudioEngine — graph and lifecycle', () => {
  it('builds nothing for a scene without sound nodes', () => {
    const h = makeHarness();
    h.engine.attachScene(h.root);
    expect(h.engine.hasSoundNodes()).toBe(false);
    expect(h.engine.getState().state).toBe('unavailable');
    expect(h.ctx.gains).toHaveLength(0);
  });

  it('attaches the listener to the camera and re-parents it when the camera is swapped', async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    const listener = h.camera.children.find((c) => c instanceof THREE.AudioListener);
    expect(listener).toBeDefined();
    const next = h.swapCamera();
    expect(next.children).toContain(listener);
    await flush();
  });

  it('plays a bed everywhere, reports it, and detachScene stops it', async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['bed']);
    expect(h.events).toEqual([{ event: 'sound-started', name: 'bed' }]);
    h.engine.detachScene();
    expect(h.engine.hasSoundNodes()).toBe(false);
    expect(h.engine.getState().playing).toEqual([]);
  });
});

describe('AudioEngine — decode order', () => {
  it('decodes the nodes audible at the opening slice before the silent ones', async () => {
    const h = makeHarness();
    const reads: string[] = [];
    const tracked = (path: string, raw: Record<string, unknown>, rows?: number[][]) => {
      const g = soundPlaceholder(path, raw, rows);
      const desc = g.userData.sound as SoundSourceDescriptor;
      desc.readClip = async () => {
        reads.push(desc.name);
        return new Uint8Array([1]);
      };
      return g;
    };
    h.root.add(tracked('/narr_story2', { trigger: 'once' }, [[2, 0, 0, 0]]));
    h.root.add(tracked('/narr_story1', { trigger: 'once' }, [[1, 0, 0, 0]]));
    h.root.add(tracked('/bed', { trigger: 'continuous' }));
    h.dims = storyDims(1);
    h.engine.attachScene(h.root);
    await flush();
    await flush();
    expect(reads.slice(0, 2).sort()).toEqual(['bed', 'narr_story1']);
    expect(reads[2]).toBe('narr_story2');
  });
});

describe('AudioEngine — slab, ducking and buses', () => {
  it('a voice-bus once clip starts on its story and ducks the ambient bus, releasing when it ends', async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous', bus: 'ambient' }));
    h.root.add(soundPlaceholder('/narr', { trigger: 'once', bus: 'voice' }, [[1, 0, 0, 0]]));
    h.engine.applySceneConfig({ duckDb: -9 });
    h.engine.attachScene(h.root);
    await flush();
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['bed']);

    // The duck gain is the one ambient routes through: find it via the listener input.
    const duck = h.ctx.gains.find(
      (g) => g.outputs.size === 1 && g !== h.ctx.gains[0] && [...g.outputs][0] === h.ctx.gains[0]
    ) as FakeGainNode;
    expect(duck).toBeDefined();

    h.setStep(1);
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['bed', 'narr']);
    expect(duck.gain.lastRampTarget()).toBeCloseTo(Math.pow(10, -9 / 20), 3);

    // Narration runs out → duck releases.
    const narrSource = h.ctx.sources.find((s) => !s.loop)!;
    narrSource.end();
    expect(h.engine.getState().playing).toEqual(['bed']);
    expect(duck.gain.lastRampTarget()).toBe(1);
    expect(h.events.map((e) => `${e.event}:${e.name}`)).toEqual([
      'sound-started:bed',
      'sound-started:narr',
      'sound-ended:narr',
    ]);
  });

  it('bus gains from the scene config land on the bus nodes and setAudio patches them live', async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.applySceneConfig({ buses: { ambient: 0.3 }, masterGain: 0.5 });
    h.engine.attachScene(h.root);
    expect(h.engine.getState().buses.ambient).toBe(0.3);
    expect(h.engine.getState().masterGain).toBe(0.5);
    h.engine.setAudio({ buses: { voice: 0.9 }, masterGain: 1.2 });
    expect(h.engine.getState().buses.voice).toBe(0.9);
    expect(h.engine.getState().masterGain).toBe(1.2);
    await flush();
  });

  it('play(name) starts a node regardless of its slab and stop(name) ends it; unknown names are false', async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/sounds/narr', { trigger: 'once', bus: 'voice' }, [[2, 0, 0, 0]]));
    h.engine.attachScene(h.root);
    await flush();
    expect(h.engine.getState().playing).toEqual([]);
    expect(h.engine.play('narr')).toBe(true);
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['narr']);
    expect(h.engine.stop('/sounds/narr')).toBe(true);
    expect(h.engine.play('nope')).toBe(false);
  });
});

describe('AudioEngine — mute, prefs and the autoplay gate', () => {
  it('mute silences the master, persists, and unmute restores', async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();
    h.engine.setMuted(true);
    expect(h.engine.isMuted()).toBe(true);
    expect(JSON.parse(localStorage.getItem(StorageKeys.audio)!)).toMatchObject({ muted: true });
    expect(h.engine.getState().playing).toEqual([]);
    h.engine.setMuted(false);
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['bed']);
  });

  it('a persisted mute wins over the scene defaults on the next engine', () => {
    localStorage.setItem(StorageKeys.audio, JSON.stringify({ muted: true, masterGain: 0.4 }));
    const h = makeHarness();
    h.engine.applySceneConfig({ masterGain: 0.9 });
    expect(h.engine.isMuted()).toBe(true);
    expect(h.engine.getMasterGain()).toBe(0.4);
  });

  it('scene enabled:false reads as muted and shows no gate; the rail unmute re-enables', () => {
    const h = makeHarness('suspended');
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.applySceneConfig({ enabled: false });
    h.engine.attachScene(h.root);
    expect(h.engine.isMuted()).toBe(true);
    expect(document.querySelector('.luxar-audio-gate')).toBeNull();
    h.engine.setMuted(false);
    expect(h.engine.isMuted()).toBe(false);
  });

  it('shows the gate only when the context stays suspended, and a tap resumes + starts the deferred sounds', async () => {
    const h = makeHarness('suspended');
    h.ctx.resumeSucceeds = false;
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.root.add(soundPlaceholder('/narr', { trigger: 'once', bus: 'voice' }, [[0, 0, 0, 0]]));
    h.engine.attachScene(h.root);
    await flush();
    expect(document.querySelector('.luxar-audio-gate')).not.toBeNull();
    expect(h.engine.getState().state).toBe('suspended');
    expect(h.ctx.sources).toHaveLength(0);

    h.ctx.resumeSucceeds = true;
    document.dispatchEvent(new Event('pointerdown'));
    await flush();
    vi.advanceTimersByTime(1);
    expect(document.querySelector('.luxar-audio-gate')).toBeNull();
    expect(h.engine.getState().state).toBe('running');
    // Both the bed AND the opening `once` narration start on the tap.
    expect(h.engine.getState().playing.sort()).toEqual(['bed', 'narr']);
  });

  it('a muted scene never shows the gate', async () => {
    localStorage.setItem(StorageKeys.audio, JSON.stringify({ muted: true, masterGain: 0.8 }));
    const h = makeHarness('suspended');
    h.ctx.resumeSucceeds = false;
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();
    expect(document.querySelector('.luxar-audio-gate')).toBeNull();
  });

  it('dispose closes the context', async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();
    h.engine.dispose();
    expect(h.ctx.state).toBe('closed');
    expect(h.camera.children.some((c) => c instanceof THREE.AudioListener)).toBe(false);
  });
});
