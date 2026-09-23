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
import type { SceneNode } from '../../../data/data-loader-types';
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
    resolveNodeCenter: (name) => (name === 'blob' ? new THREE.Vector3(4, 5, 6) : null),
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

  it('notifyWaypoint fires the on_arrive / on_depart nodes whose rows belong to the waypoint', async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/n1', { trigger: 'on_arrive', bus: 'voice' }, [[1, 0, 0, 0]]));
    h.root.add(soundPlaceholder('/n2', { trigger: 'on_arrive', bus: 'voice' }, [[2, 0, 0, 0]]));
    h.root.add(soundPlaceholder('/bye', { trigger: 'on_depart', bus: 'effects' }));
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['bed']);

    h.setStep(1);
    h.engine.notifyWaypoint('arrive', { story: 1 });
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['bed', 'n1']);

    // Departure: the row-less on_depart node belongs to every waypoint.
    h.engine.notifyWaypoint('depart', { story: 1 });
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toContain('bye');
    expect(h.engine.getState().playing).not.toContain('n2');
  });

  it('an arrive event does not discard an on_depart clip still decoding', async () => {
    const h = makeHarness();
    let release!: () => void;
    const clipReady = new Promise<void>((resolve) => {
      release = resolve;
    });
    const departing = soundPlaceholder('/bye', { trigger: 'on_depart', bus: 'voice' });
    const desc = departing.userData.sound as SoundSourceDescriptor;
    desc.readClip = async () => {
      await clipReady;
      return new Uint8Array([1, 2, 3, 4]);
    };
    h.root.add(departing);
    h.engine.attachScene(h.root);

    h.engine.notifyWaypoint('depart', { story: 0 });
    h.engine.notifyWaypoint('arrive', { story: 1 });
    release();
    await flush();
    vi.advanceTimersByTime(1);

    expect(h.engine.getState().playing).toEqual(['bye']);
  });

  it('setNodeMuted mutes one node by path or a whole subtree by ancestor, independently of the rail', async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/story/bed', { trigger: 'continuous' }));
    h.root.add(soundPlaceholder('/story/hum', { trigger: 'continuous' }));
    h.root.add(soundPlaceholder('/other', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['bed', 'hum', 'other']);

    h.engine.setNodeMuted('/story/hum', true);
    expect(h.engine.getState().playing).toEqual(['bed', 'other']);
    h.engine.setNodeMuted('/story', true);
    expect(h.engine.getState().playing).toEqual(['other']);
    // Un-hiding the group does not unmute the node's own eye.
    h.engine.setNodeMuted('/story', false);
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['bed', 'other']);
    h.engine.setNodeMuted('/story/hum', false);
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['bed', 'hum', 'other']);
    expect(h.engine.isMuted()).toBe(false);
  });

  it('keeps a nested sound muted until every hidden ancestor is visible', async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/a/b/hum', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['hum']);

    h.engine.setNodeMuted('/a', true);
    h.engine.setNodeMuted('/a/b', true);
    h.engine.setNodeMuted('/a/b', false);
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual([]);

    h.engine.setNodeMuted('/a', false);
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['hum']);
  });

  it('setNodeGain ramps a playing node and is the value later starts use', async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous', gain: 0.4 }));
    h.engine.attachScene(h.root);
    await flush();
    expect(h.engine.getNodeGain('/bed')).toBe(0.4);
    const voiceGain = h.ctx.gains.find((g) => g.gain.lastRampTarget() === 0.4)!;
    expect(voiceGain).toBeDefined();
    h.engine.setNodeGain('/bed', 1.1);
    expect(voiceGain.gain.lastRampTarget()).toBeCloseTo(1.1);
    expect(h.engine.getNodeGain('/bed')).toBeCloseTo(1.1);
    expect(h.engine.getNodeGain('/nope')).toBeUndefined();
  });

  it('an attach_to target without geometry yet falls back to its scene-graph position_bounds centre, then refreshes', async () => {
    const h = makeHarness();
    // No live centre for this name (the port only knows 'blob'), but the scene
    // graph carries the node's authored bounds: story dim 0, then x/y/z.
    const graph = {
      path: '',
      type: 'scene',
      attrs: {},
      children: [
        {
          path: '/Story 1: Hb',
          type: 'points',
          attrs: { position_bounds: { min: [1, 2, -8, 4], max: [1, 6, -6, 10] } },
          children: [],
        },
      ],
    } as unknown as SceneNode;
    (h.engine as unknown as { deps: AudioEngineDeps }).deps.getSceneGraph = () => graph;
    h.root.add(soundPlaceholder('/hum', { trigger: 'continuous', attach_to: 'Story 1: Hb' }));
    h.engine.attachScene(h.root);
    await flush();
    const voice = h.root.children[0].children.find(
      (c) => c instanceof THREE.PositionalAudio
    ) as THREE.PositionalAudio;
    expect(voice.position.toArray()).toEqual([4, -7, 7]);
    // Once the target has a live box, the per-frame refresh moves the voice there.
    (h.engine as unknown as { deps: AudioEngineDeps }).deps.resolveNodeCenter = () =>
      new THREE.Vector3(9, 9, 9);
    for (let i = 0; i < 30; i++) h.camera.updateMatrixWorld(true);
    expect(voice.position.toArray()).toEqual([9, 9, 9]);
  });

  it('an attach_to source sits at the named node centre', async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/hum', { trigger: 'continuous', attach_to: 'blob' }));
    h.engine.attachScene(h.root);
    await flush();
    const voice = h.root.children[0].children.find(
      (c) => c instanceof THREE.PositionalAudio
    ) as THREE.PositionalAudio;
    expect(voice).toBeDefined();
    expect(voice.position.toArray()).toEqual([4, 5, 6]);
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

  it('resets omitted scene mix fields to defaults when the dataset changes', () => {
    const h = makeHarness();
    h.engine.applySceneConfig({
      masterGain: 0.2,
      buses: { voice: 0.1 },
      panningModel: 'HRTF',
      duckDb: -40,
    });
    h.engine.applySceneConfig({});
    expect(h.engine.getMasterGain()).toBe(0.8);
    expect(h.engine.getBusGains()).toEqual({ ambient: 0.6, voice: 1, effects: 0.8 });
    expect(h.engine.getState().panningModel).toBe('equalpower');
  });

  it('does not notify the UI when a geometry visibility update reaches an empty engine', () => {
    const h = makeHarness();
    const before = h.uiChanges;
    h.engine.setNodeMuted('/cloud', true);
    expect(h.uiChanges).toBe(before);
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

  it('muting does not persist the current scene-authored master gain', () => {
    const h = makeHarness();
    h.engine.applySceneConfig({ masterGain: 0.3 });
    h.engine.setMuted(true);
    h.engine.setMuted(false);
    expect(JSON.parse(localStorage.getItem(StorageKeys.audio)!)).toEqual({ muted: false });
    h.engine.applySceneConfig({ masterGain: 0.9 });
    expect(h.engine.getMasterGain()).toBe(0.9);
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

  it('keeps the gate closed when a tap resume resolves but the context stays suspended', async () => {
    const h = makeHarness('suspended');
    h.ctx.resumeSucceeds = false;
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();

    document.dispatchEvent(new Event('pointerdown'));
    await flush();
    expect(h.engine.isBlocked()).toBe(true);
    expect(h.engine.getState().playing).toEqual([]);

    h.ctx.resumeSucceeds = true;
    document.dispatchEvent(new Event('keydown'));
    await flush();
    vi.advanceTimersByTime(1);
    expect(h.engine.isBlocked()).toBe(false);
    expect(h.engine.getState().playing).toEqual(['bed']);
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

  it("rotates ambisonic fields against the camera on the listener's per-frame update", async () => {
    const h = makeHarness();
    h.root.add(soundPlaceholder('/field', { trigger: 'continuous', ambisonic: 'foa' }));
    h.engine.attachScene(h.root);
    await flush();
    const before = h.ctx.gains.reduce((n, g) => n + g.gain.calls.length, 0);
    // Camera turns 90° left; three refreshes the listener from the camera each frame.
    h.camera.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
    h.camera.updateMatrixWorld(true);
    const after = h.ctx.gains.reduce((n, g) => n + g.gain.calls.length, 0);
    expect(after - before).toBe(9);
    // Still: no further automation.
    h.camera.updateMatrixWorld(true);
    expect(h.ctx.gains.reduce((n, g) => n + g.gain.calls.length, 0)).toBe(after);
  });

  it('acquireCaptureStream taps the master gain; release disconnects and stops the tracks', async () => {
    const h = makeHarness();
    expect(h.engine.acquireCaptureStream()).toBeNull(); // no graph yet
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();
    const stream = h.engine.acquireCaptureStream()!;
    expect(stream).not.toBeNull();
    const destination = h.ctx.streamDestinations[0];
    const master = h.ctx.gains[0];
    expect(master.outputs.has(destination)).toBe(true);
    expect(master.outputs.has(h.ctx.destination)).toBe(true);
    h.engine.releaseCaptureStream(stream);
    expect(master.outputs.has(destination)).toBe(false);
    expect(destination.track.stopped).toBe(true);
    // Unknown streams are ignored; dispose releases anything still held.
    h.engine.releaseCaptureStream(stream);
    h.engine.acquireCaptureStream();
    h.engine.dispose();
    expect(h.ctx.streamDestinations[1].track.stopped).toBe(true);
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

describe('AudioEngine — a context the browser has not released', () => {
  it('holds ONE deferred narration across many stops, not one per stop', async () => {
    const h = makeHarness('suspended');
    h.ctx.resumeNeverSettles = true;
    for (let i = 0; i < 4; i++) {
      h.root.add(
        soundPlaceholder(`/narr${i}`, { trigger: 'on_arrive', bus: 'voice' }, [[i, 0, 0, 0]])
      );
    }
    h.engine.attachScene(h.root);
    await flush();

    // Walk the tour with sound still blocked. Every arrival defers.
    for (let i = 0; i < 4; i++) {
      h.setStep(i);
      h.engine.notifyWaypoint('arrive', { story: i });
    }
    await flush();
    expect(h.engine.getState().playing).toEqual([]);

    // The browser lets the context through. Only the stop the listener is
    // actually on may speak; the three walked past must stay silent.
    h.ctx.unblock();
    await flush();
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['narr3']);
  });

  it('shows the gate when resume() never settles', async () => {
    const h = makeHarness('suspended');
    h.ctx.resumeNeverSettles = true;
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();
    expect(document.querySelector('.luxar-audio-gate')).not.toBeNull();
    expect(h.engine.isBlocked()).toBe(true);
    expect(h.engine.getState().playing).toEqual([]);
  });

  it('a gesture anywhere recovers, without the listener finding the mute toggle', async () => {
    const h = makeHarness('suspended');
    h.ctx.resumeNeverSettles = true;
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();
    expect(h.engine.getState().playing).toEqual([]);

    h.ctx.resumeNeverSettles = false;
    document.dispatchEvent(new Event('keydown'));
    await flush();
    vi.advanceTimersByTime(1);
    expect(h.engine.isBlocked()).toBe(false);
    expect(h.engine.getState().playing).toEqual(['bed']);
  });

  it('a context that reaches running on its own opens the gate', async () => {
    const h = makeHarness('suspended');
    h.ctx.resumeNeverSettles = true;
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();
    expect(h.engine.getState().playing).toEqual([]);

    // No call of ours: the browser simply released it, which is reported
    // through onstatechange and nothing else.
    h.ctx.unblock();
    await flush();
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['bed']);
    expect(document.querySelector('.luxar-audio-gate')).toBeNull();
  });

  it('blocked is not muted: the listener never chose it, and unmuting is not the cure', async () => {
    const h = makeHarness('suspended');
    h.ctx.resumeNeverSettles = true;
    h.root.add(soundPlaceholder('/bed', { trigger: 'continuous' }));
    h.engine.attachScene(h.root);
    await flush();
    expect(h.engine.isBlocked()).toBe(true);
    expect(h.engine.isMuted()).toBe(false);

    h.ctx.resumeNeverSettles = false;
    expect(await h.engine.enableSound()).toBe(true);
    expect(h.engine.isBlocked()).toBe(false);
    vi.advanceTimersByTime(1);
    expect(h.engine.getState().playing).toEqual(['bed']);
  });
});
