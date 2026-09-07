/**
 * `SoundNode` — bus routing, the edges, fades, gate deferral, spatial voices.
 * Runs in node against the fake AudioContext three is pointed at.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import * as THREE from 'three';
import { SoundNode, type SoundNodeDeps } from '../../../audio/sound-node';
import { FoaAudio, type FoaDecoder } from '../../../audio/foa-decoder';
import { buildSoundBaseViewState } from '../../../audio/audibility';
import { parseSoundNodeAttrs } from '../../../audio/sound-attrs';
import type { SoundSourceDescriptor } from '../../../types/audio';
import type { SimpleDims } from '../../../types/dims';
import {
  FakeAudioBuffer,
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

function descriptor(
  overrides: Partial<SoundSourceDescriptor> & { rows?: number[][] } = {}
): SoundSourceDescriptor {
  const rows = overrides.rows;
  const positions = rows ? Float32Array.from(rows.flat()) : null;
  return {
    path: '/sounds/clip',
    name: 'clip',
    rawAttrs: {},
    positions,
    nPositions: rows ? rows.length : 0,
    ndim: 4,
    readClip: async () => new Uint8Array([1, 2, 3]),
    ...overrides,
  };
}

let ctx: FakeAudioContext;
let listener: THREE.AudioListener;
let bus: FakeGainNode;
let parent: THREE.Group;
let started: string[];
let ended: string[];

function makeDeps(): SoundNodeDeps {
  return {
    listener,
    bus: bus as unknown as GainNode,
    panningModel: 'equalpower',
    parent,
    onStarted: (n) => started.push(n),
    onEnded: (n) => ended.push(n),
  };
}

function makeNode(raw: Record<string, unknown>, desc = descriptor()): SoundNode {
  const attrs = parseSoundNodeAttrs(desc.path, { audio_file: 'audio.mp3', ...raw }, 100);
  return new SoundNode(desc, attrs, makeDeps());
}

beforeEach(() => {
  vi.useFakeTimers();
  ctx = installFakeAudioContext('running');
  listener = new THREE.AudioListener();
  bus = ctx.createGain();
  parent = new THREE.Group();
  started = [];
  ended = [];
});

afterEach(() => {
  vi.useRealTimers();
});

describe('SoundNode — routing and voices', () => {
  it('routes the voice gain into the bus, not the listener input', () => {
    const node = makeNode({ trigger: 'continuous' });
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    const voice = parent.children[0] as THREE.Audio<GainNode>;
    expect(voice).toBeInstanceOf(THREE.Audio);
    const gain = voice.gain as unknown as FakeGainNode;
    expect(gain.outputs.has(bus)).toBe(true);
    expect(gain.outputs.has(listener.getInput() as unknown as FakeGainNode)).toBe(false);
  });

  it('a spatial node gets one PositionalAudio per row at the displayed xyz, panning overwritten', () => {
    const desc = descriptor({
      rows: [
        [1, 10, 20, 30],
        [2, -5, 0, 5],
      ],
    });
    const node = makeNode({ spatial: true, ref_distance: 3, max_distance: 40 }, desc);
    node.setViewState(buildSoundBaseViewState(storyDims(0)), null);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    expect(parent.children).toHaveLength(2);
    const v0 = parent.children[0] as THREE.PositionalAudio;
    expect(v0).toBeInstanceOf(THREE.PositionalAudio);
    expect([v0.position.x, v0.position.y, v0.position.z]).toEqual([10, 20, 30]);
    expect(v0.panner.panningModel).toBe('equalpower');
    expect(v0.panner.refDistance).toBe(3);
    expect(v0.panner.maxDistance).toBe(40);
  });

  it('uses xyz columns before the first view-state update', () => {
    const node = makeNode({ spatial: true }, descriptor({ rows: [[10, 20, 30]], ndim: 3 }));
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    const voice = parent.children[0] as THREE.PositionalAudio;
    expect(voice.position.toArray()).toEqual([10, 20, 30]);
  });
});

describe('SoundNode — waypoint triggers and attach_to', () => {
  it('on_arrive ignores the rising slab edge, fires on the trigger, and fades out when its story is left', () => {
    const desc = descriptor({ rows: [[1, 0, 0, 0]] });
    const node = makeNode({ trigger: 'on_arrive', fade_out_ms: 400 }, desc);
    node.setGateOpen(true);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    node.setViewState(buildSoundBaseViewState(storyDims(1)), null);
    // Audible by the slab, but nothing starts: the driver owns the start.
    expect(ctx.sources).toHaveLength(0);
    expect(node.isWaypointTriggered).toBe(true);

    expect(node.triggerFromWaypoint('depart', null)).toBe(false);
    expect(node.triggerFromWaypoint('arrive', null)).toBe(true);
    expect(ctx.sources).toHaveLength(1);
    expect(ctx.sources[0].loop).toBe(false);

    // Moving on cuts the narration with its fade.
    node.setViewState(buildSoundBaseViewState(storyDims(2)), null);
    expect(ctx.sources[0].stops[0]).toBeCloseTo(0.4);
    expect(node.isPlaying).toBe(false);
  });

  it('on_depart plays out although its own row has just left the slab', () => {
    const desc = descriptor({ rows: [[1, 0, 0, 0]] });
    const node = makeNode({ trigger: 'on_depart' }, desc);
    node.setGateOpen(true);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    node.setViewState(buildSoundBaseViewState(storyDims(1)), null);
    // The story is left: the slab falls and the departure fires.
    node.setViewState(buildSoundBaseViewState(storyDims(2)), null);
    node.triggerFromWaypoint('depart', null);
    expect(ctx.sources).toHaveLength(1);
    expect(node.isPlaying).toBe(true);
    // Further slab changes do not stop it.
    node.setViewState(buildSoundBaseViewState(storyDims(0)), null);
    expect(node.isPlaying).toBe(true);
  });

  it('a trigger before the gate opens, or before the clip decodes, is kept pending', () => {
    const early = makeNode({ trigger: 'on_arrive' });
    early.triggerFromWaypoint('arrive', null); // no buffer, no voices yet
    early.setGateOpen(true);
    expect(ctx.sources).toHaveLength(0);
    early.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    expect(ctx.sources).toHaveLength(1);

    const gated = makeNode({ trigger: 'on_arrive' });
    gated.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    gated.triggerFromWaypoint('arrive', null); // gate closed
    expect(ctx.sources).toHaveLength(1);
    gated.setGateOpen(true);
    expect(ctx.sources).toHaveLength(2);
  });

  it('a spatial trigger starts only the rows that belong to the waypoint', () => {
    const desc = descriptor({
      rows: [
        [1, 0, 0, 0],
        [2, 5, 5, 5],
      ],
    });
    const node = makeNode({ trigger: 'on_arrive', spatial: true }, desc);
    node.setGateOpen(true);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    node.triggerFromWaypoint('arrive', Uint8Array.from([0, 1]));
    expect(ctx.sources).toHaveLength(1);
    expect((parent.children[1] as THREE.PositionalAudio).isPlaying).toBe(true);
  });

  it('attach_to places every voice at the target centre in the placeholder frame, re-resolved on evaluate', () => {
    const desc = descriptor({ rows: [[1, 0, 0, 0]] });
    parent.position.set(10, 0, 0);
    let centre: THREE.Vector3 | null = null;
    const attrs = parseSoundNodeAttrs(
      desc.path,
      { audio_file: 'audio.mp3', attach_to: 'blob', trigger: 'continuous' },
      100
    );
    expect(attrs.spatial).toBe(true);
    const node = new SoundNode(desc, attrs, { ...makeDeps(), resolveAttachCenter: () => centre });
    node.setGateOpen(true);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    const voice = parent.children[0] as THREE.PositionalAudio;
    expect(voice).toBeInstanceOf(THREE.PositionalAudio);
    // Unknown target: stays at the row's own displayed xyz.
    expect(voice.position.toArray()).toEqual([1, 0, 0]);
    centre = new THREE.Vector3(13, 2, 1);
    node.setViewState(buildSoundBaseViewState(storyDims(1)), null);
    expect(voice.position.toArray()).toEqual([3, 2, 1]);
  });
});

describe('SoundNode — ambisonic field', () => {
  it('an ambisonic node gets one FoaAudio voice, registers its decoder, and is never spatial', () => {
    const registered: unknown[] = [];
    const desc = descriptor({ rows: [[1, 0, 0, 0]] });
    const attrs = parseSoundNodeAttrs(
      desc.path,
      { audio_file: 'audio.m4a', ambisonic: 'foa', spatial: true, trigger: 'continuous' },
      100
    );
    expect(attrs.ambisonic).toBe('foa');
    expect(attrs.spatial).toBe(false);
    const node = new SoundNode(desc, attrs, {
      ...makeDeps(),
      registerFoaDecoder: (d) => registered.push(d),
      unregisterFoaDecoder: (d) => registered.splice(registered.indexOf(d), 1),
    });
    node.setGateOpen(true);
    node.setBuffer(new FakeAudioBuffer(1, 48000, 4) as unknown as AudioBuffer);
    expect(parent.children).toHaveLength(1);
    expect(parent.children[0]).toBeInstanceOf(FoaAudio);
    expect(registered).toHaveLength(1);
    node.setViewState(buildSoundBaseViewState(storyDims(1)), null);
    expect(node.isPlaying).toBe(true);
    // The source feeds the decoder, not the gain directly.
    expect(ctx.sources[0].outputs.has((registered[0] as FoaDecoder).input as never)).toBe(true);
    node.dispose();
    expect(registered).toHaveLength(0);
  });
});

describe('SoundNode — edges', () => {
  it('continuous: rising edge loops, starts after delay and ramps in; falling edge ramps out and stops', () => {
    const desc = descriptor({ rows: [[1, 0, 0, 0]] });
    const node = makeNode(
      { trigger: 'continuous', delay_ms: 250, fade_in_ms: 1000, fade_out_ms: 500, gain: 0.7 },
      desc
    );
    node.setGateOpen(true);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    node.setViewState(buildSoundBaseViewState(storyDims(0)), null);
    expect(ctx.sources).toHaveLength(0);

    node.setViewState(buildSoundBaseViewState(storyDims(1)), null);
    expect(ctx.sources).toHaveLength(1);
    const src = ctx.sources[0];
    expect(src.loop).toBe(true);
    expect(src.starts[0].when).toBeCloseTo(0.25);
    const gain = (parent.children[0] as THREE.Audio<GainNode>).gain as unknown as FakeGainNode;
    expect(gain.gain.lastRampTarget()).toBeCloseTo(0.7);
    expect(node.isPlaying).toBe(true);
    vi.advanceTimersByTime(250);
    expect(started).toEqual(['clip']);

    node.setViewState(buildSoundBaseViewState(storyDims(2)), null);
    expect(gain.gain.lastRampTarget()).toBe(0);
    expect(src.stops[0]).toBeCloseTo(0.5);
    expect(node.isPlaying).toBe(false);
    vi.advanceTimersByTime(500);
    expect(ended).toEqual(['clip']);
  });

  it('stopping during the start delay emits neither lifecycle event', () => {
    const node = makeNode(
      { trigger: 'continuous', delay_ms: 600 },
      descriptor({ rows: [[1, 0, 0, 0]] })
    );
    node.setGateOpen(true);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    node.setViewState(buildSoundBaseViewState(storyDims(1)), null);
    vi.advanceTimersByTime(100);
    node.setViewState(buildSoundBaseViewState(storyDims(2)), null);
    vi.runAllTimers();
    expect(started).toEqual([]);
    expect(ended).toEqual([]);
  });

  it('once: fires once per rising edge and not again while staying audible', () => {
    const desc = descriptor({ rows: [[1, 0, 0, 0]] });
    const node = makeNode({ trigger: 'once' }, desc);
    node.setGateOpen(true);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    node.setViewState(buildSoundBaseViewState(storyDims(1)), null);
    expect(ctx.sources).toHaveLength(1);
    expect(ctx.sources[0].loop).toBe(false);
    vi.advanceTimersByTime(0);
    // Same slice again (a scrub inside the slab): no restart.
    node.setViewState(buildSoundBaseViewState(storyDims(1)), null);
    expect(ctx.sources).toHaveLength(1);
    // The clip runs out → ended once.
    ctx.sources[0].end();
    expect(ended).toEqual(['clip']);
    // Leave and come back → fires again.
    node.setViewState(buildSoundBaseViewState(storyDims(0)), null);
    node.setViewState(buildSoundBaseViewState(storyDims(1)), null);
    expect(ctx.sources).toHaveLength(2);
  });

  it('a node without positions is audible everywhere and starts as soon as the buffer lands', () => {
    const node = makeNode({ trigger: 'continuous' });
    node.setGateOpen(true);
    node.setViewState(buildSoundBaseViewState(storyDims(0)), null);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    expect(ctx.sources).toHaveLength(1);
    node.setViewState(buildSoundBaseViewState(storyDims(2)), null);
    expect(ctx.sources).toHaveLength(1);
    expect(node.isPlaying).toBe(true);
  });

  it('while the gate is closed nothing starts; opening replays the pending rising edge (once included)', () => {
    const desc = descriptor({ rows: [[1, 0, 0, 0]] });
    const node = makeNode({ trigger: 'once' }, desc);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    node.setViewState(buildSoundBaseViewState(storyDims(1)), null);
    expect(ctx.sources).toHaveLength(0);
    node.setGateOpen(true);
    expect(ctx.sources).toHaveLength(1);
  });

  it('mute fades out and unmute replays the audible voices', () => {
    const node = makeNode({ trigger: 'continuous', fade_out_ms: 300 });
    node.setGateOpen(true);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    node.setViewState(buildSoundBaseViewState(storyDims(0)), null);
    expect(node.isPlaying).toBe(true);
    node.setMuted(true);
    expect(node.isPlaying).toBe(false);
    expect(ctx.sources[0].stops[0]).toBeCloseTo(0.3);
    node.setMuted(false);
    expect(ctx.sources).toHaveLength(2);
    expect(node.isPlaying).toBe(true);
  });

  it('manual play() starts regardless of the slab and stop() fades out', () => {
    const desc = descriptor({ rows: [[1, 0, 0, 0]] });
    const node = makeNode({ trigger: 'once' }, desc);
    node.setGateOpen(true);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    node.setViewState(buildSoundBaseViewState(storyDims(0)), null); // silent slice
    expect(node.play()).toBe(true);
    expect(ctx.sources).toHaveLength(1);
    expect(node.stop()).toBe(true);
    expect(node.isPlaying).toBe(false);
    expect(node.stop()).toBe(false);
  });

  it('dispose stops and detaches every voice', () => {
    const node = makeNode({ trigger: 'continuous' });
    node.setGateOpen(true);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    node.setViewState(buildSoundBaseViewState(storyDims(0)), null);
    node.dispose();
    expect(parent.children).toHaveLength(0);
    expect(node.isPlaying).toBe(false);
  });
});
