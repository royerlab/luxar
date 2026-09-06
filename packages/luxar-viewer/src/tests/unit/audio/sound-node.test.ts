/**
 * `SoundNode` — bus routing, the edges, fades, gate deferral, spatial voices.
 * Runs in node against the fake AudioContext three is pointed at.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import * as THREE from 'three';
import { SoundNode, type SoundNodeDeps } from '../../../audio/sound-node';
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

  it('once: fires once per rising edge and not again while staying audible', () => {
    const desc = descriptor({ rows: [[1, 0, 0, 0]] });
    const node = makeNode({ trigger: 'once' }, desc);
    node.setGateOpen(true);
    node.setBuffer(new FakeAudioBuffer() as unknown as AudioBuffer);
    node.setViewState(buildSoundBaseViewState(storyDims(1)), null);
    expect(ctx.sources).toHaveLength(1);
    expect(ctx.sources[0].loop).toBe(false);
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
