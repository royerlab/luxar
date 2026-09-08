/**
 * A fake Web Audio `AudioContext` for the sound-layer unit tests.
 *
 * Node has no Web Audio; three's `AudioListener` / `Audio` / `PositionalAudio`
 * only need the handful of node factories and params below, and
 * `THREE.AudioContext.setContext(fake)` makes three build on it. Every
 * `AudioParam` records its automation calls so a test can assert a ramp was
 * scheduled (value, time) without simulating the audio clock; `advance()` moves
 * `currentTime` so delayed starts can be reasoned about.
 */

import * as THREE from 'three';

/** An `AudioParam` that records every automation call. */
export class FakeAudioParam {
  value: number;
  readonly calls: Array<{ method: string; value?: number; time: number }> = [];
  constructor(initial = 1) {
    this.value = initial;
  }
  setValueAtTime(value: number, time: number): this {
    this.calls.push({ method: 'setValueAtTime', value, time });
    this.value = value;
    return this;
  }
  linearRampToValueAtTime(value: number, time: number): this {
    this.calls.push({ method: 'linearRampToValueAtTime', value, time });
    this.value = value;
    return this;
  }
  exponentialRampToValueAtTime(value: number, time: number): this {
    this.calls.push({ method: 'exponentialRampToValueAtTime', value, time });
    this.value = value;
    return this;
  }
  setTargetAtTime(value: number, time: number, _tc: number): this {
    this.calls.push({ method: 'setTargetAtTime', value, time });
    this.value = value;
    return this;
  }
  cancelScheduledValues(time: number): this {
    this.calls.push({ method: 'cancelScheduledValues', time });
    return this;
  }
  /** The last ramp target scheduled on this param, or undefined. */
  lastRampTarget(): number | undefined {
    for (let i = this.calls.length - 1; i >= 0; i--) {
      const c = this.calls[i];
      if (c.method === 'linearRampToValueAtTime') return c.value;
    }
    return undefined;
  }
}

/** Common connect/disconnect bookkeeping. */
export class FakeAudioNode {
  readonly outputs = new Set<FakeAudioNode | AudioDestinationNode>();
  /** Every connect call with its output/input indices (channel routing tests). */
  readonly connections: Array<{
    target: FakeAudioNode | AudioDestinationNode;
    output: number;
    input: number;
  }> = [];
  readonly context: FakeAudioContext;
  constructor(context: FakeAudioContext) {
    this.context = context;
  }
  connect(
    target: FakeAudioNode | AudioDestinationNode,
    output = 0,
    input = 0
  ): FakeAudioNode | AudioDestinationNode {
    this.outputs.add(target);
    this.connections.push({ target, output, input });
    return target;
  }
  disconnect(target?: FakeAudioNode | AudioDestinationNode): void {
    if (target) this.outputs.delete(target);
    else this.outputs.clear();
  }
}

/** `GainNode` stand-in. */
export class FakeGainNode extends FakeAudioNode {
  readonly gain = new FakeAudioParam(1);
}

/** `PannerNode` stand-in with the params three writes. */
export class FakePannerNode extends FakeAudioNode {
  panningModel = 'equalpower';
  distanceModel = 'inverse';
  refDistance = 1;
  maxDistance = 10000;
  rolloffFactor = 1;
  coneInnerAngle = 360;
  coneOuterAngle = 360;
  coneOuterGain = 0;
  readonly positionX = new FakeAudioParam(0);
  readonly positionY = new FakeAudioParam(0);
  readonly positionZ = new FakeAudioParam(0);
  readonly orientationX = new FakeAudioParam(1);
  readonly orientationY = new FakeAudioParam(0);
  readonly orientationZ = new FakeAudioParam(0);
}

/** `AudioBufferSourceNode` stand-in recording starts/stops; `end()` fires `onended`. */
export class FakeBufferSource extends FakeAudioNode {
  buffer: FakeAudioBuffer | null = null;
  loop = false;
  loopStart = 0;
  loopEnd = 0;
  readonly playbackRate = new FakeAudioParam(1);
  readonly detune = new FakeAudioParam(0);
  onended: (() => void) | null = null;
  readonly starts: Array<{ when: number; offset?: number; duration?: number }> = [];
  readonly stops: number[] = [];
  start(when = 0, offset?: number, duration?: number): void {
    this.starts.push({ when, offset, duration });
    this.context.sources.push(this);
  }
  stop(when = 0): void {
    this.stops.push(when);
  }
  /** Simulate the clip running out. */
  end(): void {
    this.onended?.();
  }
}

/** `ChannelSplitterNode` / `ChannelMergerNode` stand-in recording per-channel wiring. */
export class FakeChannelNode extends FakeAudioNode {
  /** `(output index, target, input index)` triples for every connect call. */
  readonly channelLinks: Array<{ output: number; target: FakeAudioNode; input: number }> = [];
  constructor(
    context: FakeAudioContext,
    readonly channels: number
  ) {
    super(context);
  }
  connect(
    target: FakeAudioNode | AudioDestinationNode,
    output = 0,
    input = 0
  ): FakeAudioNode | AudioDestinationNode {
    this.channelLinks.push({ output, target: target as FakeAudioNode, input });
    return super.connect(target);
  }
}

/** `MediaStreamAudioDestinationNode` stand-in: one fake audio track. */
export class FakeMediaStreamDestination extends FakeAudioNode {
  readonly track = { kind: 'audio', stopped: false, stop: (): void => undefined };
  readonly stream: MediaStream;
  constructor(context: FakeAudioContext) {
    super(context);
    const track = this.track;
    track.stop = () => {
      track.stopped = true;
    };
    this.stream = {
      getAudioTracks: () => [track],
      getTracks: () => [track],
    } as unknown as MediaStream;
  }
}

/** `AudioBuffer` stand-in (duration only). */
export class FakeAudioBuffer {
  constructor(
    readonly duration = 1,
    readonly sampleRate = 48000,
    readonly numberOfChannels = 2
  ) {}
  get length(): number {
    return Math.round(this.duration * this.sampleRate);
  }
}

/** The context listener params three's `AudioListener.updateMatrixWorld` writes. */
export class FakeAudioListener {
  readonly positionX = new FakeAudioParam(0);
  readonly positionY = new FakeAudioParam(0);
  readonly positionZ = new FakeAudioParam(0);
  readonly forwardX = new FakeAudioParam(0);
  readonly forwardY = new FakeAudioParam(0);
  readonly forwardZ = new FakeAudioParam(-1);
  readonly upX = new FakeAudioParam(0);
  readonly upY = new FakeAudioParam(1);
  readonly upZ = new FakeAudioParam(0);
}

/** The fake context: node factories, decode, resume/suspend/close, a manual clock. */
export class FakeAudioContext {
  currentTime = 0;
  state: 'suspended' | 'running' | 'closed';
  readonly destination = { name: 'destination' } as unknown as AudioDestinationNode;
  readonly listener = new FakeAudioListener();
  readonly sources: FakeBufferSource[] = [];
  readonly gains: FakeGainNode[] = [];
  readonly panners: FakePannerNode[] = [];
  readonly streamDestinations: FakeMediaStreamDestination[] = [];
  onstatechange: (() => void) | null = null;
  resumeCalls = 0;
  /** What `resume()` does: flip to running (default) or stay suspended. */
  resumeSucceeds = true;
  decodeCalls = 0;

  constructor(initialState: 'suspended' | 'running' = 'running') {
    this.state = initialState;
  }
  createGain(): FakeGainNode {
    const g = new FakeGainNode(this);
    this.gains.push(g);
    return g;
  }
  createPanner(): FakePannerNode {
    const p = new FakePannerNode(this);
    this.panners.push(p);
    return p;
  }
  createBufferSource(): FakeBufferSource {
    return new FakeBufferSource(this);
  }
  createChannelSplitter(channels = 6): FakeChannelNode {
    return new FakeChannelNode(this, channels);
  }
  createChannelMerger(channels = 6): FakeChannelNode {
    return new FakeChannelNode(this, channels);
  }
  createMediaStreamDestination(): FakeMediaStreamDestination {
    const d = new FakeMediaStreamDestination(this);
    this.streamDestinations.push(d);
    return d;
  }
  async decodeAudioData(_data: ArrayBuffer): Promise<FakeAudioBuffer> {
    this.decodeCalls++;
    return new FakeAudioBuffer(2);
  }
  async resume(): Promise<void> {
    this.resumeCalls++;
    if (this.resumeSucceeds && this.state === 'suspended') {
      this.state = 'running';
      this.onstatechange?.();
    }
  }
  async suspend(): Promise<void> {
    this.state = 'suspended';
    this.onstatechange?.();
  }
  async close(): Promise<void> {
    this.state = 'closed';
    this.onstatechange?.();
  }
  advance(seconds: number): void {
    this.currentTime += seconds;
  }
}

/**
 * Install a fresh fake as three's module-level context and return it. Call in
 * `beforeEach`; three caches the context, so a stale one from a previous test
 * would otherwise leak across files.
 */
export function installFakeAudioContext(
  initialState: 'suspended' | 'running' = 'running'
): FakeAudioContext {
  const fake = new FakeAudioContext(initialState);
  THREE.AudioContext.setContext(fake as unknown as AudioContext);
  return fake;
}
