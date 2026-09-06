/**
 * One playing sound node: K `THREE.PositionalAudio` voices (spatial) or one
 * `THREE.Audio` (non-spatial), its buffer decoded once, fades on the node gain,
 * and the audible ⇄ silent edges the slab rule produces.
 *
 * Edge rules (`SOUND_SPEC.md` §3.1 / §4.1), per voice:
 *
 * - rising + `continuous` → loop, start after `delay_ms`, ramp 0 → `gain` over
 *   `fade_in_ms`;
 * - rising + `once` → start once after `delay_ms`, ramp in, report the end;
 * - falling (either) → ramp to 0 over `fade_out_ms`, then stop;
 * - a rising edge during a fade-out restarts the voice.
 *
 * While the autoplay gate is closed, edges are still tracked but nothing starts;
 * opening the gate re-runs the rising edges — `once` nodes included, so the
 * opening story's narration is not lost to the tap (a deliberate reading of
 * §4.4, which only names `continuous`).
 *
 * Three's `Audio` wires its gain node straight to the listener input in its
 * constructor; `play()` only reconnects the SOURCE side. So the bus routing here
 * (`gain.disconnect(); gain.connect(bus)`) is done once and survives every
 * start. `PositionalAudio` hard-codes `panningModel = 'HRTF'`; the engine's
 * choice overwrites it.
 *
 * @module audio/sound-node
 */

import * as THREE from 'three';
import type { SceneNode, ViewState } from '../data/data-loader-types';
import type { PanningModel, SoundNodeAttrs, SoundSourceDescriptor } from '../types/audio';
import { computeRowAudibility, displayedXYZ } from './audibility';
import { rampGain } from './fades';

/** Everything a node needs from the engine, as ports. */
export interface SoundNodeDeps {
  listener: THREE.AudioListener;
  /** The bus gain node this node routes into (already attrs.bus). */
  bus: GainNode;
  panningModel: PanningModel;
  /** The node's placeholder group (carries the node transform); voices attach here. */
  parent: THREE.Object3D;
  onStarted(name: string): void;
  onEnded(name: string): void;
}

/** One playing source: a three `Audio` (non-spatial) or `PositionalAudio` (one per row). */
export interface Voice {
  audio: THREE.Audio<GainNode> | THREE.PositionalAudio;
  row: number;
  audible: boolean;
  /** True from a start until the voice ended or a stop was scheduled. */
  playing: boolean;
  startTimer?: ReturnType<typeof setTimeout>;
  stopTimer?: ReturnType<typeof setTimeout>;
}

const _forward = new THREE.Vector3(0, 0, 1);

/** One `sound` scene node's voices, edges and fades (see the module docs). */
export class SoundNode {
  readonly name: string;
  readonly path: string;
  private buffer: AudioBuffer | null = null;
  private readonly voices: Voice[] = [];
  private readonly mask: Uint8Array;
  private gateOpen = false;
  private muted = false;
  private disposed = false;
  /** Last derived per-row audibility, kept so a deferred start can replay it. */
  private lastBase: ViewState | null = null;
  private lastSceneGraph: SceneNode | null = null;

  constructor(
    readonly desc: SoundSourceDescriptor,
    readonly attrs: SoundNodeAttrs,
    private readonly deps: SoundNodeDeps
  ) {
    this.name = desc.name;
    this.path = desc.path;
    this.mask = new Uint8Array(Math.max(1, desc.nPositions));
  }

  /** Number of currently playing voices. */
  get isPlaying(): boolean {
    return this.voices.some((v) => v.playing);
  }

  /**
   * True when the current slice makes this node audible (before or after its
   * buffer arrived). The engine decodes audible nodes first, so the opening
   * story's bed and narration are not queued behind ten silent clips.
   */
  get isAudibleNow(): boolean {
    const rows = this.rowAudibility();
    if (rows === null) return true;
    for (let i = 0; i < rows.length; i++) if (rows[i]) return true;
    return false;
  }

  private get context(): AudioContext {
    return this.deps.listener.context;
  }

  /**
   * Hand the decoded clip over. Voices are created here (lazily, once) so a node
   * whose clip never arrives costs nothing but a placeholder. Pending rising
   * edges recorded before the buffer arrived are replayed.
   */
  setBuffer(buffer: AudioBuffer): void {
    if (this.disposed) return;
    this.buffer = buffer;
    if (this.voices.length === 0) this.createVoices();
    for (const v of this.voices) v.audio.setBuffer(buffer);
    this.applyEdges(true);
  }

  private createVoices(): void {
    const spatial = this.attrs.spatial && this.desc.positions !== null && this.desc.nPositions > 0;
    const count = spatial ? this.desc.nPositions : 1;
    for (let row = 0; row < count; row++) {
      const audio = spatial
        ? new THREE.PositionalAudio(this.deps.listener)
        : new THREE.Audio(this.deps.listener);
      // Route past the listener input into the bus (see module docs).
      audio.gain.disconnect();
      audio.gain.connect(this.deps.bus);
      audio.gain.gain.value = 0;
      audio.setLoop(this.attrs.loop);
      if (audio instanceof THREE.PositionalAudio) this.configurePanner(audio);
      audio.name = `${this.path}#${row}`;
      this.deps.parent.add(audio);
      const voice: Voice = { audio, row, audible: false, playing: false };
      // Three binds `source.onended` to `this.onEnded` at play() time, so an
      // instance override is what runs when a `once` clip runs out.
      audio.onEnded = () => {
        THREE.Audio.prototype.onEnded.call(audio);
        if (voice.playing) {
          voice.playing = false;
          this.deps.onEnded(this.name);
        }
      };
      this.voices.push(voice);
    }
    this.repositionVoices();
  }

  private configurePanner(audio: THREE.PositionalAudio): void {
    const a = this.attrs;
    audio.panner.panningModel = this.deps.panningModel;
    audio.setDistanceModel(a.distance_model);
    audio.setRefDistance(a.ref_distance);
    audio.setMaxDistance(a.max_distance);
    audio.setRolloffFactor(a.rolloff);
    if (a.cone_inner_deg !== undefined && a.cone_outer_deg !== undefined) {
      audio.setDirectionalCone(a.cone_inner_deg, a.cone_outer_deg, a.cone_outer_gain ?? 0);
    }
    if (a.orientation) {
      const dir = new THREE.Vector3(...a.orientation).normalize();
      audio.quaternion.setFromUnitVectors(_forward, dir);
    }
  }

  /** Place each spatial voice at its row's displayed x/y/z. */
  private repositionVoices(): void {
    const positions = this.desc.positions;
    const displayDims =
      this.lastBase?.displayDims ?? [1, 2, 3].slice(0, Math.min(3, this.desc.ndim));
    if (!positions) return;
    for (const v of this.voices) {
      if (!(v.audio instanceof THREE.PositionalAudio)) continue;
      const [x, y, z] = displayedXYZ(positions, v.row, this.desc.ndim, displayDims);
      v.audio.position.set(x, y, z);
    }
  }

  /**
   * Re-evaluate audibility for the current slice and apply the edges.
   * Called by the engine on every dimension change (and once at attach).
   */
  setViewState(base: ViewState, sceneGraph: SceneNode | null): void {
    if (this.disposed) return;
    this.lastBase = base;
    this.lastSceneGraph = sceneGraph;
    this.repositionVoices();
    this.applyEdges(false);
  }

  /** Per-row audibility for the last view state (null = live everywhere). */
  private rowAudibility(): Uint8Array | null {
    if (!this.desc.positions || this.desc.nPositions === 0) return null;
    if (!this.lastBase) {
      this.mask.fill(0);
      return this.mask;
    }
    computeRowAudibility(
      this.desc,
      this.attrs.extend_to_all,
      this.lastBase,
      this.lastSceneGraph,
      this.mask
    );
    return this.mask;
  }

  /**
   * Compare the current audibility against each voice's recorded state and act
   * on the edges. `replayRising` treats every audible voice as a rising edge
   * (used when the gate opens or the buffer arrives), so deferred starts fire.
   */
  private applyEdges(replayRising: boolean): void {
    const rows = this.rowAudibility();
    const spatialVoices = this.voices.length > 1 || (this.attrs.spatial && rows !== null);
    let anyAudible = false;
    if (rows) {
      for (let i = 0; i < rows.length; i++) if (rows[i]) anyAudible = true;
    } else {
      anyAudible = true;
    }
    for (const v of this.voices) {
      const audible = spatialVoices && rows ? rows[v.row] === 1 : anyAudible;
      const was = v.audible;
      v.audible = audible;
      if (audible && (!was || (replayRising && !v.playing))) {
        this.startVoice(v);
      } else if (!audible && was) {
        this.stopVoice(v, this.attrs.fade_out_ms);
      }
    }
  }

  private startVoice(v: Voice): void {
    if (!this.buffer || !this.gateOpen || this.muted) return;
    if (v.stopTimer) {
      clearTimeout(v.stopTimer);
      v.stopTimer = undefined;
    }
    if (v.audio.isPlaying || v.playing) {
      // Restart (a re-rise during a fade-out, or a `once` re-triggered).
      v.audio.stop();
      v.playing = false;
    }
    const ctx = this.context;
    const now = ctx.currentTime;
    const delaySec = this.attrs.delay_ms / 1000;
    const startAt = now + delaySec;
    v.audio.setLoop(this.attrs.loop);
    rampGain(v.audio.gain.gain, now, startAt, this.attrs.gain, this.attrs.fade_in_ms, 0);
    v.audio.play(delaySec);
    v.playing = true;
    if (v.startTimer) clearTimeout(v.startTimer);
    v.startTimer = setTimeout(() => {
      v.startTimer = undefined;
      if (v.playing) this.deps.onStarted(this.name);
    }, this.attrs.delay_ms);
  }

  private stopVoice(v: Voice, fadeMs: number): void {
    if (v.startTimer) {
      clearTimeout(v.startTimer);
      v.startTimer = undefined;
    }
    if (!v.playing && !v.audio.isPlaying) return;
    const ctx = this.context;
    const now = ctx.currentTime;
    const end = rampGain(v.audio.gain.gain, now, now, 0, fadeMs);
    const wasPlaying = v.playing;
    v.playing = false;
    if (v.audio.isPlaying) v.audio.stop(end - now);
    if (v.stopTimer) clearTimeout(v.stopTimer);
    v.stopTimer = setTimeout(
      () => {
        v.stopTimer = undefined;
        if (wasPlaying) this.deps.onEnded(this.name);
      },
      Math.ceil((end - now) * 1000)
    );
  }

  /** Autoplay gate: while closed nothing starts; opening replays pending edges. */
  setGateOpen(open: boolean): void {
    if (this.gateOpen === open) return;
    this.gateOpen = open;
    if (open) this.applyEdges(true);
  }

  /** Mute stops the voices with a fade; unmute replays the audible ones. */
  setMuted(muted: boolean): void {
    if (this.muted === muted) return;
    this.muted = muted;
    if (muted) {
      for (const v of this.voices) this.stopVoice(v, this.attrs.fade_out_ms);
    } else {
      this.applyEdges(true);
    }
  }

  setPanningModel(model: PanningModel): void {
    for (const v of this.voices) {
      if (v.audio instanceof THREE.PositionalAudio) v.audio.panner.panningModel = model;
    }
  }

  /** Manual start (embedder API): plays regardless of the slab. */
  play(): boolean {
    if (!this.buffer || this.voices.length === 0) return false;
    for (const v of this.voices) {
      v.audible = true;
      this.startVoice(v);
    }
    return true;
  }

  /** Manual stop (embedder API), with the node's fade-out. */
  stop(): boolean {
    let any = false;
    for (const v of this.voices) {
      if (v.playing || v.audio.isPlaying) any = true;
      this.stopVoice(v, this.attrs.fade_out_ms);
    }
    return any;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const v of this.voices) {
      if (v.startTimer) clearTimeout(v.startTimer);
      if (v.stopTimer) clearTimeout(v.stopTimer);
      try {
        if (v.audio.isPlaying) v.audio.stop();
      } catch {
        /* a closed context throws on stop; nothing to release */
      }
      v.audio.gain.disconnect();
      v.audio.removeFromParent();
    }
    this.voices.length = 0;
    this.buffer = null;
  }
}
