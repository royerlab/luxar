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
 * `on_depart` / `on_arrive` nodes ignore the slab's RISING edge: the engine
 * calls {@link SoundNode.triggerFromWaypoint} when the waypoint driver reports
 * the event, for the rows that belong to that waypoint (§4.3). An `on_arrive`
 * clip still fades out on the slab's falling edge (the visitor moved on, the
 * next story's narration must not overlap); an `on_depart` clip plays out —
 * by definition it starts when its own row has just stopped being audible.
 * A trigger that lands while the gate is closed, or before the clip decoded,
 * is kept pending and fires when either arrives.
 *
 * `attach_to` moves every voice to the named node's bounding-box centre (in
 * this node's local frame) instead of its rows' displayed x/y/z; the centre is
 * re-resolved on every evaluation, so a node that loads later is picked up.
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
import { FOA_CHANNELS, FoaAudio, FoaDecoder } from './foa-decoder';
import { MIN_FADE_MS } from '../types/audio';
import { log, Modules } from '../utils/log';

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
  /** World-space centre of the `attach_to` node, or null while it is unknown. */
  resolveAttachCenter?(): THREE.Vector3 | null;
  /** An ambisonic node's decoder, for the engine to rotate against the camera each frame. */
  registerFoaDecoder?(decoder: FoaDecoder): void;
  unregisterFoaDecoder?(decoder: FoaDecoder): void;
}

/** One playing source: a three `Audio` (non-spatial) or `PositionalAudio` (one per row). */
export interface Voice {
  audio: THREE.Audio<GainNode> | THREE.PositionalAudio;
  row: number;
  audible: boolean;
  /** True from a start until the voice ended or a stop was scheduled. */
  playing: boolean;
  /** Whether this start has emitted the public started event. */
  started: boolean;
  /** Whether a reported start still needs its matching ended event. */
  endPending: boolean;
  /** A waypoint trigger recorded while nothing could start (gate closed / no buffer). */
  pending: boolean;
  startTimer?: ReturnType<typeof setTimeout>;
  stopTimer?: ReturnType<typeof setTimeout>;
}

const _forward = new THREE.Vector3(0, 0, 1);
const _center = new THREE.Vector3();

/** One `sound` scene node's voices, edges and fades (see the module docs). */
export class SoundNode {
  readonly name: string;
  readonly path: string;
  private buffer: AudioBuffer | null = null;
  private readonly voices: Voice[] = [];
  private readonly mask: Uint8Array;
  private gateOpen = false;
  /** Engine-wide mute (the rail). */
  private globalMuted = false;
  /** This node's own Layers-panel eye. */
  private nodeMuted = false;
  /** Layers-panel eyes on ancestor groups. */
  private readonly mutingAncestors = new Set<string>();
  private disposed = false;
  /** Live per-node linear gain (the Layers-panel slider); starts at the authored value. */
  gain: number;
  /** The FOA rotate-and-decode graph of an ambisonic node (one voice). */
  private foaDecoder: FoaDecoder | null = null;
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
    this.gain = attrs.gain;
    this.mask = new Uint8Array(Math.max(1, desc.nPositions));
  }

  private get muted(): boolean {
    return this.globalMuted || this.nodeMuted || this.mutingAncestors.size > 0;
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

  /** True for the two triggers the waypoint driver drives (not the slab). */
  get isWaypointTriggered(): boolean {
    return this.attrs.trigger === 'on_depart' || this.attrs.trigger === 'on_arrive';
  }

  /**
   * Hand the decoded clip over. Voices are created here (lazily, once) so a node
   * whose clip never arrives costs nothing but a placeholder. Pending rising
   * edges recorded before the buffer arrived are replayed.
   */
  setBuffer(buffer: AudioBuffer): void {
    if (this.disposed) return;
    this.buffer = buffer;
    if (this.attrs.ambisonic === 'foa' && buffer.numberOfChannels !== FOA_CHANNELS) {
      log.warning(
        Modules.AUDIO,
        `${this.path}: ambisonic "foa" needs a ${FOA_CHANNELS}-channel clip, got ` +
          `${buffer.numberOfChannels}; the missing channels play silent.`
      );
    }
    if (this.voices.length === 0) {
      this.createVoices();
      this.restorePendingVoices();
    }
    for (const v of this.voices) v.audio.setBuffer(buffer);
    this.applyEdges(true);
  }

  private restorePendingVoices(): void {
    if (this.pendingBeforeVoices === undefined) return;
    const rows = this.pendingBeforeVoices;
    const multiVoice = this.attrs.spatial && this.desc.nPositions > 1;
    this.pendingBeforeVoices = undefined;
    for (const v of this.voices) {
      if (!rows || !multiVoice || rows[v.row] === 1) v.pending = true;
    }
  }

  private createVoices(): void {
    const hasRows = this.desc.positions !== null && this.desc.nPositions > 0;
    const spatial = this.attrs.spatial && (hasRows || this.attrs.attach_to !== undefined);
    const count = spatial && hasRows ? this.desc.nPositions : 1;
    for (let row = 0; row < count; row++) {
      const audio = this.createVoiceAudio(spatial);
      // Route past the listener input into the bus (see module docs).
      audio.gain.disconnect();
      audio.gain.connect(this.deps.bus);
      audio.gain.gain.value = 0;
      audio.setLoop(this.attrs.loop);
      if (audio instanceof THREE.PositionalAudio) this.configurePanner(audio);
      audio.name = `${this.path}#${row}`;
      this.deps.parent.add(audio);
      const voice: Voice = {
        audio,
        row,
        audible: false,
        playing: false,
        started: false,
        endPending: false,
        pending: false,
      };
      // Three binds `source.onended` to `this.onEnded` at play() time, so an
      // instance override is what runs when a `once` clip runs out.
      audio.onEnded = () => {
        THREE.Audio.prototype.onEnded.call(audio);
        if (voice.playing) {
          const wasStarted = voice.endPending || voice.started;
          voice.playing = false;
          voice.started = false;
          voice.endPending = false;
          if (wasStarted) this.deps.onEnded(this.name);
        }
      };
      this.voices.push(voice);
    }
    this.repositionVoices();
  }

  /** One voice's three `Audio`: a FOA field, a positional source, or a plain clip. */
  private createVoiceAudio(spatial: boolean): THREE.Audio<GainNode> | THREE.PositionalAudio {
    if (this.attrs.ambisonic === 'foa') {
      // A field, not a point: one voice through the rotate-and-decode graph.
      const decoder = new FoaDecoder(this.context);
      this.foaDecoder = decoder;
      this.deps.registerFoaDecoder?.(decoder);
      return new FoaAudio(this.deps.listener, decoder);
    }
    if (spatial) return new THREE.PositionalAudio(this.deps.listener);
    return new THREE.Audio<GainNode>(this.deps.listener);
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

  /**
   * Place each spatial voice: at the `attach_to` node's centre when it resolves,
   * else at its row's displayed x/y/z.
   */
  private repositionVoices(): void {
    if (!this.placeAtAttachCenter()) this.placeAtRows();
  }

  /** True when an `attach_to` centre resolved and every voice now sits there. */
  private placeAtAttachCenter(): boolean {
    if (this.attrs.attach_to === undefined || !this.deps.resolveAttachCenter) return false;
    const world = this.deps.resolveAttachCenter();
    if (!world) return false;
    // Voices are children of the placeholder, which carries the node's own
    // transform; the centre arrives in world space.
    this.deps.parent.updateWorldMatrix(true, false);
    this.deps.parent.worldToLocal(_center.copy(world));
    for (const v of this.voices) {
      if (v.audio instanceof THREE.PositionalAudio) v.audio.position.copy(_center);
    }
    return true;
  }

  private placeAtRows(): void {
    const positions = this.desc.positions;
    if (!positions) return;
    const displayDims =
      this.lastBase?.displayDims ?? [0, 1, 2].slice(0, Math.min(3, this.desc.ndim));
    for (const v of this.voices) {
      if (!(v.audio instanceof THREE.PositionalAudio)) continue;
      const [x, y, z] = displayedXYZ(positions, v.row, this.desc.ndim, displayDims);
      v.audio.position.set(x, y, z);
    }
  }

  /** Re-place the voices (an `attach_to` target may have loaded or moved). */
  refreshPlacement(): void {
    if (this.disposed || this.voices.length === 0) return;
    this.repositionVoices();
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
    const perVoice = this.voices.length > 1 || (this.attrs.spatial && rows !== null);
    const anyAudible = rows === null || rows.some((r) => r === 1);
    for (const v of this.voices) {
      const audible = perVoice && rows ? rows[v.row] === 1 : anyAudible;
      const was = v.audible;
      v.audible = audible;
      if (this.isWaypointTriggered) this.applyWaypointEdge(v, audible, was, replayRising);
      else this.applySlabEdge(v, audible, was, replayRising);
    }
  }

  /** `continuous` / `once`: the slab's rising edge starts, its falling edge stops. */
  private applySlabEdge(v: Voice, audible: boolean, was: boolean, replayRising: boolean): void {
    if (audible && (!was || (replayRising && !v.playing))) this.startVoice(v);
    else if (!audible && was) this.stopVoice(v, this.attrs.fade_out_ms);
  }

  /**
   * `on_depart` / `on_arrive`: the waypoint driver owns the start (a deferred
   * trigger replays here); the slab only ends an `on_arrive` clip once its
   * story is left (see the module docs).
   *
   * Staleness is NOT decided here. Audibility is unknown until a view state
   * arrives, and a waypoint may address one row of a spatial node explicitly
   * while the slab reports that row silent, so both look identical to a clip
   * whose story has been left. The engine expires deferred triggers instead,
   * on the next waypoint event, where the tour having moved on is a fact
   * rather than an inference.
   */
  private applyWaypointEdge(v: Voice, audible: boolean, was: boolean, replayRising: boolean): void {
    if (replayRising && v.pending) {
      this.startVoice(v);
      return;
    }
    const leftStory = this.attrs.trigger === 'on_arrive' && !audible && was && !v.pending;
    if (leftStory) this.stopVoice(v, this.attrs.fade_out_ms);
  }

  /**
   * Forget any deferred waypoint trigger, without touching what is playing.
   *
   * The engine calls this on every new waypoint event so at most ONE deferred
   * trigger — the most recent — is ever waiting. Navigating N stops while the
   * autoplay gate is shut otherwise leaves N clips armed, and opening the gate
   * starts all N together.
   */
  clearDeferredTrigger(): void {
    for (const v of this.voices) v.pending = false;
    this.pendingBeforeVoices = undefined;
  }

  /**
   * The waypoint driver's event for a waypoint this node belongs to. `rows`
   * marks the belonging rows (null = every voice). Returns true when the node
   * takes the event — its trigger is the matching `on_*` kind.
   */
  triggerFromWaypoint(kind: 'depart' | 'arrive', rows: Uint8Array | null): boolean {
    if (this.disposed || this.attrs.trigger !== `on_${kind}`) return false;
    for (const v of this.voices) {
      if (rows && this.voices.length > 1 && rows[v.row] !== 1) continue;
      this.startVoice(v);
    }
    if (this.voices.length === 0) {
      this.pendingBeforeVoices = rows ? Uint8Array.from(rows) : null;
    }
    return true;
  }

  /** A waypoint trigger that arrived before the clip decoded (no voices yet). */
  private pendingBeforeVoices: Uint8Array | null | undefined;

  private reportEnded(v: Voice): void {
    if (!v.endPending && !v.started) return;
    v.endPending = false;
    v.started = false;
    this.deps.onEnded(this.name);
  }

  private startVoice(v: Voice): boolean {
    if (!this.buffer || !this.gateOpen || this.muted) {
      if (this.isWaypointTriggered) v.pending = true;
      return false;
    }
    v.pending = false;
    if (v.stopTimer) {
      clearTimeout(v.stopTimer);
      v.stopTimer = undefined;
      this.reportEnded(v);
      v.audio.stop();
    }
    if (v.audio.isPlaying || v.playing) {
      // Restart (a re-rise during a fade-out, or a `once` re-triggered).
      this.reportEnded(v);
      v.playing = false;
      if (v.audio.isPlaying) v.audio.stop();
    }
    const ctx = this.context;
    const now = ctx.currentTime;
    const delaySec = this.attrs.delay_ms / 1000;
    const startAt = now + delaySec;
    v.audio.setLoop(this.attrs.loop);
    rampGain(v.audio.gain.gain, {
      now,
      startAt,
      target: this.gain,
      ms: this.attrs.fade_in_ms,
      from: 0,
    });
    v.audio.play(delaySec);
    v.playing = true;
    v.started = false;
    if (v.startTimer) clearTimeout(v.startTimer);
    v.startTimer = setTimeout(() => {
      v.startTimer = undefined;
      if (v.playing) {
        v.started = true;
        v.endPending = true;
        this.deps.onStarted(this.name);
      }
    }, this.attrs.delay_ms);
    return true;
  }

  private stopVoice(v: Voice, fadeMs: number): void {
    if (v.startTimer) {
      clearTimeout(v.startTimer);
      v.startTimer = undefined;
    }
    if (!v.playing && !v.audio.isPlaying) return;
    const ctx = this.context;
    const now = ctx.currentTime;
    const end = rampGain(v.audio.gain.gain, { now, target: 0, ms: fadeMs });
    v.endPending = v.endPending || v.started;
    v.playing = false;
    v.started = false;
    if (v.audio.isPlaying) v.audio.stop(end - now);
    if (v.stopTimer) clearTimeout(v.stopTimer);
    v.stopTimer = setTimeout(
      () => {
        v.stopTimer = undefined;
        this.reportEnded(v);
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

  /** Engine-wide mute: stops the voices with a fade; unmute replays the audible ones. */
  setMuted(muted: boolean): void {
    this.applyMuteChange(() => {
      this.globalMuted = muted;
    });
  }

  /** The node's own Layers-panel eye. */
  setNodeMuted(muted: boolean): void {
    this.applyMuteChange(() => {
      this.nodeMuted = muted;
    });
  }

  /** An ancestor group's Layers-panel eye. */
  setAncestorMuted(path: string, muted: boolean): void {
    this.applyMuteChange(() => {
      if (muted) this.mutingAncestors.add(path);
      else this.mutingAncestors.delete(path);
    });
  }

  private applyMuteChange(mutate: () => void): void {
    const was = this.muted;
    mutate();
    const now = this.muted;
    if (was === now) return;
    if (now) {
      for (const v of this.voices) this.stopVoice(v, this.attrs.fade_out_ms);
    } else {
      this.applyEdges(true);
    }
  }

  /** Live per-node gain: playing voices ramp to it (anti-click), later starts use it. */
  setGain(gain: number): void {
    if (!Number.isFinite(gain)) return;
    this.gain = Math.min(2, Math.max(0, gain));
    const now = this.context.currentTime;
    for (const v of this.voices) {
      if (v.playing) rampGain(v.audio.gain.gain, { now, target: this.gain, ms: MIN_FADE_MS });
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
    let any = false;
    for (const v of this.voices) {
      v.audible = true;
      any = this.startVoice(v) || any;
    }
    return any;
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
    if (this.foaDecoder) {
      this.deps.unregisterFoaDecoder?.(this.foaDecoder);
      this.foaDecoder.dispose();
      this.foaDecoder = null;
    }
  }
}
