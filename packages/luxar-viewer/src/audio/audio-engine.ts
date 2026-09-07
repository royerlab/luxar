/**
 * The viewer's audio engine: owns the `AudioContext` (through three's
 * `AudioListener`, which is the master gain), the three buses and the ducker,
 * the autoplay gate, and one {@link SoundNode} per `sound` node in the loaded
 * scene (`SOUND_SPEC.md` §4.1).
 *
 * Everything the engine needs from the rest of the viewer arrives as PORTS —
 * the camera, the dims manager, the scene graph, the container, the embedder
 * emitter — so `audio/` sits between `data` and `scene` in the layer order and
 * the engine runs in a node test with a fake `AudioContext` injected through
 * `THREE.AudioContext.setContext`.
 *
 * Nothing is constructed until a scene with at least one sound node attaches:
 * a viewer showing a plain points scene never creates an `AudioContext`.
 *
 * @module audio/audio-engine
 */

import * as THREE from 'three';
import type { SceneNode } from '../data/data-loader-types';
import type { SimpleDims } from '../types/dims';
import {
  AUDIO_BUS_NAMES,
  type AudioBusName,
  type AudioConfigOverrides,
  type AudioContextState,
  type AudioPatch,
  type AudioState,
  type PanningModel,
  type SoundSourceDescriptor,
  type SoundWaypointCondition,
  type WaypointEventKind,
} from '../types/audio';
import { log, Modules } from '../utils/log';
import {
  buildSoundBaseViewState,
  buildWaypointViewState,
  computeRowAudibility,
} from './audibility';
import { loadAudioPrefs, saveAudioPrefs } from './audio-prefs';
import { AutoplayGate } from './autoplay-gate';
import { dbToGain, rampGain } from './fades';
import type { FoaDecoder } from './foa-decoder';
import { parseSoundNodeAttrs } from './sound-attrs';
import { SoundNode } from './sound-node';

/** Disposer returned by the engine's subscription ports. */
export type Unsubscribe = () => void;

/** The pieces of the viewer the engine reads, as ports (see module docs). */
export interface AudioEngineDeps {
  getCamera(): THREE.Object3D;
  /** Fires after the camera OBJECT is replaced (perspective ⇄ ortho); the listener re-parents. */
  onCameraReplaced(cb: () => void): Unsubscribe;
  getDims(): SimpleDims | null;
  /** Fires synchronously on every dimension change. */
  onDimsChanged(cb: () => void): Unsubscribe;
  getSceneGraph(): SceneNode | null;
  /** Scene characteristic size — the distance defaults derive from it. */
  getSceneScale(): number;
  /** Where the autoplay gate mounts. */
  container(): HTMLElement;
  emit(event: 'sound-started' | 'sound-ended', payload: { name: string }): void;
  /** Tell the UI (the rail) the engine's state changed. */
  notifyUiChanged(): void;
  /**
   * World-space bounding-box centre of the scene node called `name`, or null
   * when it is unknown or not loaded yet — what `attach_to` sources follow.
   */
  resolveNodeCenter?(name: string): THREE.Vector3 | null;
}

const DEFAULT_BUSES: Record<AudioBusName, number> = { ambient: 0.6, voice: 1.0, effects: 0.8 };
/** Rendered frames between re-placements of `attach_to` sources (~0.5 s at 60 Hz). */
const ATTACH_REFRESH_FRAMES = 30;

/**
 * Centre of an nD `position_bounds` attr over the displayed dimensions (0 where a
 * displayed axis is missing or non-finite); null when the attr is not a bounds pair.
 */
function boundsCenter(bounds: unknown, displayed: readonly number[]): THREE.Vector3 | null {
  const b = bounds as { min?: unknown; max?: unknown } | undefined;
  if (!Array.isArray(b?.min) || !Array.isArray(b?.max)) return null;
  const min = b.min as number[];
  const max = b.max as number[];
  const pick = (i: number): number => {
    const d = displayed[i];
    if (d === undefined) return 0;
    const lo = min[d];
    const hi = max[d];
    return Number.isFinite(lo) && Number.isFinite(hi) ? (lo + hi) / 2 : 0;
  };
  return new THREE.Vector3(pick(0), pick(1), pick(2));
}

/** Depth-first lookup of a scene-graph node by name (last path segment) or full path. */
function findSceneNode(root: SceneNode, name: string): SceneNode | null {
  const stack: SceneNode[] = [root];
  while (stack.length > 0) {
    const node = stack.pop()!;
    const last = node.path.split('/').pop();
    if (node.path === name || last === name) return node;
    if (node.children) for (const child of node.children) stack.push(child);
  }
  return null;
}
const DEFAULT_MASTER_GAIN = 0.8;
const DEFAULT_DUCK_DB = -9;
const DUCK_RAMP_MS = 150;

/**
 * The sound layer's engine: one per `LuxarApp`, nodes attached per scene.
 * See the module docs for the graph it builds and the ports it takes.
 */
export class AudioEngine {
  private listener: THREE.AudioListener | null = null;
  private busNodes: Record<AudioBusName, GainNode> | null = null;
  private duck: GainNode | null = null;
  private readonly nodes = new Map<string, SoundNode>();
  private readonly gate: AutoplayGate;
  private unsubscribeDims: Unsubscribe | null = null;
  private unsubscribeCamera: Unsubscribe | null = null;

  private masterGain = DEFAULT_MASTER_GAIN;
  private muted = false;
  private sceneEnabled = true;
  private panningModel: PanningModel = 'equalpower';
  private busGains: Record<AudioBusName, number> = { ...DEFAULT_BUSES };
  private duckDb = DEFAULT_DUCK_DB;
  private activeVoiceClips = 0;
  private disposed = false;
  /** Ambisonic decoders to rotate against the camera (see `ensureGraph`). */
  private readonly foaDecoders = new Set<FoaDecoder>();
  private readonly listenerQuaternion = new THREE.Quaternion();

  constructor(private readonly deps: AudioEngineDeps) {
    const prefs = loadAudioPrefs();
    if (prefs.muted !== undefined) this.muted = prefs.muted;
    if (prefs.masterGain !== undefined) this.masterGain = prefs.masterGain;
    this.gate = new AutoplayGate(deps.container, () => this.onGateTapped());
  }

  // ── Graph ────────────────────────────────────────────────────────────

  private get context(): AudioContext | null {
    return this.listener?.context ?? null;
  }

  /** Build the listener, buses and ducker on first use. */
  private ensureGraph(): THREE.AudioListener {
    if (this.listener && this.busNodes && this.duck) return this.listener;
    const listener = new THREE.AudioListener();
    const ctx = listener.context;
    const input = listener.getInput();
    const duck = ctx.createGain();
    duck.gain.value = 1;
    duck.connect(input);
    const buses = {} as Record<AudioBusName, GainNode>;
    for (const name of AUDIO_BUS_NAMES) {
      const g = ctx.createGain();
      g.gain.value = this.busGains[name];
      g.connect(name === 'ambient' ? duck : input);
      buses[name] = g;
    }
    listener.setMasterVolume(this.effectiveMaster());
    // Three refreshes the listener from the camera's world matrix every frame
    // it renders; the ambisonic fields ride the same update (no per-frame
    // callback of our own, and nothing runs while the camera is still).
    const updateMatrixWorld = listener.updateMatrixWorld.bind(listener);
    let frame = 0;
    listener.updateMatrixWorld = (force?: boolean): void => {
      updateMatrixWorld(force);
      if (this.foaDecoders.size > 0) this.rotateFoaFields(listener);
      // Attached sources follow a target that may load or move later; a
      // bounding-box read every ~half second of rendering is cheap enough.
      if (++frame % ATTACH_REFRESH_FRAMES === 0) this.refreshAttachments();
    };
    this.listener = listener;
    this.busNodes = buses;
    this.duck = duck;
    ctx.onstatechange = () => this.deps.notifyUiChanged();
    this.attachListener();
    this.unsubscribeCamera?.();
    this.unsubscribeCamera = this.deps.onCameraReplaced(() => this.attachListener());
    return listener;
  }

  private attachListener(): void {
    if (!this.listener) return;
    const camera = this.deps.getCamera();
    if (this.listener.parent !== camera) camera.add(this.listener);
  }

  private effectiveMaster(): number {
    return this.muted || !this.sceneEnabled ? 0 : this.masterGain;
  }

  // ── Scene lifecycle ──────────────────────────────────────────────────

  /**
   * Find every sound node under `root` (their placeholders carry
   * `userData.sound`), build the graph if needed, evaluate the slab once,
   * and start decoding clips. Idempotent per scene: call `detachScene` first
   * when a new dataset loads.
   */
  attachScene(root: THREE.Object3D): void {
    if (this.disposed) return;
    const found: Array<{ desc: SoundSourceDescriptor; parent: THREE.Object3D }> = [];
    root.traverse((obj) => {
      const desc = obj.userData?.sound as SoundSourceDescriptor | undefined;
      if (obj.userData?.nodeType === 'sound' && desc) found.push({ desc, parent: obj });
    });
    if (found.length === 0) {
      this.deps.notifyUiChanged();
      return;
    }
    const listener = this.ensureGraph();
    const buses = this.busNodes!;
    const scale = this.deps.getSceneScale();
    for (const { desc, parent } of found) {
      if (this.nodes.has(desc.path)) continue;
      const attrs = parseSoundNodeAttrs(desc.path, desc.rawAttrs, scale);
      const node = new SoundNode(desc, attrs, {
        listener,
        bus: buses[attrs.bus],
        panningModel: this.panningModel,
        parent,
        onStarted: (name) => this.onNodeStarted(name, attrs.bus),
        onEnded: (name) => this.onNodeEnded(name, attrs.bus),
        resolveAttachCenter:
          attrs.attach_to !== undefined
            ? () => this.resolveAttachCenter(attrs.attach_to as string)
            : undefined,
        registerFoaDecoder: (decoder) => {
          this.foaDecoders.add(decoder);
          if (this.listener) this.rotateFoaFields(this.listener);
        },
        unregisterFoaDecoder: (decoder) => this.foaDecoders.delete(decoder),
      });
      node.setMuted(this.muted || !this.sceneEnabled);
      this.nodes.set(desc.path, node);
    }
    log.custom('🔈', Modules.AUDIO, `${this.nodes.size} sound node(s) attached`);

    this.unsubscribeDims?.();
    this.unsubscribeDims = this.deps.onDimsChanged(() => this.evaluate());
    this.evaluate();
    this.checkGate();
    void this.decodeAll();
    this.deps.notifyUiChanged();
  }

  /** Stop and drop every node; keep the context and buses for the next scene. */
  detachScene(): void {
    this.unsubscribeDims?.();
    this.unsubscribeDims = null;
    for (const node of this.nodes.values()) node.dispose();
    this.nodes.clear();
    this.activeVoiceClips = 0;
    this.setDuck(1);
    this.gate.hide();
    this.deps.notifyUiChanged();
  }

  /**
   * Where an `attach_to` source sits: the target's live bounding-box centre
   * when its geometry is loaded, else the centre of the `position_bounds` its
   * scene-graph node carries (a story-bound cluster has no geometry until the
   * slice reaches its story, but its authored bounds are on disk from the
   * start). The fallback reads the displayed dimensions and assumes an identity
   * target transform; the live box takes over once the node loads.
   */
  private resolveAttachCenter(name: string): THREE.Vector3 | null {
    const live = this.deps.resolveNodeCenter?.(name) ?? null;
    if (live) return live;
    const graph = this.deps.getSceneGraph();
    const dims = this.deps.getDims();
    if (!graph || !dims) return null;
    const node = findSceneNode(graph, name);
    return node ? boundsCenter(node.attrs?.position_bounds, dims.displayed) : null;
  }

  /** Re-place every `attach_to` source (its target may have loaded since). */
  private refreshAttachments(): void {
    for (const node of this.nodes.values()) {
      if (node.attrs.attach_to !== undefined) node.refreshPlacement();
    }
  }

  /** Point every ambisonic field at the camera the listener sits on. */
  private rotateFoaFields(listener: THREE.AudioListener): void {
    // The listener has no rotation of its own, so its world rotation is the camera's.
    this.listenerQuaternion.setFromRotationMatrix(listener.matrixWorld);
    for (const decoder of this.foaDecoders) decoder.setCameraQuaternion(this.listenerQuaternion);
  }

  private evaluate(): void {
    const dims = this.deps.getDims();
    if (!dims || this.nodes.size === 0) return;
    const base = buildSoundBaseViewState(dims);
    const sceneGraph = this.deps.getSceneGraph();
    for (const node of this.nodes.values()) node.setViewState(base, sceneGraph);
  }

  /**
   * The waypoint driver left (`depart`) or reached (`arrive`) the waypoint whose
   * `when` clause this is. Every `on_depart` / `on_arrive` node whose rows belong
   * to that waypoint fires (`SOUND_SPEC.md` §4.3); a node without rows belongs
   * to every waypoint. Called by the app from the driver's event port.
   */
  notifyWaypoint(kind: WaypointEventKind, when: SoundWaypointCondition): void {
    if (this.disposed || this.nodes.size === 0) return;
    const dims = this.deps.getDims();
    const sceneGraph = this.deps.getSceneGraph();
    let fired = 0;
    for (const node of this.nodes.values()) {
      if (node.attrs.trigger !== `on_${kind}`) continue;
      const rows = this.belongingRows(node, when, dims, sceneGraph);
      if (rows === undefined) continue;
      if (node.triggerFromWaypoint(kind, rows)) fired++;
    }
    if (fired > 0) {
      log.custom('🔈', Modules.AUDIO, `waypoint ${kind}: ${fired} sound node(s) triggered`);
    }
  }

  /**
   * The rows of `node` that belong to the waypoint: null for a row-less node
   * (it belongs to every waypoint), a mask otherwise, `undefined` when no row
   * belongs (or the dims are not ready) — nothing to trigger.
   */
  private belongingRows(
    node: SoundNode,
    when: SoundWaypointCondition,
    dims: SimpleDims | null,
    sceneGraph: SceneNode | null
  ): Uint8Array | null | undefined {
    if (!node.desc.positions || node.desc.nPositions === 0) return null;
    if (!dims) return undefined;
    const rows = new Uint8Array(node.desc.nPositions);
    const query = buildWaypointViewState(when, dims);
    const count = computeRowAudibility(
      node.desc,
      node.attrs.extend_to_all,
      query,
      sceneGraph,
      rows
    );
    return count === 0 ? undefined : rows;
  }

  private async decodeAll(): Promise<void> {
    const ctx = this.context;
    if (!ctx) return;
    // Sequential on purpose: decoding is CPU-heavy and a scene with a dozen
    // clips would otherwise stall the main thread in one burst. Audible-first,
    // so the opening slice's bed and narration land before the silent clips.
    const ordered = Array.from(this.nodes.values()).sort(
      (a, b) => Number(b.isAudibleNow) - Number(a.isAudibleNow)
    );
    for (const node of ordered) {
      if (this.disposed || !this.nodes.has(node.path)) return;
      try {
        const bytes = await node.desc.readClip();
        if (!bytes) {
          log.warning(Modules.AUDIO, `${node.path}: clip ${node.attrs.audio_file} not found`);
          continue;
        }
        const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        const buffer = await ctx.decodeAudioData(copy as ArrayBuffer);
        if (this.disposed || !this.nodes.has(node.path)) return;
        node.setBuffer(buffer);
      } catch (error) {
        log.warning(Modules.AUDIO, `${node.path}: could not decode clip`, error);
      }
    }
  }

  // ── Autoplay gate ────────────────────────────────────────────────────

  private checkGate(): void {
    const ctx = this.context;
    if (!ctx || this.nodes.size === 0) return;
    if (this.muted || !this.sceneEnabled) {
      this.gate.hide();
      return;
    }
    if (ctx.state === 'running') {
      this.openGate();
      return;
    }
    if (ctx.state === 'suspended') {
      // A kiosk with the autoplay flag resumes here; a browser without a
      // gesture rejects or stays suspended, and the gate takes over.
      void ctx
        .resume()
        .then(() => {
          if (ctx.state === 'running') this.openGate();
          else this.gate.show();
        })
        .catch(() => this.gate.show());
    }
  }

  private onGateTapped(): void {
    const ctx = this.context;
    if (!ctx) return;
    void ctx
      .resume()
      .then(() => this.openGate())
      .catch((error) => log.warning(Modules.AUDIO, 'AudioContext.resume() failed', error));
  }

  private openGate(): void {
    this.gate.hide();
    for (const node of this.nodes.values()) node.setGateOpen(true);
    this.deps.notifyUiChanged();
  }

  // ── Ducking + events ─────────────────────────────────────────────────

  private onNodeStarted(name: string, bus: AudioBusName): void {
    if (bus === 'voice') {
      this.activeVoiceClips++;
      if (this.activeVoiceClips === 1) this.setDuck(dbToGain(this.duckDb));
    }
    this.deps.emit('sound-started', { name });
    this.deps.notifyUiChanged();
  }

  private onNodeEnded(name: string, bus: AudioBusName): void {
    if (bus === 'voice') {
      this.activeVoiceClips = Math.max(0, this.activeVoiceClips - 1);
      if (this.activeVoiceClips === 0) this.setDuck(1);
    }
    this.deps.emit('sound-ended', { name });
    this.deps.notifyUiChanged();
  }

  private setDuck(target: number): void {
    const ctx = this.context;
    if (!ctx || !this.duck) return;
    rampGain(this.duck.gain, { now: ctx.currentTime, target, ms: DUCK_RAMP_MS });
  }

  // ── Configuration ────────────────────────────────────────────────────

  /**
   * Apply a scene's authored `viewer_config.audio`. The listener's own
   * persisted preferences (mute, master gain) win over the scene's defaults.
   */
  applySceneConfig(overrides: AudioConfigOverrides): void {
    const prefs = loadAudioPrefs();
    this.masterGain = prefs.masterGain ?? DEFAULT_MASTER_GAIN;
    this.applyBusGains(DEFAULT_BUSES);
    this.setPanningModel('equalpower');
    this.duckDb = DEFAULT_DUCK_DB;
    this.sceneEnabled = overrides.enabled !== false;
    if (overrides.masterGain !== undefined && prefs.masterGain === undefined) {
      this.masterGain = overrides.masterGain;
    }
    if (overrides.panningModel) this.setPanningModel(overrides.panningModel);
    this.applyBusGains(overrides.buses);
    if (overrides.duckDb !== undefined) this.duckDb = overrides.duckDb;
    this.listener?.setMasterVolume(this.effectiveMaster());
    for (const node of this.nodes.values()) node.setMuted(this.muted || !this.sceneEnabled);
    this.deps.notifyUiChanged();
  }

  /** Live patch from the embedder API or the rail popover. */
  setAudio(patch: AudioPatch): void {
    if (patch.masterGain !== undefined) this.setMasterGain(patch.masterGain);
    if (patch.muted !== undefined) this.setMuted(patch.muted);
    if (patch.panningModel) this.setPanningModel(patch.panningModel);
    this.applyBusGains(patch.buses);
  }

  private applyBusGains(buses: Partial<Record<AudioBusName, number>> | undefined): void {
    if (!buses) return;
    for (const name of AUDIO_BUS_NAMES) {
      const v = buses[name];
      if (v !== undefined) this.setBusGain(name, v);
    }
  }

  setMasterGain(value: number): void {
    if (!Number.isFinite(value)) return;
    this.masterGain = Math.min(2, Math.max(0, value));
    this.listener?.setMasterVolume(this.effectiveMaster());
    saveAudioPrefs({ masterGain: this.masterGain });
    this.deps.notifyUiChanged();
  }

  /**
   * Mute everything (with each node's fade-out) or unmute. Unmuting also
   * re-enables a scene that authored `enabled: false` for this session, since
   * the listener asked for sound explicitly. A muted scene never shows the gate.
   */
  setMuted(muted: boolean): void {
    this.muted = muted;
    if (!muted) this.sceneEnabled = true;
    this.listener?.setMasterVolume(this.effectiveMaster());
    for (const node of this.nodes.values()) node.setMuted(muted);
    saveAudioPrefs({ muted: this.muted });
    if (muted) this.gate.hide();
    else this.checkGate();
    this.deps.notifyUiChanged();
  }

  setPanningModel(model: PanningModel): void {
    if (model !== 'equalpower' && model !== 'HRTF') return;
    this.panningModel = model;
    for (const node of this.nodes.values()) node.setPanningModel(model);
  }

  setBusGain(bus: AudioBusName, value: number): void {
    if (!Number.isFinite(value)) return;
    const v = Math.min(2, Math.max(0, value));
    this.busGains[bus] = v;
    const ctx = this.context;
    if (ctx && this.busNodes) {
      rampGain(this.busNodes[bus].gain, { now: ctx.currentTime, target: v, ms: DUCK_RAMP_MS });
    }
    this.deps.notifyUiChanged();
  }

  // ── Capture (the Recording panel) ───────────────────────────────────

  private readonly captureDestinations = new Map<MediaStream, MediaStreamAudioDestinationNode>();

  /**
   * A `MediaStream` carrying the mix the listener hears (post master gain, so a
   * muted viewer records silence), for the Recording panel to add to its
   * canvas capture (`SOUND_SPEC.md` §6, Phase 3). Null before the graph exists
   * — a scene without sound nodes records a silent video as before. Release
   * with {@link releaseCaptureStream} when the recording ends.
   */
  acquireCaptureStream(): MediaStream | null {
    const ctx = this.context;
    if (!ctx || !this.listener || this.disposed) return null;
    const destination = ctx.createMediaStreamDestination();
    // The listener's gain is three's master: its only output is the context
    // destination, so tapping it here captures exactly what the speakers get.
    this.listener.gain.connect(destination);
    this.captureDestinations.set(destination.stream, destination);
    return destination.stream;
  }

  /** Detach a capture stream from the master gain and stop its tracks. */
  releaseCaptureStream(stream: MediaStream): void {
    const destination = this.captureDestinations.get(stream);
    if (!destination) return;
    this.captureDestinations.delete(stream);
    try {
      this.listener?.gain.disconnect(destination);
    } catch {
      /* already disconnected (context closed) */
    }
    for (const track of stream.getTracks()) track.stop();
  }

  // ── Per-node control (the Layers panel's rows) ──────────────────────

  /**
   * Mute one node (`path` names it) or every node under a group (`path` is an
   * ancestor). The node's own eye and an ancestor's eye are tracked separately,
   * as a hidden group hides its children whatever their own flag says.
   */
  setNodeMuted(path: string, muted: boolean): void {
    if (this.nodes.size === 0) return;
    const prefix = path.endsWith('/') ? path : `${path}/`;
    for (const node of this.nodes.values()) {
      if (node.path === path) node.setNodeMuted(muted);
      else if (node.path.startsWith(prefix)) node.setAncestorMuted(muted);
    }
    this.deps.notifyUiChanged();
  }

  /** Live per-node linear gain (`[0, 2]`); ramps a playing node. */
  setNodeGain(path: string, gain: number): void {
    this.nodes.get(path)?.setGain(gain);
  }

  /** The live per-node gain, or undefined for an unknown path. */
  getNodeGain(path: string): number | undefined {
    return this.nodes.get(path)?.gain;
  }

  // ── Queries + manual control ─────────────────────────────────────────

  private findNode(name: string): SoundNode | undefined {
    const direct = this.nodes.get(name);
    if (direct) return direct;
    for (const node of this.nodes.values()) if (node.name === name) return node;
    return undefined;
  }

  /** Start `name` (basename or full path) regardless of the slab. */
  play(name: string): boolean {
    const node = this.findNode(name);
    return node ? node.play() : false;
  }

  /** Stop `name` with its fade-out. */
  stop(name: string): boolean {
    const node = this.findNode(name);
    return node ? node.stop() : false;
  }

  hasSoundNodes(): boolean {
    return this.nodes.size > 0;
  }

  isMuted(): boolean {
    return this.muted || !this.sceneEnabled;
  }

  getMasterGain(): number {
    return this.masterGain;
  }

  getBusGains(): Record<AudioBusName, number> {
    return { ...this.busGains };
  }

  getState(): AudioState {
    const ctx = this.context;
    const state: AudioContextState = ctx ? (ctx.state as AudioContextState) : 'unavailable';
    return {
      state,
      muted: this.isMuted(),
      masterGain: this.masterGain,
      panningModel: this.panningModel,
      buses: { ...this.busGains },
      // Live voice state, not the event log: a node fading out is already gone here
      // while its `sound-ended` fires when the fade completes.
      playing: Array.from(this.nodes.values())
        .filter((n) => n.isPlaying)
        .map((n) => n.name)
        .sort(),
      hasSoundNodes: this.nodes.size > 0,
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.detachScene();
    for (const stream of Array.from(this.captureDestinations.keys())) {
      this.releaseCaptureStream(stream);
    }
    this.disposed = true;
    this.unsubscribeCamera?.();
    this.unsubscribeCamera = null;
    this.gate.dispose();
    const ctx = this.context;
    this.listener?.removeFromParent();
    this.listener = null;
    this.busNodes = null;
    this.duck = null;
    if (ctx) {
      ctx.onstatechange = null;
      void ctx.close().catch(() => undefined);
      // Three caches the context in a module singleton; a re-init must not
      // inherit a closed one.
      THREE.AudioContext.setContext(undefined as unknown as AudioContext);
    }
  }
}
