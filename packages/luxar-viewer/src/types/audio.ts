/**
 * Shared vocabulary for the sound layer — the `sound` node type's attrs as the
 * data layer hands them to the audio layer, and the public audio state / patch
 * shapes the embedder API exposes.
 *
 * `sound` is a node type but NOT a geometry type (`format-contract/contract.yaml`
 * lists it in `node_types` only): it carries no elements, no blending mode, no
 * LOD and no picking, so none of the geometry tables have a row for it and the
 * scene loader dispatches on `type === 'sound'` alone. Design:
 * `docs/guides/specs/SOUND_SPEC.md`.
 *
 * @module types/audio
 */

/** The three mixer buses a scene may set a gain for. */
export const AUDIO_BUS_NAMES = ['ambient', 'voice', 'effects'] as const;
/** One of the three mixer buses. */
export type AudioBusName = (typeof AUDIO_BUS_NAMES)[number];

/**
 * Playback triggers. `on_depart` / `on_arrive` are spec vocabulary (waypoint
 * events, Phase 2); the Python writer refuses them today and the viewer parses
 * them only to warn and treat the node as `once`.
 */
export type SoundTrigger = 'continuous' | 'once' | 'on_depart' | 'on_arrive';

/** `PannerNode.panningModel`: equal-power for room speakers, HRTF for headphones. */
export type PanningModel = 'equalpower' | 'HRTF';

/** `PannerNode.distanceModel`, verbatim. */
export type DistanceModel = 'inverse' | 'linear' | 'exponential';

/** Every start and stop goes through a gain ramp at least this long (anti-click). */
export const MIN_FADE_MS = 30;

/**
 * A sound node's attrs after `parseSoundNodeAttrs` (audio/sound-attrs.ts) filled the defaults —
 * what the engine reads. Mirrors the on-disk attrs in `SOUND_SPEC.md` §3.2.
 */
export interface SoundNodeAttrs {
  spatial: boolean;
  trigger: SoundTrigger;
  delay_ms: number;
  gain: number;
  bus: AudioBusName;
  loop: boolean;
  fade_in_ms: number;
  fade_out_ms: number;
  distance_model: DistanceModel;
  ref_distance: number;
  max_distance: number;
  rolloff: number;
  cone_inner_deg?: number;
  cone_outer_deg?: number;
  cone_outer_gain?: number;
  orientation?: [number, number, number];
  format?: string;
  duration_ms?: number;
  license?: string;
  attribution?: string;
  source_url?: string;
  /** The plain store key holding the clip, e.g. `audio.mp3`. */
  audio_file: string;
  extend_to_all?: string[];
}

/**
 * What the data layer hands the audio layer for one sound node: the raw attrs,
 * the optional `(K, ndim)` positions rows and a lazy clip reader bound to the
 * scene's store. Built by `data/scene-loader/nodes/load-sound-node.ts`, stashed
 * on the node's placeholder as `userData.sound`, consumed by the engine.
 */
export interface SoundSourceDescriptor {
  /** Scene-graph path (`/sounds/hum`). */
  path: string;
  /** Last path segment — the name `playSound(name)` accepts. */
  name: string;
  /** Raw zarr attrs; defaults are filled by the engine (it knows the scene scale). */
  rawAttrs: Record<string, unknown>;
  /** Row-major `(K, ndim)` positions, or `null` for a clip live everywhere. */
  positions: Float32Array | null;
  nPositions: number;
  ndim: number;
  /** Read the encoded clip bytes from the store (undefined when the key is missing). */
  readClip(): Promise<Uint8Array | undefined>;
}

/** `AudioContext.state`, plus `unavailable` before the context exists. */
export type AudioContextState = 'unavailable' | 'suspended' | 'running' | 'closed';

/** The audio block of the embedder `ViewerState` — what a remote controller mirrors. */
export interface AudioState {
  state: AudioContextState;
  /** True when the rail mute is on OR the scene authored `enabled: false`. */
  muted: boolean;
  masterGain: number;
  panningModel: PanningModel;
  buses: Record<AudioBusName, number>;
  /** Names of the nodes currently playing. */
  playing: string[];
  /** True when the loaded scene has at least one sound node. */
  hasSoundNodes: boolean;
}

/** Partial live update the embedder API's `setAudio()` takes. */
export interface AudioPatch {
  masterGain?: number;
  muted?: boolean;
  buses?: Partial<Record<AudioBusName, number>>;
  panningModel?: PanningModel;
}

/** Authored defaults from `viewer_config.audio`, already validated and clamped. */
export interface AudioConfigOverrides extends AudioPatch {
  enabled?: boolean;
  duckDb?: number;
}
