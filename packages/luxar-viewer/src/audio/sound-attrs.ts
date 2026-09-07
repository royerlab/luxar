/**
 * Fill a sound node's raw zarr attrs with the viewer defaults.
 *
 * The Python writer stores only what the author chose (`SOUND_SPEC.md` §3.1):
 * distances left `None` are ABSENT on disk so the viewer can default them from
 * the scene scale like the fly speed does (`ref_distance = scale / 20`,
 * `max_distance = scale`). Everything else defaults to the spec's values.
 * Unknown enum values fall back to the default with a warning rather than
 * failing the node.
 *
 * @module audio/sound-attrs
 */

import { log, Modules } from '../utils/log';
import {
  AUDIO_BUS_NAMES,
  type AudioBusName,
  type DistanceModel,
  type SoundNodeAttrs,
  type SoundTrigger,
} from '../types/audio';

const DISTANCE_MODELS: readonly DistanceModel[] = ['inverse', 'linear', 'exponential'];
const TRIGGERS: readonly SoundTrigger[] = ['continuous', 'once', 'on_depart', 'on_arrive'];

function num(value: unknown, fallback: number, min = 0): number {
  return typeof value === 'number' && Number.isFinite(value) && value >= min ? value : fallback;
}

function optNum(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function optStr(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

/** A raw attr value for a warning: strings verbatim, anything else as JSON. */
function shown(value: unknown): string {
  return typeof value === 'string' ? value : (JSON.stringify(value) ?? 'undefined');
}

function parseTrigger(path: string, raw: unknown): SoundTrigger {
  if (TRIGGERS.includes(raw as SoundTrigger)) return raw as SoundTrigger;
  if (raw !== undefined) {
    log.warning(Modules.AUDIO, `${path}: unknown trigger "${shown(raw)}", using "continuous".`);
  }
  return 'continuous';
}

function parseBus(path: string, raw: unknown): AudioBusName {
  if (AUDIO_BUS_NAMES.includes(raw as AudioBusName)) return raw as AudioBusName;
  if (raw !== undefined) {
    log.warning(Modules.AUDIO, `${path}: unknown bus "${shown(raw)}", using "ambient".`);
  }
  return 'ambient';
}

function parseDistanceModel(raw: unknown): DistanceModel {
  return DISTANCE_MODELS.includes(raw as DistanceModel) ? (raw as DistanceModel) : 'inverse';
}

function parseOrientation(raw: unknown): [number, number, number] | undefined {
  if (!Array.isArray(raw) || raw.length !== 3) return undefined;
  return [Number(raw[0]), Number(raw[1]), Number(raw[2])];
}

function parseAmbisonic(path: string, raw: unknown): 'foa' | undefined {
  if (raw === 'foa') return 'foa';
  if (raw !== undefined && raw !== null) {
    log.warning(
      Modules.AUDIO,
      `${path}: unknown ambisonic layout "${shown(raw)}"; playing the clip as plain audio.`
    );
  }
  return undefined;
}

function parseAttachTo(raw: unknown): string | undefined {
  return typeof raw === 'string' && raw.trim() ? raw.trim() : undefined;
}

function parseFormat(path: string, raw: unknown): string | undefined {
  const format = optStr(raw);
  if (format === 'ogg' || format === 'opus') {
    log.warning(Modules.AUDIO, `${path}: ${format} clips do not decode on Safari.`);
  }
  return format;
}

/** A positive authored distance, else the scene-scale default. */
function distance(raw: unknown, fallback: number): number {
  const value = optNum(raw);
  return value !== undefined && value > 0 ? value : fallback;
}

/** A field (`ambisonic`) is never a point source; `attach_to` implies one. */
function parseSpatial(
  raw: unknown,
  attachTo: string | undefined,
  ambisonic: 'foa' | undefined
): boolean {
  if (ambisonic !== undefined) return false;
  if (raw === true) return true;
  return raw === undefined && attachTo !== undefined;
}

function parseExtendToAll(raw: unknown): string[] | undefined {
  return Array.isArray(raw) ? (raw.filter((d) => typeof d === 'string') as string[]) : undefined;
}

/**
 * Parse `raw` (a sound node's zarr attrs) into {@link SoundNodeAttrs}.
 *
 * `sceneScale` is the scene's characteristic size (`SceneManager.getSceneScale`)
 * the distance defaults derive from.
 */
export function parseSoundNodeAttrs(
  path: string,
  raw: Record<string, unknown>,
  sceneScale: number
): SoundNodeAttrs {
  const scale = Number.isFinite(sceneScale) && sceneScale > 0 ? sceneScale : 100;
  const trigger = parseTrigger(path, raw.trigger);
  const attachTo = parseAttachTo(raw.attach_to);
  const ambisonic = parseAmbisonic(path, raw.ambisonic);
  return {
    spatial: parseSpatial(raw.spatial, attachTo, ambisonic),
    trigger,
    delay_ms: num(raw.delay_ms, 0),
    gain: num(raw.gain, 1),
    bus: parseBus(path, raw.bus),
    loop: typeof raw.loop === 'boolean' ? raw.loop : trigger === 'continuous',
    fade_in_ms: num(raw.fade_in_ms, 0),
    fade_out_ms: num(raw.fade_out_ms, 0),
    distance_model: parseDistanceModel(raw.distance_model),
    ref_distance: distance(raw.ref_distance, scale / 20),
    max_distance: distance(raw.max_distance, scale),
    rolloff: num(raw.rolloff, 1),
    cone_inner_deg: optNum(raw.cone_inner_deg),
    cone_outer_deg: optNum(raw.cone_outer_deg),
    cone_outer_gain: optNum(raw.cone_outer_gain),
    orientation: parseOrientation(raw.orientation),
    format: parseFormat(path, raw.format),
    duration_ms: optNum(raw.duration_ms),
    license: optStr(raw.license),
    attribution: optStr(raw.attribution),
    source_url: optStr(raw.source_url),
    audio_file: optStr(raw.audio_file) ?? 'audio.mp3',
    extend_to_all: parseExtendToAll(raw.extend_to_all),
    attach_to: attachTo,
    ambisonic,
  };
}
