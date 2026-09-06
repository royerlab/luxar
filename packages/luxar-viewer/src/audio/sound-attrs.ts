/**
 * Fill a sound node's raw zarr attrs with the viewer defaults.
 *
 * The Python writer stores only what the author chose (`SOUND_SPEC.md` §3.1):
 * distances left `None` are ABSENT on disk so the viewer can default them from
 * the scene scale like the fly speed does (`ref_distance = scale / 20`,
 * `max_distance = scale`). Everything else defaults to the spec's values.
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

/**
 * Parse `raw` (a sound node's zarr attrs) into {@link SoundNodeAttrs}.
 *
 * `sceneScale` is the scene's characteristic size (`SceneManager.getSceneScale`)
 * the distance defaults derive from. Unknown enum values fall back to the
 * default with a warning rather than failing the node.
 */
export function parseSoundNodeAttrs(
  path: string,
  raw: Record<string, unknown>,
  sceneScale: number
): SoundNodeAttrs {
  const scale = Number.isFinite(sceneScale) && sceneScale > 0 ? sceneScale : 100;

  let trigger = raw.trigger as SoundTrigger;
  if (!TRIGGERS.includes(trigger)) {
    trigger = 'continuous';
  } else if (trigger === 'on_depart' || trigger === 'on_arrive') {
    log.warning(
      Modules.AUDIO,
      `${path}: trigger "${trigger}" needs the waypoint events (sound layer Phase 2); playing it as "once".`
    );
    trigger = 'once';
  }

  let bus = raw.bus as AudioBusName;
  if (!AUDIO_BUS_NAMES.includes(bus)) {
    if (raw.bus !== undefined) {
      log.warning(Modules.AUDIO, `${path}: unknown bus "${String(raw.bus)}", using "ambient".`);
    }
    bus = 'ambient';
  }

  let distanceModel = raw.distance_model as DistanceModel;
  if (!DISTANCE_MODELS.includes(distanceModel)) distanceModel = 'inverse';

  const format = typeof raw.format === 'string' ? raw.format : undefined;
  if (format === 'ogg' || format === 'opus') {
    log.warning(Modules.AUDIO, `${path}: ${format} clips do not decode on Safari.`);
  }

  const refDistance = optNum(raw.ref_distance);
  const maxDistance = optNum(raw.max_distance);
  const orientation =
    Array.isArray(raw.orientation) && raw.orientation.length === 3
      ? ([raw.orientation[0], raw.orientation[1], raw.orientation[2]].map((c) => Number(c)) as [
          number,
          number,
          number,
        ])
      : undefined;
  const audioFile = typeof raw.audio_file === 'string' ? raw.audio_file : 'audio.mp3';
  const extendToAll = Array.isArray(raw.extend_to_all)
    ? (raw.extend_to_all.filter((d) => typeof d === 'string') as string[])
    : undefined;

  return {
    spatial: raw.spatial === true,
    trigger,
    delay_ms: num(raw.delay_ms, 0),
    gain: num(raw.gain, 1),
    bus,
    loop: typeof raw.loop === 'boolean' ? raw.loop : trigger === 'continuous',
    fade_in_ms: num(raw.fade_in_ms, 0),
    fade_out_ms: num(raw.fade_out_ms, 0),
    distance_model: distanceModel,
    ref_distance: refDistance !== undefined && refDistance > 0 ? refDistance : scale / 20,
    max_distance: maxDistance !== undefined && maxDistance > 0 ? maxDistance : scale,
    rolloff: num(raw.rolloff, 1),
    cone_inner_deg: optNum(raw.cone_inner_deg),
    cone_outer_deg: optNum(raw.cone_outer_deg),
    cone_outer_gain: optNum(raw.cone_outer_gain),
    orientation,
    format,
    duration_ms: optNum(raw.duration_ms),
    license: typeof raw.license === 'string' ? raw.license : undefined,
    attribution: typeof raw.attribution === 'string' ? raw.attribution : undefined,
    source_url: typeof raw.source_url === 'string' ? raw.source_url : undefined,
    audio_file: audioFile,
    extend_to_all: extendToAll,
  };
}
