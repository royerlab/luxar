/**
 * Validate the zarr `viewer_config.audio` block (Python `AudioConfig`) into the
 * engine's {@link AudioConfigOverrides}. Same posture as
 * `extractRenderingOverrides`: only fields that are set come through, bad
 * values are dropped with a warning (never thrown — a typo in a store must not
 * take the scene down), and gains are clamped.
 *
 * @module config/zarr-bridge/audio-config
 */

import { log, Modules } from '../../utils/log';
import {
  AUDIO_BUS_NAMES,
  type AudioBusName,
  type AudioConfigOverrides,
  type PanningModel,
} from '../../types/audio';
import type { ZarrAudioConfig } from '../../types/zarr';

const GAIN_MAX = 2;

function clampGain(value: unknown, label: string): number | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    log.warning(Modules.CONFIG, `viewer_config.audio.${label} is not a number; ignored.`);
    return undefined;
  }
  return Math.min(GAIN_MAX, Math.max(0, value));
}

function readPanningModel(cfg: ZarrAudioConfig): PanningModel | undefined {
  const value = cfg.panning_model;
  if (value === undefined || value === null) return undefined;
  if (value === 'equalpower' || value === 'HRTF') return value;
  log.warning(
    Modules.CONFIG,
    `viewer_config.audio.panning_model "${String(value)}" is not "equalpower" or "HRTF"; ignored.`
  );
  return undefined;
}

function readBuses(cfg: ZarrAudioConfig): Partial<Record<AudioBusName, number>> | undefined {
  if (!cfg.buses || typeof cfg.buses !== 'object') return undefined;
  const buses: Partial<Record<AudioBusName, number>> = {};
  for (const [name, value] of Object.entries(cfg.buses)) {
    if (!(AUDIO_BUS_NAMES as readonly string[]).includes(name)) {
      log.warning(Modules.CONFIG, `viewer_config.audio.buses has unknown bus "${name}"; ignored.`);
      continue;
    }
    const gain = clampGain(value, `buses.${name}`);
    if (gain !== undefined) buses[name as AudioBusName] = gain;
  }
  return Object.keys(buses).length > 0 ? buses : undefined;
}

function readDuckDb(cfg: ZarrAudioConfig): number | undefined {
  const value = cfg.duck_db;
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'number' && Number.isFinite(value)) return Math.min(0, Math.max(-60, value));
  log.warning(Modules.CONFIG, 'viewer_config.audio.duck_db is not a number; ignored.');
  return undefined;
}

/** Extract and validate the authored audio defaults; `{}` when absent or malformed. */
export function extractAudioConfig(raw: unknown): AudioConfigOverrides {
  if (!raw || typeof raw !== 'object') return {};
  const cfg = raw as ZarrAudioConfig;
  const out: AudioConfigOverrides = {};
  if (typeof cfg.enabled === 'boolean') out.enabled = cfg.enabled;
  const master = clampGain(cfg.master_gain, 'master_gain');
  if (master !== undefined) out.masterGain = master;
  const panning = readPanningModel(cfg);
  if (panning) out.panningModel = panning;
  const buses = readBuses(cfg);
  if (buses) out.buses = buses;
  const duckDb = readDuckDb(cfg);
  if (duckDb !== undefined) out.duckDb = duckDb;
  return out;
}
