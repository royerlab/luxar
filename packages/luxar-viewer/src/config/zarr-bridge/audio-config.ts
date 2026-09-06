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

/** Extract and validate the authored audio defaults; `{}` when absent or malformed. */
export function extractAudioConfig(raw: unknown): AudioConfigOverrides {
  if (!raw || typeof raw !== 'object') return {};
  const cfg = raw as ZarrAudioConfig;
  const out: AudioConfigOverrides = {};

  if (typeof cfg.enabled === 'boolean') out.enabled = cfg.enabled;

  const master = clampGain(cfg.master_gain, 'master_gain');
  if (master !== undefined) out.masterGain = master;

  if (cfg.panning_model !== undefined && cfg.panning_model !== null) {
    if (cfg.panning_model === 'equalpower' || cfg.panning_model === 'HRTF') {
      out.panningModel = cfg.panning_model as PanningModel;
    } else {
      log.warning(
        Modules.CONFIG,
        `viewer_config.audio.panning_model "${String(cfg.panning_model)}" is not "equalpower" or "HRTF"; ignored.`
      );
    }
  }

  if (cfg.buses && typeof cfg.buses === 'object') {
    const buses: Partial<Record<AudioBusName, number>> = {};
    for (const [name, value] of Object.entries(cfg.buses)) {
      if (!(AUDIO_BUS_NAMES as readonly string[]).includes(name)) {
        log.warning(
          Modules.CONFIG,
          `viewer_config.audio.buses has unknown bus "${name}"; ignored.`
        );
        continue;
      }
      const gain = clampGain(value, `buses.${name}`);
      if (gain !== undefined) buses[name as AudioBusName] = gain;
    }
    if (Object.keys(buses).length > 0) out.buses = buses;
  }

  if (cfg.duck_db !== undefined && cfg.duck_db !== null) {
    if (typeof cfg.duck_db === 'number' && Number.isFinite(cfg.duck_db)) {
      out.duckDb = Math.min(0, Math.max(-60, cfg.duck_db));
    } else {
      log.warning(Modules.CONFIG, 'viewer_config.audio.duck_db is not a number; ignored.');
    }
  }
  return out;
}
