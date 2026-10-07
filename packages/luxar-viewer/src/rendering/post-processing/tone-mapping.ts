/**
 * Tone-mapping name ↔ `THREE.ToneMapping` mapping — the single source
 * of truth for every consumer.
 *
 * The viewer names tone-mapping modes with strings ('ACES', 'AgX', …)
 * because that is what travels through the GUI, `localStorage`, and the
 * zarr `viewer_config`. THREE wants its own enum. That translation used
 * to be re-implemented in four places (the GUI dropdown, the settings
 * applier, the cinematic toggle, the post-processing resource builder)
 * plus two hardcoded `?? THREE.ACESFilmicToneMapping` fallbacks inside
 * the mega-shader materials — five copies to keep in step with the
 * `renderingControls.defaults.toneMapping` default. They now all route
 * through this module, as does the mega-shader's own compact mode id
 * ({@link luxarToneMappingMode}), shared by the GLSL and TSL materials.
 *
 * @module rendering/post-processing/tone-mapping
 */

import * as THREE from 'three';
import { config } from '../../config';
import type { ToneMappingName } from '../../types/format-contract';

/**
 * Tone-mapping mode names, as spelled in config / GUI / zarr metadata.
 * Single-sourced from `format-contract/contract.yaml::tone_mappings` (the
 * Python `VALID_TONE_MAPPINGS` is the same projection); re-exported here so
 * the historical import site keeps working.
 */
export type { ToneMappingName };

/** Name → THREE enum. Exhaustive over {@link ToneMappingName} by type. */
export const TONE_MAPPING_BY_NAME: Record<ToneMappingName, THREE.ToneMapping> = {
  None: THREE.NoToneMapping,
  Linear: THREE.LinearToneMapping,
  Reinhard: THREE.ReinhardToneMapping,
  Cineon: THREE.CineonToneMapping,
  ACES: THREE.ACESFilmicToneMapping,
  AgX: THREE.AgXToneMapping,
  Neutral: THREE.NeutralToneMapping,
};

/**
 * Selectable mode names in GUI order (least → most processed).
 * Drives the tone-mapping dropdown so it can never drift from the map.
 */
export const TONE_MAPPING_NAMES = Object.keys(TONE_MAPPING_BY_NAME) as ToneMappingName[];

/**
 * The configured default (`renderingControls.defaults.toneMapping`) as a
 * THREE enum. Every "no tone mapping was specified" path resolves here,
 * so changing the config default moves the whole viewer at once.
 */
export function resolveToneMappingDefault(): THREE.ToneMapping {
  return TONE_MAPPING_BY_NAME[config.renderingControls.defaults.toneMapping];
}

/**
 * Resolve a mode name coming from untrusted state (persisted settings,
 * zarr `viewer_config`, a URL parameter). Unknown names fall back to the
 * configured default rather than leaving tone mapping undefined.
 */
export function toneMappingFromName(name: string): THREE.ToneMapping {
  // `Object.hasOwn` guard, not `?? default`: bracket-indexing a plain object
  // with an untrusted name would return inherited members ('constructor',
  // 'toString', …) instead of falling back — those keys are truthy, so `??`
  // wouldn't catch them.
  return Object.hasOwn(TONE_MAPPING_BY_NAME, name)
    ? TONE_MAPPING_BY_NAME[name as ToneMappingName]
    : resolveToneMappingDefault();
}

/**
 * The mega-shader's compact tone-mapping mode: the `LUXAR_TONE_MAPPING_MODE`
 * define of the GLSL material and the `toneMappingMode` of the TSL factory.
 * Luxar-internal IDs (1=Linear, 2=Reinhard, 3=Cineon, 4=ACES, 5=AgX,
 * 6=Neutral), NOT THREE's enum values.
 */
export type LuxarToneMappingMode = 1 | 2 | 3 | 4 | 5 | 6;

/** The THREE enum of each mega-shader mode, in mode order (mode = index + 1). */
const TONE_MAPPING_BY_LUXAR_MODE: readonly THREE.ToneMapping[] = [
  THREE.LinearToneMapping,
  THREE.ReinhardToneMapping,
  THREE.CineonToneMapping,
  THREE.ACESFilmicToneMapping,
  THREE.AgXToneMapping,
  THREE.NeutralToneMapping,
];

/**
 * THREE enum → mega-shader mode, for both backends' materials.
 * `NoToneMapping` aliases to Linear, so the shader still clamps to [0, 1];
 * an enum with no mode (`CustomToneMapping`) gets Neutral, a gentle
 * hue-preserving rolloff.
 */
export function luxarToneMappingMode(mapping: THREE.ToneMapping): LuxarToneMappingMode {
  if (mapping === THREE.NoToneMapping) return 1;
  const index = TONE_MAPPING_BY_LUXAR_MODE.indexOf(mapping);
  return (index < 0 ? 6 : index + 1) as LuxarToneMappingMode;
}

/**
 * Mega-shader mode → THREE enum: the inverse of {@link luxarToneMappingMode},
 * so `NoToneMapping` reads back as `LinearToneMapping`. An out-of-range mode
 * reads back as Neutral.
 */
export function toneMappingFromLuxarMode(mode: number): THREE.ToneMapping {
  return TONE_MAPPING_BY_LUXAR_MODE[mode - 1] ?? THREE.NeutralToneMapping;
}
