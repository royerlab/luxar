/**
 * Tone-mapping handler for the post-processing pipeline.
 *
 * Pure mode-map tables (THREE ↔ pmndrs `ToneMappingMode`) plus thin operations
 * over a {@link LuxarToneMappingEffect}. Extracted from
 * `post-processing-manager.ts` so the conversions and the apply/update calls
 * are testable without an `EffectComposer` or a WebGL context.
 *
 * @module rendering/post-processing/tone-mapping-handler
 */

import * as THREE from 'three';
import { ToneMappingMode } from 'postprocessing';
import { log, Modules } from '../../utils/log';
import type { LuxarToneMappingEffect } from './luxar-tone-mapping-effect';
import { toneMappingModeName } from './tone-mapping-mode-names';

/**
 * Structural minimum we need from a tone-mapping effect: the four
 * mutable properties this handler reads or writes. Using a structural
 * type here (rather than the concrete {@link LuxarToneMappingEffect})
 * keeps the helpers trivially testable with a plain object stub.
 */
export type ToneMappingTarget = Pick<
  LuxarToneMappingEffect,
  'mode' | 'exposure' | 'globalOffset' | 'globalGamma'
>;

/**
 * Map from THREE.js tone-mapping constants to pmndrs `ToneMappingMode`.
 * Anything outside this table falls back to {@link ToneMappingMode.ACES_FILMIC}.
 */
const THREE_TO_PMNDRS: Record<number, ToneMappingMode> = {
  [THREE.NoToneMapping]: ToneMappingMode.LINEAR,
  [THREE.LinearToneMapping]: ToneMappingMode.LINEAR,
  [THREE.ReinhardToneMapping]: ToneMappingMode.REINHARD,
  [THREE.CineonToneMapping]: ToneMappingMode.OPTIMIZED_CINEON,
  [THREE.ACESFilmicToneMapping]: ToneMappingMode.ACES_FILMIC,
  [THREE.AgXToneMapping]: ToneMappingMode.AGX,
  [THREE.NeutralToneMapping]: ToneMappingMode.NEUTRAL,
};

/**
 * Map from pmndrs `ToneMappingMode` to THREE.js tone-mapping constants.
 *
 * Several pmndrs modes (`REINHARD2`, `REINHARD2_ADAPTIVE`, `UNCHARTED2`,
 * `CINEON`) collapse onto the closest THREE constant, since THREE has a
 * coarser enum.
 */
const PMNDRS_TO_THREE: Record<ToneMappingMode, THREE.ToneMapping> = {
  [ToneMappingMode.LINEAR]: THREE.LinearToneMapping,
  [ToneMappingMode.REINHARD]: THREE.ReinhardToneMapping,
  [ToneMappingMode.REINHARD2]: THREE.ReinhardToneMapping,
  [ToneMappingMode.REINHARD2_ADAPTIVE]: THREE.ReinhardToneMapping,
  [ToneMappingMode.UNCHARTED2]: THREE.CineonToneMapping,
  [ToneMappingMode.OPTIMIZED_CINEON]: THREE.CineonToneMapping,
  [ToneMappingMode.CINEON]: THREE.CineonToneMapping,
  [ToneMappingMode.ACES_FILMIC]: THREE.ACESFilmicToneMapping,
  [ToneMappingMode.AGX]: THREE.AgXToneMapping,
  [ToneMappingMode.NEUTRAL]: THREE.NeutralToneMapping,
};

/**
 * Convert a THREE.js tone-mapping constant to the equivalent pmndrs
 * `ToneMappingMode`. Falls back to `ACES_FILMIC` for unknown values.
 */
export function threeToPmndrsToneMapping(mode: THREE.ToneMapping): ToneMappingMode {
  return THREE_TO_PMNDRS[mode] ?? ToneMappingMode.ACES_FILMIC;
}

/**
 * Convert a pmndrs `ToneMappingMode` to the closest THREE.js tone-mapping
 * constant. Falls back to `ACESFilmicToneMapping` for unknown values.
 */
export function pmndrsToThreeToneMapping(mode: ToneMappingMode): THREE.ToneMapping {
  return PMNDRS_TO_THREE[mode] ?? THREE.ACESFilmicToneMapping;
}

/**
 * Apply a THREE tone-mapping mode to the given effect (no-op if the effect
 * is absent, e.g. between dispose and rebuild). Logs the resulting pmndrs
 * mode by canonical name.
 */
export function applyToneMapping(
  effect: ToneMappingTarget | null | undefined,
  mode: THREE.ToneMapping
): void {
  if (!effect) return;
  const mapped = threeToPmndrsToneMapping(mode);
  effect.mode = mapped;
  log.update(Modules.POST_PROCESSING, `Tone mapping set to: ${toneMappingModeName(mapped)}`);
}

/**
 * Read the current tone-mapping mode back as a THREE constant. Returns
 * `ACESFilmicToneMapping` when the effect is absent.
 */
export function readToneMapping(effect: ToneMappingTarget | null | undefined): THREE.ToneMapping {
  if (!effect) return THREE.ACESFilmicToneMapping;
  return pmndrsToThreeToneMapping(effect.mode);
}

/**
 * Set the global exposure (log2 stops) on the tone-mapping effect.
 * No-op when the effect is absent.
 */
export function applyExposure(effect: ToneMappingTarget | null | undefined, value: number): void {
  if (effect) effect.exposure = value;
}

/**
 * Set the global additive offset on the tone-mapping effect.
 * No-op when the effect is absent.
 */
export function applyGlobalOffset(
  effect: ToneMappingTarget | null | undefined,
  value: number
): void {
  if (effect) effect.globalOffset = value;
}

/**
 * Set the global gamma value on the tone-mapping effect.
 * No-op when the effect is absent.
 */
export function applyGlobalGamma(
  effect: ToneMappingTarget | null | undefined,
  value: number
): void {
  if (effect) effect.globalGamma = value;
}
