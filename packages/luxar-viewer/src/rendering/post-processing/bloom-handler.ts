/**
 * Bloom-effect handler for the post-processing pipeline.
 *
 * Pure helpers + thin operations over a {@link BloomEffectTyped}. The
 * actual `new BloomEffect(...)` construction stays in
 * `post-processing-manager.ts` (it requires the live pmndrs class), but
 * everything *around* the construction — the parameter resolution, the
 * settings read-back used during a levels-change rebuild, and the
 * settings update path — lives here so it can be unit-tested without an
 * `EffectComposer` or a WebGL context.
 *
 * @module rendering/post-processing/bloom-handler
 */

import { BlendFunction, KernelSize } from 'postprocessing';
import type { BloomEffect } from 'postprocessing';
import { log, Modules } from '../../utils/log';
import { config } from '../../config';
import type { BloomEffectTyped } from './postprocessing-types';
import { isBloomEffectTyped } from './postprocessing-types';

/**
 * Resolved bloom parameter triple (the three knobs the UI exposes).
 */
export interface BloomSettings {
  /** Bloom intensity (a.k.a. "strength"). */
  intensity: number;
  /** Mipmap-blur radius. */
  radius: number;
  /** Luminance threshold (only pixels above this contribute). */
  threshold: number;
}

/**
 * Constructor options for `new BloomEffect(...)`. Mirrors the shape we
 * actually pass at the call site so the manager can spread the result
 * directly.
 */
export interface BloomConstructorOptions {
  intensity: number;
  luminanceThreshold: number;
  luminanceSmoothing: number;
  mipmapBlur: boolean;
  kernelSize: KernelSize;
  blendFunction: BlendFunction;
  levels: number;
}

/**
 * Structural slice of the parts of `BloomEffectTyped` we read or write.
 * Using a Pick-style structural type keeps the helpers easy to test
 * with plain object stubs.
 */
export interface BloomTarget {
  intensity: number;
  mipmapBlurPass?: { radius: number };
  luminanceMaterial?: { threshold: number };
}

/**
 * Default bloom parameters from the rendering-controls config.
 */
export function getDefaultBloomSettings(): BloomSettings {
  return {
    intensity: config.renderingControls.defaults.bloomStrength,
    radius: config.renderingControls.defaults.bloomRadius,
    threshold: config.renderingControls.defaults.bloomThreshold,
  };
}

/**
 * Resolve a partial parameter set against the defaults — used both when
 * creating a new {@link BloomEffect} and when re-creating one (level
 * change) where we want to preserve current values.
 *
 * Each parameter falls back to its corresponding default if undefined.
 */
export function resolveBloomSettings(
  partial: Partial<BloomSettings> = {},
  base: BloomSettings = getDefaultBloomSettings()
): BloomSettings {
  return {
    intensity: partial.intensity ?? base.intensity,
    radius: partial.radius ?? base.radius,
    threshold: partial.threshold ?? base.threshold,
  };
}

/**
 * Read current settings off a {@link BloomTarget}, falling back to the
 * defaults for any part the effect doesn't expose. Used during the
 * levels-change rebuild to preserve the user's current settings.
 */
export function readBloomSettings(
  effect: BloomTarget,
  defaults: BloomSettings = getDefaultBloomSettings()
): BloomSettings {
  return {
    intensity: effect.intensity || defaults.intensity,
    radius: effect.mipmapBlurPass?.radius ?? defaults.radius,
    threshold: effect.luminanceMaterial?.threshold ?? defaults.threshold,
  };
}

/**
 * Build the constructor options for a fresh {@link BloomEffect}.
 *
 * Note: `radius` is **not** part of the constructor options (pmndrs
 * sets it post-construction via `mipmapBlurPass.radius`). The caller is
 * responsible for calling {@link applyBloomRadius} after `new`.
 */
export function buildBloomConstructorOptions(
  settings: BloomSettings,
  levels: number
): BloomConstructorOptions {
  return {
    intensity: settings.intensity,
    luminanceThreshold: settings.threshold,
    luminanceSmoothing: 0.01,
    mipmapBlur: true,
    kernelSize: KernelSize.LARGE,
    blendFunction: BlendFunction.ADD,
    levels,
  };
}

/**
 * Set the radius on the bloom effect's mipmap-blur pass. Returns
 * `true` when the assignment was made; `false` if the effect lacks a
 * `mipmapBlurPass` (e.g. mid-disposal or a non-mipmap bloom).
 */
export function applyBloomRadius(
  effect: { mipmapBlurPass?: { radius: number } },
  radius: number
): boolean {
  if (!effect.mipmapBlurPass) return false;
  effect.mipmapBlurPass.radius = radius;
  return true;
}

/**
 * Apply a partial settings update to a bloom effect. Returns the new
 * resolved triple so the caller can log or surface the values, and
 * `null` when the effect is absent or doesn't look like a BloomEffect.
 */
export function applyBloomSettings(
  effect: BloomEffectTyped | BloomEffect | null | undefined,
  partial: Partial<BloomSettings>
): BloomSettings | null {
  if (!effect) {
    log.warning(Modules.POST_PROCESSING, 'Bloom effect not initialized');
    return null;
  }
  if (!isBloomEffectTyped(effect)) {
    log.error(Modules.POST_PROCESSING, 'Invalid bloom effect type');
    return null;
  }
  const target = effect as unknown as BloomTarget;

  if (partial.intensity !== undefined) target.intensity = partial.intensity;
  if (partial.radius !== undefined && target.mipmapBlurPass) {
    target.mipmapBlurPass.radius = partial.radius;
  }
  if (partial.threshold !== undefined && target.luminanceMaterial) {
    target.luminanceMaterial.threshold = partial.threshold;
  }

  const resolved: BloomSettings = {
    intensity: target.intensity || 0,
    radius: target.mipmapBlurPass?.radius ?? 0,
    threshold: target.luminanceMaterial?.threshold ?? 0,
  };

  log.update(
    Modules.POST_PROCESSING,
    `Bloom updated: strength=${resolved.intensity.toFixed(2)}, ` +
      `radius=${resolved.radius.toFixed(2)}, threshold=${resolved.threshold.toFixed(2)}`
  );
  return resolved;
}

/**
 * Clamp a bloom-levels value to the supported range (1..12) and round
 * to the nearest integer. Pure.
 */
export function clampBloomLevels(levels: number): number {
  return Math.round(Math.max(1, Math.min(12, levels)));
}
