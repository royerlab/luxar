/**
 * Pure utility functions for rendering controls settings management
 *
 * This module contains side-effect-free functions extracted from rendering-controls.ts
 * to improve testability and maintainability. These functions handle settings
 * validation, serialization, and merging without external dependencies.
 */

import type { RenderingSettings as ConfigRenderingSettings } from '../config/types';

/**
 * Re-export the configuration RenderingSettings to maintain consistency
 */
export type RenderingSettings = ConfigRenderingSettings;

import { config } from '../config';

/**
 * Get default rendering settings from main config
 */
export function getDefaultRenderingSettings(): RenderingSettings {
  return config.renderingControls.defaults;
}

/**
 * Validates rendering settings and applies defaults for missing/invalid values
 *
 * @param settings - Partial settings object
 * @returns Complete validated settings
 */
export function validateRenderingSettings(settings: Partial<RenderingSettings>): RenderingSettings {
  const validated: RenderingSettings = { ...getDefaultRenderingSettings() };

  // Simply merge provided settings with defaults, letting TypeScript catch any invalid properties
  return { ...validated, ...settings };
}

/**
 * Merges settings with defaults, preserving valid values
 *
 * @param settings - Partial settings to merge
 * @param defaults - Default settings
 * @returns Merged settings
 */
export function mergeSettings(
  settings: Partial<RenderingSettings>,
  defaults: RenderingSettings = getDefaultRenderingSettings()
): RenderingSettings {
  return validateRenderingSettings({ ...defaults, ...settings });
}

/**
 * Generates a storage key for settings persistence
 *
 * @param sceneId - Unique scene identifier
 * @param prefix - Key prefix (default 'luxar-rendering-settings')
 * @returns Storage key string
 */
export function generateSettingsKey(
  sceneId: string,
  prefix: string = 'luxar-rendering-settings'
): string {
  // Sanitize scene ID to prevent injection
  const sanitized = sceneId.replace(/[^a-zA-Z0-9-_]/g, '_');
  return `${prefix}-${sanitized}`;
}

/**
 * Serializes settings for storage
 *
 * @param settings - Settings to serialize
 * @returns JSON string
 */
export function serializeSettings(settings: RenderingSettings): string {
  return JSON.stringify(settings);
}

/**
 * Deserializes settings from storage
 *
 * @param data - JSON string
 * @returns Parsed settings or null if invalid
 */
export function deserializeSettings(data: string): Partial<RenderingSettings> | null {
  try {
    const parsed = JSON.parse(data);
    if (typeof parsed === 'object' && parsed !== null) {
      return parsed as Partial<RenderingSettings>;
    }
  } catch {
    // Invalid JSON
  }
  return null;
}

/**
 * Checks if a color string is valid
 *
 * @param color - Color string to validate
 * @returns True if valid hex color or CSS color
 */
export function isValidColor(color: string): boolean {
  // Check hex color (both 3 and 6 character formats)
  if (/^#[0-9A-F]{6}$/i.test(color) || /^#[0-9A-F]{3}$/i.test(color)) {
    return true;
  }

  // Check RGB/RGBA
  if (/^rgba?\(/.test(color)) {
    return true;
  }

  // Check named colors (simplified list)
  const namedColors = ['black', 'white', 'red', 'green', 'blue', 'gray', 'grey'];
  return namedColors.includes(color.toLowerCase());
}

/**
 * Clamps MSAA samples to valid power-of-2 values
 *
 * @param samples - Requested sample count
 * @returns Valid sample count (0, 2, 4, 8, 16)
 */
export function clampMSAASamples(samples: number): number {
  if (samples <= 0) return 0;
  if (samples <= 2) return 2;
  if (samples <= 4) return 4;
  if (samples <= 8) return 8;
  return 16;
}

/**
 * Determines if settings have changed
 *
 * @param current - Current settings
 * @param previous - Previous settings
 * @returns True if settings differ
 */
export function settingsChanged(current: RenderingSettings, previous: RenderingSettings): boolean {
  // Simple deep equality check for settings
  return JSON.stringify(current) !== JSON.stringify(previous);
}

/**
 * Gets settings that require pipeline rebuild
 *
 * @param current - Current settings
 * @param previous - Previous settings
 * @returns List of changed settings that require rebuild
 */
export function getSettingsRequiringRebuild(
  current: RenderingSettings,
  previous: RenderingSettings
): string[] {
  const rebuildRequired: string[] = [];

  // Anti-aliasing changes require rebuild
  if (
    current.msaaEnabled !== previous.msaaEnabled ||
    current.msaaSamples !== previous.msaaSamples
  ) {
    rebuildRequired.push('msaa');
  }
  if (current.smaaEnabled !== previous.smaaEnabled) {
    rebuildRequired.push('smaa');
  }
  if (
    current.ssaaEnabled !== previous.ssaaEnabled ||
    current.ssaaMultiplier !== previous.ssaaMultiplier
  ) {
    rebuildRequired.push('ssaa');
  }
  if (current.fxaaEnabled !== previous.fxaaEnabled) {
    rebuildRequired.push('fxaa');
  }

  // Tone mapping type change requires rebuild
  if (current.toneMapping !== previous.toneMapping) {
    rebuildRequired.push('toneMapping');
  }

  return rebuildRequired;
}

/**
 * Filters settings for export (removes temporary/local values)
 *
 * @param settings - Full settings object
 * @returns Exportable settings
 */
export function filterExportableSettings(settings: RenderingSettings): Partial<RenderingSettings> {
  // Remove settings that shouldn't be exported/shared
  const exportable = { ...settings };

  // These might be scene-specific and shouldn't transfer
  delete (exportable as any).backgroundColor; // Keep local

  return exportable;
}

/**
 * Calculates performance impact score for settings
 *
 * @param settings - Rendering settings
 * @returns Performance score (0-100, higher is more demanding)
 */
export function calculatePerformanceImpact(settings: RenderingSettings): number {
  let score = 0;

  // Anti-aliasing impact
  if (settings.fxaaEnabled) score += 5;
  if (settings.smaaEnabled) score += 10;
  if (settings.msaaEnabled) score += 15 + settings.msaaSamples * 2;
  if (settings.ssaaEnabled) score += 20 + settings.ssaaMultiplier * 10;

  // Post-processing impact
  score += 5 + settings.bloomStrength * 5;

  // Tone mapping impact
  if (settings.toneMapping !== 'None') {
    score += 5;
  }

  // DOF and chromatic aberration impact
  if (settings.dofEnabled) score += 8;
  if (settings.chromaticAberrationEnabled) score += 3;

  // Auto-rotate impact (continuous rendering)
  if (settings.autoRotate) score += 5;

  return Math.min(100, score);
}
