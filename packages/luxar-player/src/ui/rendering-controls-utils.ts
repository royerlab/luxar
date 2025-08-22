/**
 * Pure utility functions for rendering controls settings management
 *
 * This module contains side-effect-free functions extracted from rendering-controls.ts
 * to improve testability and maintainability. These functions handle settings
 * validation, serialization, and merging without external dependencies.
 */

/**
 * Rendering settings structure
 */
export interface RenderingSettings {
  // Display
  showStats: boolean;
  showAxes: boolean;
  showGrid: boolean;
  backgroundColor: string;

  // Anti-aliasing
  antialiasing: 'none' | 'fxaa' | 'smaa' | 'msaa' | 'ssaa';
  msaaSamples: number;
  ssaaMultiplier: number;

  // Post-processing
  bloomEnabled: boolean;
  bloomIntensity: number;
  bloomThreshold: number;
  bloomRadius: number;

  // Tone mapping
  toneMapping: string;
  exposure: number;

  // Camera controls
  controlType: 'orbit' | 'arcball' | 'fly';
  autoRotate: boolean;
  autoRotateSpeed: number;

  // Fly controls
  flyInertialMode: boolean;
  flyMovementSpeed: number;
  flyDamping: number;
  flyRotationDamping: number;
}

/**
 * Default rendering settings
 */
export const DEFAULT_RENDERING_SETTINGS: RenderingSettings = {
  // Display
  showStats: false,
  showAxes: false,
  showGrid: false,
  backgroundColor: '#000000',

  // Anti-aliasing
  antialiasing: 'smaa',
  msaaSamples: 4,
  ssaaMultiplier: 1.5,

  // Post-processing
  bloomEnabled: true,
  bloomIntensity: 1.0,
  bloomThreshold: 0.9,
  bloomRadius: 0.4,

  // Tone mapping
  toneMapping: 'aces',
  exposure: 1.0,

  // Camera controls
  controlType: 'orbit',
  autoRotate: false,
  autoRotateSpeed: 2.0,

  // Fly controls
  flyInertialMode: true,
  flyMovementSpeed: 100,
  flyDamping: 0.9,
  flyRotationDamping: 0.95,
};

/**
 * Validates rendering settings and applies defaults for missing/invalid values
 *
 * @param settings - Partial settings object
 * @returns Complete validated settings
 */
export function validateRenderingSettings(settings: Partial<RenderingSettings>): RenderingSettings {
  const validated: RenderingSettings = { ...DEFAULT_RENDERING_SETTINGS };

  // Display settings
  if (typeof settings.showStats === 'boolean') {
    validated.showStats = settings.showStats;
  }
  if (typeof settings.showAxes === 'boolean') {
    validated.showAxes = settings.showAxes;
  }
  if (typeof settings.showGrid === 'boolean') {
    validated.showGrid = settings.showGrid;
  }
  if (typeof settings.backgroundColor === 'string' && isValidColor(settings.backgroundColor)) {
    validated.backgroundColor = settings.backgroundColor;
  }

  // Anti-aliasing
  const validAA = ['none', 'fxaa', 'smaa', 'msaa', 'ssaa'];
  if (validAA.includes(settings.antialiasing as string)) {
    validated.antialiasing = settings.antialiasing as any;
  }
  if (typeof settings.msaaSamples === 'number') {
    validated.msaaSamples = clampMSAASamples(settings.msaaSamples);
  }
  if (typeof settings.ssaaMultiplier === 'number') {
    validated.ssaaMultiplier = Math.max(1.0, Math.min(4.0, settings.ssaaMultiplier));
  }

  // Post-processing
  if (typeof settings.bloomEnabled === 'boolean') {
    validated.bloomEnabled = settings.bloomEnabled;
  }
  if (typeof settings.bloomIntensity === 'number') {
    validated.bloomIntensity = Math.max(0, Math.min(5, settings.bloomIntensity));
  }
  if (typeof settings.bloomThreshold === 'number') {
    validated.bloomThreshold = Math.max(0, Math.min(2, settings.bloomThreshold));
  }
  if (typeof settings.bloomRadius === 'number') {
    validated.bloomRadius = Math.max(0, Math.min(2, settings.bloomRadius));
  }

  // Tone mapping
  const validToneMappings = ['none', 'linear', 'reinhard', 'cineon', 'aces', 'agx', 'neutral'];
  if (validToneMappings.includes(settings.toneMapping as string)) {
    validated.toneMapping = settings.toneMapping as string;
  }
  if (typeof settings.exposure === 'number') {
    validated.exposure = Math.max(0.1, Math.min(10, settings.exposure));
  }

  // Camera controls
  const validControls = ['orbit', 'arcball', 'fly'];
  if (validControls.includes(settings.controlType as string)) {
    validated.controlType = settings.controlType as any;
  }
  if (typeof settings.autoRotate === 'boolean') {
    validated.autoRotate = settings.autoRotate;
  }
  if (typeof settings.autoRotateSpeed === 'number') {
    validated.autoRotateSpeed = Math.max(-10, Math.min(10, settings.autoRotateSpeed));
  }

  // Fly controls
  if (typeof settings.flyInertialMode === 'boolean') {
    validated.flyInertialMode = settings.flyInertialMode;
  }
  if (typeof settings.flyMovementSpeed === 'number') {
    validated.flyMovementSpeed = Math.max(0.1, Math.min(10000, settings.flyMovementSpeed));
  }
  if (typeof settings.flyDamping === 'number') {
    validated.flyDamping = Math.max(0, Math.min(1, settings.flyDamping));
  }
  if (typeof settings.flyRotationDamping === 'number') {
    validated.flyRotationDamping = Math.max(0, Math.min(1, settings.flyRotationDamping));
  }

  return validated;
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
  defaults: RenderingSettings = DEFAULT_RENDERING_SETTINGS
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
  // Check hex color
  if (/^#[0-9A-F]{6}$/i.test(color)) {
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
  if (current.antialiasing !== previous.antialiasing) {
    rebuildRequired.push('antialiasing');
  }
  if (current.msaaSamples !== previous.msaaSamples) {
    rebuildRequired.push('msaaSamples');
  }
  if (current.ssaaMultiplier !== previous.ssaaMultiplier) {
    rebuildRequired.push('ssaaMultiplier');
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
  switch (settings.antialiasing) {
    case 'none':
      score += 0;
      break;
    case 'fxaa':
      score += 5;
      break;
    case 'smaa':
      score += 10;
      break;
    case 'msaa':
      score += 15 + settings.msaaSamples * 2;
      break;
    case 'ssaa':
      score += 20 + settings.ssaaMultiplier * 10;
      break;
  }

  // Post-processing impact
  if (settings.bloomEnabled) {
    score += 10 + settings.bloomIntensity * 5;
  }

  // Tone mapping impact
  if (settings.toneMapping !== 'none') {
    score += 5;
  }

  // Visual helpers impact
  if (settings.showStats) score += 2;
  if (settings.showAxes) score += 3;
  if (settings.showGrid) score += 3;

  // Auto-rotate impact (continuous rendering)
  if (settings.autoRotate) score += 5;

  return Math.min(100, score);
}
