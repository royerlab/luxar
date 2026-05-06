/**
 * Luxar Logging Style Utility
 *
 * Provides consistent, structured logging that works with the console interceptor.
 * All logs follow the format: [emoji] [Module] message
 *
 * This is a lightweight wrapper that ensures consistent formatting while
 * allowing the console interceptor to capture all output.
 */

/**
 * Standard emojis for different log categories
 */
export const LogEmoji = {
  // Status
  START: '🚀',
  SUCCESS: '✅',
  ERROR: '❌',
  WARNING: '⚠️',
  INFO: 'ℹ️',

  // Actions
  LOAD: '📥',
  SAVE: '💾',
  UPDATE: '🔄',
  DELETE: '🗑️',
  SEARCH: '🔍',
  QUERY: '🔍',
  CLEAN: '🧹',
  BROADCAST: '📡',
  TARGET: '🎯',
  ROCKET: '🚀',

  // Data
  DATA: '📊',
  CACHE: '💾',
  NETWORK: '🌐',
  FILE: '📄',
  SCENE: '🎬',

  // Rendering
  RENDER: '🎨',
  RESIZE: '📐',
  FULLSCREEN: '🖥️',
  HDR: '🌟',
  EFFECT: '✨',

  // Controls
  CONTROLS: '🎮',
  INPUT: '⌨️',

  // UI
  UI: '🖼️',
  WINDOW: '🪟',
  PANEL: '📋',

  // Debug
  DEBUG: '🐛',
  CONSOLE: '🔧',
  MONITOR: '📊',
  PERFORMANCE: '⚡',
  MEMORY: '💾',
} as const;

/**
 * Format a log message with consistent style
 * @param emoji - Emoji prefix (use LogEmoji constants)
 * @param module - Module name to appear in brackets
 * @param message - The log message
 */
export function formatLog(emoji: string, module: string, message: string): string {
  return `[${emoji}] [${module}] ${message}`;
}

/**
 * Quick logging functions that maintain consistent format
 * These work directly with console methods so the interceptor captures them
 */
export const log = {
  // Basic logging with module
  info: (module: string, message: string, ...args: unknown[]) => {
    console.log(`[${LogEmoji.INFO}] [${module}] ${message}`, ...args);
  },

  success: (module: string, message: string, ...args: unknown[]) => {
    console.log(`[${LogEmoji.SUCCESS}] [${module}] ${message}`, ...args);
  },

  error: (module: string, message: string, ...args: unknown[]) => {
    console.error(`[${LogEmoji.ERROR}] [${module}] ${message}`, ...args);
  },

  warning: (module: string, message: string, ...args: unknown[]) => {
    console.warn(`[${LogEmoji.WARNING}] [${module}] ${message}`, ...args);
  },

  // Action-specific logging
  load: (module: string, message: string, ...args: unknown[]) => {
    console.log(`[${LogEmoji.LOAD}] [${module}] ${message}`, ...args);
  },

  update: (module: string, message: string, ...args: unknown[]) => {
    console.log(`[${LogEmoji.UPDATE}] [${module}] ${message}`, ...args);
  },

  query: (module: string, message: string, ...args: unknown[]) => {
    console.log(`[${LogEmoji.QUERY}] [${module}] ${message}`, ...args);
  },

  data: (module: string, message: string, ...args: unknown[]) => {
    console.log(`[${LogEmoji.DATA}] [${module}] ${message}`, ...args);
  },

  // Custom emoji logging
  custom: (emoji: string, module: string, message: string, ...args: unknown[]) => {
    console.log(`[${emoji}] [${module}] ${message}`, ...args);
  },

  // Raw console access (already formatted)
  raw: (formattedMessage: string, ...args: unknown[]) => {
    console.log(formattedMessage, ...args);
  },
};

/**
 * Module names for consistent identification
 */
export const Modules = {
  // Core
  LUXAR: 'Luxar',
  APP: 'App',
  MAIN: 'Main',

  // Data Loading
  SCENE_LOADER: 'SceneLoader',
  SPATIAL_INDEX_LOADER: 'PointSpatialIndexLoader',
  GSPLATS_SPATIAL_INDEX_LOADER: 'GSplatsSpatialIndexLoader',
  SPATIAL_INDEX: 'PointSpatialIndex',
  DATA_MONITOR: 'DataMonitor',
  DATA_ACCUMULATOR: 'DataAccumulator',
  WORKER_POOL: 'WorkerPool',
  ZARR_LOADER: 'ZarrLoader',
  RANGE_CACHE: 'RangeCache',
  CACHE: 'Cache',
  LINES_LOADER: 'LinesSpatialIndexLoader',
  WASM: 'WASM',
  SCENE_DIMS: 'SceneDims',

  // Rendering
  RENDERER: 'Renderer',
  GPU_BUFFER_POOL: 'GPUBufferPool',
  POST_PROCESSING: 'PostProcessing',
  HDR: 'HDR',
  SCENE_MANAGER: 'SceneManager',

  // Controls
  CONTROLS: 'Controls',
  ORBIT_CONTROLS: 'OrbitControls',
  FLY_CONTROLS: 'FlyControls',
  INPUT: 'Input',
  INPUT_CONTEXT: 'InputContext',

  // UI
  UI: 'UI',
  DEBUG_CONSOLE: 'DebugConsole',
  DATA_LOADING_MONITOR: 'DataLoadingMonitor',
  RENDERING_CONTROLS: 'RenderingControls',
  RECORDING: 'Recording',
  ANIMATION: 'DimAnimation',

  // Utils
  MEMORY: 'Memory',
  PERFORMANCE: 'Performance',
  CONSOLE_INTERCEPTOR: 'ConsoleInterceptor',
  ADAPTIVE_DPR: 'AdaptiveDPR',
  EVENT_GROUP: 'EventGroup',
} as const;

/**
 * Helper to create module-specific loggers
 */
export function createModuleLogger(module: string) {
  return {
    log: (message: string, ...args: unknown[]) => log.info(module, message, ...args),
    info: (message: string, ...args: unknown[]) => log.info(module, message, ...args),
    success: (message: string, ...args: unknown[]) => log.success(module, message, ...args),
    error: (message: string, ...args: unknown[]) => log.error(module, message, ...args),
    warning: (message: string, ...args: unknown[]) => log.warning(module, message, ...args),
    load: (message: string, ...args: unknown[]) => log.load(module, message, ...args),
    update: (message: string, ...args: unknown[]) => log.update(module, message, ...args),
    query: (message: string, ...args: unknown[]) => log.query(module, message, ...args),
    data: (message: string, ...args: unknown[]) => log.data(module, message, ...args),
    custom: (emoji: string, message: string, ...args: unknown[]) =>
      log.custom(emoji, module, message, ...args),
  };
}
