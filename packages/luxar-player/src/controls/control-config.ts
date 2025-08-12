/**
 * Central configuration for all control system constants
 * 
 * This file consolidates all magic numbers and configuration values
 * that were previously scattered throughout the codebase.
 */

import type { ControlConfig } from './types';

/**
 * Central control system configuration
 * All values are validated and documented
 */
export const CONTROL_CONFIG: ControlConfig = {
  fly: {
    movement: {
      speed: {
        min: 0.5,
        max: 50.0,
        default: 5.0,
        step: 0.1
      },
      acceleration: {
        min: 0.1,
        max: 2.0,
        default: 0.5,
        step: 0.1
      },
      damping: {
        min: 0.9,
        max: 0.9999,
        default: 0.999,
        step: 0.0001
      }
    },
    look: {
      mouseSpeed: {
        default: 0.002  // Radians per pixel
      },
      keyboardSpeed: {
        default: 0.15  // Radians per second (1.5 / 10 for precise control)
      }
    },
    physics: {
      velocityThreshold: 1e-5,  // Below this velocity, movement stops completely
      dampingPower: 60  // Normalization factor for 60fps damping calculation
    }
  },
  orbit: {
    autoRotate: {
      speed: {
        min: 0.1,
        max: 5.0,
        default: 0.25,
        step: 0.1
      }
    },
    zoom: {
      minDistance: 0.1,
      maxDistance: 1000,
      speed: {
        min: 0.5,
        max: 2.0,
        default: 1.0,
        step: 0.1
      }
    },
    damping: {
      enabled: true,
      factor: {
        min: 0.01,
        max: 0.3,
        default: 0.05,
        step: 0.01
      }
    }
  }
} as const;

/**
 * Rendering configuration constants
 */
export const RENDERING_CONFIG = {
  animation: {
    idleTimeoutMs: 2000  // Pause rendering after 2 seconds of inactivity
  },
  performance: {
    targetFPS: 60,
    minFPS: 30
  }
} as const;

/**
 * Input handling configuration
 */
export const INPUT_CONFIG = {
  keyboard: {
    // Key bindings for various modes
    shortcuts: {
      toggleFullscreen: ' ',      // Space
      toggleHelp: 'h',
      toggleDimensions: 'n',       // Changed from 'd' to avoid WASD conflict
      toggleDatasetBrowser: 'o',
      togglePerformance: 'p',
      toggleRendering: 'r',
      toggleCenter: 'c',
      toggleDebugConsole: 'ctrl+l',
      recenterCamera: 'f',
      toggleControlMode: 'v',      // Orbit <-> Fly
      toggleInertialMode: 'i'      // Fly mode only
    },
    // Keys used for fly mode movement (should be disabled in orbit mode)
    flyModeKeys: ['w', 'a', 's', 'd', 'W', 'A', 'S', 'D'],
    // Keys for dimension navigation
    dimensionKeys: ['[', ']', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  },
  mouse: {
    doubleClickDelay: 300  // ms
  }
} as const;

/**
 * Helper function to get config value with validation
 */
export function getConfigValue<T extends number>(
  value: T | undefined,
  config: { min: number; max: number; default: T }
): T {
  if (value === undefined) return config.default;
  return Math.max(config.min, Math.min(config.max, value)) as T;
}

/**
 * Helper to validate and clamp a value within range
 */
export function clampValue(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Get human-readable description for config values
 */
export function getConfigDescription(path: string): string {
  const descriptions: Record<string, string> = {
    'fly.movement.speed': 'Movement speed in units per second',
    'fly.movement.damping': 'How quickly movement slows down (0.9 = quick stop, 0.999 = long drift)',
    'fly.look.keyboardSpeed': 'Camera rotation speed with arrow keys',
    'orbit.autoRotate.speed': 'Auto-rotation speed (revolutions per minute)',
    'orbit.zoom.speed': 'Mouse wheel zoom sensitivity'
  };
  return descriptions[path] || 'Configuration value';
}