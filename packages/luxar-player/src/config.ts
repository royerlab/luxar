// Configuration constants for the Luxar scene player
//
// This module centralizes all configuration values to ensure consistency
// and make the application easy to customize. All magic numbers and
// configuration parameters are defined here with clear documentation.

/**
 * Main configuration object containing all application settings
 * 
 * Organized by functional area for easy navigation and maintenance.
 * Uses 'as const' to ensure TypeScript treats these as literal values
 * rather than generic types, enabling better type checking.
 */
export const CONFIG = {
  /** Camera configuration for 3D perspective and navigation */
  CAMERA: {
    /** Field of view in degrees - 60° provides natural human-like viewing angle */
    FOV: 60,
    
    /** Near clipping plane distance - objects closer than this are not rendered */
    NEAR: 0.1,
    
    /** Far clipping plane distance - objects further than this are not rendered */
    FAR: 1000,
    
    /** Initial camera position in 3D space (world coordinates) */
    INITIAL_POSITION: { x: 0, y: 0, z: 8 },
    
    /** Minimum field of view for zoom limits - prevents excessive zoom-in */
    FOV_MIN: 10,
    
    /** Maximum field of view for zoom limits - prevents excessive zoom-out */
    FOV_MAX: 200,
    
    /** FOV change sensitivity for Shift+wheel input - lower = finer control */
    FOV_SENSITIVITY: 0.05,
  },
  
  /** Animation loop and performance optimization settings */
  ANIMATION: {
    /** Time in milliseconds before pausing animation when idle - saves power */
    IDLE_TIMEOUT_MS: 2000,
  },
  
  /** 3D scene visual configuration */
  SCENE: {
    /** Background color in hexadecimal - dark gray for good contrast with point clouds */
    BACKGROUND_COLOR: 0x111111,
  },
  
  /** HDR post-processing and rendering configuration */
  RENDERING: {
    /** Enable HDR post-processing pipeline with bloom effects */
    HDR_ENABLED: true,
    
    /** Bloom effect settings */
    BLOOM: {
      /** Bloom intensity - how strong the glow effect appears */
      STRENGTH: 0.1,
      
      /** Bloom radius - how far the glow spreads from bright areas */
      RADIUS: 0.5,
      
      /** Bloom threshold - brightness level required to trigger bloom (0.0 = everything glows) */
      THRESHOLD: 0.0,
      
      /** Resolution scale for bloom pass - higher = faster but lower quality */
      RESOLUTION_SCALE: 4,
    },
    
    /** Advanced point rendering settings */
    POINTS: {
      /** Base point size in screen pixels */
      SIZE: 4.0,
      
      /** HDR color multiplier for driving bloom effects */
      HDR_MULTIPLIER: 13.0,
      
      /** Base alpha intensity for point visibility */
      BASE_ALPHA: 0.01,
      
      /** Gaussian falloff steepness for smooth point edges */
      FALLOFF_STEEPNESS: 40.0,
    },
  },
  
  /** Default path to demo Zarr data when no source is specified */
  DEFAULT_ZARR_PATH: "/data/demo.zarr",
} as const;

/** HTML element ID for the canvas where 3D rendering occurs */
export const CANVAS_ID = 'app';

/** Accessibility configuration for screen readers and assistive technology */
export const ACCESSIBILITY = {
  /** ARIA label describing the 3D canvas and its controls for screen readers */
  CANVAS_ARIA_LABEL: '3D scene viewer - Use mouse to navigate, Shift+wheel to zoom',
} as const;