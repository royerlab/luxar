/**
 * WebGL context attributes
 */
export interface WebGLContextAttributes {
  alpha: boolean;
  antialias: boolean;
  depth: boolean;
  stencil: boolean;
  powerPreference: 'high-performance' | 'low-power' | 'default';
  preserveDrawingBuffer: boolean;
  desynchronized: boolean;
  premultipliedAlpha: boolean;
  failIfMajorPerformanceCaveat: boolean;
}

/**
 * THREE.WebGLRenderer configuration (renderer-specific settings only).
 * Shared attributes (alpha, antialias, depth, stencil, powerPreference, preserveDrawingBuffer,
 * premultipliedAlpha) live in WebGLContextAttributes and are spread
 * alongside these at renderer creation time.
 */
export interface WebGLRendererConfig {
  logarithmicDepthBuffer: boolean;
  precision: 'highp' | 'mediump' | 'lowp';
  shadowMap: {
    enabled: boolean;
    type: number;
  };
}

/**
 * WebGL render target configuration
 */
export interface WebGLRenderTargetConfig {
  depthBuffer: boolean;
  stencilBuffer: boolean;
  samples: number;
}

/**
 * WebGL configuration
 */
export interface WebGLConfig {
  context: WebGLContextAttributes;
  renderer: WebGLRendererConfig;
  renderTarget: WebGLRenderTargetConfig;
}
