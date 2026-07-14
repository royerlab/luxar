import * as THREE from 'three';
import type { WebGLConfig } from './types';

/**
 * WebGL context and renderer configuration
 */
export const webglConfig: WebGLConfig = {
  // WebGL2 context attributes for canvas
  context: {
    alpha: false, // No transparency in canvas background
    // The scene never renders to the backbuffer — everything goes through
    // the HDR render target (renderTarget.samples controls real MSAA), so
    // an antialiased backbuffer is a dead multisample allocation.
    antialias: false,
    depth: true, // Enable depth buffer for 3D rendering
    stencil: false, // No stencil buffer needed (saves memory)
    powerPreference: 'high-performance' as const, // Request high-performance GPU
    // NOTE: no `colorSpace` here — it is NOT a WebGL context attribute
    // (drawing-buffer color space is `gl.drawingBufferColorSpace`) and the
    // previous 'display-p3' entry was silently ignored. Output color
    // handling lives in the HDR pipeline (post-processing-manager).
    preserveDrawingBuffer: false, // Don't preserve buffer (better performance)
    desynchronized: true, // Better performance with async updates
    premultipliedAlpha: true, // Standard alpha blending
    failIfMajorPerformanceCaveat: false, // Don't fail on slow GPUs
  },

  // THREE.WebGLRenderer specific settings (renderer-only; shared attributes
  // like antialias, powerPreference, preserveDrawingBuffer, premultipliedAlpha
  // are sourced from webgl.context and spread at renderer creation time)
  renderer: {
    logarithmicDepthBuffer: false, // Standard depth buffer (faster)
    precision: 'highp' as const, // High precision for better quality
    shadowMap: {
      enabled: false, // No shadows needed for points
      type: THREE.PCFShadowMap, // PCF is soft by default since three r182
    },
  },

  // Render target configuration for post-processing
  renderTarget: {
    depthBuffer: true, // Needed for depth testing
    stencilBuffer: false, // Not needed, saves memory
    samples: 0, // MSAA samples (0 = disabled for additive blending compatibility)
  },
};
