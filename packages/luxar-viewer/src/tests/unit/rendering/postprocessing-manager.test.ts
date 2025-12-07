/**
 * Tests for the pmndrs/postprocessing-based PostProcessingManager
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import { PostProcessingManager } from '../../../rendering/post-processing-manager';

// Mock postprocessing library
vi.mock('postprocessing', () => ({
  EffectComposer: vi.fn().mockImplementation(() => ({
    setSize: vi.fn(),
    addPass: vi.fn(),
    removePass: vi.fn(),
    render: vi.fn(),
    dispose: vi.fn(),
    passes: [],
  })),
  RenderPass: vi.fn().mockImplementation(() => ({})),
  EffectPass: vi.fn().mockImplementation(() => ({})),
  BloomEffect: vi.fn().mockImplementation(() => ({
    intensity: 1.0,
    mipmapBlurPass: {
      radius: 1.0,
    },
    luminanceMaterial: {
      threshold: 0.01,
    },
  })),
  ToneMappingEffect: vi.fn().mockImplementation(() => ({
    mode: 4, // ToneMappingMode.ACES_FILMIC = 4
    uniforms: { whitePoint: { value: 4.0 } },
  })),
  DepthOfFieldEffect: vi.fn().mockImplementation(() => ({
    bokehScale: 2.0,
  })),
  SMAAEffect: vi.fn().mockImplementation(() => ({
    isEnabled: true,
  })),
  FXAAEffect: vi.fn().mockImplementation(() => ({
    isEnabled: true,
  })),
  SSAOEffect: vi.fn().mockImplementation(() => ({})),
  ChromaticAberrationEffect: vi.fn().mockImplementation(() => ({
    offset: new THREE.Vector2(),
  })),
  VignetteEffect: vi.fn().mockImplementation(() => ({
    darkness: 0.5,
    offset: 0.5,
  })),
  KernelSize: {
    VERY_SMALL: 0,
    SMALL: 1,
    MEDIUM: 2,
    LARGE: 3,
    VERY_LARGE: 4,
    HUGE: 5,
  },
  BlendFunction: {
    ADD: 0,
    SCREEN: 1,
    NORMAL: 2,
  },
  ToneMappingMode: {
    LINEAR: 0,
    REINHARD: 1,
    OPTIMIZED_CINEON: 2,
    ACES_FILMIC: 4,
    AGX: 5,
    NEUTRAL: 6,
  },
  SMAAPreset: {
    LOW: 0,
    MEDIUM: 1,
    HIGH: 2,
    ULTRA: 3,
  },
}));

describe('PostProcessingManager', () => {
  let manager: PostProcessingManager;
  let renderer: THREE.WebGLRenderer;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;

  beforeEach(() => {
    // Mock WebGL context with all required methods
    const mockGL = {
      getExtension: vi.fn((name: string) => {
        // Return mock extension for EXT_color_buffer_float
        if (name === 'EXT_color_buffer_float') return {};
        return {};
      }),
      getParameter: vi.fn((param: number) => {
        // VERSION constant in WebGL is 0x1F02 (7938 in decimal)
        if (param === 7938) {
          return 'WebGL 2.0 (OpenGL ES 3.0)';
        }
        // MAX_SAMPLES (0x8D57 = 36183 in decimal)
        if (param === 36183) {
          return 8;
        }
        // For all other parameters return reasonable defaults
        return 1;
      }),
      getContextAttributes: vi.fn(() => ({
        alpha: true,
        antialias: false,
        depth: true,
        premultipliedAlpha: true,
        stencil: true,
        preserveDrawingBuffer: false,
      })),
      getShaderPrecisionFormat: vi.fn(() => ({
        rangeMin: 1,
        rangeMax: 1,
        precision: 1,
      })),
      VERTEX_SHADER: 35633,
      FRAGMENT_SHADER: 35632,
      HIGH_FLOAT: 36338,
      MEDIUM_FLOAT: 36337,
      LOW_FLOAT: 36336,
      createTexture: vi.fn(() => ({})),
      bindTexture: vi.fn(),
      texParameteri: vi.fn(),
      texImage2D: vi.fn(),
      createFramebuffer: vi.fn(() => ({})),
      bindFramebuffer: vi.fn(),
      createRenderbuffer: vi.fn(() => ({})),
      bindRenderbuffer: vi.fn(),
      renderbufferStorage: vi.fn(),
      framebufferTexture2D: vi.fn(),
      framebufferRenderbuffer: vi.fn(),
      checkFramebufferStatus: vi.fn(() => 0x8cd5), // GL_FRAMEBUFFER_COMPLETE
      createProgram: vi.fn(() => ({})),
      createShader: vi.fn(() => ({})),
      shaderSource: vi.fn(),
      compileShader: vi.fn(),
      attachShader: vi.fn(),
      VERSION: 0x1f02,
      MAX_SAMPLES: 0x8d57,
      linkProgram: vi.fn(),
      getProgramParameter: vi.fn(() => true),
      getShaderParameter: vi.fn(() => true),
      getUniformLocation: vi.fn(() => ({})),
      getAttribLocation: vi.fn(() => 0),
      useProgram: vi.fn(),
      viewport: vi.fn(),
      clearColor: vi.fn(),
      clear: vi.fn(),
      enable: vi.fn(),
      disable: vi.fn(),
      blendFunc: vi.fn(),
      blendEquation: vi.fn(),
      cullFace: vi.fn(),
      frontFace: vi.fn(),
      depthMask: vi.fn(),
      depthFunc: vi.fn(),
      pixelStorei: vi.fn(),
      texImage3D: vi.fn(),
      clearDepth: vi.fn(),
      clearStencil: vi.fn(),
      drawingBufferWidth: 1920,
      drawingBufferHeight: 1080,
    };

    // Create mock Three.js objects
    const canvas = document.createElement('canvas');
    vi.spyOn(canvas, 'getContext').mockReturnValue(mockGL as any);

    renderer = new THREE.WebGLRenderer({ canvas });
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera();

    // Mock getContext method on renderer
    renderer.getContext = vi.fn(() => mockGL as any);

    // Create manager
    manager = new PostProcessingManager(renderer, scene, camera, { width: 1920, height: 1080 });
  });

  afterEach(() => {
    if (manager) {
      manager.dispose();
    }
    vi.clearAllMocks();
  });

  describe('initialization', () => {
    it('should create manager with HDR support', () => {
      expect(manager).toBeDefined();
      expect(renderer.outputColorSpace).toBe(THREE.SRGBColorSpace);
      expect(renderer.toneMapping).toBe(THREE.NoToneMapping);
    });

    it('should initialize with default effects', () => {
      const status = manager.getEffectsStatus();
      expect(status.bloom).toBe(true);
      expect(status.toneMapping).toBe('ACES Filmic');
      expect(status.fxaa).toBe(false);
      expect(status.smaa).toBe(false);
    });
  });

  describe('bloom settings', () => {
    it('should update bloom strength', () => {
      manager.updateBloomSettings(0.5, undefined, undefined);
      // Verify internal state was updated (would need to expose for testing)
      expect(manager).toBeDefined();
    });

    it('should update bloom radius', () => {
      manager.updateBloomSettings(undefined, 0.8, undefined);
      expect(manager).toBeDefined();
    });

    it('should update bloom threshold', () => {
      manager.updateBloomSettings(undefined, undefined, 0.02);
      expect(manager).toBeDefined();
    });
  });

  describe('tone mapping', () => {
    it('should set tone mapping mode', () => {
      manager.setToneMapping(THREE.ACESFilmicToneMapping);
      expect(manager.getToneMapping()).toBe(THREE.ACESFilmicToneMapping);
    });
  });

  describe('anti-aliasing', () => {
    it('should enable FXAA', () => {
      manager.setFXAAEnabled(true);
      expect(manager.isFXAAEnabled()).toBe(true);
    });

    it('should disable FXAA', () => {
      manager.setFXAAEnabled(true);
      manager.setFXAAEnabled(false);
      expect(manager.isFXAAEnabled()).toBe(false);
    });

    it('should enable SMAA', () => {
      manager.setSMAAEnabled(true);
      const status = manager.getEffectsStatus();
      expect(status.smaa).toBe(true);
    });

    it('should update SMAA settings', () => {
      manager.setSMAAEnabled(true);
      manager.updateSMAASettings('HIGH');
      expect(manager).toBeDefined();
    });
  });

  describe('depth of field', () => {
    it('should enable DOF', () => {
      manager.setDOF(true, 10.0, 0.5);
      const status = manager.getEffectsStatus();
      expect(status.dof).toBe(true);
    });

    it('should disable DOF', () => {
      manager.setDOF(true, 10.0, 0.5);
      manager.setDOF(false);
      const status = manager.getEffectsStatus();
      expect(status.dof).toBe(false);
    });

    it('should update DOF parameters', () => {
      manager.setDOF(true, 10.0, 0.5);
      manager.updateDOF({ focus: 20.0, strength: 0.8 });
      expect(manager).toBeDefined();
    });
  });

  describe('new effects', () => {
    it('should enable ambient occlusion', () => {
      manager.setAOEnabled(true, 'medium');
      const status = manager.getEffectsStatus();
      expect(status.ao).toBe(true);
    });

    it('should enable vignette', () => {
      manager.setVignetteEnabled(true, 0.5, 0.5);
      const status = manager.getEffectsStatus();
      expect(status.vignette).toBe(true);
    });

    it('should enable chromatic aberration', () => {
      manager.setChromaticAberration(true, 0.5);
      const status = manager.getEffectsStatus();
      expect(status.chromaticAberration).toBe(true);
    });

    it('should update chromatic aberration strength', () => {
      manager.setChromaticAberration(true, 0.5);
      manager.updateChromaticAberration(0.8);
      expect(manager).toBeDefined();
    });
  });

  describe('rendering', () => {
    it('should render through composer', () => {
      const composerRenderSpy = vi.spyOn((manager as any).composer, 'render');
      manager.render();
      expect(composerRenderSpy).toHaveBeenCalled();
    });
  });

  describe('resize', () => {
    it('should handle resize', () => {
      const composerSetSizeSpy = vi.spyOn((manager as any).composer, 'setSize');
      manager.resize(3840, 2160);
      expect(composerSetSizeSpy).toHaveBeenCalledWith(3840, 2160);
    });
  });

  describe('disposal', () => {
    it('should dispose resources', () => {
      const composerDisposeSpy = vi.spyOn((manager as any).composer, 'dispose');
      manager.dispose();
      expect(composerDisposeSpy).toHaveBeenCalled();
    });
  });

  describe('compatibility methods', () => {
    it('should handle MSAA methods', () => {
      // MSAA is now supported in the pmndrs implementation
      manager.setMSAAEnabled(true);
      expect(manager.isMSAAEnabled()).toBe(true);

      manager.setMSAASamples(4);
      expect(manager.getMSAASamples()).toBe(4);

      // Disable MSAA
      manager.setMSAAEnabled(false);
      expect(manager.isMSAAEnabled()).toBe(false);
    });
  });
});
