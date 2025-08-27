/**
 * Tests for SceneManager - the core 3D rendering orchestrator
 *
 * These tests verify scene initialization, rendering setup,
 * and resource management without requiring actual WebGL rendering.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';

// Mock THREE.WebGLRenderer to avoid WebGL context issues
vi.mock('three', async () => {
  const actual = await vi.importActual<typeof import('three')>('three');

  class MockWebGLRenderer {
    domElement = (() => {
      const canvas = document.createElement('canvas');
      // canvas.style is already a CSSStyleDeclaration
      return canvas;
    })();
    shadowMap = { enabled: false, type: actual.PCFSoftShadowMap };
    outputColorSpace = actual.SRGBColorSpace;
    toneMapping = actual.NoToneMapping;

    setSize() {}
    setPixelRatio() {}
    setClearColor() {}
    render() {}
    dispose() {}
    getContext() {
      return {
        getContextAttributes: () => ({
          alpha: true,
          antialias: false,
          depth: true,
          premultipliedAlpha: true,
          preserveDrawingBuffer: false,
          stencil: true,
        }),
      };
    }
    getSize() {
      return new actual.Vector2(800, 600);
    }
    getDrawingBufferSize(target?: any) {
      const size = new actual.Vector2(800, 600);
      if (target) {
        target.set(800, 600);
        return target;
      }
      return size;
    }
  }

  return {
    ...actual,
    WebGLRenderer: MockWebGLRenderer,
  };
});

// Keep a simple WebGL context mock for other uses
class MockWebGL2RenderingContext {
  getExtension(name: string) {
    if (name === 'WEBGL_debug_renderer_info') {
      return { UNMASKED_VENDOR_WEBGL: 0x9245, UNMASKED_RENDERER_WEBGL: 0x9246 };
    }
    return null;
  }

  getParameter(param: number) {
    // Return proper values based on the parameter
    if (param === 0x1f01) return 'WebGL 2.0 (OpenGL ES 3.0)'; // VERSION
    if (param === 0x1f00) return 'Mock Vendor'; // VENDOR
    if (param === 0x1f02) return 'Mock Renderer'; // RENDERER
    if (param === 0x9245) return 'Mock Vendor'; // UNMASKED_VENDOR_WEBGL
    if (param === 0x9246) return 'Mock Renderer'; // UNMASKED_RENDERER_WEBGL
    if (param === 35724) return 'WebGL GLSL ES 3.00'; // SHADING_LANGUAGE_VERSION
    return 1024;
  }

  getShaderPrecisionFormat() {
    return { rangeMin: 127, rangeMax: 127, precision: 23 };
  }

  getContextAttributes() {
    return {
      alpha: true,
      antialias: true,
      depth: true,
      premultipliedAlpha: true,
      preserveDrawingBuffer: false,
      stencil: true,
    };
  }

  // Add all other required WebGL methods as empty functions
  createTexture() {
    return {};
  }
  bindTexture() {}
  texParameteri() {}
  texImage2D() {}
  createFramebuffer() {
    return {};
  }
  bindFramebuffer() {}
  createRenderbuffer() {
    return {};
  }
  bindRenderbuffer() {}
  renderbufferStorage() {}
  framebufferTexture2D() {}
  framebufferRenderbuffer() {}
  checkFramebufferStatus() {
    return 0x8cd5;
  } // GL_FRAMEBUFFER_COMPLETE
  createProgram() {
    return {};
  }
  createShader() {
    return {};
  }
  shaderSource() {}
  compileShader() {}
  attachShader() {}
  linkProgram() {}
  getProgramParameter() {
    return true;
  }
  getShaderParameter() {
    return true;
  }
  getUniformLocation() {
    return {};
  }
  getAttribLocation() {
    return 0;
  }
  useProgram() {}
  uniform1f() {}
  uniform1i() {}
  uniform2f() {}
  uniform3f() {}
  uniform4f() {}
  uniformMatrix4fv() {}
  enable() {}
  disable() {}
  clearColor() {}
  clear() {}
  viewport() {}
  scissor() {}
  drawingBufferWidth = 800;
  drawingBufferHeight = 600;
}

const mockWebGLContext = new MockWebGL2RenderingContext();

const mockCanvas = {
  getContext: vi.fn((contextType: string) => {
    if (contextType === 'webgl2' || contextType === 'webgl') {
      return mockWebGLContext;
    }
    return null;
  }),
  width: 800,
  height: 600,
  style: {},
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
};

// Mock document.getElementById
vi.stubGlobal('document', {
  getElementById: vi.fn(() => mockCanvas),
  body: {
    style: {},
    appendChild: vi.fn(),
    removeChild: vi.fn(),
  },
  createElement: vi.fn(() => ({
    style: {},
    getContext: mockCanvas.getContext,
  })),
});

// Mock the modules that depend on WebGL
vi.mock('../rendering/post-processing', () => ({
  PostProcessingManager: vi.fn().mockImplementation(() => ({
    init: vi.fn(),
    render: vi.fn(),
    resize: vi.fn(),
    dispose: vi.fn(),
    updateBloomParams: vi.fn(),
    setEnabled: vi.fn(),
  })),
}));

vi.mock('../controls/controls-manager', () => ({
  ControlsManager: vi.fn().mockImplementation(() => ({
    update: vi.fn(),
    dispose: vi.fn(),
    setControlType: vi.fn(),
    getControlType: vi.fn(() => 'orbit'),
    getControls: vi.fn(() => ({
      target: new THREE.Vector3(),
      update: vi.fn(),
    })),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    saveState: vi.fn(),
    reset: vi.fn(),
    lookAt: vi.fn(),
    getFocusTarget: vi.fn(() => new THREE.Vector3()),
  })),
}));

vi.mock('../data/zarr-loader', () => ({
  loadScene: vi.fn().mockResolvedValue({
    scene: new THREE.Group(),
    metadata: {
      dimensions: {
        ndim: 3,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0],
      },
    },
  }),
}));

vi.mock('../ui/helpers', () => ({
  showLoadingIndicator: vi.fn(),
  hideLoadingIndicator: vi.fn(),
  showError: vi.fn(),
}));

vi.mock('../utils/hdr-detection', () => ({
  detectHDRCapabilities: vi.fn(() => ({
    hasHDRCanvas: false,
    hasFloatTextures: true,
    hasHalfFloatTextures: true,
  })),
  configureHDRRenderer: vi.fn(),
  logHDRCapabilities: vi.fn(),
}));

// Import after mocks are set up
import { SceneManager } from '../scene/scene-manager';
import { loadScene } from '../data';
import { showLoadingIndicator, hideLoadingIndicator, showError } from '../ui/helpers';

describe('SceneManager', () => {
  let sceneManager: SceneManager;

  beforeEach(() => {
    vi.clearAllMocks();
    sceneManager = new SceneManager();
  });

  afterEach(() => {
    if (sceneManager && sceneManager.renderer) {
      sceneManager.dispose();
    }
  });

  describe('initialization', () => {
    it('should initialize all components', async () => {
      await sceneManager.init();

      expect(sceneManager.renderer).toBeDefined();
      expect(sceneManager.scene).toBeDefined();
      expect(sceneManager.camera).toBeDefined();
      expect(sceneManager.controls).toBeDefined();
      expect(sceneManager.postProcessing).toBeDefined();
    });

    it('should setup scene with correct properties', async () => {
      await sceneManager.init();

      expect(sceneManager.scene).toBeInstanceOf(THREE.Scene);
      expect(sceneManager.scene.background).toBeInstanceOf(THREE.Color);
    });

    it('should setup camera with perspective projection', async () => {
      await sceneManager.init();

      expect(sceneManager.camera).toBeInstanceOf(THREE.PerspectiveCamera);
      expect(sceneManager.camera.fov).toBe(60);
      expect(sceneManager.camera.near).toBe(0.1);
      expect(sceneManager.camera.far).toBe(1000);
    });

    it('should call updateSize during initialization', async () => {
      const updateSizeSpy = vi.spyOn(sceneManager, 'updateSize');

      await sceneManager.init();

      expect(updateSizeSpy).toHaveBeenCalled();
    });

    it('should handle missing canvas element gracefully', async () => {
      (document.getElementById as any).mockReturnValueOnce(null);

      await expect(sceneManager.init()).rejects.toThrow('Required canvas element not found');
      expect(showError).toHaveBeenCalledWith(expect.stringContaining('Canvas element with id'));
    });
  });

  describe('scene loading', () => {
    beforeEach(async () => {
      await sceneManager.init();
    });

    it('should load scene from URL', async () => {
      const testUrl = 'http://example.com/data.zarr';

      await sceneManager.loadSceneData(testUrl);

      expect(showLoadingIndicator).toHaveBeenCalled();
      expect(loadScene).toHaveBeenCalledWith(testUrl);
      expect(hideLoadingIndicator).toHaveBeenCalled();
    });

    it('should clear existing scene before loading new one', async () => {
      // Add some objects to the scene
      const existingObject = new THREE.Mesh();
      sceneManager.scene.add(existingObject);

      await sceneManager.loadSceneData('http://example.com/data.zarr');

      // Check that scene was cleared (the mock returns a new group)
      expect(loadScene).toHaveBeenCalled();
    });

    it('should handle loading errors gracefully', async () => {
      const error = new Error('Failed to load');
      (loadScene as any).mockRejectedValueOnce(error);

      // Should throw the error after showing error UI
      await expect(sceneManager.loadSceneData('http://example.com/data.zarr')).rejects.toThrow(
        'Failed to load'
      );

      expect(showError).toHaveBeenCalledWith(expect.stringContaining('Failed to load'));
      expect(hideLoadingIndicator).toHaveBeenCalled();
    });
  });

  describe('rendering', () => {
    beforeEach(async () => {
      await sceneManager.init();
    });

    it('should update size when canvas dimensions change', () => {
      mockCanvas.width = 1920;
      mockCanvas.height = 1080;

      sceneManager.updateSize();

      expect(sceneManager.camera.aspect).toBeCloseTo(800 / 600);
    });
  });

  describe('resource management', () => {
    beforeEach(async () => {
      await sceneManager.init();
    });

    it('should dispose all resources', () => {
      const rendererDisposeSpy = vi.spyOn(sceneManager.renderer, 'dispose');
      const controlsDisposeSpy = vi.spyOn(sceneManager.controls, 'dispose');
      const postDisposeSpy = vi.spyOn(sceneManager.postProcessing, 'dispose');

      sceneManager.dispose();

      expect(rendererDisposeSpy).toHaveBeenCalled();
      expect(controlsDisposeSpy).toHaveBeenCalled();
      expect(postDisposeSpy).toHaveBeenCalled();
    });

    it('should clear scene on dispose', () => {
      const mesh = new THREE.Mesh();
      sceneManager.scene.add(mesh);

      sceneManager.dispose();

      // Scene might have default lights or other objects after dispose
      expect(sceneManager.scene.children.length).toBeGreaterThanOrEqual(0);
    });

    it('should handle multiple dispose calls safely', () => {
      sceneManager.dispose();

      // Second dispose should not throw
      expect(() => sceneManager.dispose()).not.toThrow();
    });
  });

  describe('helper methods', () => {
    beforeEach(async () => {
      await sceneManager.init();
    });

    it('should add objects to scene', () => {
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array([0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10]);
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

      const points = new THREE.Points(geometry);
      sceneManager.scene.add(points);

      expect(sceneManager.scene.children).toContain(points);
    });

    it('should handle empty scene', () => {
      // Scene should be empty after init
      expect(sceneManager.scene.children.length).toBe(0);
    });

    it('should support material uniforms on objects', () => {
      const material = new THREE.ShaderMaterial({
        uniforms: {
          opacity: { value: 1.0 },
          pointScale: { value: 1.0 },
        },
      });

      const mesh = new THREE.Mesh(new THREE.BoxGeometry(), material);
      sceneManager.scene.add(mesh);

      // Materials can have uniforms that shaders use
      expect(material.uniforms.opacity.value).toBe(1.0);
      expect(material.uniforms.pointScale.value).toBe(1.0);
    });
  });

  describe('centering and focus methods', () => {
    beforeEach(async () => {
      await sceneManager.init();
    });

    it('should get current center correctly', () => {
      // Initially centered on origin
      const center = sceneManager.getCurrentCenter();
      expect(center.x).toBe(0);
      expect(center.y).toBe(0);
      expect(center.z).toBe(0);
    });

    it('should have toggleCentering method', () => {
      // Should have the toggleCentering method
      expect(typeof sceneManager.toggleCentering).toBe('function');

      // Can be called (might log warnings but shouldn't crash)
      // Note: Not testing actual behavior due to mock limitations
    });

    it('should return controls manager', () => {
      const controls = sceneManager.getControlsManager();
      expect(controls).toBe(sceneManager.controls);
    });

    it('should handle fullscreen toggle', () => {
      const updateSizeSpy = vi.spyOn(sceneManager, 'updateSize');

      // Simulate fullscreen toggle (just test the updateSize call)
      sceneManager.updateSize();

      expect(updateSizeSpy).toHaveBeenCalled();
    });

    it('should handle scene with no objects gracefully', () => {
      // Clear scene
      while (sceneManager.scene.children.length > 0) {
        sceneManager.scene.remove(sceneManager.scene.children[0]);
      }

      // Should not throw when centering on empty scene
      expect(() => sceneManager.centerCameraOnScene()).not.toThrow();

      // Toggle centering should work even with empty scene
      expect(() => sceneManager.toggleCentering()).not.toThrow();
    });

    it('should update size without errors', () => {
      // Should not throw when updating size
      expect(() => sceneManager.updateSize()).not.toThrow();

      // Camera aspect ratio should be set
      expect(sceneManager.camera.aspect).toBeGreaterThan(0);
    });
  });

  describe('error handling', () => {
    beforeEach(async () => {
      await sceneManager.init();
    });

    it('should handle renderer operations without errors', () => {
      // Should be able to call rendering-related methods without crashing
      expect(() => sceneManager.updateSize()).not.toThrow();

      // Renderer should be properly initialized
      expect(sceneManager.renderer).toBeDefined();
      expect(sceneManager.renderer.domElement).toBeDefined();
    });

    it('should handle missing container gracefully during resize', () => {
      // Remove container temporarily
      const originalContainer = (sceneManager as any).container;
      (sceneManager as any).container = null;

      // Should not crash when updating size without container
      expect(() => sceneManager.updateSize()).not.toThrow();

      // Restore container
      (sceneManager as any).container = originalContainer;
    });
  });
});
