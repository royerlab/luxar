/**
 * Tests for SceneManager - the core 3D rendering orchestrator
 *
 * These tests verify scene initialization, rendering setup,
 * and resource management without requiring actual WebGL rendering.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import {
  boundingBoxToSphere,
  calculateClippingPlanesFromSphere,
} from '../../../scene/scene-manager/clipping/bounds-math';

// Mock THREE.WebGLRenderer to avoid WebGL context issues
vi.mock('three', async () => {
  const actual = await vi.importActual<typeof import('three')>('three');

  class MockWebGLRenderer {
    // Positive flag that the real `THREE.WebGLRenderer` sets on
    // `this`. `renderer-capabilities.ts::isWebGLRenderer` reads it
    // to pick the WebGL2 vs WebGPU branch for the `api` field;
    // without this, the api detection mis-classifies the mock as WebGPU
    // and the WebGL-specific code (`setupContextLossHandling`, raw-GL
    // probes) silently skips.
    isWebGLRenderer = true;
    domElement = (() => {
      const canvas = document.createElement('canvas') as any;
      // Ensure canvas has required methods for OrbitControls
      canvas.addEventListener = canvas.addEventListener || vi.fn();
      canvas.removeEventListener = canvas.removeEventListener || vi.fn();
      canvas.getBoundingClientRect =
        canvas.getBoundingClientRect ||
        vi.fn(() => ({ left: 0, top: 0, width: 800, height: 600, right: 800, bottom: 600 }));
      return canvas;
    })();
    shadowMap = { enabled: false, type: actual.PCFShadowMap };
    outputColorSpace = actual.SRGBColorSpace;
    toneMapping = actual.NoToneMapping;

    setSize() {}
    setPixelRatio() {}
    getPixelRatio() {
      return 1;
    }
    setClearColor() {}
    clear() {}
    render() {}
    resetState() {}
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
        getExtension: (name: string) => {
          // Support HDR extensions
          if (
            name === 'EXT_color_buffer_float' ||
            name === 'EXT_color_buffer_half_float' ||
            name === 'WEBGL_color_buffer_float'
          ) {
            return {}; // Return truthy to indicate support
          }
          return null;
        },
        getParameter: (param: number) => {
          if (param === 0x1f02) return 'WebGL 2.0';
          if (param === 0x1f00) return 'Mock Vendor';
          return 1024;
        },
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

// Mock document.getElementById - returns null unless it's 'app' (the canvas id used by main.ts)
vi.stubGlobal('document', {
  getElementById: vi.fn((id) => {
    if (id === 'app') return mockCanvas; // 'app' is the canvas id resolved by main.ts
    return null; // All other IDs return null (loading-indicator, error-message, etc.)
  }),
  head: {
    appendChild: vi.fn(),
  },
  body: {
    style: {},
    appendChild: vi.fn(),
    removeChild: vi.fn(),
  },
  createElement: vi.fn(() => ({
    id: '',
    style: {},
    textContent: '',
    innerHTML: '',
    className: '',
    title: '',
    onclick: null,
    onmouseover: null,
    onmouseout: null,
    parentNode: null,
    getContext: mockCanvas.getContext,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    appendChild: vi.fn(), // Add appendChild for DOM elements
    remove: vi.fn(), // Add remove method for DOM elements
    contains: vi.fn(() => false),
    getBoundingClientRect: vi.fn(() => ({
      left: 0,
      top: 0,
      width: 800,
      height: 600,
      right: 800,
      bottom: 600,
    })),
    getRootNode: vi.fn(() => ({ addEventListener: vi.fn(), removeEventListener: vi.fn() })),
    ownerDocument: { addEventListener: vi.fn(), removeEventListener: vi.fn() },
  })),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
});

// Mock PostProcessingManager so the WebGL-dependent pipeline (bloom
// chain, mega-shader allocation, EXR encoder) doesn't try to run under
// jsdom. The stub mirrors only the methods the test path actually hits:
// resize + DPR scaling during init / resizer.resizeNow, render +
// continuous-animation query from the animation loop (if exercised),
// rebuildAfterContextRestore for the context-restored handler, and
// dispose for teardown. Setters (updateExposure / setBloomEnabled /
// etc.) are not used by this test suite but are added defensively so a
// future test calling them through `sceneManager.*` won't crash.
vi.mock('../../../rendering/post-processing/post-processing-manager', () => ({
  PostProcessingManager: vi.fn().mockImplementation(() => ({
    // Resize / DPR / camera plumbing (resize-orchestrator + camera-mode)
    resize: vi.fn(),
    setDPRScale: vi.fn(),
    setCamera: vi.fn(),
    // Render loop (animation-controller)
    render: vi.fn(),
    needsContinuousAnimation: vi.fn(() => false),
    // Lifecycle: dispose (scene-manager.dispose) +
    // rebuildAfterContextRestore (webgl-context-recovery).
    dispose: vi.fn(),
    rebuildAfterContextRestore: vi.fn(),
    // EOG sliders (scene-manager getters/setters at ~line 956)
    updateExposure: vi.fn(),
    updateGlobalOffset: vi.fn(),
    updateGlobalGamma: vi.fn(),
  })),
}));

// [scene.md/W10][P3] Fix mock path: previously '../controls/controls-manager'
// resolved relative to this test file (src/tests/unit/scene/), which does
// not exist. vi.mock resolves paths relative to the call site, so the
// correct path from this file to the production module is '../../../controls/controls-manager'.
// The dead mock was silently ignored — SceneManager was being constructed
// with the real ControlsManager, defeating test isolation.
vi.mock('../../../controls/controls-manager', () => ({
  ControlsManager: vi.fn().mockImplementation(() => ({
    update: vi.fn(),
    dispose: vi.fn(),
    setCamera: vi.fn(),
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
    setSceneScale: vi.fn(),
    setTarget: vi.fn(),
    setDistanceLimits: vi.fn(),
    setZoomLimits: vi.fn(),
    reinitialize: vi.fn(),
  })),
}));

// Mock the data module (scene-manager imports from '../data', not '../data/zarr-loader')
vi.mock('../../../data', () => ({
  loadScene: vi.fn().mockImplementation(async () => {
    const THREE = await import('three');
    const group = new THREE.Group();
    group.name = 'LuxarScene';
    group.userData = {
      sceneDimensions: {
        dimensions: [
          { name: 'x', unit: 'um', range: [0, 100], display: true },
          { name: 'y', unit: 'um', range: [0, 100], display: true },
          { name: 'z', unit: 'um', range: [0, 100], display: true },
        ],
      },
      maxRadius: 0.5,
    };
    return group;
  }),
  // Need to provide other exports from the module
  SceneLoader: vi.fn(),
  PointsSpatialIndexLoader: vi.fn(),
  updateView: vi.fn(),
  updateSceneForDimensions: vi.fn(),
  dispose: vi.fn(),
}));

// SceneManager now talks to UI feedback through `notifier`. Mock the
// notifier surface so tests can assert on showLoading/hideLoading/error.
// vi.mock hoists, so the spies must come from vi.hoisted to be defined
// when the factory runs.
const notifierMocks = vi.hoisted(() => ({
  showLoading: vi.fn(),
  hideLoading: vi.fn(),
  error: vi.fn(),
}));
vi.mock('../../../utils/cross-layer/notifier', () => ({
  notifier: {
    showLoading: notifierMocks.showLoading,
    hideLoading: notifierMocks.hideLoading,
    error: notifierMocks.error,
    toast: vi.fn(),
    showHelp: vi.fn(),
    hideHelp: vi.fn(),
    clearError: vi.fn(),
  },
}));
const mockShowLoading = notifierMocks.showLoading;
const mockHideLoading = notifierMocks.hideLoading;
const mockShowError = notifierMocks.error;

vi.mock('../../../utils/hdr/hdr-detection', () => ({
  detectDisplayCapabilities: vi.fn(() => ({
    p3Gamut: false,
    rec2020Gamut: false,
    hdr: false,
    deepColor: false,
    floatTextures: false,
    colorDepth: { red: 8, green: 8, blue: 8 },
    recommendedColorSpace: 'srgb',
  })),
  configureHDRRenderer: vi.fn(),
  logHDRCapabilities: vi.fn(),
}));

// Wrap renderer-setup with importActual so the real helpers pass through
// by default, but make createWebGPURenderer a vi.fn() that tests can
// override (e.g. to force the WebGPU→WebGL fallback path).
vi.mock('../../../scene/scene-manager/render-pipeline/renderer-setup', async () => {
  const actual = await vi.importActual<
    typeof import('../../../scene/scene-manager/render-pipeline/renderer-setup')
  >('../../../scene/scene-manager/render-pipeline/renderer-setup');
  return {
    ...actual,
    createWebGPURenderer: vi.fn(actual.createWebGPURenderer),
  };
});

// Import after mocks are set up
import { SceneManager } from '../../../scene/scene-manager';
import { loadScene as mockLoadScene } from '../../../data';
import { createWebGPURenderer as mockedCreateWebGPURenderer } from '../../../scene/scene-manager/render-pipeline/renderer-setup';
const mockShowLoadingIndicator = mockShowLoading;
const mockHideLoadingIndicator = mockHideLoading;

describe('SceneManager', () => {
  let sceneManager: SceneManager;

  beforeEach(() => {
    vi.clearAllMocks();
    // Pin the unit tests to the WebGL path. This is the production
    // default — these tests exercise scene composition / disposal /
    // position bounds and are renderer-agnostic in intent. Stubbing
    // the env flag keeps the test hermetic against any future change
    // that introduces a new opt-in to WebGPU, and the GLSL path is
    // the well-mocked one (`THREE.WebGLRenderer` is stubbed in this
    // file; the `three/webgpu` `WebGPURenderer` is not).
    vi.stubEnv('VITE_LUXAR_USE_LEGACY_WEBGL', '1');
    sceneManager = new SceneManager();
  });

  afterEach(() => {
    if (sceneManager && sceneManager.renderer) {
      sceneManager.dispose();
    }
    vi.unstubAllEnvs();
  });

  describe('initialization', () => {
    it('initializes all components with the correct concrete types', async () => {
      // scene.md W1 fix: previously asserted only .toBeDefined() on each
      // field. A regression that set `this.scene = new THREE.Group()`
      // instead of `new THREE.Scene()` would have survived. The next test
      // ('should setup scene with correct properties') does check
      // instanceof for scene; here we extend the same discipline to all
      // five fields.
      await sceneManager.init({ canvas: mockCanvas as any });

      expect(sceneManager.renderer).toBeInstanceOf(THREE.WebGLRenderer);
      expect(sceneManager.scene).toBeInstanceOf(THREE.Scene);
      expect(sceneManager.camera).toBeInstanceOf(THREE.Camera);
      // W1 (controls/postProcessing): `toBeDefined()` would survive a mutant
      // that assigned an empty object `{}`. ControlsManager and
      // PostProcessingManager aren't trivially instanceof-checkable here
      // without extra imports, so pin their characteristic methods instead.
      expect(sceneManager.controls).toBeDefined();
      expect(typeof sceneManager.controls.setControlType).toBe('function');
      expect(typeof sceneManager.controls.dispose).toBe('function');
      expect(sceneManager.postProcessing).toBeDefined();
      expect(typeof sceneManager.postProcessing.setCamera).toBe('function');
      expect(typeof sceneManager.postProcessing.dispose).toBe('function');
    });

    it('should setup scene with correct properties', async () => {
      await sceneManager.init({ canvas: mockCanvas as any });

      expect(sceneManager.scene).toBeInstanceOf(THREE.Scene);
      expect(sceneManager.scene.background).toBeInstanceOf(THREE.Color);
    });

    it('should setup camera with perspective projection', async () => {
      await sceneManager.init({ canvas: mockCanvas as any });

      expect(sceneManager.camera).toBeInstanceOf(THREE.PerspectiveCamera);
      expect((sceneManager.camera as THREE.PerspectiveCamera).fov).toBe(47);
      expect(sceneManager.camera.near).toBe(0.1);
      expect(sceneManager.camera.far).toBe(1000);
    });

    it('applies size directly during initialization (no debounce)', async () => {
      // The init path calls resizer.resizeNow() so the renderer is sized
      // before the first paint, bypassing the rAF coalescing path that
      // updateSize() uses.
      const resizer = (sceneManager as unknown as { resizer: { resizeNow: () => void } }).resizer;
      const resizeNowSpy = vi.spyOn(resizer, 'resizeNow');

      await sceneManager.init({ canvas: mockCanvas as any });

      expect(resizeNowSpy).toHaveBeenCalled();
      // First arg is width, second is height; verify against window dims.
      const [width, height] = resizeNowSpy.mock.calls[0] as unknown as [number, number];
      expect(width).toBe(window.innerWidth);
      expect(height).toBe(window.innerHeight);
    });
  });

  describe('scene loading', () => {
    beforeEach(async () => {
      await sceneManager.init({ canvas: mockCanvas as any });
    });

    it('should load scene from URL', async () => {
      const testUrl = 'http://example.com/data.zarr';

      await sceneManager.loadSceneData(testUrl);

      expect(mockShowLoadingIndicator).toHaveBeenCalled();
      expect(mockLoadScene).toHaveBeenCalledWith(testUrl, undefined);
      expect(mockHideLoadingIndicator).toHaveBeenCalled();
    });

    it('should clear existing scene before loading new one', async () => {
      // Add some objects to the scene
      const existingObject = new THREE.Mesh();
      sceneManager.scene.add(existingObject);
      expect(sceneManager.scene.children).toContain(existingObject);

      await sceneManager.loadSceneData('http://example.com/data.zarr');

      expect(mockLoadScene).toHaveBeenCalled();
      // W2: don't just assert loadScene fired — verify the pre-existing
      // (non-light, non-background) object was actually removed. A mutant that
      // deleted the clear step would leave it attached and survive the
      // call-only assertion.
      expect(sceneManager.scene.children).not.toContain(existingObject);
    });

    it('should handle loading errors gracefully', async () => {
      const error = new Error('Failed to load');
      // Use the direct mock reference - cast to any to use mock methods
      (mockLoadScene as any).mockRejectedValue(error);

      // Should throw the error after showing error UI
      await expect(sceneManager.loadSceneData('http://example.com/data.zarr')).rejects.toThrow(
        'Failed to load'
      );

      expect(mockShowError).toHaveBeenCalledWith(expect.stringContaining('Failed to load'));
      expect(mockHideLoadingIndicator).toHaveBeenCalled();
    });
  });

  describe('loadSceneData orchestration', () => {
    // Pin the 7-step call chain and the positionApplied conditional. These
    // tests complement the basic smoke tests in `scene loading` — they spy
    // on the private collaborators that loadSceneData orchestrates and
    // assert ordering / conditional branches that aren't visible through
    // the public-method assertions above.

    beforeEach(async () => {
      // Reset the loadScene mock back to its default success behaviour
      // (the error test in the previous block leaves it in rejected
      // state across cases otherwise).
      (mockLoadScene as any).mockImplementation(async () => {
        const T = await import('three');
        const group = new T.Group();
        group.name = 'LuxarScene';
        return group;
      });
      await sceneManager.init({ canvas: mockCanvas as any });
    });

    /**
     * Set up the seven spies the orchestration tests share. If
     * `opts.viewerConfig` is provided, it's attached to the root group
     * returned by `mockLoadScene` so that the `viewerConfig` read in
     * `loadSceneData` picks it up *before* `applyZarrViewerConfig` is
     * called (the production order is: capture viewerConfig, then call
     * the helper, then read camOverrides from the captured value).
     */
    function installSpies(opts: { positionApplied?: boolean; viewerConfig?: unknown } = {}) {
      const internals = sceneManager as unknown as {
        clearSceneContent(): void;
        resetControls(): void;
        applyZarrViewerConfig(root: THREE.Group): { positionApplied: boolean };
        autoFrameCamera(preserveTarget?: boolean): void;
      };
      if (opts.viewerConfig !== undefined) {
        (mockLoadScene as any).mockImplementationOnce(async () => {
          const T = await import('three');
          const group = new T.Group();
          group.name = 'LuxarScene';
          group.userData.viewerConfig = opts.viewerConfig;
          return group;
        });
      }
      const clearSceneContent = vi.spyOn(internals, 'clearSceneContent');
      const resetControls = vi.spyOn(internals, 'resetControls');
      const updateMaterialsForCurrentCamera = vi.spyOn(
        sceneManager,
        'updateMaterialsForCurrentCamera'
      );
      const applyZarrViewerConfig = vi
        .spyOn(internals, 'applyZarrViewerConfig')
        .mockImplementation(() => ({ positionApplied: opts.positionApplied ?? false }));
      const autoFrameCamera = vi.spyOn(internals, 'autoFrameCamera').mockImplementation(() => {});
      const autoAdjustClippingPlanes = vi
        .spyOn(sceneManager, 'autoAdjustClippingPlanes')
        .mockImplementation(() => ({ near: 0.1, far: 1000 }));
      return {
        clearSceneContent,
        resetControls,
        updateMaterialsForCurrentCamera,
        applyZarrViewerConfig,
        autoFrameCamera,
        autoAdjustClippingPlanes,
      };
    }

    it('invokes collaborators in the documented order', async () => {
      const spies = installSpies();

      await sceneManager.loadSceneData('http://example.com/data.zarr');

      // Expected sequence: clearSceneContent → resetControls →
      // updateMaterialsForCurrentCamera → loadScene → applyZarrViewerConfig →
      // autoFrameCamera (because positionApplied=false by default) →
      // autoAdjustClippingPlanes.
      const order = [
        spies.clearSceneContent.mock.invocationCallOrder[0],
        spies.resetControls.mock.invocationCallOrder[0],
        spies.updateMaterialsForCurrentCamera.mock.invocationCallOrder[0],
        (mockLoadScene as unknown as { mock: { invocationCallOrder: number[] } }).mock
          .invocationCallOrder[0],
        spies.applyZarrViewerConfig.mock.invocationCallOrder[0],
        spies.autoFrameCamera.mock.invocationCallOrder[0],
        spies.autoAdjustClippingPlanes.mock.invocationCallOrder[0],
      ];
      for (let i = 1; i < order.length; i++) {
        expect(order[i]).toBeGreaterThan(order[i - 1]);
      }
    });

    it('skips autoFrameCamera when applyZarrViewerConfig reports positionApplied=true', async () => {
      const spies = installSpies({
        positionApplied: true,
        viewerConfig: { camera: { position: [10, 20, 30] } },
      });

      await sceneManager.loadSceneData('http://example.com/data.zarr');

      // Author specified an explicit camera position — auto-framing must
      // not overwrite it. Clipping still runs (it follows camera position
      // regardless of source).
      expect(spies.autoFrameCamera).not.toHaveBeenCalled();
      expect(spies.autoAdjustClippingPlanes).toHaveBeenCalledTimes(1);
    });

    it('calls autoFrameCamera(true) when author set target/targetNode but no position', async () => {
      const spies = installSpies({
        positionApplied: false,
        viewerConfig: { camera: { target: [5, 5, 5] } },
      });

      await sceneManager.loadSceneData('http://example.com/data.zarr');

      expect(spies.autoFrameCamera).toHaveBeenCalledTimes(1);
      // preserveTarget=true → the helper preserves the author's look-at
      // point instead of overwriting it with bounding-box center.
      expect(spies.autoFrameCamera).toHaveBeenCalledWith(true);
    });

    it('establishes scene-scale distance limits BEFORE applying the author camera (regression: a far establishing-shot camera must not be clamped to the orbit default maxDistance)', async () => {
      // Give the scene non-trivial metadata bounds so the scale step fires.
      const internals = sceneManager as unknown as {
        getSceneBoundsFromMetadata(): unknown;
      };
      vi.spyOn(internals, 'getSceneBoundsFromMetadata').mockReturnValue({
        min: { x: -100000, y: -100, z: -100 },
        max: { x: 100000, y: 100, z: 100 },
      });
      const spies = installSpies({
        positionApplied: true,
        viewerConfig: { camera: { position: [-27622, 813, 5231], target: [103584, 0, 0] } },
      });
      const setSceneScale = (
        sceneManager as unknown as { controls: { setSceneScale: ReturnType<typeof vi.fn> } }
      ).controls.setSceneScale;

      await sceneManager.loadSceneData('http://example.com/data.zarr');

      // setSceneScale must run, and run BEFORE applyZarrViewerConfig. Otherwise
      // the orbit controls still hold their small default maxDistance when the
      // author camera is applied, so reinitialize()+update() clamps a wide
      // establishing-shot distance down to the default and snaps the camera in
      // toward the target (the embryo-line demo regression).
      expect(setSceneScale).toHaveBeenCalled();
      expect(setSceneScale.mock.invocationCallOrder[0]).toBeLessThan(
        spies.applyZarrViewerConfig.mock.invocationCallOrder[0]
      );
    });

    it('error path: skips autoFrame/autoAdjust + reports through notifier + rethrows', async () => {
      const spies = installSpies();
      const error = new Error('synthetic load failure');
      (mockLoadScene as any).mockRejectedValueOnce(error);

      await expect(sceneManager.loadSceneData('http://example.com/data.zarr')).rejects.toThrow(
        'synthetic load failure'
      );

      // Pre-load steps still ran...
      expect(spies.clearSceneContent).toHaveBeenCalledTimes(1);
      expect(spies.resetControls).toHaveBeenCalledTimes(1);
      // ...but the post-load orchestration was skipped after the throw.
      expect(spies.applyZarrViewerConfig).not.toHaveBeenCalled();
      expect(spies.autoFrameCamera).not.toHaveBeenCalled();
      expect(spies.autoAdjustClippingPlanes).not.toHaveBeenCalled();
      // notifier surface was invoked correctly.
      expect(mockHideLoadingIndicator).toHaveBeenCalled();
      expect(mockShowError).toHaveBeenCalledWith(expect.stringContaining('Failed to load'));
    });
  });

  describe('setControlType', () => {
    // Event dispatch is contractual: the 'camera-changed' event must
    // fire from the SceneManager call site when (and only when) the
    // projection mode swaps. The camera-mode helper's swap logic is
    // covered separately in camera-mode.test.ts.

    beforeEach(async () => {
      await sceneManager.init({ canvas: mockCanvas as any });
    });

    it("dispatches 'camera-changed' when switching to ortho swaps the camera", () => {
      const listener = vi.fn();
      sceneManager.addEventListener('camera-changed', listener);
      // Start in perspective (the default after init).
      expect(sceneManager.camera).toBeInstanceOf(THREE.PerspectiveCamera);

      sceneManager.setControlType('ortho');

      expect(sceneManager.camera).toBeInstanceOf(THREE.OrthographicCamera);
      expect(listener).toHaveBeenCalledTimes(1);
    });

    it("does NOT dispatch 'camera-changed' when control type changes without a projection swap", () => {
      const listener = vi.fn();
      sceneManager.addEventListener('camera-changed', listener);

      // 'fly' uses the existing perspective camera — no swap.
      sceneManager.setControlType('fly');

      expect(sceneManager.camera).toBeInstanceOf(THREE.PerspectiveCamera);
      expect(listener).not.toHaveBeenCalled();
    });
  });

  describe('toggleCentering', () => {
    // toggleCentering is a 2-state machine: bbox-center ↔ origin. Each
    // toggle invokes the matching centering helper (private) and flips
    // the internal isCenteredOnBoundingBox flag. The flag isn't directly
    // observable, but a second toggle's behaviour proves the state flip.

    beforeEach(async () => {
      await sceneManager.init({ canvas: mockCanvas as any });
    });

    it('first toggle from default (origin) calls centerCameraOnScene and flips to bbox', () => {
      const internals = sceneManager as unknown as {
        centerOnOrigin(): void;
        isCenteredOnBoundingBox: boolean;
      };
      const onOriginSpy = vi.spyOn(internals, 'centerOnOrigin').mockImplementation(() => {});
      const onSceneSpy = vi.spyOn(sceneManager, 'centerCameraOnScene').mockImplementation(() => {});
      // Default after construction is isCenteredOnBoundingBox=false.
      expect(internals.isCenteredOnBoundingBox).toBe(false);

      sceneManager.toggleCentering();

      expect(onSceneSpy).toHaveBeenCalledTimes(1);
      expect(onOriginSpy).not.toHaveBeenCalled();
      expect(internals.isCenteredOnBoundingBox).toBe(true);
    });

    it('second toggle from bbox calls centerOnOrigin and flips back to origin', () => {
      const internals = sceneManager as unknown as {
        centerOnOrigin(): void;
        isCenteredOnBoundingBox: boolean;
      };
      const onOriginSpy = vi.spyOn(internals, 'centerOnOrigin').mockImplementation(() => {});
      const onSceneSpy = vi.spyOn(sceneManager, 'centerCameraOnScene').mockImplementation(() => {});
      // Force the bbox-centered state directly.
      internals.isCenteredOnBoundingBox = true;

      sceneManager.toggleCentering();

      expect(onOriginSpy).toHaveBeenCalledTimes(1);
      expect(onSceneSpy).not.toHaveBeenCalled();
      expect(internals.isCenteredOnBoundingBox).toBe(false);
    });
  });

  describe('rendering', () => {
    beforeEach(async () => {
      await sceneManager.init({ canvas: mockCanvas as any });
    });

    it('updates the camera aspect ratio to match the new viewport dimensions', () => {
      // C1 (scene.md): the previous version mutated `mockCanvas.width/height`,
      // which updateSize() never reads (it resizes from window dimensions via
      // the orchestrator), and then asserted the aspect equalled 800/600 — a
      // value that coincidentally matched the 4:3 init aspect, so the test
      // passed vacuously. Drive the real resize path with an explicit, DISTINCT
      // (16:9) viewport and assert the aspect actually changes to match.
      const internals = sceneManager as unknown as {
        resizer: { resizeNow: (w: number, h: number, ctx: unknown) => void };
        makeResizeCtx: () => unknown;
      };
      const before = (sceneManager.camera as THREE.PerspectiveCamera).aspect;
      expect(before).toBeCloseTo(4 / 3, 5); // jsdom init viewport is 1024x768

      internals.resizer.resizeNow(1600, 900, internals.makeResizeCtx());

      const after = (sceneManager.camera as THREE.PerspectiveCamera).aspect;
      expect(after).toBeCloseTo(1600 / 900, 5); // 16:9, distinct from 4:3
      expect(after).not.toBeCloseTo(before, 2);
    });

    it('preserves manual/adaptive DPR override across window resize', () => {
      const setPixelRatioSpy = vi.spyOn(sceneManager.renderer, 'setPixelRatio');

      sceneManager.setAdaptivePixelRatio(0.5);
      setPixelRatioSpy.mockClear();

      // Drive the resize path via the orchestrator's synchronous entry point.
      const internals = sceneManager as unknown as {
        resizer: { resizeNow: (w: number, h: number, ctx: unknown) => void };
        makeResizeCtx: () => unknown;
      };
      internals.resizer.resizeNow(800, 600, internals.makeResizeCtx());

      expect(setPixelRatioSpy).toHaveBeenCalledWith(0.5);
    });
  });

  describe('resource management', () => {
    beforeEach(async () => {
      await sceneManager.init({ canvas: mockCanvas as any });
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

    it('should dispose geometry and material resources on dispose', () => {
      const geometry = new THREE.BufferGeometry();
      const material = new THREE.MeshBasicMaterial();
      const geometryDisposeSpy = vi.spyOn(geometry, 'dispose');
      const materialDisposeSpy = vi.spyOn(material, 'dispose');

      const mesh = new THREE.Mesh(geometry, material);
      sceneManager.scene.add(mesh);

      sceneManager.dispose();

      // dispose() traverses the scene graph and disposes all geometry/material GPU resources
      expect(geometryDisposeSpy).toHaveBeenCalled();
      expect(materialDisposeSpy).toHaveBeenCalled();
    });

    it('should handle multiple dispose calls safely', () => {
      sceneManager.dispose();

      // Second dispose should not throw
      expect(() => sceneManager.dispose()).not.toThrow();
    });

    it('dispose() clears the colormap texture cache', async () => {
      const { getColormapTexture } = await import('../../../rendering/colormap-textures');
      // Touch the cache so there's something to dispose.
      const beforeTex = getColormapTexture('viridis');
      sceneManager.dispose();
      // After dispose, next access returns a fresh instance (cache was cleared).
      const afterTex = getColormapTexture('viridis');
      expect(afterTex).toBeDefined();
      expect(afterTex).not.toBe(beforeTex);
    });

    it('should mark geometry attributes and materials dirty on context restore', async () => {
      const geometry = new THREE.BufferGeometry();
      const position = new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3);
      geometry.setAttribute('position', position);
      const material = new THREE.MeshBasicMaterial();
      const mesh = new THREE.Mesh(geometry, material);
      sceneManager.scene.add(mesh);
      const initialPositionVersion = position.version;
      const initialMaterialVersion = material.version;

      const restoredHandler = mockCanvas.addEventListener.mock.calls.find(
        (call) => call[0] === 'webglcontextrestored'
      )?.[1] as ((event: Event) => Promise<void>) | undefined;

      expect(restoredHandler).toBeDefined();
      await restoredHandler?.(new Event('webglcontextrestored'));

      expect(position.version).toBeGreaterThan(initialPositionVersion);
      expect(material.version).toBeGreaterThan(initialMaterialVersion);
    });

    it('preserves PostProcessingManager identity across context restore (CR-1)', async () => {
      // Cached references in PickingSystem / AnimationController /
      // RenderingControls must remain valid after context restore. The
      // rebuild path must NOT replace the manager instance — it should
      // rebuild GPU-bound resources in place via
      // `rebuildAfterContextRestore()`.
      const ppBefore = sceneManager.postProcessing;
      expect(ppBefore).toBeDefined();

      const restoredHandler = mockCanvas.addEventListener.mock.calls.find(
        (call) => call[0] === 'webglcontextrestored'
      )?.[1] as ((event: Event) => Promise<void>) | undefined;

      expect(restoredHandler).toBeDefined();
      await restoredHandler?.(new Event('webglcontextrestored'));

      const ppAfter = sceneManager.postProcessing;
      // Identity preserved: same object, not a replacement.
      expect(ppAfter).toBe(ppBefore);
    });
  });

  describe('helper methods', () => {
    beforeEach(async () => {
      await sceneManager.init({ canvas: mockCanvas as any });
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
      await sceneManager.init({ canvas: mockCanvas as any });
    });

    it('should get current center correctly', () => {
      // Initially centered on origin
      const center = sceneManager.getCurrentCenter();
      expect(center.x).toBe(0);
      expect(center.y).toBe(0);
      expect(center.z).toBe(0);
    });

    it('toggleCentering toggles the internal centering flag', () => {
      // [scene.md/W2][P2] Previously asserted only typeof === 'function'.
      // A method that did nothing would survive. Verify the call actually
      // flips observable state — getCurrentCenter exposes the result
      // through the controls focus target accessor.
      const initialControls = sceneManager.getControlsManager();
      expect(initialControls).toBe(sceneManager.controls);

      // Calling toggleCentering twice should be idempotent (back to start).
      expect(() => {
        sceneManager.toggleCentering();
        sceneManager.toggleCentering();
      }).not.toThrow();

      // After toggling, controls reference is still the same object
      // (toggleCentering must not swap the controls manager).
      expect(sceneManager.getControlsManager()).toBe(initialControls);
    });

    it('should return controls manager', () => {
      const controls = sceneManager.getControlsManager();
      expect(controls).toBe(sceneManager.controls);
    });

    // [scene.md/O2][P9] Renamed from 'should handle fullscreen toggle' —
    // the body never exercises fullscreen, it just verifies that
    // `updateSize()` is a real (spyable, callable) method on SceneManager.
    it('updateSize() is a callable method observable via vi.spyOn', () => {
      const updateSizeSpy = vi.spyOn(sceneManager, 'updateSize');

      sceneManager.updateSize();

      expect(updateSizeSpy).toHaveBeenCalled();
    });

    it('centering on an empty scene is a no-op that preserves the camera pose', () => {
      // Clear scene
      while (sceneManager.scene.children.length > 0) {
        sceneManager.scene.remove(sceneManager.scene.children[0]);
      }

      const posBefore = sceneManager.camera.position.clone();

      // W4: "doesn't throw" alone would survive a mutant that moved the camera
      // to NaN/origin on an empty scene. centerCameraOnScene finds no geometry,
      // so it must leave the camera pose untouched.
      expect(() => sceneManager.centerCameraOnScene()).not.toThrow();
      expect(sceneManager.camera.position.equals(posBefore)).toBe(true);

      // Toggle centering should also be a safe no-op with an empty scene.
      expect(() => sceneManager.toggleCentering()).not.toThrow();
      expect(Number.isFinite(sceneManager.camera.position.x)).toBe(true);
    });

    it('updateSize keeps camera aspect equal to viewport aspect', () => {
      // [scene.md/W3][P2] Previously asserted only !toThrow() and aspect>0.
      // A mutation that set aspect to a constant (e.g. 1) would survive.
      // Pin the actual relationship: camera aspect should equal width/height.
      sceneManager.updateSize();

      const cam = sceneManager.camera as THREE.PerspectiveCamera;
      const expectedAspect = window.innerWidth / window.innerHeight;
      // Allow a tiny tolerance for device-pixel-ratio rounding inside resize.
      expect(cam.aspect).toBeCloseTo(expectedAspect, 4);
      expect(Number.isFinite(cam.aspect)).toBe(true);
      expect(cam.aspect).toBeGreaterThan(0);
    });
  });

  describe('error handling', () => {
    beforeEach(async () => {
      await sceneManager.init({ canvas: mockCanvas as any });
    });

    it('renderer.domElement is a canvas-like element that is stable across updateSize()', () => {
      // [scene.md/W4][P2] Previously asserted only that renderer.domElement
      // is defined. A mock that set domElement to `{}` (truthy) would pass.
      // Pin: domElement must have `addEventListener` + `getBoundingClientRect`
      // (the contract that downstream controls / picking code relies on),
      // and the reference must not be swapped by updateSize().
      const beforeDom = sceneManager.renderer.domElement;
      expect(beforeDom).toBeTruthy();
      expect(typeof beforeDom.addEventListener).toBe('function');
      expect(typeof beforeDom.getBoundingClientRect).toBe('function');
      // getBoundingClientRect returns a real DOMRect-shaped object —
      // distinguishes from a plain {} stub.
      const rect = beforeDom.getBoundingClientRect();
      expect(typeof rect.width).toBe('number');
      expect(typeof rect.height).toBe('number');

      sceneManager.updateSize();

      // Reference identity must be preserved: a resize path that swapped
      // the canvas (e.g. recreating the WebGLRenderer) would break event
      // bindings already attached by ControlsManager.
      expect(sceneManager.renderer.domElement).toBe(beforeDom);
    });
  });

  describe('position bounds from metadata', () => {
    beforeEach(async () => {
      await sceneManager.init({ canvas: mockCanvas as any });
    });

    it('should return null when no position bounds in scene', () => {
      // Clear scene and add empty group
      while (sceneManager.scene.children.length > 0) {
        sceneManager.scene.remove(sceneManager.scene.children[0]);
      }
      const group = new THREE.Group();
      sceneManager.scene.add(group);

      // Use the private method via type assertion
      const bounds = (sceneManager as any).getSceneBoundsFromMetadata();
      expect(bounds).toBeNull();
    });

    it('should find position bounds from scene userData', () => {
      // Add group with position bounds
      const group = new THREE.Group();
      group.userData.positionBounds = {
        min: [0, 0, 0],
        max: [10, 20, 30],
      };
      sceneManager.scene.add(group);

      // Use the private method via type assertion
      const bounds = (sceneManager as any).getSceneBoundsFromMetadata();

      expect(bounds).not.toBeNull();
      expect(bounds.min.x).toBe(0);
      expect(bounds.min.y).toBe(0);
      expect(bounds.min.z).toBe(0);
      expect(bounds.max.x).toBe(10);
      expect(bounds.max.y).toBe(20);
      expect(bounds.max.z).toBe(30);
    });

    it('should use position bounds in autoAdjustClippingPlanes', () => {
      // Add group with position bounds
      const group = new THREE.Group();
      group.userData.positionBounds = {
        min: [-5, -5, -5],
        max: [5, 5, 5],
      };
      sceneManager.scene.add(group);

      // Call autoAdjustClippingPlanes
      const result = sceneManager.autoAdjustClippingPlanes();

      // Should return valid near/far values
      expect(result.near).toBeGreaterThan(0);
      expect(result.far).toBeGreaterThan(result.near);

      // Camera should be updated
      expect(sceneManager.camera.near).toBe(result.near);
      expect(sceneManager.camera.far).toBe(result.far);

      // G8: pin that near/far are DERIVED from the bounds + camera position
      // via the sphere formula — not arbitrary positive values. Compute the
      // expectation from the same [-5,5] bounds and the camera's actual pose.
      const cam = sceneManager.camera.position;
      const expected = calculateClippingPlanesFromSphere(
        boundingBoxToSphere({ min: { x: -5, y: -5, z: -5 }, max: { x: 5, y: 5, z: 5 } }),
        { x: cam.x, y: cam.y, z: cam.z }
      );
      expect(result.near).toBeCloseTo(expected.near, 4);
      expect(result.far).toBeCloseTo(expected.far, 4);
    });

    it('should fall back to geometry bounds when metadata not available', () => {
      // Clear scene and add geometry without metadata bounds
      while (sceneManager.scene.children.length > 0) {
        sceneManager.scene.remove(sceneManager.scene.children[0]);
      }

      // Add points with geometry but no metadata bounds
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array([0, 0, 0, 10, 10, 10]);
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const material = new THREE.PointsMaterial();
      const points = new THREE.Points(geometry, material);
      sceneManager.scene.add(points);

      // Call autoAdjustClippingPlanes - should use geometry bounds
      const result = sceneManager.autoAdjustClippingPlanes();

      // Should still return valid near/far values
      expect(result.near).toBeGreaterThan(0);
      expect(result.far).toBeGreaterThan(result.near);
    });

    it('should handle nD bounds by projecting to display dimensions', () => {
      // Add group with 5D position bounds
      const group = new THREE.Group();
      group.userData.positionBounds = {
        min: [0, 10, 20, 30, 40], // 5D
        max: [5, 15, 25, 35, 45],
      };
      sceneManager.scene.add(group);

      // Use the private method via type assertion
      // By default, display dims are [0, 1, 2] so X=dim0, Y=dim1, Z=dim2
      const bounds = (sceneManager as any).getSceneBoundsFromMetadata();

      expect(bounds).not.toBeNull();
      // Should project first 3 dimensions to X, Y, Z
      expect(bounds.min.x).toBe(0);
      expect(bounds.min.y).toBe(10);
      expect(bounds.min.z).toBe(20);
      expect(bounds.max.x).toBe(5);
      expect(bounds.max.y).toBe(15);
      expect(bounds.max.z).toBe(25);
    });
  });

  describe('WebGPU → WebGL fallback wiring', () => {
    // When createWebGPURenderer returns { fallback: true } (adapter below
    // the WebGPU spec minimum and no explicit override), SceneManager's
    // setupWebGPURenderer must drop down to setupWebGLRenderer and the
    // final this.renderer must be a THREE.WebGLRenderer.
    //
    // The createWebGPURenderer import is wrapped in a vi.fn() at the top
    // of this file (see vi.mock for renderer-setup) so individual tests
    // can override its resolution per case.

    it('setupWebGPURenderer delegates to setupWebGLRenderer when createWebGPURenderer reports fallback', async () => {
      // Force the fallback signal.
      vi.mocked(mockedCreateWebGPURenderer).mockResolvedValueOnce({ fallback: true });

      // Pass renderer:'webgpu' so selectBackend picks the WebGPU branch.
      // The mock immediately returns fallback, so setupWebGPURenderer
      // recurses into setupWebGLRenderer.
      await sceneManager.init({ canvas: mockCanvas as any, renderer: 'webgpu' });

      // After the fallback path, the renderer is the WebGL mock — the
      // same one the rest of the suite exercises.
      expect(sceneManager.renderer).toBeDefined();
      expect((sceneManager.renderer as { isWebGLRenderer?: boolean }).isWebGLRenderer).toBe(true);
      // And createWebGPURenderer was called exactly once before falling
      // through; setupWebGPURenderer doesn't re-attempt the WebGPU path.
      expect(mockedCreateWebGPURenderer).toHaveBeenCalledTimes(1);
    });
  });
});
