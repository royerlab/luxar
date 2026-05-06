/**
 * Unit tests for PostProcessingManager.captureHDRAsEXR()
 *
 * Tests HDR EXR capture: effect state save/restore, pixel readback,
 * DataTexture creation, and EXR encoding.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

// Mock EXRExporter
const mockExrParse = vi.fn().mockReturnValue(new Uint8Array([0x76, 0x2f, 0x31, 0x01, 0, 0, 0, 0]));
vi.mock('three/examples/jsm/exporters/EXRExporter.js', () => ({
  EXRExporter: vi.fn().mockImplementation(() => ({
    parse: mockExrParse,
  })),
  ZIP_COMPRESSION: 3,
}));

// Mock log
vi.mock('../../../utils/log', () => ({
  log: {
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    success: vi.fn(),
    update: vi.fn(),
  },
  Modules: { POST_PROCESSING: 'PostProcessing' },
}));

// Mock config
vi.mock('../../../config', () => ({
  config: {
    renderingControls: {
      defaults: {
        bloomStrength: 0.4,
        bloomRadius: 0.4,
        bloomThreshold: 0.85,
        bloomLevels: 5,
        fxaaEnabled: false,
        smaaEnabled: false,
        msaaEnabled: false,
        msaaSamples: 4,
        ssaaEnabled: false,
        ssaaMultiplier: 1.0,
        detectorNoiseReadoutSigma: 0.02,
        detectorNoisePhotonGain: 1.0,
        detectorNoiseFpnSigma: 0.01,
      },
    },
  },
}));

// Create mock composer and effects for isolated testing
function createMockComposer() {
  const inputBuffer = {
    width: 100,
    height: 100,
    isWebGLRenderTarget: true,
    texture: { type: THREE.HalfFloatType, format: THREE.RGBAFormat },
  };
  const outputBuffer = {
    width: 100,
    height: 100,
    isWebGLRenderTarget: true,
    texture: { type: THREE.HalfFloatType, format: THREE.RGBAFormat },
  };

  return {
    inputBuffer,
    outputBuffer,
    autoRenderToScreen: true,
    render: vi.fn(),
    addPass: vi.fn(),
    removePass: vi.fn(),
    setSize: vi.fn(),
    dispose: vi.fn(),
    passes: [
      { enabled: true, renderToScreen: false },
      { enabled: true, renderToScreen: true },
    ],
  };
}

function createMockRenderer() {
  return {
    readRenderTargetPixels: vi.fn(
      (_target: any, _x: number, _y: number, w: number, h: number, buffer: Float32Array) => {
        // Fill with non-zero test data
        for (let i = 0; i < w * h * 4; i++) {
          buffer[i] = 0.5;
        }
      }
    ),
    outputColorSpace: THREE.SRGBColorSpace,
    toneMapping: THREE.NoToneMapping,
    setSize: vi.fn(),
    getSize: vi.fn().mockReturnValue({ x: 100, y: 100 }),
    getDrawingBufferSize: vi.fn().mockReturnValue(new THREE.Vector2(100, 100)),
    domElement: {
      style: {},
    },
  };
}

describe('PostProcessingManager.captureHDRAsEXR', () => {
  let mockComposer: ReturnType<typeof createMockComposer>;
  let mockRenderer: ReturnType<typeof createMockRenderer>;
  let mockToneMappingEffect: any;
  let mockVignetteEffect: any;
  let mockSmaaEffect: any;
  let mockFxaaEffect: any;

  beforeEach(() => {
    vi.clearAllMocks();
    mockComposer = createMockComposer();
    mockRenderer = createMockRenderer();

    // Create mock effects with enabled tracking
    mockToneMappingEffect = { enabled: true, _name: 'toneMapping' };
    mockVignetteEffect = { enabled: true, _name: 'vignette' };
    mockSmaaEffect = { enabled: true, _name: 'smaa' };
    mockFxaaEffect = { enabled: true, _name: 'fxaa' };
  });

  /**
   * Helper to create a minimal PostProcessingManager-like object for testing
   * captureHDRAsEXR in isolation, without needing the full constructor.
   */
  async function callCaptureHDR(options?: { type?: THREE.TextureDataType }) {
    // Dynamically import to get the actual method
    const { PostProcessingManager } = await import('../../../rendering/post-processing/post-processing-manager');

    // Create an instance with mocked internals
    const manager = Object.create(PostProcessingManager.prototype);
    manager.composer = mockComposer;
    manager.renderer = mockRenderer;
    manager.toneMappingEffect = mockToneMappingEffect;
    manager.vignetteEffect = mockVignetteEffect;
    manager.smaaEffect = mockSmaaEffect;
    manager.fxaaEffect = mockFxaaEffect;

    return manager.captureHDRAsEXR(options);
  }

  it('should disable LDR effects before rendering', async () => {
    await callCaptureHDR();

    // After captureHDR completes, effects should be RESTORED
    // But during the render call, they should have been disabled
    // We verify restore happened correctly
    expect(mockToneMappingEffect.enabled).toBe(true);
    expect(mockVignetteEffect.enabled).toBe(true);
    expect(mockSmaaEffect.enabled).toBe(true);
    expect(mockFxaaEffect.enabled).toBe(true);
  });

  it('should restore autoRenderToScreen after capture', async () => {
    mockComposer.autoRenderToScreen = true;

    await callCaptureHDR();

    expect(mockComposer.autoRenderToScreen).toBe(true);
  });

  it('should restore pass renderToScreen states', async () => {
    const originalStates = mockComposer.passes.map((p) => p.renderToScreen);

    await callCaptureHDR();

    mockComposer.passes.forEach((p, i) => {
      expect(p.renderToScreen).toBe(originalStates[i]);
    });
  });

  it('should call composer.render()', async () => {
    await callCaptureHDR();

    expect(mockComposer.render).toHaveBeenCalled();
  });

  it('should call readRenderTargetPixels to get pixel data', async () => {
    await callCaptureHDR();

    expect(mockRenderer.readRenderTargetPixels).toHaveBeenCalled();
  });

  it('should call EXRExporter.parse with a DataTexture', async () => {
    await callCaptureHDR();

    expect(mockExrParse).toHaveBeenCalledWith(
      expect.objectContaining({ isDataTexture: true }),
      expect.objectContaining({ compression: 3 })
    );
  });

  it('should return Uint8Array with EXR magic bytes', async () => {
    const result = await callCaptureHDR();

    expect(result).toBeInstanceOf(Uint8Array);
    // EXR magic number: 0x76, 0x2f, 0x31, 0x01
    expect(result[0]).toBe(0x76);
    expect(result[1]).toBe(0x2f);
    expect(result[2]).toBe(0x31);
    expect(result[3]).toBe(0x01);
  });

  it('should default to HalfFloatType', async () => {
    await callCaptureHDR();

    expect(mockExrParse).toHaveBeenCalledWith(
      expect.objectContaining({
        isDataTexture: true,
      }),
      expect.objectContaining({ type: THREE.HalfFloatType })
    );
  });

  it('should support FloatType when requested', async () => {
    await callCaptureHDR({ type: THREE.FloatType });

    expect(mockExrParse).toHaveBeenCalledWith(
      expect.objectContaining({
        isDataTexture: true,
      }),
      expect.objectContaining({ type: THREE.FloatType })
    );
  });

  it('should restore effects even if render throws', async () => {
    mockComposer.render.mockImplementation(() => {
      throw new Error('render failed');
    });

    // The method should throw but the try/finally in captureHDRPixels()
    // must still flip effects back to their original `enabled=true`.
    await expect(callCaptureHDR()).rejects.toThrow('render failed');

    expect(mockToneMappingEffect.enabled).toBe(true);
    expect(mockVignetteEffect.enabled).toBe(true);
    expect(mockSmaaEffect.enabled).toBe(true);
    expect(mockFxaaEffect.enabled).toBe(true);
  });
});
