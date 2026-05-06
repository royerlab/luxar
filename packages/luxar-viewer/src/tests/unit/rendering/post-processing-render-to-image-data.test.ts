/**
 * Unit tests for PostProcessingManager.renderToImageData() and getResultBuffer()
 *
 * Tests LDR image capture via gl.readPixels, vertical flipping, and the
 * deterministic ping-pong buffer selection based on enabled pass count.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Polyfill ImageData for jsdom/node environments
if (typeof globalThis.ImageData === 'undefined') {
  (globalThis as any).ImageData = class ImageData {
    data: Uint8ClampedArray;
    width: number;
    height: number;
    constructor(data: Uint8ClampedArray, width: number, height?: number) {
      this.data = data;
      this.width = width;
      this.height = height ?? data.length / (4 * width);
    }
  };
}

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

/**
 * Creates a mock WebGL context with configurable framebuffer dimensions.
 * The readPixels mock can be customized per-test to fill specific pixel data.
 */
function createMockGL(width: number, height: number) {
  return {
    drawingBufferWidth: width,
    drawingBufferHeight: height,
    RGBA: 0x1908,
    UNSIGNED_BYTE: 0x1401,
    readPixels: vi.fn(),
  };
}

/**
 * Creates a mock renderer whose getContext() returns the given GL mock.
 */
function createMockRenderer(gl: ReturnType<typeof createMockGL>) {
  return {
    getContext: vi.fn().mockReturnValue(gl),
    readRenderTargetPixels: vi.fn(),
    outputColorSpace: 'srgb',
    toneMapping: 0,
    setSize: vi.fn(),
    getSize: vi.fn().mockReturnValue({ x: gl.drawingBufferWidth, y: gl.drawingBufferHeight }),
    getDrawingBufferSize: vi.fn(),
    domElement: { style: {} },
  };
}

/**
 * Creates a mock EffectComposer with configurable pass list and buffers.
 */
function createMockComposer(passes: Array<{ enabled: boolean; needsSwap?: boolean }> = []) {
  return {
    render: vi.fn(),
    passes,
    inputBuffer: { _tag: 'inputBuffer' },
    outputBuffer: { _tag: 'outputBuffer' },
    autoRenderToScreen: true,
    addPass: vi.fn(),
    removePass: vi.fn(),
    setSize: vi.fn(),
    dispose: vi.fn(),
  };
}

describe('PostProcessingManager.renderToImageData', () => {
  let mockGL: ReturnType<typeof createMockGL>;
  let mockRenderer: ReturnType<typeof createMockRenderer>;
  let mockComposer: ReturnType<typeof createMockComposer>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockGL = createMockGL(4, 4);
    mockRenderer = createMockRenderer(mockGL);
    mockComposer = createMockComposer();
  });

  /**
   * Helper: creates a PostProcessingManager instance via Object.create
   * (bypasses constructor) and injects mock internals.
   */
  async function callRenderToImageData() {
    const { PostProcessingManager } = await import('../../../rendering/post-processing/post-processing-manager');
    const instance = Object.create(PostProcessingManager.prototype);
    (instance as any).renderer = mockRenderer;
    (instance as any).composer = mockComposer;
    return instance.renderToImageData();
  }

  it('should return ImageData with correct width and height', async () => {
    const result = await callRenderToImageData();

    expect(result).toBeInstanceOf(ImageData);
    expect(result.width).toBe(4);
    expect(result.height).toBe(4);
  });

  it('should call composer.render() exactly once', async () => {
    await callRenderToImageData();

    expect(mockComposer.render).toHaveBeenCalledTimes(1);
  });

  it('should call gl.readPixels with correct parameters', async () => {
    await callRenderToImageData();

    expect(mockGL.readPixels).toHaveBeenCalledTimes(1);
    expect(mockGL.readPixels).toHaveBeenCalledWith(
      0,
      0,
      4,
      4,
      mockGL.RGBA,
      mockGL.UNSIGNED_BYTE,
      expect.any(Uint8Array)
    );
  });

  it('should flip the image vertically', async () => {
    // Set up a 2x2 image where each row has distinct RGBA values:
    //   WebGL row 0 (bottom): [255, 0, 0, 255] per pixel  (red)
    //   WebGL row 1 (top):    [0, 255, 0, 255] per pixel  (green)
    // After flip, ImageData row 0 should be green, row 1 should be red.
    mockGL = createMockGL(2, 2);
    mockRenderer = createMockRenderer(mockGL);

    mockGL.readPixels.mockImplementation(
      (
        _x: number,
        _y: number,
        _w: number,
        _h: number,
        _fmt: number,
        _type: number,
        buf: Uint8Array
      ) => {
        // Row 0 (bottom in WebGL): red
        buf[0] = 255;
        buf[1] = 0;
        buf[2] = 0;
        buf[3] = 255;
        buf[4] = 255;
        buf[5] = 0;
        buf[6] = 0;
        buf[7] = 255;
        // Row 1 (top in WebGL): green
        buf[8] = 0;
        buf[9] = 255;
        buf[10] = 0;
        buf[11] = 255;
        buf[12] = 0;
        buf[13] = 255;
        buf[14] = 0;
        buf[15] = 255;
      }
    );

    const result = await callRenderToImageData();
    const data = result.data;

    // After vertical flip: row 0 in ImageData should be green (was WebGL row 1)
    expect(data[0]).toBe(0);
    expect(data[1]).toBe(255);
    expect(data[2]).toBe(0);
    expect(data[3]).toBe(255);

    // Row 1 in ImageData should be red (was WebGL row 0)
    expect(data[8]).toBe(255);
    expect(data[9]).toBe(0);
    expect(data[10]).toBe(0);
    expect(data[11]).toBe(255);
  });

  it('should handle non-square dimensions', async () => {
    // 6 wide, 4 tall
    mockGL = createMockGL(6, 4);
    mockRenderer = createMockRenderer(mockGL);

    const result = await callRenderToImageData();

    expect(result.width).toBe(6);
    expect(result.height).toBe(4);
    expect(result.data.length).toBe(6 * 4 * 4);
  });
});

describe('PostProcessingManager.getResultBuffer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  /**
   * Helper: creates an instance with the given passes array and returns
   * the result of the private getResultBuffer() method.
   */
  async function callGetResultBuffer(passes: Array<{ enabled: boolean; needsSwap?: boolean }>) {
    const { PostProcessingManager } = await import('../../../rendering/post-processing/post-processing-manager');
    const instance = Object.create(PostProcessingManager.prototype);
    const composer = createMockComposer(passes);
    (instance as any).composer = composer;
    const result = (instance as any).getResultBuffer();
    return { result, composer };
  }

  it('should return inputBuffer when even number of swaps (2)', async () => {
    const { result, composer } = await callGetResultBuffer([
      { enabled: true, needsSwap: true },
      { enabled: true, needsSwap: true },
    ]);
    expect(result).toBe(composer.inputBuffer);
  });

  it('should return outputBuffer when odd number of swaps (3)', async () => {
    const { result, composer } = await callGetResultBuffer([
      { enabled: true, needsSwap: true },
      { enabled: true, needsSwap: true },
      { enabled: true, needsSwap: true },
    ]);
    expect(result).toBe(composer.outputBuffer);
  });

  it('should not count disabled passes', async () => {
    const { result, composer } = await callGetResultBuffer([
      { enabled: true, needsSwap: true },
      { enabled: false, needsSwap: true },
      { enabled: true, needsSwap: true },
    ]);
    expect(result).toBe(composer.inputBuffer);
  });

  it('should not count passes with needsSwap=false', async () => {
    // 3 enabled passes but only 2 have needsSwap=true → even → inputBuffer
    const { result, composer } = await callGetResultBuffer([
      { enabled: true, needsSwap: true },
      { enabled: true, needsSwap: false },
      { enabled: true, needsSwap: true },
    ]);
    expect(result).toBe(composer.inputBuffer);
  });

  it('should return inputBuffer when 0 swaps (0 is even)', async () => {
    const { result, composer } = await callGetResultBuffer([
      { enabled: false, needsSwap: true },
      { enabled: true, needsSwap: false },
    ]);
    expect(result).toBe(composer.inputBuffer);
  });
});
