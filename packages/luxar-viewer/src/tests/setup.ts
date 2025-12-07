/**
 * Test setup file for Vitest
 * Configures WebGL mocks and global test utilities
 */

import { vi } from 'vitest';

// Mock WebGL context
class MockWebGLRenderingContext {
  canvas = {
    width: 800,
    height: 600,
    style: {},
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
  };

  // Extension support
  getExtension = vi.fn((name: string) => {
    // Support HDR extensions for hdr-detection.ts
    if (
      name === 'EXT_color_buffer_float' ||
      name === 'EXT_color_buffer_half_float' ||
      name === 'WEBGL_color_buffer_float'
    ) {
      return {}; // Return a truthy object to indicate support
    }
    if (name === 'WEBGL_debug_renderer_info') {
      return { UNMASKED_VENDOR_WEBGL: 0x9245, UNMASKED_RENDERER_WEBGL: 0x9246 };
    }
    return null;
  });

  getParameter = vi.fn((param: number) => {
    // VERSION constant in WebGL is 0x1F02
    if (param === 0x1f02) return 'WebGL 2.0 (OpenGL ES 3.0)';
    if (param === 0x1f00) return 'Mock Vendor'; // VENDOR
    if (param === 0x1f01) return 'WebGL 2.0'; // VERSION
    if (param === 0x9245) return 'Mock Vendor'; // UNMASKED_VENDOR_WEBGL
    if (param === 0x9246) return 'Mock Renderer'; // UNMASKED_RENDERER_WEBGL
    if (param === 35724) return 'WebGL GLSL ES 3.00'; // SHADING_LANGUAGE_VERSION
    return 1024;
  });

  getShaderPrecisionFormat = vi.fn(() => ({
    rangeMin: 127,
    rangeMax: 127,
    precision: 23,
  }));

  getContextAttributes = vi.fn(() => ({
    alpha: true,
    antialias: false,
    depth: true,
    premultipliedAlpha: true,
    preserveDrawingBuffer: false,
    stencil: true,
  }));

  // Program and shader methods
  createProgram = vi.fn(() => ({}));
  createShader = vi.fn(() => ({}));
  shaderSource = vi.fn();
  compileShader = vi.fn();
  attachShader = vi.fn();
  linkProgram = vi.fn();
  getProgramParameter = vi.fn(() => true);
  getShaderParameter = vi.fn(() => true);
  useProgram = vi.fn();
  deleteProgram = vi.fn();
  deleteShader = vi.fn();

  // Buffer methods
  createBuffer = vi.fn(() => ({}));
  bindBuffer = vi.fn();
  bufferData = vi.fn();
  deleteBuffer = vi.fn();

  // Texture methods
  createTexture = vi.fn(() => ({}));
  bindTexture = vi.fn();
  texImage2D = vi.fn();
  texParameteri = vi.fn();
  deleteTexture = vi.fn();

  // Framebuffer methods
  createFramebuffer = vi.fn(() => ({}));
  bindFramebuffer = vi.fn();
  createRenderbuffer = vi.fn(() => ({}));
  bindRenderbuffer = vi.fn();
  renderbufferStorage = vi.fn();
  framebufferTexture2D = vi.fn();
  framebufferRenderbuffer = vi.fn();
  checkFramebufferStatus = vi.fn(() => 0x8cd5); // GL_FRAMEBUFFER_COMPLETE

  // Uniform and attribute methods
  getUniformLocation = vi.fn(() => ({}));
  getAttribLocation = vi.fn(() => 0);
  uniform1f = vi.fn();
  uniform2f = vi.fn();
  uniform3f = vi.fn();
  uniform4f = vi.fn();
  uniform1i = vi.fn();
  uniformMatrix4fv = vi.fn();

  // Vertex attribute methods
  enableVertexAttribArray = vi.fn();
  disableVertexAttribArray = vi.fn();
  vertexAttribPointer = vi.fn();

  // Drawing methods
  clear = vi.fn();
  clearColor = vi.fn();
  viewport = vi.fn();
  scissor = vi.fn();
  drawArrays = vi.fn();
  drawElements = vi.fn();

  // State methods
  enable = vi.fn();
  disable = vi.fn();
  blendFunc = vi.fn();
  depthFunc = vi.fn();

  // Drawing buffer size
  drawingBufferWidth = 800;
  drawingBufferHeight = 600;

  // Constants
  VERTEX_SHADER = 35633;
  FRAGMENT_SHADER = 35632;
  ARRAY_BUFFER = 34962;
  ELEMENT_ARRAY_BUFFER = 34963;
  STATIC_DRAW = 35044;
  FLOAT = 5126;
  TEXTURE_2D = 3553;
  RGBA = 6408;
  UNSIGNED_BYTE = 5121;
  BLEND = 3042;
  DEPTH_TEST = 2929;
}

// Mock HTMLCanvasElement.getContext
HTMLCanvasElement.prototype.getContext = vi.fn((contextType: string) => {
  if (contextType === 'webgl' || contextType === 'webgl2') {
    return new MockWebGLRenderingContext() as any;
  }
  return null;
});

// Mock requestAnimationFrame
(globalThis as any).requestAnimationFrame = vi.fn((cb: any) => setTimeout(cb, 16));
(globalThis as any).cancelAnimationFrame = vi.fn((id: any) => clearTimeout(id));

// Mock performance.now()
(globalThis as any).performance = {
  now: vi.fn(() => Date.now()),
};

// Mock ResizeObserver
(globalThis as any).ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock IntersectionObserver
(globalThis as any).IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock window.matchMedia for HDR detection
(globalThis as any).window.matchMedia = vi.fn().mockImplementation((query: string) => ({
  matches: false, // Default to false (no HDR/P3 support in tests)
  media: query,
  onchange: null,
  addListener: vi.fn(),
  removeListener: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  dispatchEvent: vi.fn(),
}));

// Mock OPFS (Origin Private File System) API for cache tests
(globalThis as any).navigator.storage = {
  getDirectory: vi.fn().mockRejectedValue(new Error('OPFS not available in test environment')),
  estimate: vi.fn().mockResolvedValue({ quota: 0, usage: 0 }),
};
