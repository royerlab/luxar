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
  drawArrays = vi.fn();
  drawElements = vi.fn();

  // State methods
  enable = vi.fn();
  disable = vi.fn();
  blendFunc = vi.fn();
  depthFunc = vi.fn();

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
