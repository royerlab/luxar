/**
 * Mock Index
 *
 * Central export point for all test mocks.
 * Import from here to get consistent mocking across all tests.
 *
 * Usage:
 *   import { installAllMocks } from './mocks';
 *   installAllMocks(); // In setup.ts or individual test files
 */

// Re-export all mocks
export * from './webgl.mock';
export * from './browser-apis.mock';
export * from './opfs.mock';

// Convenience imports
import { installWebGLMock } from './webgl.mock';
import { installAllBrowserMocks } from './browser-apis.mock';
import { installOPFSMock } from './opfs.mock';

/**
 * Install all mocks at once
 *
 * This is the recommended way to set up mocks in setup.ts
 */
export function installAllMocks(): void {
  installWebGLMock();
  installAllBrowserMocks();
  installOPFSMock();
}
