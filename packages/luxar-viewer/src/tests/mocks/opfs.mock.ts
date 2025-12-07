/**
 * OPFS (Origin Private File System) Mock
 *
 * Mocks the browser's Origin Private File System API used by
 * the caching layer. Since OPFS is only available in browsers,
 * we mock it to fail gracefully in tests.
 */

import { vi } from 'vitest';

/**
 * Install OPFS mock
 *
 * Configures navigator.storage to reject with appropriate error
 * This allows cache tests to verify fallback behavior
 */
export function installOPFSMock(): void {
  (globalThis as any).navigator.storage = {
    getDirectory: vi
      .fn()
      .mockRejectedValue(new Error('OPFS not available in test environment')),
    estimate: vi.fn().mockResolvedValue({ quota: 0, usage: 0 }),
  };
}
