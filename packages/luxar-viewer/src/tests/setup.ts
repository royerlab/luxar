/**
 * Test setup file for Vitest
 *
 * Configures test environment with all necessary mocks:
 * - WebGL rendering context
 * - Browser APIs (matchMedia, ResizeObserver, etc.)
 * - OPFS (Origin Private File System)
 *
 * All mocks are now organized in separate files under ./mocks/
 */

import { installAllMocks } from './mocks';

// Install all mocks
installAllMocks();
