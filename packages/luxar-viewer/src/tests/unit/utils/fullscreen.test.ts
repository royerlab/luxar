/**
 * Unit tests for utils/fullscreen.ts.
 *
 * Regression: the branch enables webkit-prefixed fullscreen entry (Safari
 * <16.4), so every "are we fullscreen?" consumer must recognise
 * `webkitFullscreenElement`, not just the standard `fullscreenElement`.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { isDocumentFullscreen } from '../../../utils/fullscreen';

function setFullscreen(standard: boolean, webkit: boolean): void {
  Object.defineProperty(document, 'fullscreenElement', {
    configurable: true,
    get: () => (standard ? document.body : null),
  });
  Object.defineProperty(document, 'webkitFullscreenElement', {
    configurable: true,
    get: () => (webkit ? document.body : null),
  });
}

describe('isDocumentFullscreen', () => {
  afterEach(() => setFullscreen(false, false));

  it('is false when no element is fullscreen', () => {
    setFullscreen(false, false);
    expect(isDocumentFullscreen()).toBe(false);
  });

  it('is true via the standard document.fullscreenElement', () => {
    setFullscreen(true, false);
    expect(isDocumentFullscreen()).toBe(true);
  });

  it('is true via webkitFullscreenElement (Safari <16.4) — the whole point', () => {
    setFullscreen(false, true);
    expect(isDocumentFullscreen()).toBe(true);
  });
});
