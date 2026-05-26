/**
 * Unit tests for HDR detection utilities.
 *
 * The capability-detection function reaches out to `window.matchMedia` and a
 * WebGL context; we mock those at the boundary so the pure decision logic
 * (recommendedColorSpace, isHDRDisplay, getOptimalRenderTargetType) is
 * exercised directly without a real display or GPU.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import {
  detectDisplayCapabilities,
  isHDRDisplay,
  getOptimalRenderTargetType,
  configureHDRRenderer,
  logHDRCapabilities,
  type HDRCapabilities,
} from '../../../../utils/hdr/hdr-detection';

// ---------------------------------------------------------------------
// matchMedia mock — return true only for the queries the test names.
// ---------------------------------------------------------------------

function installMatchMedia(matching: Set<string> = new Set()) {
  const original = window.matchMedia;
  const stub = vi.fn().mockImplementation((q: string) => ({
    matches: matching.has(q),
    media: q,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
  Object.defineProperty(window, 'matchMedia', { configurable: true, value: stub });
  return () => {
    Object.defineProperty(window, 'matchMedia', { configurable: true, value: original });
  };
}

function makeCaps(over: Partial<HDRCapabilities> = {}): HDRCapabilities {
  return {
    p3Gamut: false,
    rec2020Gamut: false,
    hdr: false,
    deepColor: false,
    floatTextures: false,
    colorDepth: { red: 8, green: 8, blue: 8 },
    recommendedColorSpace: 'srgb',
    ...over,
  };
}

// ---------------------------------------------------------------------
// detectDisplayCapabilities — CSS-media-query branch only.
// Renderer-side GL probes are covered by renderer-capabilities.test.ts.
// ---------------------------------------------------------------------

describe('detectDisplayCapabilities', () => {
  it('returns sRGB defaults when no media query matches', () => {
    const restore = installMatchMedia(new Set());
    try {
      const caps = detectDisplayCapabilities();
      expect(caps).toEqual({
        p3Gamut: false,
        rec2020Gamut: false,
        hdr: false,
        deepColor: false,
        floatTextures: false,
        colorDepth: { red: 8, green: 8, blue: 8 },
        recommendedColorSpace: 'srgb',
      });
    } finally {
      restore();
    }
  });

  it('reports p3Gamut when (color-gamut: p3) matches', () => {
    const restore = installMatchMedia(new Set(['(color-gamut: p3)']));
    try {
      const caps = detectDisplayCapabilities();
      expect(caps.p3Gamut).toBe(true);
      expect(caps.rec2020Gamut).toBe(false);
      expect(caps.recommendedColorSpace).toBe('display-p3');
    } finally {
      restore();
    }
  });

  it('prefers Rec2020 when both Rec2020 gamut AND HDR are detected', () => {
    const restore = installMatchMedia(new Set(['(color-gamut: rec2020)', '(dynamic-range: high)']));
    try {
      const caps = detectDisplayCapabilities();
      expect(caps.rec2020Gamut).toBe(true);
      expect(caps.hdr).toBe(true);
      expect(caps.recommendedColorSpace).toBe('rec2020');
    } finally {
      restore();
    }
  });

  it('falls back to display-p3 when Rec2020 is matched but HDR is not', () => {
    // Rec2020 without HDR signal → not promoted, P3 (matched here too) wins.
    const restore = installMatchMedia(new Set(['(color-gamut: rec2020)', '(color-gamut: p3)']));
    try {
      const caps = detectDisplayCapabilities();
      expect(caps.recommendedColorSpace).toBe('display-p3');
    } finally {
      restore();
    }
  });

  it('reports deep color when EITHER (color: 48) or (color: 30) matches', () => {
    for (const q of ['(color: 48)', '(color: 30)']) {
      const restore = installMatchMedia(new Set([q]));
      try {
        const caps = detectDisplayCapabilities();
        expect(caps.deepColor).toBe(true);
      } finally {
        restore();
      }
    }
  });
});

// ---------------------------------------------------------------------
// isHDRDisplay — pure boolean predicate
// ---------------------------------------------------------------------

describe('isHDRDisplay', () => {
  it('returns true only when ALL four signals are present and a wide gamut is matched', () => {
    expect(
      isHDRDisplay(makeCaps({ hdr: true, deepColor: true, floatTextures: true, p3Gamut: true }))
    ).toBe(true);
    expect(
      isHDRDisplay(
        makeCaps({ hdr: true, deepColor: true, floatTextures: true, rec2020Gamut: true })
      )
    ).toBe(true);
  });

  // utils.md O8 / Phase E22: previously one `it` bundled 4 independent
  // "single missing requirement → not HDR" assertions in sequence. A
  // regression that silently treated `floatTextures=false` as HDR-ok
  // would surface as a generic "returns false when any single
  // requirement is missing" failure without naming the offending
  // capability. Parametrize via `it.each` over `{missing, caps}` so each
  // failure names the specific capability that should have gated.
  it.each<{ missing: string; caps: Parameters<typeof isHDRDisplay>[0] }>([
    {
      missing: 'hdr',
      caps: makeCaps({ hdr: false, deepColor: true, floatTextures: true, p3Gamut: true }),
    },
    {
      missing: 'deepColor',
      caps: makeCaps({ hdr: true, deepColor: false, floatTextures: true, p3Gamut: true }),
    },
    {
      missing: 'floatTextures',
      caps: makeCaps({ hdr: true, deepColor: true, floatTextures: false, p3Gamut: true }),
    },
    {
      missing: 'p3Gamut (wide gamut)',
      caps: makeCaps({ hdr: true, deepColor: true, floatTextures: true }),
    },
  ])('isHDRDisplay returns false when $missing is missing', ({ caps }) => {
    expect(isHDRDisplay(caps)).toBe(false);
  });

  it('returns false for the all-defaults capabilities', () => {
    expect(isHDRDisplay(makeCaps())).toBe(false);
  });
});

// ---------------------------------------------------------------------
// getOptimalRenderTargetType — pure switch
// ---------------------------------------------------------------------

describe('getOptimalRenderTargetType', () => {
  it('returns HalfFloatType when both floatTextures AND hdr are present', () => {
    expect(getOptimalRenderTargetType(makeCaps({ floatTextures: true, hdr: true }))).toBe(
      THREE.HalfFloatType
    );
  });

  it('returns HalfFloatType when only floatTextures is present (better SDR gradients)', () => {
    expect(getOptimalRenderTargetType(makeCaps({ floatTextures: true }))).toBe(THREE.HalfFloatType);
  });

  it('falls back to UnsignedByteType when floatTextures is missing', () => {
    expect(getOptimalRenderTargetType(makeCaps({ floatTextures: false }))).toBe(
      THREE.UnsignedByteType
    );
  });
});

// ---------------------------------------------------------------------
// configureHDRRenderer / logHDRCapabilities — logging only
// ---------------------------------------------------------------------

describe('configureHDRRenderer', () => {
  let logSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);
  });

  afterEach(() => {
    logSpy.mockRestore();
  });

  it('logs the Rec2020+HDR success message', () => {
    configureHDRRenderer({} as THREE.WebGLRenderer, makeCaps({ rec2020Gamut: true, hdr: true }));
    expect(logSpy).toHaveBeenCalled();
    const messages: string[] = logSpy.mock.calls.map((c: unknown[]) => c[0] as string);
    expect(messages.some((m) => m.includes('Rec2020 gamut'))).toBe(true);
  });

  it('logs the P3 success message when only P3 is detected', () => {
    configureHDRRenderer({} as THREE.WebGLRenderer, makeCaps({ p3Gamut: true }));
    const messages: string[] = logSpy.mock.calls.map((c: unknown[]) => c[0] as string);
    expect(messages.some((m) => m.includes('P3 gamut'))).toBe(true);
  });

  it('logs the standard sRGB message when nothing exotic is detected', () => {
    configureHDRRenderer({} as THREE.WebGLRenderer, makeCaps());
    const messages: string[] = logSpy.mock.calls.map((c: unknown[]) => c[0] as string);
    expect(messages.some((m) => m.includes('Standard sRGB display'))).toBe(true);
  });
});

describe('logHDRCapabilities', () => {
  let logSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);
  });

  afterEach(() => {
    logSpy.mockRestore();
  });

  it('emits one line per capability and the recommended-colorspace summary', () => {
    logHDRCapabilities(
      makeCaps({ p3Gamut: true, hdr: true, deepColor: true, floatTextures: true })
    );
    // 1 header + 5 capability lines + 1 color-buffer line + 1 recommendation line.
    expect(logSpy.mock.calls.length).toBeGreaterThanOrEqual(8);
    const all = logSpy.mock.calls.map((c: unknown[]) => c[0] as string).join('\n');
    expect(all).toContain('P3 Wide Gamut');
    expect(all).toContain('Rec2020 Gamut');
    expect(all).toContain('High Dynamic Range');
    expect(all).toContain('10-bit+ Deep Color');
    expect(all).toContain('Float Textures');
    expect(all).toContain('Color Buffer Depth');
    expect(all).toContain('Recommended Color Space');
  });
});
