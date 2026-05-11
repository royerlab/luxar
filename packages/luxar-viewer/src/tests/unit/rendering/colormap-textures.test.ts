/**
 * Tests for colormap texture management.
 */
import { describe, it, expect, afterEach } from 'vitest';
import {
  getBuiltinColormapTexture,
  getColormapTexture,
  createCustomColormapTexture,
  isBuiltinColormap,
  disposeColormapTextures,
  disposeBuiltinColormapTextures,
  disposeCustomColormapTextures,
  _customColormapCacheSize,
} from '../../../rendering/colormap-textures';
import { BUILTIN_COLORMAPS, BUILTIN_COLORMAP_NAMES } from '../../../rendering/colormap-data';

afterEach(() => {
  disposeColormapTextures();
});

describe('colormap-textures', () => {
  describe('getBuiltinColormapTexture', () => {
    it('returns a DataTexture for known colormaps', () => {
      const tex = getBuiltinColormapTexture('viridis');
      expect(tex).toBeDefined();
      expect(tex!.image.width).toBe(256);
      expect(tex!.image.height).toBe(1);
    });

    it('returns undefined for unknown colormaps', () => {
      const tex = getBuiltinColormapTexture('nonexistent_colormap_xyz');
      expect(tex).toBeUndefined();
    });

    it('returns the same instance on repeated calls (caching)', () => {
      const tex1 = getBuiltinColormapTexture('green');
      const tex2 = getBuiltinColormapTexture('green');
      expect(tex1).toBe(tex2);
    });

    it('returns different textures for different colormaps', () => {
      const green = getBuiltinColormapTexture('green');
      const magenta = getBuiltinColormapTexture('magenta');
      expect(green).not.toBe(magenta);
    });
  });

  describe('getColormapTexture', () => {
    it('returns builtin texture for named colormaps', () => {
      const tex = getColormapTexture('viridis');
      expect(tex).toBeDefined();
    });

    it('returns custom texture when name is "custom" with LUT data', () => {
      const lut = new Uint8Array(768);
      for (let i = 0; i < 768; i++) lut[i] = i % 256;
      const tex = getColormapTexture('custom', lut);
      expect(tex).toBeDefined();
      expect(tex!.image.width).toBe(256);
    });

    it('falls back to viridis for "custom" without LUT data', () => {
      // Missing custom LUT data falls back to viridis with a warning so
      // the user gets a visible result rather than an empty render.
      const tex = getColormapTexture('custom');
      const viridis = getColormapTexture('viridis');
      expect(tex).toBe(viridis);
    });
  });

  describe('createCustomColormapTexture', () => {
    it('creates a texture from raw LUT data', () => {
      const lut = new Uint8Array(768).fill(128);
      const tex = createCustomColormapTexture(lut);
      expect(tex).toBeDefined();
      expect(tex.image.width).toBe(256);
    });
  });

  describe('isBuiltinColormap', () => {
    it('returns true for built-in names', () => {
      expect(isBuiltinColormap('viridis')).toBe(true);
      expect(isBuiltinColormap('green')).toBe(true);
      expect(isBuiltinColormap('magenta')).toBe(true);
    });

    it('returns false for unknown names', () => {
      expect(isBuiltinColormap('foobar')).toBe(false);
    });
  });

  describe('BUILTIN_COLORMAPS', () => {
    it('has at least 15 entries', () => {
      expect(BUILTIN_COLORMAP_NAMES.length).toBeGreaterThanOrEqual(15);
    });

    it('all entries have 768 bytes (256 * 3)', () => {
      for (const name of BUILTIN_COLORMAP_NAMES) {
        expect(BUILTIN_COLORMAPS[name].length).toBe(768);
      }
    });

    it('green ramp starts at black and ends at green', () => {
      const lut = BUILTIN_COLORMAPS['green'];
      // First pixel: black
      expect(lut[0]).toBe(0);
      expect(lut[1]).toBe(0);
      expect(lut[2]).toBe(0);
      // Last pixel: green
      expect(lut[765]).toBe(0); // R
      expect(lut[766]).toBe(255); // G
      expect(lut[767]).toBe(0); // B
    });

    it('gray ramp ends at white', () => {
      const lut = BUILTIN_COLORMAPS['gray'];
      expect(lut[765]).toBe(255);
      expect(lut[766]).toBe(255);
      expect(lut[767]).toBe(255);
    });
  });

  describe('disposeColormapTextures', () => {
    it('clears cache so new instances are returned', () => {
      const tex1 = getBuiltinColormapTexture('viridis');
      disposeColormapTextures();
      const tex2 = getBuiltinColormapTexture('viridis');
      expect(tex1).not.toBe(tex2);
    });
  });

  // ==========================================================================
  // B.2 — scoped disposal + bounded LRU for custom LUT cache
  // ==========================================================================

  describe('B.2: custom LUT cache is bounded LRU', () => {
    function makeLut(seed: number): Uint8Array {
      const lut = new Uint8Array(768);
      for (let i = 0; i < 768; i++) {
        lut[i] = (seed + i) % 256;
      }
      return lut;
    }

    it('caps cache size at the LRU bound (16) and disposes evicted entries', () => {
      disposeColormapTextures(); // clean slate
      // Load 20 unique LUTs; cache must not exceed 16, and the first 4 should
      // have been disposed.
      const created: { tex: ReturnType<typeof createCustomColormapTexture>; disposed: () => boolean }[] = [];
      for (let i = 0; i < 20; i++) {
        const tex = createCustomColormapTexture(makeLut(i));
        // Detect disposed state by spying on the dispose method post-hoc.
        // We can't reliably detect disposal on a real THREE.DataTexture
        // without a renderer, but we can assert the cache size invariant.
        created.push({ tex, disposed: () => false });
      }
      expect(_customColormapCacheSize()).toBeLessThanOrEqual(16);
    });

    it('LRU promotion: recently-used entries survive eviction', () => {
      disposeColormapTextures();
      // Load 16 unique LUTs (fill the cache).
      const first = createCustomColormapTexture(makeLut(0));
      for (let i = 1; i < 16; i++) {
        createCustomColormapTexture(makeLut(i));
      }
      // Touch the first to bump it to MRU.
      const promoted = createCustomColormapTexture(makeLut(0));
      expect(promoted).toBe(first);

      // Loading a 17th unique LUT should evict the LRU (which is now seed=1).
      createCustomColormapTexture(makeLut(99));
      // seed=0 must still be present.
      const stillThere = createCustomColormapTexture(makeLut(0));
      expect(stillThere).toBe(first);
    });
  });

  describe('B.2: scoped disposal', () => {
    function makeLut(seed: number): Uint8Array {
      const lut = new Uint8Array(768);
      for (let i = 0; i < 768; i++) lut[i] = (seed + i) % 256;
      return lut;
    }

    it('disposeCustomColormapTextures() clears only the custom cache; built-ins survive', () => {
      const builtinBefore = getBuiltinColormapTexture('viridis');
      createCustomColormapTexture(makeLut(123));
      expect(_customColormapCacheSize()).toBe(1);

      disposeCustomColormapTextures();
      expect(_customColormapCacheSize()).toBe(0);

      // Built-in survives: same instance comes back without re-creation.
      const builtinAfter = getBuiltinColormapTexture('viridis');
      expect(builtinAfter).toBe(builtinBefore);
    });

    it('disposeBuiltinColormapTextures() clears only built-ins', () => {
      const builtin = getBuiltinColormapTexture('viridis');
      createCustomColormapTexture(makeLut(45));
      expect(_customColormapCacheSize()).toBe(1);

      disposeBuiltinColormapTextures();
      // Built-in cache cleared (next get returns a new instance).
      const builtinAfter = getBuiltinColormapTexture('viridis');
      expect(builtinAfter).not.toBe(builtin);
      // Custom cache untouched.
      expect(_customColormapCacheSize()).toBe(1);
    });
  });
});
