/**
 * Property tests for the snake_case↔camelCase rendering-config mapping
 * (the renderingSettingsToZarr ↔ extractRenderingOverrides pair).
 *
 * The audit (delme/test-audit-luxar-viewer.src/config.md, G2) called this
 * the highest-leverage missing test in the entire config subpackage:
 *
 *   - 43 entries in RENDERING_SETTINGS_MAP, only ~14 directly verified by
 *     example tests today.
 *   - A mutation that swapped two map values would not be caught by examples
 *     that target only 14 specific keys.
 *
 * Property tests close that gap by enumerating the entire map.
 */
import { describe, expect, test, vi, beforeEach, afterEach } from 'vitest';
import * as fc from 'fast-check';
import {
  RENDERING_SETTINGS_MAP,
  REVERSE_SETTINGS_MAP,
  _warnedUnknownRenderingKeys,
  extractRenderingOverrides,
  renderingSettingsToZarr,
} from '../../../config/zarr-bridge/viewer-config-utils';
import type { RenderingSettings } from '../../../config/types';
import type { ZarrViewerConfig } from '../../../types/zarr';

describe('RENDERING_SETTINGS_MAP — bijection invariants', () => {
  test('every snake_case key maps to a camelCase key whose reverse maps back', () => {
    // For every snake_case key in the forward map, REVERSE_SETTINGS_MAP must
    // map its value back to the same snake_case key. A mutation that broke
    // any single entry (e.g. swapping bloomEnabled→bloomThreshold) would fail.
    for (const [snakeKey, camelKey] of Object.entries(RENDERING_SETTINGS_MAP)) {
      expect(REVERSE_SETTINGS_MAP[camelKey]).toBe(snakeKey);
    }
  });

  test('every camelCase key in REVERSE_SETTINGS_MAP is covered by RENDERING_SETTINGS_MAP', () => {
    // No orphan entries in the reverse map.
    for (const [camelKey, snakeKey] of Object.entries(REVERSE_SETTINGS_MAP)) {
      expect(RENDERING_SETTINGS_MAP[snakeKey]).toBe(camelKey);
    }
  });

  test('forward and reverse maps have equal size (no duplicate values or keys)', () => {
    expect(Object.keys(RENDERING_SETTINGS_MAP).length).toBe(
      Object.keys(REVERSE_SETTINGS_MAP).length
    );
  });
});

describe('renderingSettingsToZarr ↔ extractRenderingOverrides — roundtrip invariants', () => {
  // Build an arbitrary that picks an arbitrary subset of camelCase keys and
  // assigns each a primitive value of an appropriate-ish type.
  const camelKeys = Object.values(RENDERING_SETTINGS_MAP);

  const settingsArb = fc.dictionary(
    fc.constantFrom(...camelKeys),
    fc.oneof(
      fc.float({
        min: Math.fround(-1000),
        max: Math.fround(1000),
        noNaN: true,
        noDefaultInfinity: true,
      }),
      fc.boolean(),
      fc.constantFrom('low', 'medium', 'high', 'sRGB', 'webgl2')
    ),
    { minKeys: 1, maxKeys: Math.min(20, camelKeys.length) }
  ) as fc.Arbitrary<Partial<RenderingSettings>>;

  test('roundtrip: extractRenderingOverrides(renderingSettingsToZarr(s)) === s (for mappable keys)', () => {
    fc.assert(
      fc.property(settingsArb, (settings) => {
        const zarr = renderingSettingsToZarr(settings);
        const roundtripped = extractRenderingOverrides(zarr as ZarrViewerConfig);

        // For every key in `settings` that is in the map, the roundtrip
        // should preserve the value (modulo whatever the renderer does at the
        // edges — but the mapping itself is identity).
        for (const [camelKey, value] of Object.entries(settings)) {
          if (camelKey in REVERSE_SETTINGS_MAP) {
            expect((roundtripped as Record<string, unknown>)[camelKey]).toBe(value);
          }
        }
      }),
      { numRuns: 200 }
    );
  });

  test('undefined fields in settings are filtered out by renderingSettingsToZarr', () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(...camelKeys), { minLength: 1, maxLength: 10 }),
        (keys) => {
          // Build a settings object where every chosen key is undefined.
          const settings: Partial<RenderingSettings> = {};
          for (const k of keys) (settings as Record<string, unknown>)[k] = undefined;

          const zarr = renderingSettingsToZarr(settings);
          // The result should have NO entries for undefined inputs.
          for (const k of keys) {
            const snake = REVERSE_SETTINGS_MAP[k];
            if (snake) {
              expect((zarr as Record<string, unknown>)[snake]).toBeUndefined();
            }
          }
        }
      ),
      { numRuns: 100 }
    );
  });

  test('extractRenderingOverrides on empty zarr config returns empty overrides', () => {
    const empty: ZarrViewerConfig = {};
    const result = extractRenderingOverrides(empty);
    expect(Object.keys(result)).toHaveLength(0);
  });

  test('extractRenderingOverrides ignores keys NOT in RENDERING_SETTINGS_MAP', () => {
    fc.assert(
      fc.property(
        fc.dictionary(
          fc.string({ minLength: 1, maxLength: 30 }).filter((s) => !(s in RENDERING_SETTINGS_MAP)),
          fc.anything(),
          { maxKeys: 5 }
        ),
        (unknownKeys) => {
          const result = extractRenderingOverrides(unknownKeys as ZarrViewerConfig);
          // Unknown keys are silently dropped.
          expect(Object.keys(result)).toHaveLength(0);
        }
      ),
      { numRuns: 100 }
    );
  });
});

describe('renderingSettingsToZarr — unknown-key warning (visibility fix)', () => {
  // The previous behaviour silently dropped any RenderingSettings field not
  // listed in RENDERING_SETTINGS_MAP. Future-added fields would have been
  // invisibly lost. The fix preserves the drop (older zarr files keep loading)
  // but surfaces a single console.warn per unknown camelCase key.

  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    _warnedUnknownRenderingKeys.clear();
  });

  afterEach(() => {
    warnSpy.mockRestore();
    _warnedUnknownRenderingKeys.clear();
  });

  test('unknown camelCase keys are dropped from the result (backwards-compat)', () => {
    const settings = {
      bloomEnabled: true,
      futureUnknownField: 42,
    } as unknown as Partial<RenderingSettings>;
    const result = renderingSettingsToZarr(settings) as Record<string, unknown>;
    // bloomEnabled maps to bloom_enabled and is preserved.
    expect(result.bloom_enabled).toBe(true);
    // The unknown camelCase key is NOT in the output under either spelling.
    expect(result.futureUnknownField).toBeUndefined();
    expect(result.future_unknown_field).toBeUndefined();
  });

  test('log.warning is called with the unknown camelCase key name', () => {
    const settings = {
      bloomEnabled: true,
      futureUnknownField: 42,
    } as unknown as Partial<RenderingSettings>;
    renderingSettingsToZarr(settings);
    // log.warning routes through console.warn — the message must mention the
    // unknown key by name so the engineer can find and fix the bridge map.
    expect(warnSpy).toHaveBeenCalled();
    const messages = warnSpy.mock.calls.map((c: unknown[]) => c.join(' '));
    expect(messages.some((m: string) => m.includes('futureUnknownField'))).toBe(true);
    // Known key (bloomEnabled) must NOT have triggered a warning.
    expect(messages.some((m: string) => m.includes('bloomEnabled'))).toBe(false);
  });

  test('the same unknown key warns only once across repeated calls', () => {
    const settings = { futureUnknownField: 1 } as unknown as Partial<RenderingSettings>;
    renderingSettingsToZarr(settings);
    renderingSettingsToZarr(settings);
    renderingSettingsToZarr(settings);
    const matching = warnSpy.mock.calls
      .map((c: unknown[]) => c.join(' '))
      .filter((m: string) => m.includes('futureUnknownField'));
    expect(matching).toHaveLength(1);
  });
});
