import { describe, expect, it } from 'vitest';
import { StorageKeys } from '../../../utils/storage-keys';

describe('StorageKeys', () => {
  it('exposes the theme and debug keys as constants', () => {
    expect(StorageKeys.theme).toBe('luxar.theme');
    expect(StorageKeys.debug).toBe('luxar.debug');
  });

  it('exposes the control-rail keys under the luxar.controlRail namespace', () => {
    expect(StorageKeys.controlRailHintDismissed).toBe('luxar.controlRail.hintDismissed');
    expect(StorageKeys.controlRailCollapsed).toBe('luxar.controlRail.collapsed');
  });

  it('builds rendering keys under the luxar.rendering namespace', () => {
    expect(StorageKeys.rendering('my-scene')).toBe('luxar.rendering.my-scene');
  });

  it('sanitizes scene ids to keep keys safe', () => {
    expect(StorageKeys.rendering('test@scene!')).toBe('luxar.rendering.test_scene_');
    expect(StorageKeys.rendering('http://example.com/data.zarr')).toBe(
      'luxar.rendering.http___example_com_data_zarr'
    );
  });

  it('preserves alphanumerics, hyphens, and underscores', () => {
    expect(StorageKeys.rendering('a-b_c-1')).toBe('luxar.rendering.a-b_c-1');
  });

  // [api.md OOS] Frozen so embedders cannot mutate the namespacing
  // contract. `as const` is a TypeScript-only readonly marker; the
  // runtime object was a plain mutable record pre-fix. Object.freeze
  // locks it at module load.
  describe('immutability (Object.freeze contract)', () => {
    it('Object.isFrozen reports the constant as frozen', () => {
      expect(Object.isFrozen(StorageKeys)).toBe(true);
    });

    it('writing to an existing key throws in strict mode (test runs in strict by default)', () => {
      // Vitest runs ESM with strict mode → frozen-object writes throw
      // TypeError instead of silently no-op'ing.
      const tamper = () => {
        (StorageKeys as { theme: string }).theme = 'pwned.theme';
      };
      expect(tamper).toThrow(TypeError);
      // Original value preserved.
      expect(StorageKeys.theme).toBe('luxar.theme');
    });

    it('adding a new property throws in strict mode', () => {
      const tamper = () => {
        (StorageKeys as Record<string, unknown>).injected = 'pwned.injected';
      };
      expect(tamper).toThrow(TypeError);
    });

    it('deleting a property throws in strict mode', () => {
      const tamper = () => {
        delete (StorageKeys as { debug?: string }).debug;
      };
      expect(tamper).toThrow(TypeError);
      // Original value preserved.
      expect(StorageKeys.debug).toBe('luxar.debug');
    });

    it('rendering() function is still callable (freeze does not affect method invocation)', () => {
      // The freeze locks the property descriptors but does not break
      // method binding — `rendering` is still callable.
      expect(StorageKeys.rendering('after-freeze')).toBe('luxar.rendering.after-freeze');
    });
  });
});
