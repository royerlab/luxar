import { describe, expect, it } from 'vitest';
import { StorageKeys } from '../../../utils/storage-keys';

describe('StorageKeys', () => {
  it('exposes the theme and debug keys as constants', () => {
    expect(StorageKeys.theme).toBe('luxar.theme');
    expect(StorageKeys.debug).toBe('luxar.debug');
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
});
