import { describe, expect, it } from 'vitest';
import { config } from '../../../../config';
import { syncCameraFovState } from '../../../../ui/rendering-controls/sync-current-state';

describe('syncCameraFovState', () => {
  it('updates only FOV state and derives the matching preset', () => {
    const settings = { ...config.renderingControls.defaults, autoRotate: true };

    syncCameraFovState(settings, { currentFov: 63 } as never);

    expect(settings.fov).toBe(63);
    expect(settings.fovPreset).toBe('35mm');
    expect(settings.autoRotate).toBe(true);
  });

  it('marks an unmatched live FOV as custom', () => {
    const settings = { ...config.renderingControls.defaults };

    syncCameraFovState(settings, { currentFov: 61 } as never);

    expect(settings.fov).toBe(61);
    expect(settings.fovPreset).toBe('Custom');
  });
});
