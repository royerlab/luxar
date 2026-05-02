import { describe, expect, it, vi } from 'vitest';

vi.mock('../../../utils/log', () => ({
  log: {
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    success: vi.fn(),
    update: vi.fn(),
  },
  Modules: { POST_PROCESSING: 'PostProcessing' },
}));

import { PostProcessingManager } from '../../../rendering/post-processing-manager';

type Disposable = { dispose: ReturnType<typeof vi.fn> };
type DisposableEffectKey =
  | 'bloomEffect'
  | 'detectorNoiseEffect'
  | 'dofEffect'
  | 'aoEffect'
  | 'vignetteEffect'
  | 'chromaticLensDistortionEffect'
  | 'smaaEffect'
  | 'fxaaEffect'
  | 'toneMappingEffect';
type TestPostProcessingManager = Record<DisposableEffectKey, Disposable | undefined> & {
  effectPass?: Disposable;
  secondaryPass?: Disposable;
  composer: Disposable;
};

function disposableEffect(): Disposable {
  return { dispose: vi.fn() };
}

describe('PostProcessingManager.dispose', () => {
  it('disposes owned effects explicitly in addition to passes and composer', () => {
    const manager = Object.create(PostProcessingManager.prototype) as TestPostProcessingManager;
    const effects: Record<DisposableEffectKey, Disposable> = {
      bloomEffect: disposableEffect(),
      detectorNoiseEffect: disposableEffect(),
      dofEffect: disposableEffect(),
      aoEffect: disposableEffect(),
      vignetteEffect: disposableEffect(),
      chromaticLensDistortionEffect: disposableEffect(),
      smaaEffect: disposableEffect(),
      fxaaEffect: disposableEffect(),
      toneMappingEffect: disposableEffect(),
    };
    const effectPass = disposableEffect();
    const secondaryPass = disposableEffect();
    const composer = disposableEffect();

    Object.assign(manager, {
      ...effects,
      effectPass,
      secondaryPass,
      composer,
    });

    const dispose = PostProcessingManager.prototype.dispose as unknown as (
      this: TestPostProcessingManager
    ) => void;
    dispose.call(manager);

    expect(effectPass.dispose).toHaveBeenCalledTimes(1);
    expect(secondaryPass.dispose).toHaveBeenCalledTimes(1);
    for (const effect of Object.values(effects)) {
      expect(effect.dispose).toHaveBeenCalledTimes(1);
    }
    expect(composer.dispose).toHaveBeenCalledTimes(1);
    expect(manager.bloomEffect).toBeUndefined();
    expect(manager.fxaaEffect).toBeUndefined();
  });
});
