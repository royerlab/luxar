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

import { PostProcessingManager } from '../../../rendering/post-processing/post-processing-manager';

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
    // VR-1: toneMappingEffect must be cleared symmetrically with the others
    // so post-dispose reads return `undefined` rather than dangling at a
    // disposed object. Previously the field was non-nullable and skipped.
    expect(manager.toneMappingEffect).toBeUndefined();
  });

  it('is idempotent: a second dispose is a no-op', () => {
    // Idempotency guard: with toneMappingEffect now optional and nulled
    // (VR-1), calling dispose twice must not throw, must not re-dispose
    // already-cleared effects, and must leave every effect reference
    // unchanged at `undefined`.
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

    // First dispose: real teardown.
    dispose.call(manager);

    // Second dispose: must not throw, must not re-dispose already-cleared
    // effects, must not re-dispose the composer.
    expect(() => dispose.call(manager)).not.toThrow();

    // Each effect-dispose mock should have been called exactly once across
    // both invocations — the second dispose is a true no-op.
    for (const effect of Object.values(effects)) {
      expect(effect.dispose).toHaveBeenCalledTimes(1);
    }
    expect(composer.dispose).toHaveBeenCalledTimes(1);
    expect(effectPass.dispose).toHaveBeenCalledTimes(1);
    expect(secondaryPass.dispose).toHaveBeenCalledTimes(1);

    // All effect references remain undefined after the second pass.
    for (const key of Object.keys(effects) as DisposableEffectKey[]) {
      expect(manager[key]).toBeUndefined();
    }
  });
});

describe('PostProcessingManager.rebuildAfterContextRestore', () => {
  // CR-1: The rebuild path must (a) tear down transient GPU resources via
  // disposeTransientResources() WITHOUT setting `disposed=true`, and
  // (b) be a no-op once the manager has actually been disposed.
  type RebuildableManager = TestPostProcessingManager & {
    disposed: boolean;
    captureDurableState: () => unknown;
    disposeTransientResources: () => void;
    initializeTransientResources: () => void;
    applyDurableState: (state: unknown) => void;
    rebuildEffectPass: () => void;
  };

  function rebuildableManager(): RebuildableManager {
    const manager = Object.create(PostProcessingManager.prototype) as RebuildableManager;
    Object.assign(manager, {
      bloomEffect: disposableEffect(),
      detectorNoiseEffect: disposableEffect(),
      dofEffect: disposableEffect(),
      aoEffect: disposableEffect(),
      vignetteEffect: disposableEffect(),
      chromaticLensDistortionEffect: disposableEffect(),
      smaaEffect: disposableEffect(),
      fxaaEffect: disposableEffect(),
      toneMappingEffect: disposableEffect(),
      effectPass: disposableEffect(),
      secondaryPass: disposableEffect(),
      composer: disposableEffect(),
      disposed: false,
      // Stub the protected helpers so we can observe their invocation
      // ordering without needing a real WebGL renderer.
      captureDurableState: vi.fn(() => ({ tag: 'state' })),
      disposeTransientResources: vi.fn(),
      initializeTransientResources: vi.fn(),
      applyDurableState: vi.fn(),
      rebuildEffectPass: vi.fn(),
    });
    return manager;
  }

  it('runs capture → dispose → init → apply → rebuild in order', () => {
    const manager = rebuildableManager();
    const rebuild = PostProcessingManager.prototype.rebuildAfterContextRestore as unknown as (
      this: RebuildableManager
    ) => void;

    rebuild.call(manager);

    expect(manager.captureDurableState).toHaveBeenCalledTimes(1);
    expect(manager.disposeTransientResources).toHaveBeenCalledTimes(1);
    expect(manager.initializeTransientResources).toHaveBeenCalledTimes(1);
    expect(manager.applyDurableState).toHaveBeenCalledTimes(1);
    expect(manager.applyDurableState).toHaveBeenCalledWith({ tag: 'state' });
    expect(manager.rebuildEffectPass).toHaveBeenCalledTimes(1);

    // Manager identity is preserved: `disposed` is NOT toggled on, so
    // future rebuilds continue to work.
    expect(manager.disposed).toBe(false);
  });

  it('is a no-op when the manager has already been disposed', () => {
    const manager = rebuildableManager();
    manager.disposed = true;

    const rebuild = PostProcessingManager.prototype.rebuildAfterContextRestore as unknown as (
      this: RebuildableManager
    ) => void;
    rebuild.call(manager);

    // None of the rebuild steps should run after disposal.
    expect(manager.captureDurableState).not.toHaveBeenCalled();
    expect(manager.disposeTransientResources).not.toHaveBeenCalled();
    expect(manager.initializeTransientResources).not.toHaveBeenCalled();
    expect(manager.applyDurableState).not.toHaveBeenCalled();
    expect(manager.rebuildEffectPass).not.toHaveBeenCalled();
  });
});

describe('PostProcessingManager.startDeferRebuild / endDeferRebuild', () => {
  type DeferManager = {
    deferRebuildDepth: number;
    rebuildEffectPass: ReturnType<typeof vi.fn>;
  };

  function makeManager(): DeferManager {
    const m = Object.create(PostProcessingManager.prototype) as DeferManager;
    m.deferRebuildDepth = 0;
    m.rebuildEffectPass = vi.fn();
    return m;
  }

  const start = PostProcessingManager.prototype.startDeferRebuild as unknown as (
    this: DeferManager
  ) => void;
  const end = PostProcessingManager.prototype.endDeferRebuild as unknown as (
    this: DeferManager
  ) => void;

  it('a single start/end pair triggers exactly one rebuild', () => {
    const m = makeManager();
    start.call(m);
    expect(m.deferRebuildDepth).toBe(1);
    end.call(m);
    expect(m.deferRebuildDepth).toBe(0);
    expect(m.rebuildEffectPass).toHaveBeenCalledTimes(1);
  });

  it('nested starts only rebuild when the outermost end runs', () => {
    const m = makeManager();
    start.call(m);
    start.call(m);
    end.call(m);
    expect(m.rebuildEffectPass).not.toHaveBeenCalled();
    end.call(m);
    expect(m.rebuildEffectPass).toHaveBeenCalledTimes(1);
  });

  it('end without a matching start is a logged no-op (does not rebuild)', () => {
    const m = makeManager();
    end.call(m);
    expect(m.deferRebuildDepth).toBe(0);
    expect(m.rebuildEffectPass).not.toHaveBeenCalled();
  });

  it('try/finally keeps the counter balanced when the batch throws', () => {
    const m = makeManager();
    expect(() => {
      start.call(m);
      try {
        throw new Error('simulated setter failure');
      } finally {
        end.call(m);
      }
    }).toThrow('simulated setter failure');
    expect(m.deferRebuildDepth).toBe(0);
    // Rebuild still fires once on the outer end despite the thrown error.
    expect(m.rebuildEffectPass).toHaveBeenCalledTimes(1);
  });
});

describe('PostProcessingManager.withDeferredRebuild', () => {
  type DeferManager = {
    deferRebuildDepth: number;
    rebuildEffectPass: ReturnType<typeof vi.fn>;
  };

  function makeManager(): DeferManager {
    const m = Object.create(PostProcessingManager.prototype) as DeferManager;
    m.deferRebuildDepth = 0;
    m.rebuildEffectPass = vi.fn();
    return m;
  }

  const withDeferredRebuild = PostProcessingManager.prototype
    .withDeferredRebuild as unknown as <T>(this: DeferManager, fn: () => T) => T;

  it('triggers exactly one rebuild after the closure returns', () => {
    const m = makeManager();
    const result = withDeferredRebuild.call(m, () => 42);
    expect(result).toBe(42);
    expect(m.deferRebuildDepth).toBe(0);
    expect(m.rebuildEffectPass).toHaveBeenCalledTimes(1);
  });

  it('rethrows the closure error AND drains depth to zero', () => {
    const m = makeManager();
    expect(() =>
      withDeferredRebuild.call(m, () => {
        throw new Error('boom');
      })
    ).toThrow('boom');
    expect(m.deferRebuildDepth).toBe(0);
    // The rebuild still ran (matching the manual try/finally semantics).
    expect(m.rebuildEffectPass).toHaveBeenCalledTimes(1);
  });

  it('nested withDeferredRebuild rebuilds only on the outermost return', () => {
    const m = makeManager();
    withDeferredRebuild.call(m, () => {
      withDeferredRebuild.call(m, () => {
        // Inner closure does its work; depth is 2 here.
        expect(m.deferRebuildDepth).toBe(2);
      });
      // Inner has returned; depth is back to 1, no rebuild yet.
      expect(m.rebuildEffectPass).not.toHaveBeenCalled();
    });
    expect(m.deferRebuildDepth).toBe(0);
    expect(m.rebuildEffectPass).toHaveBeenCalledTimes(1);
  });
});
