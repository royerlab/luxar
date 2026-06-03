/**
 * Unit tests for core/app/factories.ts (G10).
 *
 * Covers the two pure helpers:
 *   - `defaultFactories.*`  — each entry calls `new X(...)` and returns
 *     the constructed instance. Verified against `vi.mock`ed constructors.
 *   - `resolveFactories(overrides)` — merges user overrides with
 *     `defaultFactories`, returning a fully populated `Required<AppFactories>`
 *     record. Critical because every factory must be callable: a missing
 *     key would crash the init pipeline mid-construction.
 *
 * The merge semantics matter for the LuxarApp init pipeline (P3): user
 * overrides must take precedence per-key while every unset key still
 * falls back to a real constructor.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock every heavy constructor so `new X(...)` is instrumented without
// touching THREE.js, the DOM, or the real subsystems. The mocked
// constructor's return value isn't checked — we only verify that the
// default factory invokes it exactly once with the right arguments and
// returns whatever `new` produced.
vi.mock('../../../../scene/scene-manager', () => ({
  SceneManager: vi.fn().mockImplementation(() => ({ kind: 'scene-manager' })),
}));
vi.mock('../../../../scene/animation/animation-controller', () => ({
  AnimationController: vi.fn().mockImplementation((controls, postProcessing) => ({
    kind: 'animation-controller',
    controls,
    postProcessing,
  })),
}));
vi.mock('../../../../ui/rendering-controls', () => ({
  RenderingControls: vi.fn().mockImplementation((postProcessing, sceneManager) => ({
    kind: 'rendering-controls',
    postProcessing,
    sceneManager,
  })),
}));
vi.mock('../../../../ui/recording-panel', () => ({
  RecordingPanel: vi.fn().mockImplementation((sm, ac) => ({
    kind: 'recording-panel',
    sm,
    ac,
  })),
}));
vi.mock('../../../../ui/layers', () => ({
  LayersPanel: vi.fn().mockImplementation((parent, ac) => ({
    kind: 'layers-panel',
    parent,
    ac,
  })),
}));

import {
  defaultFactories,
  resolveFactories,
  type AppFactories,
} from '../../../../core/app/factories';
import { SceneManager } from '../../../../scene/scene-manager';
import { AnimationController } from '../../../../scene/animation/animation-controller';
import { RenderingControls } from '../../../../ui/rendering-controls';
import { RecordingPanel } from '../../../../ui/recording-panel';
import { LayersPanel } from '../../../../ui/layers';

describe('defaultFactories', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('sceneManager factory invokes new SceneManager() exactly once with no args', () => {
    const result = defaultFactories.sceneManager();

    expect(SceneManager).toHaveBeenCalledExactlyOnceWith();
    // Confirm the factory hands back the constructed instance, not undefined.
    expect((result as unknown as { kind: string }).kind).toBe('scene-manager');
  });

  it('animationController factory forwards (controls, postProcessing) to new AnimationController', () => {
    const controls = { kind: 'ctrl' } as never;
    const pp = { kind: 'pp' } as never;

    const result = defaultFactories.animationController(controls, pp);

    expect(AnimationController).toHaveBeenCalledExactlyOnceWith(controls, pp);
    // Arg order is load-bearing — controls first, postProcessing second.
    expect((result as unknown as { controls: unknown; postProcessing: unknown }).controls).toBe(
      controls
    );
    expect(
      (result as unknown as { controls: unknown; postProcessing: unknown }).postProcessing
    ).toBe(pp);
  });

  it('renderingControls factory forwards (postProcessing, sceneManager) — arg ORDER matters', () => {
    const pp = { kind: 'pp' } as never;
    const sm = { kind: 'sm' } as never;

    const result = defaultFactories.renderingControls(pp, sm);

    // postProcessing is FIRST, sceneManager SECOND — swapping would
    // pass type-check (both are `unknown` to the factory call) but
    // break the constructor's internal field assignment.
    expect(RenderingControls).toHaveBeenCalledExactlyOnceWith(pp, sm);
    expect(
      (result as unknown as { postProcessing: unknown; sceneManager: unknown }).postProcessing
    ).toBe(pp);
    expect(
      (result as unknown as { postProcessing: unknown; sceneManager: unknown }).sceneManager
    ).toBe(sm);
  });

  it('recordingPanel factory forwards (sceneManager, animationController)', () => {
    const sm = { kind: 'sm' } as never;
    const ac = { kind: 'ac' } as never;

    const result = defaultFactories.recordingPanel(sm, ac);

    expect(RecordingPanel).toHaveBeenCalledExactlyOnceWith(sm, ac);
    expect((result as unknown as { sm: unknown; ac: unknown }).sm).toBe(sm);
    expect((result as unknown as { sm: unknown; ac: unknown }).ac).toBe(ac);
  });

  it('layersPanel factory forwards (parent, animationController)', () => {
    const parent = document.createElement('div');
    const ac = { kind: 'ac' } as never;

    const result = defaultFactories.layersPanel(parent, ac);

    expect(LayersPanel).toHaveBeenCalledExactlyOnceWith(parent, ac);
    expect((result as unknown as { parent: unknown; ac: unknown }).parent).toBe(parent);
    expect((result as unknown as { parent: unknown; ac: unknown }).ac).toBe(ac);
  });

  it('every key is a non-undefined function (init pipeline relies on this)', () => {
    // `Required<AppFactories>` is the contract: every entry must be
    // callable. A regression where one entry is accidentally undefined
    // would only show up when the init pipeline tries to call the
    // factory — by then the orchestrator is mid-init and the error is
    // far from the root cause.
    expect(typeof defaultFactories.sceneManager).toBe('function');
    expect(typeof defaultFactories.animationController).toBe('function');
    expect(typeof defaultFactories.renderingControls).toBe('function');
    expect(typeof defaultFactories.recordingPanel).toBe('function');
    expect(typeof defaultFactories.layersPanel).toBe('function');
  });
});

describe('resolveFactories', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns defaultFactories unchanged when overrides is undefined', () => {
    const resolved = resolveFactories(undefined);

    expect(resolved.sceneManager).toBe(defaultFactories.sceneManager);
    expect(resolved.animationController).toBe(defaultFactories.animationController);
    expect(resolved.renderingControls).toBe(defaultFactories.renderingControls);
    expect(resolved.recordingPanel).toBe(defaultFactories.recordingPanel);
    expect(resolved.layersPanel).toBe(defaultFactories.layersPanel);
  });

  it('returns defaultFactories unchanged when overrides is omitted entirely', () => {
    // The function has a `overrides?: AppFactories` signature; the
    // call site sometimes omits it. Both paths must produce the same
    // record.
    const resolved = resolveFactories();
    expect(resolved.sceneManager).toBe(defaultFactories.sceneManager);
    expect(resolved.layersPanel).toBe(defaultFactories.layersPanel);
  });

  it('returns defaultFactories unchanged for an empty overrides object', () => {
    const resolved = resolveFactories({});
    expect(resolved.sceneManager).toBe(defaultFactories.sceneManager);
    expect(resolved.recordingPanel).toBe(defaultFactories.recordingPanel);
  });

  it('user-supplied sceneManager override replaces the default; others fall through', () => {
    const customScene = vi.fn().mockReturnValue({ kind: 'custom-scene' });
    const overrides: AppFactories = { sceneManager: customScene };

    const resolved = resolveFactories(overrides);

    expect(resolved.sceneManager).toBe(customScene);
    // Other entries must still point at the defaults — a regression
    // that re-spreads or mutates `defaultFactories` would break this.
    expect(resolved.animationController).toBe(defaultFactories.animationController);
    expect(resolved.renderingControls).toBe(defaultFactories.renderingControls);
    expect(resolved.recordingPanel).toBe(defaultFactories.recordingPanel);
    expect(resolved.layersPanel).toBe(defaultFactories.layersPanel);
  });

  it('overrides for every key are honored', () => {
    const custom: Required<AppFactories> = {
      sceneManager: vi.fn().mockReturnValue({ kind: 'sm-stub' }),
      animationController: vi.fn().mockReturnValue({ kind: 'ac-stub' }),
      renderingControls: vi.fn().mockReturnValue({ kind: 'rc-stub' }),
      recordingPanel: vi.fn().mockReturnValue({ kind: 'rp-stub' }),
      layersPanel: vi.fn().mockReturnValue({ kind: 'lp-stub' }),
    };

    const resolved = resolveFactories(custom);

    expect(resolved.sceneManager).toBe(custom.sceneManager);
    expect(resolved.animationController).toBe(custom.animationController);
    expect(resolved.renderingControls).toBe(custom.renderingControls);
    expect(resolved.recordingPanel).toBe(custom.recordingPanel);
    expect(resolved.layersPanel).toBe(custom.layersPanel);
  });

  it('does not mutate the supplied overrides object', () => {
    const overrides: AppFactories = {
      sceneManager: vi.fn(),
    };
    const before = { ...overrides };

    resolveFactories(overrides);

    expect(overrides).toEqual(before);
    // Keys not in the original input must NOT be added to it.
    expect((overrides as Record<string, unknown>).layersPanel).toBeUndefined();
  });

  it('does not mutate the defaultFactories module export', () => {
    // Guard against `Object.assign(defaultFactories, overrides)` style
    // regressions that would leak overrides across calls.
    const customScene = vi.fn();
    const snapshot = { ...defaultFactories };

    resolveFactories({ sceneManager: customScene });

    expect(defaultFactories.sceneManager).toBe(snapshot.sceneManager);
    expect(defaultFactories.animationController).toBe(snapshot.animationController);
  });
});
