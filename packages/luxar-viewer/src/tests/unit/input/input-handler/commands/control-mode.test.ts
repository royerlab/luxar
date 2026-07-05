/**
 * Unit tests for the camera control-mode commands.
 *
 * `nextControlType` is the pure cycle helper. `toggleControlMode` and
 * `toggleInertialMode` are the two command bodies extracted from
 * input-handler.ts during step 5d — they coordinate SceneManager +
 * InputContextManager + RenderingControls. Tests use minimal mocks for
 * each collaborator (only the surface the command touches) so failures
 * point at the command logic, not at the mock.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  nextControlType,
  toggleControlMode,
  setControlMode,
  toggleInertialMode,
  type ControlModeCtx,
} from '../../../../../input/input-handler/commands/control-mode';
import { InputContext } from '../../../../../input/input-handler/context-manager';

describe('nextControlType', () => {
  it('cycles orbit → fly → ortho → orbit', () => {
    expect(nextControlType('orbit')).toBe('fly');
    expect(nextControlType('fly')).toBe('ortho');
    expect(nextControlType('ortho')).toBe('orbit');
  });

  it('completes a full cycle in three steps', () => {
    let current = 'orbit' as const;
    const sequence: string[] = [current];
    for (let i = 0; i < 3; i++) {
      const next = nextControlType(current);
      sequence.push(next);
      current = next as typeof current;
    }
    expect(sequence).toEqual(['orbit', 'fly', 'ortho', 'orbit']);
  });

  it('falls back to orbit for unknown control types', () => {
    expect(nextControlType('unknown')).toBe('orbit');
    expect(nextControlType('')).toBe('orbit');
  });
});

interface FakeControls {
  type: 'orbit' | 'fly' | 'ortho';
  getControlType: ReturnType<typeof vi.fn>;
  getFlyControls: ReturnType<typeof vi.fn>;
}

interface FakeFlyControls {
  inertialMode: boolean;
  setInertialMode: ReturnType<typeof vi.fn>;
}

function makeCtx(opts: {
  initialControlType?: 'orbit' | 'fly' | 'ortho';
  flyControls?: FakeFlyControls | null;
  withRenderingControls?: boolean;
}): {
  ctx: ControlModeCtx;
  setControlType: ReturnType<typeof vi.fn>;
  setContext: ReturnType<typeof vi.fn>;
  syncCurrentState: ReturnType<typeof vi.fn>;
  saveSettings: ReturnType<typeof vi.fn>;
  controls: FakeControls;
} {
  const setControlType = vi.fn();
  const setContext = vi.fn();
  const syncCurrentState = vi.fn();
  const saveSettings = vi.fn();

  const controls: FakeControls = {
    type: opts.initialControlType ?? 'orbit',
    getControlType: vi.fn(() => controls.type),
    getFlyControls: vi.fn(() => (opts.flyControls === undefined ? null : opts.flyControls)),
  };

  const sceneManager = {
    controls: {
      getControlType: controls.getControlType,
      getFlyControls: controls.getFlyControls,
    },
    setControlType,
  } as unknown as ControlModeCtx['sceneManager'];

  const contextManager = {
    setContext,
  } as unknown as ControlModeCtx['contextManager'];

  const renderingControls = opts.withRenderingControls
    ? ({ syncCurrentState, saveSettings } as unknown as ControlModeCtx['renderingControls'])
    : undefined;

  return {
    ctx: { sceneManager, contextManager, renderingControls },
    setControlType,
    setContext,
    syncCurrentState,
    saveSettings,
    controls,
  };
}

describe('toggleControlMode', () => {
  it('cycles orbit → fly and sets FLY_CONTROLS context', () => {
    const { ctx, setControlType, setContext } = makeCtx({
      initialControlType: 'orbit',
    });

    toggleControlMode(ctx);

    expect(setControlType).toHaveBeenCalledWith('fly');
    expect(setContext).toHaveBeenCalledWith(InputContext.FLY_CONTROLS);
  });

  it('cycles fly → ortho and sets NAVIGATION context', () => {
    const { ctx, setControlType, setContext } = makeCtx({
      initialControlType: 'fly',
    });

    toggleControlMode(ctx);

    expect(setControlType).toHaveBeenCalledWith('ortho');
    expect(setContext).toHaveBeenCalledWith(InputContext.NAVIGATION);
  });

  it('cycles ortho → orbit and sets NAVIGATION context', () => {
    const { ctx, setControlType, setContext } = makeCtx({
      initialControlType: 'ortho',
    });

    toggleControlMode(ctx);

    expect(setControlType).toHaveBeenCalledWith('orbit');
    expect(setContext).toHaveBeenCalledWith(InputContext.NAVIGATION);
  });

  it('syncs renderingControls when one is wired', () => {
    const { ctx, syncCurrentState } = makeCtx({
      initialControlType: 'orbit',
      withRenderingControls: true,
    });

    toggleControlMode(ctx);

    expect(syncCurrentState).toHaveBeenCalledTimes(1);
  });

  it('persists the new mode via saveSettings() (else it reverts on reload)', () => {
    // Regression: after the "Control Type" dropdown was removed, the mode-change
    // path stopped persisting. Every mode switch must call saveSettings() so the
    // choice survives a reload (settings.controlType is applied on setSceneId).
    const { ctx, saveSettings } = makeCtx({
      initialControlType: 'orbit',
      withRenderingControls: true,
    });

    toggleControlMode(ctx);

    expect(saveSettings).toHaveBeenCalledTimes(1);
  });

  it('is a no-op on renderingControls when none is wired', () => {
    const { ctx, syncCurrentState, saveSettings } = makeCtx({
      initialControlType: 'orbit',
      withRenderingControls: false,
    });

    toggleControlMode(ctx);

    expect(syncCurrentState).not.toHaveBeenCalled();
    expect(saveSettings).not.toHaveBeenCalled();
  });
});

describe('setControlMode', () => {
  it('switches to an explicit mode + persists it, and no-ops when already there', () => {
    const { ctx, setControlType, setContext, saveSettings } = makeCtx({
      initialControlType: 'orbit',
      withRenderingControls: true,
    });

    setControlMode(ctx, 'fly');
    expect(setControlType).toHaveBeenCalledWith('fly');
    expect(setContext).toHaveBeenCalledWith(InputContext.FLY_CONTROLS);
    expect(saveSettings).toHaveBeenCalledTimes(1);
  });

  it('is a no-op when the target mode is already active', () => {
    const { ctx, setControlType, saveSettings } = makeCtx({
      initialControlType: 'fly',
      withRenderingControls: true,
    });

    setControlMode(ctx, 'fly');
    expect(setControlType).not.toHaveBeenCalled();
    expect(saveSettings).not.toHaveBeenCalled();
  });
});

describe('toggleInertialMode', () => {
  it('flips inertialMode on the fly controls', () => {
    const flyControls: FakeFlyControls = {
      inertialMode: false,
      setInertialMode: vi.fn(),
    };
    const { ctx } = makeCtx({ flyControls });

    toggleInertialMode(ctx);

    expect(flyControls.setInertialMode).toHaveBeenCalledWith(true);
  });

  it('flips inertialMode back off when currently on', () => {
    const flyControls: FakeFlyControls = {
      inertialMode: true,
      setInertialMode: vi.fn(),
    };
    const { ctx } = makeCtx({ flyControls });

    toggleInertialMode(ctx);

    expect(flyControls.setInertialMode).toHaveBeenCalledWith(false);
  });

  it('syncs renderingControls when one is wired AND fly controls are active', () => {
    const flyControls: FakeFlyControls = {
      inertialMode: false,
      setInertialMode: vi.fn(),
    };
    const { ctx, syncCurrentState } = makeCtx({
      flyControls,
      withRenderingControls: true,
    });

    toggleInertialMode(ctx);

    expect(syncCurrentState).toHaveBeenCalledTimes(1);
  });

  it('is a no-op (besides logging) when not in fly mode', () => {
    const { ctx, syncCurrentState } = makeCtx({
      flyControls: null,
      withRenderingControls: true,
    });

    // Should not throw and should not touch renderingControls.
    expect(() => toggleInertialMode(ctx)).not.toThrow();
    expect(syncCurrentState).not.toHaveBeenCalled();
  });
});
