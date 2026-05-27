/**
 * Unit tests for AnimationShortcuts.
 *
 * Strategy: stub InputContextManager.registerBinding so the test can
 * capture the 5 registered handlers, then invoke each handler and
 * assert on the side-effects (sceneDimsManager + animation-manager
 * calls + log output).
 *
 * The handler bodies are exercised with a real `sceneDimsManager`
 * mock to keep the test honest about the
 * `selectedDimension → dims → navigableDims[i]` lookup.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// AUDIT NOTE (input.md C2): this mock replaces the sceneDimsManager
// singleton (sibling internal module, not an external boundary). A
// refactor that renames any method (e.g. setDimensionValue →
// updateDimension) would silently leave this test green because the
// mock contract is whatever the test author wrote. Treating the
// sceneDimsManager as the integration seam with the scene graph is
// defensible — it's effectively the "data layer" boundary for input —
// but the alternative is exercising AnimationShortcuts against the
// real singleton in jsdom. Follow-up.
vi.mock('../../../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    getDims: vi.fn(),
    getDimensionRanges: vi.fn(),
    setDimensionValue: vi.fn(),
  },
}));

import { sceneDimsManager } from '../../../../../scene/scene-dims-manager';
import {
  AnimationShortcuts,
  type AnimationShortcutsContext,
} from '../../../../../input/input-handler/key-bindings/animation-shortcuts';
import {
  InputContext,
  type InputContextManager,
} from '../../../../../input/input-handler/context-manager';
import type { DimensionAnimationManager } from '../../../../../scene/animation/dimension-animation-manager';
import type { SimpleDims } from '../../../../../types/dims';

function makeContextManager(): {
  manager: InputContextManager;
  bindings: Array<{
    context: InputContext;
    key: string;
    modifiers?: { shift?: boolean };
    handler: () => void;
  }>;
} {
  const bindings: Array<{
    context: InputContext;
    key: string;
    modifiers?: { shift?: boolean };
    handler: () => void;
  }> = [];
  const registerBinding = vi.fn(
    (
      context: InputContext,
      binding: { key: string; modifiers?: { shift?: boolean }; handler: () => void }
    ) => {
      bindings.push({
        context,
        key: binding.key,
        modifiers: binding.modifiers,
        handler: binding.handler,
      });
    }
  );
  const manager = { registerBinding } as unknown as InputContextManager;
  return { manager, bindings };
}

function makeAnimationManager(): {
  manager: DimensionAnimationManager;
  togglePlay: ReturnType<typeof vi.fn>;
  increaseSpeed: ReturnType<typeof vi.fn>;
  decreaseSpeed: ReturnType<typeof vi.fn>;
  getState: ReturnType<typeof vi.fn>;
} {
  const togglePlay = vi.fn();
  const increaseSpeed = vi.fn();
  const decreaseSpeed = vi.fn();
  const getState = vi.fn(() => ({ targetFPS: 30 }));
  const manager = {
    togglePlay,
    increaseSpeed,
    decreaseSpeed,
    getState,
  } as unknown as DimensionAnimationManager;
  return { manager, togglePlay, increaseSpeed, decreaseSpeed, getState };
}

function makeContext(
  selectedDim: number,
  animManager?: DimensionAnimationManager
): AnimationShortcutsContext {
  return {
    getSelectedDimension: () => selectedDim,
    getAnimationManager: () => animManager,
  };
}

function makeDims(): SimpleDims {
  // 5D dataset, displayed = X, Y, Z (dims 0, 1, 2). Time and Channel
  // (dims 3, 4) are the navigable ones.
  return {
    ndim: 5,
    currentStep: [0, 0, 0, 0, 0],
    displayed: [0, 1, 2],
    metadata: undefined,
  };
}

describe('AnimationShortcuts.register', () => {
  beforeEach(() => {
    vi.mocked(sceneDimsManager.getDims).mockReturnValue(makeDims());
    vi.mocked(sceneDimsManager.getDimensionRanges).mockReset();
    vi.mocked(sceneDimsManager.setDimensionValue).mockReset();
  });

  it('registers exactly 5 bindings on the NAVIGATION context', () => {
    const { manager, bindings } = makeContextManager();
    const shortcuts = new AnimationShortcuts(manager, makeContext(0));
    shortcuts.register();
    expect(bindings).toHaveLength(5);
    expect(bindings.every((b) => b.context === InputContext.NAVIGATION)).toBe(true);
  });

  it('registers the expected key + modifier combinations', () => {
    const { manager, bindings } = makeContextManager();
    const shortcuts = new AnimationShortcuts(manager, makeContext(0));
    shortcuts.register();
    const keys = bindings.map((b) => ({ key: b.key, shift: b.modifiers?.shift ?? false }));
    expect(keys).toEqual([
      { key: 'k', shift: false },
      { key: 'Home', shift: false },
      { key: 'End', shift: false },
      { key: 'ArrowUp', shift: true },
      { key: 'ArrowDown', shift: true },
    ]);
  });
});

describe('AnimationShortcuts handlers', () => {
  beforeEach(() => {
    vi.mocked(sceneDimsManager.getDims).mockReturnValue(makeDims());
    vi.mocked(sceneDimsManager.getDimensionRanges).mockReset();
    vi.mocked(sceneDimsManager.setDimensionValue).mockReset();
  });

  function setup(selectedDim: number, animManager?: DimensionAnimationManager) {
    const { manager, bindings } = makeContextManager();
    new AnimationShortcuts(manager, makeContext(selectedDim, animManager)).register();
    const findHandler = (key: string) => bindings.find((b) => b.key === key)!.handler;
    return { findHandler };
  }

  it('K: toggles play on the selected dim (selectedDim=0 → actual dim 3)', () => {
    const { manager, togglePlay } = makeAnimationManager();
    const { findHandler } = setup(0, manager);
    findHandler('k')();
    expect(togglePlay).toHaveBeenCalledWith(3);
  });

  it('K: no-op when no dim is selected (selectedDim=-1)', () => {
    const { manager, togglePlay } = makeAnimationManager();
    const { findHandler } = setup(-1, manager);
    findHandler('k')();
    expect(togglePlay).not.toHaveBeenCalled();
  });

  it('K: no-op when the animation manager is not yet constructed', () => {
    // input.md [W1][P2] strengthening: previously `.not.toThrow()` only.
    // The contract of the no-anim-manager branch is that NO sceneDims
    // call fires (the handler must early-return before touching
    // setDimensionValue / getDimensionRanges). Pin the observable
    // side-effect-free invariant — a regression that called through to
    // sceneDimsManager would survive a `.not.toThrow()` smoke check.
    const { findHandler } = setup(0, undefined);
    findHandler('k')();
    expect(sceneDimsManager.setDimensionValue).not.toHaveBeenCalled();
    expect(sceneDimsManager.getDimensionRanges).not.toHaveBeenCalled();
  });

  it('Home: jumps the selected dim to the range minimum', () => {
    vi.mocked(sceneDimsManager.getDimensionRanges).mockReturnValue([
      [0, 10],
      [0, 10],
      [0, 10],
      [-7, 100], // min = -7
      [0, 100],
    ]);
    const { manager } = makeAnimationManager();
    const { findHandler } = setup(0, manager);
    findHandler('Home')();
    expect(sceneDimsManager.setDimensionValue).toHaveBeenCalledWith(3, -7);
  });

  it('End: jumps the selected dim to the range maximum', () => {
    vi.mocked(sceneDimsManager.getDimensionRanges).mockReturnValue([
      [0, 10],
      [0, 10],
      [0, 10],
      [0, 100],
      [0, 50], // 2nd navigable max = 50
    ]);
    const { manager } = makeAnimationManager();
    const { findHandler } = setup(1, manager); // selectedDim=1 → actual dim 4
    findHandler('End')();
    expect(sceneDimsManager.setDimensionValue).toHaveBeenCalledWith(4, 50);
  });

  it('Home: no-op when getDimensionRanges returns null', () => {
    vi.mocked(sceneDimsManager.getDimensionRanges).mockReturnValue(null);
    const { manager } = makeAnimationManager();
    const { findHandler } = setup(0, manager);
    findHandler('Home')();
    expect(sceneDimsManager.setDimensionValue).not.toHaveBeenCalled();
  });

  it('Shift+ArrowUp: increases the selected dim speed', () => {
    const { manager, increaseSpeed } = makeAnimationManager();
    const { findHandler } = setup(0, manager);
    findHandler('ArrowUp')();
    expect(increaseSpeed).toHaveBeenCalledWith(3);
  });

  it('Shift+ArrowDown: decreases the selected dim speed', () => {
    const { manager, decreaseSpeed } = makeAnimationManager();
    const { findHandler } = setup(0, manager);
    findHandler('ArrowDown')();
    expect(decreaseSpeed).toHaveBeenCalledWith(3);
  });

  it('all handlers no-op cleanly when getDims returns null (no scene loaded)', () => {
    // input.md [W1][P2] strengthening: previously 5x `.not.toThrow()`.
    // The no-scene contract is that every animation/sceneDims side
    // effect is skipped when getDims() returns null — pin all four
    // observable channels rather than just the non-throw smoke.
    vi.mocked(sceneDimsManager.getDims).mockReturnValue(null);
    const { manager, togglePlay, increaseSpeed, decreaseSpeed } = makeAnimationManager();
    const { findHandler } = setup(0, manager);
    findHandler('k')();
    findHandler('Home')();
    findHandler('End')();
    findHandler('ArrowUp')();
    findHandler('ArrowDown')();
    // togglePlay / speed-up / speed-down all gated by getDims() — none fire.
    expect(togglePlay).not.toHaveBeenCalled();
    expect(increaseSpeed).not.toHaveBeenCalled();
    expect(decreaseSpeed).not.toHaveBeenCalled();
    expect(sceneDimsManager.setDimensionValue).not.toHaveBeenCalled();
  });
});
