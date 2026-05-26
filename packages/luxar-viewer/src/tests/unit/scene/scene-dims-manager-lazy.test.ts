import { describe, expect, it, beforeEach } from 'vitest';
import {
  sceneDimsManager,
  __resetSceneDimsManagerForTests,
} from '../../../scene/scene-dims-manager';

describe('sceneDimsManager (lazy singleton)', () => {
  beforeEach(() => {
    __resetSceneDimsManagerForTests();
  });

  it('exposes the SceneDimsManager surface through the proxy', () => {
    expect(typeof sceneDimsManager.getDims).toBe('function');
    expect(typeof sceneDimsManager.addListener).toBe('function');
    expect(typeof sceneDimsManager.setDimensionValue).toBe('function');
  });

  it('returns null from getDims() before any scene has been initialized', () => {
    expect(sceneDimsManager.getDims()).toBeNull();
  });

  it('destructured methods stay bound to the underlying instance', () => {
    // Methods are bound when read through the Proxy, so destructuring or
    // capturing a reference should still target the underlying instance.
    let calls = 0;
    const listener = () => {
      calls++;
    };
    const { addListener, removeListener } = sceneDimsManager;
    addListener(listener);
    // No notification has been triggered yet, so calls should still be 0.
    expect(calls).toBe(0);
    // Removing the same reference works → addListener bound `this` correctly.
    expect(() => removeListener(listener)).not.toThrow();
  });

  it('starts fresh after __resetSceneDimsManagerForTests()', () => {
    sceneDimsManager.addListener(() => {});
    __resetSceneDimsManagerForTests();
    // After reset, the new instance has no listeners — adding the same
    // function again should not throw "duplicate" or surface old state.
    expect(() => sceneDimsManager.addListener(() => {})).not.toThrow();
  });
});
