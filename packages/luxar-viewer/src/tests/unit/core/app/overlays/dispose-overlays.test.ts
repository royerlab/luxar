/**
 * Unit tests for core/app/overlays/dispose-overlays.ts (G7).
 *
 * `disposeOverlays` is a tiny helper but it captures a precise contract:
 *   - When `manager` is defined, call `manager.dispose()` and then the
 *     `onDisposed` callback (so the orchestrator can null out its
 *     reference).
 *   - When `manager` is undefined, skip BOTH steps (don't fire the
 *     callback for a no-op).
 *   - dispose-order matters: `manager.dispose()` runs BEFORE `onDisposed`
 *     so the callback can rely on the manager actually being torn down.
 */

import { describe, it, expect, vi } from 'vitest';
import { disposeOverlays } from '../../../../../core/app/overlays/dispose-overlays';
import type { OverlayManager } from '../../../../../ui/overlay-manager';

describe('disposeOverlays', () => {
  it('calls manager.dispose() and onDisposed() when manager is defined', () => {
    const dispose = vi.fn();
    const onDisposed = vi.fn();
    const manager = { dispose } as unknown as OverlayManager;

    disposeOverlays({ manager, onDisposed });

    expect(dispose).toHaveBeenCalledOnce();
    expect(onDisposed).toHaveBeenCalledOnce();
  });

  it('does NOT call onDisposed when manager is undefined (no-op)', () => {
    const onDisposed = vi.fn();
    disposeOverlays({ manager: undefined, onDisposed });

    // Critical: a regression that fires onDisposed unconditionally
    // would null out the orchestrator's reference on every dispose
    // call, which is harmless today but obscures the no-op semantic.
    expect(onDisposed).not.toHaveBeenCalled();
  });

  it('calls dispose BEFORE onDisposed so the callback can rely on tear-down completing', () => {
    const order: string[] = [];
    const dispose = vi.fn(() => order.push('dispose'));
    const onDisposed = vi.fn(() => order.push('onDisposed'));
    const manager = { dispose } as unknown as OverlayManager;

    disposeOverlays({ manager, onDisposed });

    expect(order).toEqual(['dispose', 'onDisposed']);
  });

  it('does not swallow errors from manager.dispose() (caller decides safety)', () => {
    // The helper is intentionally bare — the orchestrator's
    // `safeDispose` wrapper is what makes dispose error-safe at the
    // call site. Pinning this contract here so a "convenience" try/catch
    // doesn't accidentally hide a real bug in the manager dispose path.
    const boom = new Error('overlay teardown failed');
    const dispose = vi.fn(() => {
      throw boom;
    });
    const onDisposed = vi.fn();
    const manager = { dispose } as unknown as OverlayManager;

    expect(() => disposeOverlays({ manager, onDisposed })).toThrow(boom);
    // onDisposed should NOT run because dispose threw — but the order
    // matters here: it threw synchronously *before* the callback.
    expect(onDisposed).not.toHaveBeenCalled();
  });

  it('passes no arguments to onDisposed', () => {
    const dispose = vi.fn();
    const onDisposed = vi.fn();
    const manager = { dispose } as unknown as OverlayManager;

    disposeOverlays({ manager, onDisposed });

    expect(onDisposed).toHaveBeenCalledExactlyOnceWith();
  });

  it('passes no arguments to manager.dispose', () => {
    const dispose = vi.fn();
    const onDisposed = vi.fn();
    const manager = { dispose } as unknown as OverlayManager;

    disposeOverlays({ manager, onDisposed });

    expect(dispose).toHaveBeenCalledExactlyOnceWith();
  });
});
