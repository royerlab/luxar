/**
 * Unit tests for core/app/overlays/init-scale-bar.ts (G8).
 *
 * `initScaleBar` (re-)builds the scale bar and wires it through the
 * input handler + animation controller. Contract:
 *   - When a previous instance exists, removePerFrameCallback('scale-bar')
 *     runs BEFORE previous.dispose() (so the per-frame closure doesn't
 *     reference a disposed instance for one frame).
 *   - Always returns a fresh ScaleBar.
 *   - Registers a per-frame callback under the key 'scale-bar'.
 *   - Wires the new scale bar into inputHandler.setScaleBar.
 *   - The per-frame callback invokes scaleBar.update() at call time.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

const mocks = vi.hoisted(() => ({
  scaleBarUpdate: vi.fn(),
  scaleBarDispose: vi.fn(),
  ScaleBarCtor: vi.fn(),
}));

vi.mock('../../../../../ui/scale-bar', () => ({
  ScaleBar: mocks.ScaleBarCtor,
}));

// Config provides target width + position; mock so we don't pull the
// full config validation graph in.
vi.mock('../../../../../config', () => ({
  config: {
    ui: {
      scaleBar: {
        targetWidthPx: 100,
        position: 'bottom-right',
      },
    },
  },
}));

import { initScaleBar } from '../../../../../core/app/overlays/init-scale-bar';

beforeEach(() => {
  mocks.ScaleBarCtor.mockReset();
  mocks.scaleBarUpdate.mockReset();
  mocks.scaleBarDispose.mockReset();
  mocks.ScaleBarCtor.mockImplementation(() => ({
    update: mocks.scaleBarUpdate,
    dispose: mocks.scaleBarDispose,
  }));
});

function makePorts(opts: { previous?: unknown } = {}) {
  const canvas = document.createElement('canvas');
  return {
    previous: opts.previous as never,
    sceneManager: {
      camera: { id: 'camera' },
      controls: { id: 'controls' },
      renderer: { domElement: canvas },
    } as never,
    animationController: {
      addPerFrameCallback: vi.fn(),
      removePerFrameCallback: vi.fn(),
    } as never,
    inputHandler: {
      setScaleBar: vi.fn(),
    } as never,
  };
}

describe('initScaleBar', () => {
  it('constructs a ScaleBar from the supplied sceneManager properties', () => {
    const ports = makePorts();
    initScaleBar(ports);

    expect(mocks.ScaleBarCtor).toHaveBeenCalledOnce();
    const args = mocks.ScaleBarCtor.mock.calls[0][0] as {
      getCamera: () => unknown;
      controls: unknown;
      canvas: unknown;
      targetWidthPx: number;
      position: string;
    };
    // getCamera is a LIVE accessor (the app swaps cameras on
    // perspective ↔ ortho) — it must read through to the scene
    // manager's current camera, not capture it.
    expect(args.getCamera()).toEqual({ id: 'camera' });
    (ports.sceneManager as unknown as { camera: unknown }).camera = { id: 'swapped' };
    expect(args.getCamera()).toEqual({ id: 'swapped' });
    expect(args.controls).toEqual({ id: 'controls' });
    expect(args.canvas).toBe(
      (ports.sceneManager as unknown as { renderer: { domElement: HTMLCanvasElement } }).renderer
        .domElement
    );
    expect(args.targetWidthPx).toBe(100);
    expect(args.position).toBe('bottom-right');
  });

  it('returns the newly-constructed ScaleBar', () => {
    const ports = makePorts();
    const result = initScaleBar(ports);
    expect(result).toBe(mocks.ScaleBarCtor.mock.results[0].value);
  });

  it('registers a per-frame callback keyed "scale-bar"', () => {
    const ports = makePorts();
    initScaleBar(ports);

    const animCtl = ports.animationController as unknown as {
      addPerFrameCallback: ReturnType<typeof vi.fn>;
    };
    expect(animCtl.addPerFrameCallback).toHaveBeenCalledOnce();
    expect(animCtl.addPerFrameCallback.mock.calls[0][0]).toBe('scale-bar');
    expect(typeof animCtl.addPerFrameCallback.mock.calls[0][1]).toBe('function');
  });

  it('the per-frame callback invokes scaleBar.update()', () => {
    const ports = makePorts();
    initScaleBar(ports);

    const animCtl = ports.animationController as unknown as {
      addPerFrameCallback: ReturnType<typeof vi.fn>;
    };
    const callback = animCtl.addPerFrameCallback.mock.calls[0][1] as () => void;
    callback();
    expect(mocks.scaleBarUpdate).toHaveBeenCalledOnce();
  });

  it('wires the new scale bar into inputHandler.setScaleBar', () => {
    const ports = makePorts();
    const result = initScaleBar(ports);

    const input = ports.inputHandler as unknown as {
      setScaleBar: ReturnType<typeof vi.fn>;
    };
    expect(input.setScaleBar).toHaveBeenCalledExactlyOnceWith(result);
  });

  describe('reload path (previous instance present)', () => {
    it('removes the previous per-frame callback and disposes the previous instance', () => {
      const previousDispose = vi.fn();
      const previous = { dispose: previousDispose, update: vi.fn() };
      const ports = makePorts({ previous });

      initScaleBar(ports);

      const animCtl = ports.animationController as unknown as {
        removePerFrameCallback: ReturnType<typeof vi.fn>;
      };
      expect(animCtl.removePerFrameCallback).toHaveBeenCalledExactlyOnceWith('scale-bar');
      expect(previousDispose).toHaveBeenCalledOnce();
    });

    it('removePerFrameCallback runs BEFORE previous.dispose (no stale closure)', () => {
      const order: string[] = [];
      const previous = {
        dispose: vi.fn(() => order.push('dispose')),
        update: vi.fn(),
      };
      const ports = makePorts({ previous });
      (
        ports.animationController as unknown as {
          removePerFrameCallback: ReturnType<typeof vi.fn>;
        }
      ).removePerFrameCallback.mockImplementation(() => order.push('removeCallback'));

      initScaleBar(ports);

      expect(order).toEqual(['removeCallback', 'dispose']);
    });

    it('does NOT call removePerFrameCallback when previous is undefined', () => {
      const ports = makePorts({ previous: undefined });
      initScaleBar(ports);

      const animCtl = ports.animationController as unknown as {
        removePerFrameCallback: ReturnType<typeof vi.fn>;
      };
      expect(animCtl.removePerFrameCallback).not.toHaveBeenCalled();
    });
  });
});
