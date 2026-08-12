/**
 * Unit tests for Dimension Sliders.
 * Tests critical fix: memory leaks from slider/dropdown event listeners.
 *
 * AUDIT NOTE (ui.md C3, resolved): the lifecycle tests previously pinned
 * EXACT event-type strings that the implementation registered (input/
 * keydown/etc.). They've been rewritten to assert observable post-dispose
 * behavior: dispatch input/change/keydown/click on the SAME element after
 * dispose and verify that `sceneDimsManager.setDimensionValue` is NOT
 * called. Each test includes a pre-dispose sanity probe so a setup-time
 * regression (handler never wired) is also caught.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { DimensionSliders } from '../../../ui/dimension-sliders';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import type { SimpleDims } from '../../../types/dims';

// Mock scene dims manager
vi.mock('../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    setDimensionValue: vi.fn(),
    getDims: vi.fn(),
    getDimensionRanges: vi.fn(),
    getDimensionNames: vi.fn(),
  },
}));

beforeEach(() => {
  document.body.innerHTML = '<div id="test-container"></div>';
});

afterEach(() => {
  document.body.innerHTML = '';
});

// ui.md O1 / Phase E39: previously named "Memory Leak Prevention" — a
// concern, not a behavior. The inner tests now cover real
// slider/dropdown lifecycle contracts (construction → re-init →
// dispose). Rename to surface what the block actually pins.
describe('DimensionSliders - slider/dropdown lifecycle (post-dispose)', () => {
  const createMockDims = (): SimpleDims => ({
    ndim: 5,
    displayed: [0, 1, 2], // X, Y, Z
    currentStep: [0, 0, 0, 5.5, 2], // Time at 5.5, Channel at 2
    metadata: [
      { name: 'X', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Y', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Z', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Time', unit: 's', scale: 1.0, discrete: false, step: 0.1 },
      {
        name: 'Channel',
        unit: '',
        scale: 1.0,
        discrete: true,
        step: 1,
        categories: ['DAPI', 'GFP', 'RFP'],
      },
    ],
  });

  describe('Slider Event Listener Cleanup', () => {
    // C3 strengthening (P1): the original tests asserted exact event-type
    // strings on removeEventListener — coupling to impl. The strengthened
    // form drives a real event after dispose and asserts the observable
    // contract: setDimensionValue is NOT invoked (callback unfired).
    it('post-dispose: slider input no longer routes to setDimensionValue', () => {
      const container = document.getElementById('test-container')!;
      const dims = createMockDims();

      const sliders = new DimensionSliders({
        container,
        dims,
        dimensionRanges: [
          [0, 100],
          [0, 100],
          [0, 100],
          [0, 10],
          [0, 2],
        ],
        dimensionNames: ['X', 'Y', 'Z', 'Time', 'Channel'],
        dimensionUnits: ['μm', 'μm', 'μm', 's', ''],
      });

      const sliderElements = document.querySelectorAll('input[type="range"]');
      expect(sliderElements.length).toBeGreaterThan(0);

      // Sanity: a pre-dispose input mutation routes through to the manager.
      vi.mocked(sceneDimsManager.setDimensionValue).mockClear();
      const probe = sliderElements[0] as HTMLInputElement;
      probe.value = String(Math.min(900, parseInt(probe.max) - 1));
      probe.dispatchEvent(new Event('input'));
      expect(sceneDimsManager.setDimensionValue).toHaveBeenCalled();

      // Dispose, then probe post-dispose behavior.
      sliders.dispose();
      vi.mocked(sceneDimsManager.setDimensionValue).mockClear();

      sliderElements.forEach((s, i) => {
        const el = s as HTMLInputElement;
        el.value = String(Math.min(700 + i, parseInt(el.max) || 1000));
        el.dispatchEvent(new Event('input'));
        el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
      });

      expect(sceneDimsManager.setDimensionValue).not.toHaveBeenCalled();
    });

    it('post-dispose: dropdown change/keydown no longer routes to setDimensionValue', () => {
      const container = document.getElementById('test-container')!;
      const dims = createMockDims();

      const sliders = new DimensionSliders({
        container,
        dims,
        dimensionRanges: [
          [0, 100],
          [0, 100],
          [0, 100],
          [0, 10],
          [0, 2],
        ],
        dimensionNames: ['X', 'Y', 'Z', 'Time', 'Channel'],
      });

      // Channel dim (< 10 categories) becomes a dropdown.
      const dropdown = document.querySelector('select') as HTMLSelectElement;
      expect(dropdown).toBeTruthy();

      // Sanity: a pre-dispose change routes through to the manager.
      vi.mocked(sceneDimsManager.setDimensionValue).mockClear();
      dropdown.value = '1';
      dropdown.dispatchEvent(new Event('change'));
      expect(sceneDimsManager.setDimensionValue).toHaveBeenCalledWith(4, 1);

      // Dispose, then probe post-dispose behavior.
      sliders.dispose();
      vi.mocked(sceneDimsManager.setDimensionValue).mockClear();

      dropdown.value = '2';
      dropdown.dispatchEvent(new Event('change'));
      dropdown.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
      dropdown.dispatchEvent(new KeyboardEvent('keydown', { key: ']' }));

      expect(sceneDimsManager.setDimensionValue).not.toHaveBeenCalled();
    });

    it('rebuilding sliders detaches old listeners (input on stale slider is a no-op)', () => {
      const container = document.getElementById('test-container')!;
      const dims = createMockDims();

      const sliders = new DimensionSliders({
        container,
        dims,
        dimensionRanges: [
          [0, 100],
          [0, 100],
          [0, 100],
          [0, 10],
          [0, 2],
        ],
        dimensionNames: ['X', 'Y', 'Z', 'Time', 'Channel'],
      });

      const initialSliders = Array.from(
        document.querySelectorAll('input[type="range"]')
      ) as HTMLInputElement[];
      expect(initialSliders.length).toBeGreaterThan(0);

      // Rebuild sliders (simulates dimension change) — old listeners must be cleaned up.
      (sliders as any).createSliders();

      // Confirm a fresh set was produced.
      const rebuiltSliders = document.querySelectorAll('input[type="range"]');
      expect(rebuiltSliders.length).toBeGreaterThan(0);

      // Dispatching on the OLD sliders must not invoke setDimensionValue.
      vi.mocked(sceneDimsManager.setDimensionValue).mockClear();
      initialSliders.forEach((el) => {
        el.value = '500';
        el.dispatchEvent(new Event('input'));
        el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
      });
      expect(sceneDimsManager.setDimensionValue).not.toHaveBeenCalled();

      sliders.dispose();
    });
  });
});

describe('DimensionSliders - Binary Toggle Controls', () => {
  const createBinaryDims = (): SimpleDims => ({
    ndim: 6,
    displayed: [0, 1, 2],
    currentStep: [0, 0, 0, 0, 1, 2],
    metadata: [
      { name: 'X', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Y', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Z', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      {
        name: 'DAPI',
        unit: '',
        scale: 1.0,
        discrete: true,
        step: 1,
        categories: ['Off', 'On'],
      },
      {
        name: 'GFP',
        unit: '',
        scale: 1.0,
        discrete: true,
        step: 1,
        categories: ['Off', 'On'],
      },
      {
        name: 'Channel',
        unit: '',
        scale: 1.0,
        discrete: true,
        step: 1,
        categories: ['DAPI', 'GFP', 'RFP'],
      },
    ],
  });

  it('should render binary categorical dimension as toggle, not dropdown', () => {
    const container = document.getElementById('test-container')!;
    const sliders = new DimensionSliders({
      container,
      dims: createBinaryDims(),
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 1],
        [0, 1],
        [0, 2],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'DAPI', 'GFP', 'Channel'],
    });

    // Binary categoricals should be toggles
    const toggles = document.querySelectorAll('.luxar-dimension-toggle');
    expect(toggles.length).toBe(2);

    // 3-category dimension should remain a dropdown
    const dropdowns = document.querySelectorAll('select');
    expect(dropdowns.length).toBe(1);

    sliders.dispose();
  });

  it('should show current value label and correct visual state', () => {
    const container = document.getElementById('test-container')!;
    const dims = createBinaryDims();
    // DAPI=0 (Off), GFP=1 (On)

    const sliders = new DimensionSliders({
      container,
      dims,
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 1],
        [0, 1],
        [0, 2],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'DAPI', 'GFP', 'Channel'],
    });

    // First toggle (DAPI, currentStep=0 → shows "Off", no --on class)
    const firstToggle = document.getElementById('luxar-dim-toggle-3')!;
    expect(firstToggle.textContent).toBe('Off');
    expect(firstToggle.classList.contains('luxar-dimension-toggle--on')).toBe(false);

    // Second toggle (GFP, currentStep=1 → shows "On", has --on class)
    const secondToggle = document.getElementById('luxar-dim-toggle-4')!;
    expect(secondToggle.textContent).toBe('On');
    expect(secondToggle.classList.contains('luxar-dimension-toggle--on')).toBe(true);

    sliders.dispose();
  });

  it('should render discrete non-categorical numeric dim as a slider (not toggle/dropdown)', () => {
    // Discrete-but-numeric dimensions (time, frame index, an unlabeled flag)
    // are ordinal, not categorical. They must get a scrubbing slider regardless
    // of how few values they have — toggles/dropdowns are reserved for dims with
    // explicit `categories`. This prevents e.g. a small-count `time` dimension
    // from being misrendered as a categorical pick-one control.
    const container = document.getElementById('test-container')!;
    const dims: SimpleDims = {
      ndim: 4,
      displayed: [0, 1, 2],
      currentStep: [0, 0, 0, 0],
      metadata: [
        { name: 'X', unit: '', scale: 1.0 },
        { name: 'Y', unit: '', scale: 1.0 },
        { name: 'Z', unit: '', scale: 1.0 },
        { name: 'Flag', unit: '', scale: 1.0, discrete: true, step: 1 },
      ],
    };

    const sliders = new DimensionSliders({
      container,
      dims,
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 1],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'Flag'],
    });

    // A discrete-numeric dim is a slider, not a toggle or dropdown.
    const toggle = document.querySelector('.luxar-dimension-toggle');
    expect(toggle).toBeNull();
    const dropdowns = document.querySelectorAll('select');
    expect(dropdowns.length).toBe(0);

    const sliderInputs = document.querySelectorAll('input[type="range"]');
    expect(sliderInputs.length).toBe(1);

    sliders.dispose();
  });

  it('should render a small-count discrete time dimension as a slider', () => {
    // Regression: a `time` dimension with only 6 frames used to become a
    // categorical-style dropdown. It must be a scrubbing slider.
    const container = document.getElementById('test-container')!;
    const dims: SimpleDims = {
      ndim: 4,
      displayed: [0, 1, 2],
      currentStep: [0, 0, 0, 0],
      metadata: [
        { name: 'x', unit: '', scale: 1.0 },
        { name: 'y', unit: '', scale: 1.0 },
        { name: 'z', unit: '', scale: 1.0 },
        { name: 'time', unit: '', scale: 1.0, discrete: true, step: 1 },
      ],
    };

    const sliders = new DimensionSliders({
      container,
      dims,
      dimensionRanges: [
        [-6, 6],
        [-6, 6],
        [-6, 6],
        [0, 5],
      ],
      dimensionNames: ['x', 'y', 'z', 'time'],
    });

    expect(document.querySelectorAll('select').length).toBe(0);
    expect(document.querySelector('.luxar-dimension-toggle')).toBeNull();
    expect(document.querySelectorAll('input[type="range"]').length).toBe(1);

    sliders.dispose();
  });

  it('post-dispose: toggle click/keydown no longer routes to setDimensionValue', () => {
    // C3 strengthening (P1): behavioral post-dispose check rather than
    // asserting that the implementation called removeEventListener with
    // specific event-type strings.
    const container = document.getElementById('test-container')!;
    const sliders = new DimensionSliders({
      container,
      dims: createBinaryDims(),
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 1],
        [0, 1],
        [0, 2],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'DAPI', 'GFP', 'Channel'],
    });

    const toggleElements = Array.from(
      document.querySelectorAll('.luxar-dimension-toggle')
    ) as HTMLElement[];
    expect(toggleElements.length).toBe(2);

    // Sanity: clicking a toggle pre-dispose calls setDimensionValue.
    vi.mocked(sceneDimsManager.setDimensionValue).mockClear();
    toggleElements[0].dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(sceneDimsManager.setDimensionValue).toHaveBeenCalled();

    sliders.dispose();
    vi.mocked(sceneDimsManager.setDimensionValue).mockClear();

    toggleElements.forEach((toggle) => {
      toggle.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      toggle.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
      toggle.dispatchEvent(new KeyboardEvent('keydown', { key: ' ' }));
    });

    expect(sceneDimsManager.setDimensionValue).not.toHaveBeenCalled();
  });
});

describe('DimensionSliders — wheel stepping + Step context-menu section', () => {
  const createDims = (): SimpleDims => ({
    ndim: 4,
    displayed: [0, 1, 2],
    currentStep: [0, 0, 0, 5.5],
    metadata: [
      { name: 'X', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Y', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Z', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      // Continuous W with an authored step — the wheel's base quantum.
      { name: 'W', unit: '', scale: 1.0, discrete: false, step: 0.5, range: [0, 10] },
    ],
  });

  function buildSliders() {
    const container = document.getElementById('test-container')!;
    return new DimensionSliders({
      container,
      dims: createDims(),
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 10],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'W'],
      dimensionUnits: ['μm', 'μm', 'μm', ''],
    });
  }

  /** Minimal DimensionAnimationManager stub for the context menu. */
  function makeAnimationManagerStub() {
    return {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      getState: vi.fn(() => undefined),
      isAnimating: vi.fn(() => false),
      getStepSize: vi.fn(() => null),
      setStepSize: vi.fn(),
      setTargetFPS: vi.fn(),
      setLoopMode: vi.fn(),
      play: vi.fn(),
      pause: vi.fn(),
      toggle: vi.fn(),
    };
  }

  it('wheel steps by the BASE step; Shift = ÷10; page scroll prevented', () => {
    const sliders = buildSliders();
    const track = document.querySelector('.luxar-dimension-slider__track')!;

    vi.mocked(sceneDimsManager.setDimensionValue).mockClear();
    const up = new WheelEvent('wheel', { deltaY: -100, bubbles: true, cancelable: true });
    track.dispatchEvent(up);
    // Base = authored step 0.5 (NOT the animation override — decoupled).
    expect(sceneDimsManager.setDimensionValue).toHaveBeenCalledWith(3, 6);
    expect(up.defaultPrevented).toBe(true);

    vi.mocked(sceneDimsManager.setDimensionValue).mockClear();
    const fineDown = new WheelEvent('wheel', {
      deltaY: 100,
      shiftKey: true,
      bubbles: true,
      cancelable: true,
    });
    track.dispatchEvent(fineDown);
    expect(sceneDimsManager.setDimensionValue).toHaveBeenCalledWith(3, 5.45);
    sliders.dispose();
  });

  it('wheel modifier tiers: Ctrl = coarse ×10, Ctrl+Shift = extra-fine ÷100', () => {
    const sliders = buildSliders();
    const track = document.querySelector('.luxar-dimension-slider__track')!;

    vi.mocked(sceneDimsManager.setDimensionValue).mockClear();
    track.dispatchEvent(
      new WheelEvent('wheel', { deltaY: 100, ctrlKey: true, bubbles: true, cancelable: true })
    );
    // Coarse down = authored step 0.5 × 10 → 5.5 − 5.
    expect(sceneDimsManager.setDimensionValue).toHaveBeenCalledWith(3, 0.5);

    vi.mocked(sceneDimsManager.setDimensionValue).mockClear();
    track.dispatchEvent(
      new WheelEvent('wheel', {
        deltaY: -100,
        ctrlKey: true,
        shiftKey: true,
        bubbles: true,
        cancelable: true,
      })
    );
    // Extra-fine up = authored step 0.5 ÷ 100 → 5.5 + 0.005.
    expect(sceneDimsManager.setDimensionValue).toHaveBeenCalledWith(3, expect.closeTo(5.505, 10));
    sliders.dispose();
  });

  it('Shift+wheel arriving on the horizontal axis (browser axis swap) still steps', () => {
    const sliders = buildSliders();
    const track = document.querySelector('.luxar-dimension-slider__track')!;

    // A standard mouse under Shift reports deltaX with deltaY = 0.
    vi.mocked(sceneDimsManager.setDimensionValue).mockClear();
    track.dispatchEvent(
      new WheelEvent('wheel', {
        deltaY: 0,
        deltaX: 100,
        shiftKey: true,
        bubbles: true,
        cancelable: true,
      })
    );
    expect(sceneDimsManager.setDimensionValue).toHaveBeenCalledWith(3, 5.45);

    // A zero-delta wheel event is a no-op, not a step.
    vi.mocked(sceneDimsManager.setDimensionValue).mockClear();
    track.dispatchEvent(new WheelEvent('wheel', { deltaY: 0, bubbles: true, cancelable: true }));
    expect(sceneDimsManager.setDimensionValue).not.toHaveBeenCalled();
    sliders.dispose();
  });

  it('wheel does not reach the window (stopPropagation guards the FOV handler)', () => {
    const sliders = buildSliders();
    const track = document.querySelector('.luxar-dimension-slider__track')!;
    const windowSpy = vi.fn();
    window.addEventListener('wheel', windowSpy);
    track.dispatchEvent(new WheelEvent('wheel', { deltaY: -100, bubbles: true, cancelable: true }));
    expect(windowSpy).not.toHaveBeenCalled();
    window.removeEventListener('wheel', windowSpy);
    sliders.dispose();
  });

  it('post-dispose: wheel no longer routes to setDimensionValue', () => {
    const sliders = buildSliders();
    const track = document.querySelector('.luxar-dimension-slider__track')!;
    sliders.dispose();
    vi.mocked(sceneDimsManager.setDimensionValue).mockClear();
    track.dispatchEvent(new WheelEvent('wheel', { deltaY: -100, bubbles: true, cancelable: true }));
    expect(sceneDimsManager.setDimensionValue).not.toHaveBeenCalled();
  });

  it('context menu gains a Step section: Auto selected, presets call setStepSize', () => {
    const sliders = buildSliders();
    const stub = makeAnimationManagerStub();
    sliders.setAnimationManager(stub as never);

    const playBtn = document.querySelector('.luxar-dimension-slider__play-btn')!;
    playBtn.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));

    const headers = Array.from(
      document.querySelectorAll('.luxar-dimension-slider__context-header')
    ).map((h) => h.textContent);
    expect(headers).toContain('Step');
    // The E2E-pinned headers keep their exact text.
    expect(headers).toContain('Speed');
    expect(headers).toContain('Loop Mode');

    const items = Array.from(document.querySelectorAll('.luxar-dimension-slider__context-item'));
    const auto = items.find((el) => el.textContent?.includes('Auto'))!;
    expect(auto.classList.contains('luxar-dimension-slider__context-item--selected')).toBe(true);

    // ×2 of base 0.5 → 1.
    const x2 = items.find((el) => el.textContent?.includes('×2'))!;
    expect(x2.textContent).toContain('(1)');
    (x2 as HTMLElement).click();
    expect(stub.setStepSize).toHaveBeenCalledWith(3, 1);
    sliders.dispose();
  });

  it('custom step input commits on Enter; invalid values do not', () => {
    const sliders = buildSliders();
    const stub = makeAnimationManagerStub();
    sliders.setAnimationManager(stub as never);
    document
      .querySelector('.luxar-dimension-slider__play-btn')!
      .dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));

    const input = document.querySelector<HTMLInputElement>(
      '.luxar-dimension-slider__context-step-input'
    )!;
    expect(input.type).toBe('number'); // NEVER type=range (E2E slider indexing)

    input.value = '-3';
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(stub.setStepSize).not.toHaveBeenCalled();

    input.value = '0.75';
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(stub.setStepSize).toHaveBeenCalledWith(3, 0.75);
    sliders.dispose();
  });
});
