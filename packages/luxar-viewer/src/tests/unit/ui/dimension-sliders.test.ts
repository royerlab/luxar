// @vitest-environment jsdom
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
import {
  resetInputProfileForTests,
  setInputProfileOverride,
} from '../../../utils/input-capabilities';

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

describe('DimensionSliders - keyboard selection indicator', () => {
  const dims: SimpleDims = {
    ndim: 5,
    displayed: [0, 1, 2],
    currentStep: [0, 0, 0, 7, 1],
    metadata: [
      { name: 'X', unit: '', scale: 1, discrete: false, step: 1 },
      { name: 'Y', unit: '', scale: 1, discrete: false, step: 1 },
      { name: 'Z', unit: '', scale: 1, discrete: false, step: 1 },
      { name: 'Frame', unit: '', scale: 1, discrete: true, step: 1 },
      {
        name: 'Channel',
        unit: '',
        scale: 1,
        discrete: true,
        step: 1,
        categories: ['RED', 'GREEN', 'BLUE'],
      },
    ],
  };

  function buildSliders(): DimensionSliders {
    return new DimensionSliders({
      container: document.getElementById('test-container')!,
      dims,
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 15],
        [0, 2],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'Frame', 'Channel'],
      selectedDimension: 0,
    });
  }

  it('shows the selected navigable key and dimension name in the panel header', () => {
    const sliders = buildSliders();
    const status = document.querySelector<HTMLElement>('.luxar-dimension-sliders__status');

    expect(status?.textContent).toBe('[/]: 1 · Frame · Display: X, Y, Z');
    expect(status?.title).toBe(status?.textContent);
    expect(status?.getAttribute('aria-live')).toBe('polite');

    sliders.setSelectedDimension(1);
    expect(status?.textContent).toBe('[/]: 2 · Channel · Display: X, Y, Z');
    expect(status?.title).toBe(status?.textContent);
    sliders.dispose();
  });

  it('does not rewrite an unchanged live-region status during dimension updates', async () => {
    const sliders = buildSliders();
    const status = document.querySelector<HTMLElement>('.luxar-dimension-sliders__status')!;
    const mutations: MutationRecord[] = [];
    const observer = new MutationObserver((records) => mutations.push(...records));
    observer.observe(status, {
      attributes: true,
      attributeFilter: ['title'],
      childList: true,
    });

    sliders.update();
    await Promise.resolve();

    expect(mutations).toEqual([]);
    observer.disconnect();
    sliders.dispose();
  });

  it('shows an unavailable target instead of silently naming another dimension', () => {
    const sliders = buildSliders();
    const status = document.querySelector('.luxar-dimension-sliders__status');

    sliders.setSelectedDimension(9);
    expect(status?.textContent).toContain('[/]: unavailable');
    expect(status?.textContent).not.toContain('[/]: 10 · Channel');
    sliders.dispose();
  });

  // A slider row's name is now CSS-ellipsised once it would claim the value's
  // reserved width, so the full text has to survive somewhere. jsdom does no
  // layout and therefore cannot see the truncation itself — what it CAN pin is
  // that nothing is left unrecoverable, which is the half that regressed when
  // the ellipsis was added.
  it('gives every slider name a tooltip carrying its full text', () => {
    const sliders = buildSliders();
    const names = Array.from(
      document.querySelectorAll<HTMLElement>('.luxar-dimension-slider__name')
    );

    expect(names.length).toBeGreaterThan(0);
    for (const name of names) {
      expect(name.title).toBe(name.textContent);
    }
    sliders.dispose();
  });

  /**
   * The tooltip text and the `--with-tooltip` dotted underline are two
   * decisions off the SAME fact, so they are asserted together: the underline
   * advertises that hovering reveals something new, and must not appear on a
   * fallback tooltip that merely repeats the visible name.
   *
   * The empty-string row is the one that matters. Compilers write
   * `description: ''` rather than omitting the key — the shipped
   * `dimension_sliders_5d_example` does — and reading it with `??` (which falls
   * through on null/undefined only) put `title=""` on every row of the real
   * viewer: the class ternary saw no description, the tooltip used it anyway,
   * and the full name became unrecoverable precisely where the CSS had begun
   * truncating it. Nothing else in the suite covers a falsy-but-present
   * description.
   */
  it.each([
    { label: 'no description key', descriptionMetadata: {}, title: 'Frame', underlined: false },
    {
      label: 'an empty description',
      descriptionMetadata: { description: '' },
      title: 'Frame',
      underlined: false,
    },
    {
      label: 'an authored description',
      descriptionMetadata: { description: 'Acquisition frame index' },
      title: 'Acquisition frame index',
      underlined: true,
    },
  ])('with $label the name reads title "$title"', ({ descriptionMetadata, title, underlined }) => {
    const described: SimpleDims = {
      ...dims,
      metadata: dims.metadata!.map((meta, index) =>
        index === 3 ? { ...meta, ...descriptionMetadata } : meta
      ),
    };
    const sliders = new DimensionSliders({
      container: document.getElementById('test-container')!,
      dims: described,
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 15],
        [0, 2],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'Frame', 'Channel'],
      selectedDimension: 0,
    });

    const frame = document.querySelector<HTMLElement>('.luxar-dimension-slider__name')!;
    expect(frame.textContent).toBe('Frame');
    expect(frame.title).toBe(title);
    expect(frame.className.includes('luxar-dimension-slider__name--with-tooltip')).toBe(underlined);
    sliders.dispose();
  });
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

  it('keeps authored categorical labels unchanged when metadata also has units', () => {
    const dims: SimpleDims = {
      ndim: 5,
      displayed: [0, 1, 2],
      currentStep: [0, 0, 0, 1, 2],
      metadata: [
        { name: 'X', unit: '', scale: 1.0 },
        { name: 'Y', unit: '', scale: 1.0 },
        { name: 'Z', unit: '', scale: 1.0 },
        { name: 'Phase', unit: 's', scale: 1.0, discrete: true, categories: ['0', '1'] },
        {
          name: 'Exposure',
          unit: 'ms',
          scale: 1.0,
          discrete: true,
          categories: ['0', '5', '10'],
        },
      ],
    };

    const sliders = new DimensionSliders({
      container: document.getElementById('test-container')!,
      dims,
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 1],
        [0, 2],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'Phase', 'Exposure'],
      dimensionUnits: ['', '', '', 's', 'ms'],
    });

    const toggle = document.getElementById('luxar-dim-toggle-3')!;
    expect(toggle.textContent).toBe('1');
    expect(toggle.title).toContain('1');
    expect(toggle.title).not.toContain('1 s');

    const options = Array.from(
      document.querySelectorAll<HTMLOptionElement>('#luxar-dim-dropdown-4 option')
    );
    expect(options.map((option) => option.textContent)).toEqual(['0', '5', '10']);
    expect(options[2].title).toContain('10');
    expect(options[2].title).not.toContain('10 ms');

    const labels = Array.from(
      document.querySelectorAll<HTMLElement>('.luxar-dimension-dropdown__label')
    );
    expect(labels.map((label) => label.title)).toEqual(['Phase', 'Exposure']);

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
      getStepSize: vi.fn((): number | null => null),
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

  it('wheel bubbles to the window after the local step is applied', () => {
    const sliders = buildSliders();
    const track = document.querySelector('.luxar-dimension-slider__track')!;
    const windowSpy = vi.fn();
    window.addEventListener('wheel', windowSpy);
    vi.mocked(sceneDimsManager.setDimensionValue).mockClear();
    track.dispatchEvent(new WheelEvent('wheel', { deltaY: -100, bubbles: true, cancelable: true }));
    expect(windowSpy).toHaveBeenCalledTimes(1);
    expect(sceneDimsManager.setDimensionValue).toHaveBeenCalledWith(3, 6);
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

    // ×2 of base 0.5 → 1; the computed value lives in the chip tooltip.
    const x2 = items.find((el) => el.textContent?.includes('×2'))!;
    expect((x2 as HTMLElement).title).toContain('= 1');
    (x2 as HTMLElement).click();
    expect(stub.setStepSize).toHaveBeenCalledWith(3, 1);
    sliders.dispose();
  });

  it('does not own document Escape while the animation context menu is open', () => {
    const sliders = buildSliders();
    sliders.setAnimationManager(makeAnimationManagerStub() as never);
    document
      .querySelector('.luxar-dimension-slider__play-btn')!
      .dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    expect(document.querySelector('.luxar-dimension-slider__context-menu')).not.toBeNull();
    sliders.closeContextMenu();
    expect(document.querySelector('.luxar-dimension-slider__context-menu')).toBeNull();
    sliders.dispose();
  });

  it('speed chips: sub-1 fps reads as a fraction; aria-checked tracks the radio state', () => {
    const sliders = buildSliders();
    const stub = makeAnimationManagerStub();
    sliders.setAnimationManager(stub as never);
    document
      .querySelector('.luxar-dimension-slider__play-btn')!
      .dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));

    const speedGroup = document.querySelector('[role="radiogroup"][aria-label="Speed"]')!;
    const chips = Array.from(speedGroup.querySelectorAll('button'));
    const half = chips.find((c) => c.textContent === '1/2')!;
    expect(half).toBeTruthy();
    expect(half.title).toBe('1 frame every 2 s');

    // Default 10 FPS is the checked radio; 1/2 is not.
    const ten = chips.find((c) => c.textContent === '10')!;
    expect(ten.getAttribute('aria-checked')).toBe('true');
    expect(half.getAttribute('aria-checked')).toBe('false');

    // Picking the fraction chip routes the REAL 0.5 to the manager.
    half.click();
    expect(stub.setTargetFPS).toHaveBeenCalledWith(3, 0.5);
    sliders.dispose();
  });

  it('custom step field: blur without editing never re-commits the 3-digit display form', () => {
    const sliders = buildSliders();
    const stub = makeAnimationManagerStub();
    // A full-precision override that matches no preset → seeds the custom field.
    stub.getStepSize = vi.fn(() => 0.123456);
    sliders.setAnimationManager(stub as never);

    const playBtn = document.querySelector('.luxar-dimension-slider__play-btn')!;
    playBtn.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));

    const input = document.querySelector(
      'input[aria-label="Custom step size"]'
    ) as HTMLInputElement;
    expect(input.value).toBe('0.123'); // seeded with the truncated display form

    // Focus-then-leave with no edit must NOT rewrite 0.123456 → 0.123.
    input.dispatchEvent(new Event('blur'));
    expect(stub.setStepSize).not.toHaveBeenCalled();

    // An actual edit still commits on blur.
    input.value = '0.2';
    input.dispatchEvent(new Event('input'));
    input.dispatchEvent(new Event('blur'));
    expect(stub.setStepSize).toHaveBeenCalledWith(3, 0.2);
    sliders.dispose();
  });

  it('a discrete dim offers only whole-cell step presets (#1520)', () => {
    const container = document.getElementById('test-container')!;
    const dims = createDims();
    // Make W discrete: the grid cannot honour a sub-cell quantum, so the
    // ×0.1 / ×0.25 / ×0.5 presets must not be offered.
    dims.metadata![3] = {
      name: 'W',
      unit: '',
      scale: 1.0,
      discrete: true,
      step: 1,
      range: [0, 10],
    };
    const sliders = new DimensionSliders({
      container,
      dims,
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 10],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'W'],
      dimensionUnits: ['μm', 'μm', 'μm', ''],
    });
    const stub = makeAnimationManagerStub();
    sliders.setAnimationManager(stub as never);

    const playBtn = document.querySelector('.luxar-dimension-slider__play-btn')!;
    playBtn.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));

    const labels = Array.from(
      document.querySelectorAll('.luxar-dimension-slider__context-item')
    ).map((el) => el.textContent ?? '');
    expect(labels.some((t) => t.includes('×1'))).toBe(true);
    expect(labels.some((t) => t.includes('×0.1'))).toBe(false);
    expect(labels.some((t) => t.includes('×0.25'))).toBe(false);
    expect(labels.some((t) => t.includes('×0.5'))).toBe(false);
    sliders.dispose();
  });

  it('a discrete dim with NO authored step keeps sub-1 multipliers whose value reaches a cell', () => {
    const container = document.getElementById('test-container')!;
    const dims = createDims();
    // No authored step → the menu's base falls back to 1% of the range
    // (10), while the snap grid defaults to 1. ×0.1 of 10 = 1 = one full
    // cell — a perfectly honorable quantum that must NOT be filtered.
    dims.metadata![3] = { name: 'W', unit: '', scale: 1.0, discrete: true, range: [0, 1000] };
    const sliders = new DimensionSliders({
      container,
      dims,
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 1000],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'W'],
      dimensionUnits: ['μm', 'μm', 'μm', ''],
    });
    const stub = makeAnimationManagerStub();
    sliders.setAnimationManager(stub as never);

    const playBtn = document.querySelector('.luxar-dimension-slider__play-btn')!;
    playBtn.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));

    const labels = Array.from(
      document.querySelectorAll('.luxar-dimension-slider__context-item')
    ).map((el) => el.textContent ?? '');
    expect(labels.some((t) => t.includes('×0.1'))).toBe(true); // 10 × 0.1 = 1 cell
    expect(labels.some((t) => t.includes('×0.5'))).toBe(true); // 10 × 0.5 = 5 cells
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
    // And NEVER inside the chips radiogroup — a radio group may only own
    // radios, and this is a spinbutton.
    expect(input.closest('[role="radiogroup"]')).toBeNull();
    expect(input.closest('.luxar-dimension-slider__context-section')).not.toBeNull();

    // Real typing fires an `input` event — the commit is gated on it (an
    // un-edited field must never re-commit its truncated display seed).
    input.value = '-3';
    input.dispatchEvent(new Event('input'));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(stub.setStepSize).not.toHaveBeenCalled();

    input.value = '0.75';
    input.dispatchEvent(new Event('input'));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(stub.setStepSize).toHaveBeenCalledWith(3, 0.75);
    sliders.dispose();
  });
});

// #1483: the panel root is a glass surface, so it must stay
// `overflow: visible` and delegate scrolling to an inner `__scroll` wrapper
// (UI_DESIGN_GUIDE §5.1.2/§7.4). Two invariants keep that working, and neither
// was pinned before: all content lives in the wrapper, and the root's `display`
// is never forced inline (the stylesheet's `display: flex` is what lets the
// wrapper's `flex: 1; min-height: 0` cap content at the panel's max-height —
// with an inline `display: block` the wrapper stops clipping and, since the
// root is `overflow: visible`, the slider rows paint outside the panel).
describe('DimensionSliders — glass-root scroll delegation and visibility (#1483)', () => {
  const createDims = (): SimpleDims => ({
    ndim: 6,
    displayed: [0, 1, 2],
    currentStep: [0, 0, 0, 1, 2, 3],
    metadata: [
      { name: 'X', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Y', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Z', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'T', unit: 's', scale: 1.0, discrete: true, step: 1 },
      { name: 'U', unit: '', scale: 1.0, discrete: true, step: 1 },
      { name: 'V', unit: '', scale: 1.0, discrete: true, step: 1 },
    ],
  });

  function buildSliders() {
    return new DimensionSliders({
      container: document.getElementById('test-container')!,
      dims: createDims(),
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 10],
        [0, 10],
        [0, 10],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'T', 'U', 'V'],
    });
  }

  const root = (): HTMLElement => document.getElementById('luxar-dimension-sliders')!;

  it('routes every piece of panel content into the __scroll wrapper', () => {
    const sliders = buildSliders();

    const scroll = root().querySelector('.luxar-dimension-sliders__scroll');
    expect(scroll).not.toBeNull();

    const header = document.querySelector('.luxar-dimension-sliders__header')!;
    expect(scroll!.contains(header)).toBe(true);

    const rows = document.querySelectorAll('.luxar-dimension-slider');
    expect(rows.length).toBe(3); // T, U, V — the non-displayed dims
    rows.forEach((row) => expect(scroll!.contains(row)).toBe(true));

    sliders.dispose();
  });

  it('leaves the glass root with no content children of its own', () => {
    const sliders = buildSliders();

    // Only the wrapper — plus, if the liquid-glass theme has injected it, the
    // `.luxar-glass-refraction` layer, which the theme owns.
    const unexpected = Array.from(root().children).filter(
      (el) =>
        !el.classList.contains('luxar-dimension-sliders__scroll') &&
        !el.classList.contains('luxar-glass-refraction')
    );
    expect(unexpected).toEqual([]);

    sliders.dispose();
  });

  it('shows by CLEARING inline display, never by writing block', () => {
    const sliders = buildSliders();

    // An inline `display: block` would override the stylesheet's flex column
    // and un-cap the wrapper, spilling slider rows out of the panel.
    sliders.setVisible(true);
    expect(root().style.display).toBe('');
    expect(sliders.getIsVisible()).toBe(true);

    sliders.setVisible(false);
    expect(root().style.display).toBe('none');
    expect(sliders.getIsVisible()).toBe(false);

    // toggle() back to visible must clear the property too, not write 'block'.
    sliders.toggle();
    expect(root().style.display).toBe('');
    expect(sliders.getIsVisible()).toBe(true);

    sliders.toggle();
    expect(root().style.display).toBe('none');
    expect(sliders.getIsVisible()).toBe(false);

    sliders.hide();
    expect(root().style.display).toBe('none');
    expect(sliders.getIsVisible()).toBe(false);

    sliders.dispose();
  });
});

describe('DimensionSliders - coarse pointer affordances', () => {
  const dims: SimpleDims = {
    ndim: 5,
    displayed: [0, 1, 2],
    currentStep: [0, 0, 0, 7, 1],
    metadata: [
      { name: 'X', unit: '', scale: 1, discrete: false, step: 1 },
      { name: 'Y', unit: '', scale: 1, discrete: false, step: 1 },
      { name: 'Z', unit: '', scale: 1, discrete: false, step: 1 },
      { name: 'Frame', unit: '', scale: 1, discrete: true, step: 1 },
      { name: 'Depth', unit: 'um', scale: 1, discrete: false, step: 0.5 },
    ],
  };
  const build = (onSelectDimension?: (i: number) => void): DimensionSliders =>
    new DimensionSliders({
      container: document.getElementById('test-container')!,
      dims,
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 15],
        [0, 10],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'Frame', 'Depth'],
      selectedDimension: 0,
      onSelectDimension,
    });

  afterEach(() => resetInputProfileForTests());

  it('adds ◀ ▶ step buttons that move the dimension by one base step', () => {
    setInputProfileOverride('touch');
    const sliders = build();
    const frameSteps = Array.from(
      document.querySelectorAll<HTMLButtonElement>('.luxar-dimension-slider__step')
    ).filter((b) => b.getAttribute('aria-label')?.endsWith('Frame'));
    expect(frameSteps.map((b) => b.getAttribute('aria-label'))).toEqual([
      'Previous Frame',
      'Next Frame',
    ]);
    frameSteps[1].click();
    expect(sceneDimsManager.setDimensionValue).toHaveBeenLastCalledWith(3, 8);
    frameSteps[0].click();
    expect(sceneDimsManager.setDimensionValue).toHaveBeenLastCalledWith(3, 6);
    sliders.dispose();
  });

  it('uses the animation Step override and cyclic wrap used by [ / ]', () => {
    setInputProfileOverride('touch');
    dims.currentStep[3] = 15;
    dims.metadata![3].cyclic = true;
    const sliders = build();
    sliders.setAnimationManager({
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      getState: vi.fn(() => undefined),
      isAnimating: vi.fn(() => false),
      getStepSize: vi.fn(() => 2),
      play: vi.fn(),
      pause: vi.fn(),
    } as never);
    vi.mocked(sceneDimsManager.setDimensionValue).mockClear();

    const nextFrame = Array.from(
      document.querySelectorAll<HTMLButtonElement>('.luxar-dimension-slider__step')
    ).find((button) => button.getAttribute('aria-label') === 'Next Frame')!;
    nextFrame.click();

    expect(sceneDimsManager.setDimensionValue).toHaveBeenCalledWith(3, 1);
    dims.currentStep[3] = 7;
    dims.metadata![3].cyclic = false;
    sliders.dispose();
  });

  it('adds a distinct play control after the coarse step controls', () => {
    setInputProfileOverride('touch');
    const sliders = build();
    sliders.setAnimationManager({
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      getState: vi.fn(() => undefined),
      isAnimating: vi.fn(() => false),
      getStepSize: vi.fn(() => null),
      play: vi.fn(),
      pause: vi.fn(),
    } as never);

    const frameWrapper = document
      .querySelector<HTMLInputElement>('#luxar-dim-slider-3')!
      .closest('.luxar-dimension-slider__controls-wrapper')!;
    expect(Array.from(frameWrapper.children).map((child) => child.className)).toEqual([
      'luxar-dimension-slider__step',
      'luxar-dimension-slider__track',
      'luxar-dimension-slider__step',
      'luxar-dimension-slider__play-btn',
    ]);
    expect(frameWrapper.lastElementChild?.textContent).toBe('⏵');
    sliders.dispose();
  });

  it('positions the thumb using its rendered coarse-pointer width', () => {
    setInputProfileOverride('touch');
    const sliders = build();
    const thumb = document.getElementById('luxar-dim-thumb-3')!;
    Object.defineProperty(thumb.parentElement!, 'offsetWidth', { value: 304 });
    Object.defineProperty(thumb, 'offsetWidth', { value: 22 });

    dims.currentStep[3] = 15;
    sliders.update();

    expect(thumb.style.left).toBe('282px');
    dims.currentStep[3] = 7;
    sliders.dispose();
  });

  it('makes the dimension name a chip that selects the [ / ] target', () => {
    setInputProfileOverride('touch');
    const onSelect = vi.fn();
    const sliders = build(onSelect);
    const depthName = Array.from(
      document.querySelectorAll<HTMLElement>('.luxar-dimension-slider__name')
    ).find((el) => el.textContent === 'Depth')!;
    expect(depthName.classList.contains('luxar-dimension-slider__name--chip')).toBe(true);
    expect(depthName.getAttribute('role')).toBe('button');
    depthName.click();
    // Depth is the second non-displayed dimension (Frame, Depth) → index 1.
    expect(onSelect).toHaveBeenCalledWith(1);
    expect(document.querySelector('.luxar-dimension-sliders__status')?.textContent ?? '').toContain(
      '[/]: 2'
    );
    sliders.dispose();
  });

  it('adds neither on a mouse machine', () => {
    setInputProfileOverride('mouse');
    const sliders = build();
    expect(document.querySelectorAll('.luxar-dimension-slider__step').length).toBe(0);
    expect(document.querySelectorAll('.luxar-dimension-slider__name--chip').length).toBe(0);
    sliders.dispose();
  });
});
