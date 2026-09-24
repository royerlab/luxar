// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { LabeledSlider } from '../../../../ui/layers/labeled-slider';
import { absorptionSliderRange } from '../../../../ui/layers/absorption-range';

describe('LabeledSlider', () => {
  let container: HTMLElement;

  beforeEach(() => {
    document.body.innerHTML = '';
    container = document.createElement('div');
    document.body.appendChild(container);
  });

  function findInput(): HTMLInputElement {
    return container.querySelector('input[type="range"]') as HTMLInputElement;
  }

  function findReadout(): HTMLElement {
    return container.querySelector('.luxar-layers-panel__control-value') as HTMLElement;
  }

  it('renders a labeled range input with initial value formatted', () => {
    new LabeledSlider({
      container,
      label: 'Gamma',
      min: 0.2,
      max: 5.0,
      step: 0.01,
      initialValue: 1.5,
      onChange: () => {},
    });

    const labelText = container.querySelector(
      '.luxar-layers-panel__control-label > span:first-child'
    );
    expect(labelText?.textContent).toBe('Gamma');

    const input = findInput();
    expect(input.min).toBe('0.2');
    expect(input.max).toBe('5');
    expect(input.step).toBe('0.0001');
    expect(input.dataset.baseStep).toBe('0.01');
    expect(input.value).toBe('1.5');
    expect(input.getAttribute('aria-label')).toBe('Gamma slider');

    expect(findReadout().textContent).toBe('1.50');
    expect(findReadout().getAttribute('aria-label')).toBe('Gamma value');
    expect(findReadout().getAttribute('role')).toBe('button');
  });

  it('supports tiered wheel and arrow stepping, reset, and base-grid dragging', () => {
    const onChange = vi.fn();
    new LabeledSlider({
      container,
      label: 'Gamma',
      min: 0.2,
      max: 5,
      step: 0.01,
      initialValue: 1,
      onChange,
    });

    const input = findInput();
    input.dispatchEvent(
      new WheelEvent('wheel', {
        deltaX: -120,
        deltaY: 0,
        shiftKey: true,
        bubbles: true,
        cancelable: true,
      })
    );
    expect(input.valueAsNumber).toBeCloseTo(1.001, 9);
    expect(findReadout().textContent).toBe('1.0010');

    input.dispatchEvent(
      new KeyboardEvent('keydown', {
        key: 'ArrowRight',
        ctrlKey: true,
        shiftKey: true,
        bubbles: true,
        cancelable: true,
      })
    );
    expect(input.valueAsNumber).toBeCloseTo(1.0011, 9);

    input.value = '1.006';
    input.dispatchEvent(new Event('input'));
    expect(input.valueAsNumber).toBeCloseTo(1.01, 9);

    input.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(input.valueAsNumber).toBe(1);
    expect(onChange).toHaveBeenLastCalledWith(1);
  });

  it('ignores slider gestures while inert', () => {
    const onChange = vi.fn();
    const slider = new LabeledSlider({
      container,
      label: 'Clearcoat roughness',
      min: 0,
      max: 1,
      step: 0.01,
      initialValue: 0.5,
      onChange,
    });
    slider.setInert('Clearcoat is disabled');

    findInput().dispatchEvent(
      new WheelEvent('wheel', { deltaY: -120, bubbles: true, cancelable: true })
    );

    expect(findInput().valueAsNumber).toBe(0.5);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('commits exact typed values and widens the track when needed', () => {
    const onChange = vi.fn();
    const slider = new LabeledSlider({
      container,
      label: 'Gamma',
      min: 0.2,
      max: 5,
      step: 0.01,
      initialValue: 1,
      onChange,
    });

    findReadout().click();
    const editor = container.querySelector('.luxar-slider-kit__inline-input') as HTMLInputElement;
    editor.value = '6.125';
    editor.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));

    expect(slider.getRange()).toEqual([0.2, 6.125]);
    expect(findInput().valueAsNumber).toBeCloseTo(6.125, 9);
    expect(onChange).toHaveBeenLastCalledWith(6.125);
  });

  it('fires onChange with constrained value on input event', () => {
    const onChange = vi.fn();
    new LabeledSlider({
      container,
      label: 'Gamma',
      min: 0.2,
      max: 5.0,
      step: 0.01,
      initialValue: 1.0,
      // Constrain clamps to [0.5, 4.0] for the test
      constrain: (v) => Math.max(0.5, Math.min(4.0, v)),
      onChange,
    });

    const input = findInput();
    input.value = '10';
    input.dispatchEvent(new Event('input'));

    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith(4.0); // clamped
    expect(findReadout().textContent).toBe('4.00');
  });

  it('uses custom format when provided', () => {
    new LabeledSlider({
      container,
      label: 'Opacity',
      min: 0,
      max: 1,
      step: 0.01,
      initialValue: 0.75,
      format: (v) => `${(v * 100).toFixed(0)}%`,
      onChange: () => {},
    });

    expect(findReadout().textContent).toBe('75%');

    const input = findInput();
    input.value = '0.42';
    input.dispatchEvent(new Event('input'));

    expect(findReadout().textContent).toBe('42%');
  });

  it('setValue updates input + readout without firing onChange', () => {
    const onChange = vi.fn();
    const slider = new LabeledSlider({
      container,
      label: 'Gamma',
      min: 0.2,
      max: 5.0,
      step: 0.01,
      initialValue: 1.0,
      onChange,
    });

    slider.setValue(2.5);

    expect(findInput().value).toBe('2.5');
    expect(findReadout().textContent).toBe('2.50');
    expect(onChange).not.toHaveBeenCalled();
  });

  it('passes raw value to onChange when no constrain is given', () => {
    const onChange = vi.fn();
    new LabeledSlider({
      container,
      label: 'Opacity',
      min: 0,
      max: 1,
      step: 0.01,
      initialValue: 0,
      onChange,
    });

    const input = findInput();
    input.value = '0.6';
    input.dispatchEvent(new Event('input'));

    expect(onChange).toHaveBeenCalledWith(0.6);
  });

  it('dispose removes the slider from DOM and detaches the input listener', () => {
    const onChange = vi.fn();
    const slider = new LabeledSlider({
      container,
      label: 'Gamma',
      min: 0.2,
      max: 5.0,
      step: 0.01,
      initialValue: 1.0,
      onChange,
    });

    expect(container.children.length).toBe(1);

    const input = findInput();
    slider.dispose();

    expect(container.children.length).toBe(0);

    // Listener detached: dispatching after dispose does not call onChange.
    // (The element is detached, but the handler reference is still gone too.)
    input.value = '2.0';
    input.dispatchEvent(new Event('input'));
    expect(onChange).not.toHaveBeenCalled();
  });

  describe('log scale', () => {
    /** A 4-decade track, the shape the Absorption slider uses. */
    function makeLog(onChange = vi.fn(), initialValue = 1) {
      const slider = new LabeledSlider({
        container,
        label: 'Absorption',
        min: 0.001,
        max: 10,
        step: 0.05, // ignored on a log track
        scale: 'log',
        initialValue,
        rangeForValue: absorptionSliderRange,
        onChange,
      });
      return { slider, onChange };
    }

    /**
     * Track geometry: position 0 is the dedicated zero stop, and
     * `[GAP, 1]` maps geometrically onto `[min, max]` (GAP = one step).
     */
    const GAP = 0.001;
    const pos = (t: number) => GAP + t * (1 - GAP);

    it('drives the DOM input in normalised position space', () => {
      makeLog();
      const input = findInput();
      expect(input.min).toBe('0');
      expect(input.max).toBe('1');
      // κ=1 on a 0.001–10 track sits 3/4 along the geometric span.
      expect(parseFloat(input.value)).toBeCloseTo(pos(0.75), 6);
    });

    it('maps positions geometrically and round-trips through setValue', () => {
      const { slider, onChange } = makeLog();
      const input = findInput();

      for (const [t, expected] of [
        [0.25, 0.01],
        [0.5, 0.1],
        [1, 10],
      ] as const) {
        input.value = String(pos(t));
        input.dispatchEvent(new Event('input'));
        expect(onChange).toHaveBeenLastCalledWith(expect.closeTo(expected, 6));
      }

      slider.setValue(0.1);
      expect(parseFloat(input.value)).toBeCloseTo(pos(0.5), 6);
    });

    it('reserves position 0 for an exact zero (the additive limit), not `min`', () => {
      const { onChange } = makeLog();
      const input = findInput();

      input.value = '0';
      input.dispatchEvent(new Event('input'));

      expect(onChange).toHaveBeenCalledWith(0);
      expect(findReadout().textContent).toBe('0.00');
    });

    it('a value AT the track floor survives being touched (does not collapse to 0)', () => {
      // Regression: with the zero stop shared with `min`, `setValue(min)`
      // parked the thumb at position 0, so the next `input` event — even a
      // click that moves nothing — pushed 0 instead of `min`.
      const { slider, onChange } = makeLog();
      const input = findInput();

      slider.setValue(0.001); // exactly the track minimum
      expect(parseFloat(input.value)).toBeCloseTo(GAP, 9);

      input.dispatchEvent(new Event('input'));
      expect(onChange).toHaveBeenLastCalledWith(expect.closeTo(0.001, 12));
    });

    it('setRange re-scales the track without moving the value', () => {
      const { slider, onChange } = makeLog();
      const input = findInput();

      slider.setValue(1);
      // A thin-geometry layer: same κ, a track that now reaches 10^4.
      slider.setRange(1, 10000);

      expect(slider.getRange()).toEqual([1, 10000]);
      // κ=1 is now the LEFT end of the geometric span (one step in from the
      // zero stop), and still κ=1.
      expect(parseFloat(input.value)).toBeCloseTo(GAP, 9);
      expect(findReadout().textContent).toBe('1.00');
      expect(onChange).not.toHaveBeenCalled();

      // The far end of the widened track now reaches the large κ a thin
      // line needs — unreachable on the original 0.001–10 track.
      input.value = '1';
      input.dispatchEvent(new Event('input'));
      expect(onChange).toHaveBeenLastCalledWith(expect.closeTo(10000, 3));
    });

    it('uses the live range when double-click resetting', () => {
      const { slider, onChange } = makeLog();
      const input = findInput();
      slider.setRange(0.0001, 10);

      input.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));

      expect(onChange).toHaveBeenLastCalledWith(1);
      expect(parseFloat(input.value)).toBeCloseTo(pos(0.8), 6);
    });

    it('keeps typed zero and extreme values within the absorption range policy', () => {
      const { slider, onChange } = makeLog();

      findReadout().click();
      let editor = container.querySelector('.luxar-slider-kit__inline-input') as HTMLInputElement;
      editor.value = '0';
      editor.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
      expect(slider.getRange()).toEqual([0.001, 10]);
      expect(onChange).toHaveBeenLastCalledWith(0);

      findReadout().click();
      editor = container.querySelector('.luxar-slider-kit__inline-input') as HTMLInputElement;
      editor.value = '1e12';
      editor.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
      const range = absorptionSliderRange(1e12);
      expect(slider.getRange()).toEqual([range.min, range.max]);

      const input = findInput();
      input.value = '0.5';
      input.dispatchEvent(new Event('input'));
      expect(onChange.mock.lastCall?.[0]).toBeGreaterThan(0);
    });

    it('a linear track keeps driving the input in value space (unchanged default)', () => {
      const slider = new LabeledSlider({
        container,
        label: 'Gamma',
        min: 0.2,
        max: 5,
        step: 0.01,
        initialValue: 1,
        onChange: vi.fn(),
      });
      expect(findInput().value).toBe('1');
      slider.setRange(0.5, 2);
      expect(findInput().min).toBe('0.5');
      expect(findInput().max).toBe('2');
      expect(findInput().value).toBe('1');
    });
  });

  it('allows multiple sliders to coexist in the same container', () => {
    const onGamma = vi.fn();
    const onOpacity = vi.fn();

    new LabeledSlider({
      container,
      label: 'Gamma',
      min: 0.2,
      max: 5.0,
      step: 0.01,
      initialValue: 1.0,
      onChange: onGamma,
    });
    new LabeledSlider({
      container,
      label: 'Opacity',
      min: 0,
      max: 1,
      step: 0.01,
      initialValue: 1,
      onChange: onOpacity,
    });

    const inputs = container.querySelectorAll('input[type="range"]');
    expect(inputs.length).toBe(2);

    (inputs[0] as HTMLInputElement).value = '2.0';
    inputs[0].dispatchEvent(new Event('input'));
    expect(onGamma).toHaveBeenCalledWith(2.0);
    expect(onOpacity).not.toHaveBeenCalled();

    (inputs[1] as HTMLInputElement).value = '0.5';
    inputs[1].dispatchEvent(new Event('input'));
    expect(onOpacity).toHaveBeenCalledWith(0.5);
  });
});
