import { describe, it, expect, beforeEach, vi } from 'vitest';
import { LabeledSlider } from '../../../../ui/layers/labeled-slider';

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
    expect(input.step).toBe('0.01');
    expect(input.value).toBe('1.5');

    expect(findReadout().textContent).toBe('1.50');
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
        onChange,
      });
      return { slider, onChange };
    }

    it('drives the DOM input in normalised position space', () => {
      makeLog();
      const input = findInput();
      expect(input.min).toBe('0');
      expect(input.max).toBe('1');
      // κ=1 on a 0.001–10 track sits at ln(1/0.001)/ln(10/0.001) = 3/4.
      expect(parseFloat(input.value)).toBeCloseTo(0.75, 6);
    });

    it('maps positions geometrically and round-trips through setValue', () => {
      const { slider, onChange } = makeLog();
      const input = findInput();

      for (const [position, expected] of [
        [0.25, 0.01],
        [0.5, 0.1],
        [1, 10],
      ] as const) {
        input.value = String(position);
        input.dispatchEvent(new Event('input'));
        expect(onChange).toHaveBeenLastCalledWith(expect.closeTo(expected, 6));
      }

      slider.setValue(0.1);
      expect(parseFloat(input.value)).toBeCloseTo(0.5, 6);
    });

    it('reserves position 0 for an exact zero (the additive limit), not `min`', () => {
      const { onChange } = makeLog();
      const input = findInput();

      input.value = '0';
      input.dispatchEvent(new Event('input'));

      expect(onChange).toHaveBeenCalledWith(0);
      expect(findReadout().textContent).toBe('0.00');
    });

    it('setRange re-scales the track without moving the value', () => {
      const { slider, onChange } = makeLog();
      const input = findInput();

      slider.setValue(1);
      // A thin-geometry layer: same κ, a track that now reaches 10^4.
      slider.setRange(1, 10000);

      expect(slider.getRange()).toEqual([1, 10000]);
      // κ=1 is now the LEFT end of the track, and still κ=1.
      expect(parseFloat(input.value)).toBeCloseTo(0, 6);
      expect(findReadout().textContent).toBe('1.00');
      expect(onChange).not.toHaveBeenCalled();

      // The far end of the widened track now reaches the large κ a thin
      // line needs — unreachable on the original 0.001–10 track.
      input.value = '1';
      input.dispatchEvent(new Event('input'));
      expect(onChange).toHaveBeenLastCalledWith(expect.closeTo(10000, 3));
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
