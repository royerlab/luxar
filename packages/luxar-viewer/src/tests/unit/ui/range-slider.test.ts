/**
 * Unit tests for RangeSlider — dual-thumb slider used by the layers
 * panel for the [min, max] display-range control.
 *
 * Drives the component through its public API (constructor + setValues +
 * setBounds + dispose) and through synthesized DOM events (input, wheel,
 * click). Confirms onChange / onBoundsChange semantics, the low ≤ high
 * invariant, the track-fill geometry, and dispose's listener cleanup.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { RangeSlider, type RangeSliderOptions } from '../../../ui/layers/range-slider';

let host: HTMLElement;

function makeSlider(opts: Partial<RangeSliderOptions> = {}) {
  const onChange = vi.fn();
  const onBoundsChange = vi.fn();
  const slider = new RangeSlider({
    container: host,
    min: 0,
    max: 1,
    valueLow: 0.2,
    valueHigh: 0.8,
    onChange,
    onBoundsChange,
    ...opts,
  });
  return { slider, onChange, onBoundsChange };
}

function getInputs(): { low: HTMLInputElement; high: HTMLInputElement } {
  const inputs = host.querySelectorAll('input[type="range"]');
  return {
    low: inputs[0] as HTMLInputElement,
    high: inputs[1] as HTMLInputElement,
  };
}

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
});

afterEach(() => {
  host.remove();
});

describe('RangeSlider — construction', () => {
  it('appends a wrapper, two range inputs, and bounds labels to the container', () => {
    makeSlider();
    expect(host.querySelector('.luxar-range-slider')).toBeTruthy();
    const inputs = host.querySelectorAll('input[type="range"]');
    expect(inputs.length).toBe(2);
    expect(host.querySelectorAll('.luxar-range-slider__bound').length).toBe(2);
  });

  it('renders an optional label row when `label` is supplied', () => {
    makeSlider({ label: 'Display range' });
    const labelEl = host.querySelector('.luxar-range-slider__label');
    expect(labelEl?.textContent).toBe('Display range');
    // Two value spans (low and high), inside the values container.
    expect(host.querySelectorAll('.luxar-range-slider__value').length).toBe(2);
  });

  it('omits the label row when `label` is not supplied', () => {
    makeSlider();
    expect(host.querySelector('.luxar-range-slider__label-row')).toBeNull();
  });

  it('uses the configured min/max/step on both inputs', () => {
    makeSlider({ min: -10, max: 10, step: 0.5, valueLow: -5, valueHigh: 5 });
    const { low, high } = getInputs();
    expect(low.min).toBe('-10');
    expect(low.max).toBe('10');
    expect(low.step).toBe('0.5');
    expect(low.value).toBe('-5');
    expect(high.value).toBe('5');
  });

  it('defaults the step to 0.001 when not provided', () => {
    makeSlider();
    const { low } = getInputs();
    expect(low.step).toBe('0.001');
  });

  it('initializes the bound labels with formatted min/max', () => {
    makeSlider({ min: 0, max: 1.234567, valueLow: 0, valueHigh: 1 });
    const bounds = host.querySelectorAll('.luxar-range-slider__bound');
    // formatValue strips trailing zeros after a 3-decimal toFixed.
    expect(bounds[0].textContent).toBe('0');
    expect(bounds[1].textContent).toBe('1.235');
  });

  // [R11/D-G1][P5] Pin behaviour for Infinity bounds. HTML5
  // <input type="range"> rejects non-finite-numeric strings on
  // `.value`/`.min`/`.max` assignment; the resulting DOM strings are
  // empty (jsdom: '') or browser default. The contract: construction
  // must NOT throw and the resulting slider must not produce NaN
  // `valueAsNumber` reads. A regression that called `String(Infinity)
  // = 'Infinity'` and then `.min = 'Infinity'` would leave the input
  // in an inconsistent state.
  it('survives Infinity bounds without throwing and yields finite valueAsNumber reads', () => {
    expect(() => makeSlider({ min: 0, max: Infinity, valueLow: 0, valueHigh: 1 })).not.toThrow();
    const { low, high } = getInputs();
    // valueAsNumber must be finite (NaN read would silently propagate).
    expect(Number.isFinite(low.valueAsNumber)).toBe(true);
    expect(Number.isFinite(high.valueAsNumber)).toBe(true);
  });

  it('survives -Infinity bounds without throwing and yields finite valueAsNumber reads', () => {
    expect(() => makeSlider({ min: -Infinity, max: 1, valueLow: 0, valueHigh: 0.5 })).not.toThrow();
    const { low, high } = getInputs();
    expect(Number.isFinite(low.valueAsNumber)).toBe(true);
    expect(Number.isFinite(high.valueAsNumber)).toBe(true);
  });
});

describe('RangeSlider — onChange invariant', () => {
  it('fires onChange when the low thumb moves', () => {
    const { onChange } = makeSlider();
    const { low } = getInputs();
    low.value = '0.4';
    low.dispatchEvent(new Event('input'));
    expect(onChange).toHaveBeenLastCalledWith(0.4, 0.8);
  });

  it('fires onChange when the high thumb moves', () => {
    const { onChange } = makeSlider();
    const { high } = getInputs();
    high.value = '0.6';
    high.dispatchEvent(new Event('input'));
    expect(onChange).toHaveBeenLastCalledWith(0.2, 0.6);
  });

  it('clamps low ≤ high when the user drags low past high', () => {
    const { onChange } = makeSlider();
    const { low } = getInputs();
    low.value = '0.95';
    low.dispatchEvent(new Event('input'));
    // low gets pinned to high (0.8); the underlying input is rewritten in place.
    expect(low.value).toBe('0.8');
    expect(onChange).toHaveBeenLastCalledWith(0.8, 0.8);
  });

  it('clamps high ≥ low when the user drags high below low', () => {
    const { onChange } = makeSlider({ valueLow: 0.4, valueHigh: 0.7 });
    const { high } = getInputs();
    high.value = '0.1';
    high.dispatchEvent(new Event('input'));
    expect(high.value).toBe('0.4');
    expect(onChange).toHaveBeenLastCalledWith(0.4, 0.4);
  });
});

describe('RangeSlider — track-fill geometry', () => {
  it('positions the fill between the two thumbs as % of [min,max]', () => {
    makeSlider({ min: 0, max: 1, valueLow: 0.25, valueHigh: 0.75 });
    const fill = host.querySelector('.luxar-range-slider__track-fill') as HTMLElement;
    expect(fill.style.left).toBe('25%');
    expect(fill.style.width).toBe('50%');
  });

  it('updates the fill geometry after setValues', () => {
    const { slider } = makeSlider({ min: 0, max: 1 });
    slider.setValues(0.1, 0.9);
    const fill = host.querySelector('.luxar-range-slider__track-fill') as HTMLElement;
    expect(fill.style.left).toBe('10%');
    expect(Math.round(parseFloat(fill.style.width))).toBe(80);
  });

  it('handles a zero-width range gracefully (no NaN, no throw)', () => {
    // W9 strengthening (P2): the prior test only asserted no-throw.
    // Strengthen by also confirming (a) the fill element's geometry is
    // not "NaN%" (the actual failure mode of a 1/(max-min) division),
    // (b) bounds labels render the actual number 5, and (c) the inputs'
    // value reflect the requested 5.
    expect(() => makeSlider({ min: 5, max: 5, valueLow: 5, valueHigh: 5 })).not.toThrow();

    const fill = host.querySelector('.luxar-range-slider__track-fill') as HTMLElement;
    expect(fill).toBeTruthy();
    expect(fill.style.left).not.toContain('NaN');
    expect(fill.style.width).not.toContain('NaN');

    const { low, high } = getInputs();
    expect(low.value).toBe('5');
    expect(high.value).toBe('5');

    const bounds = host.querySelectorAll('.luxar-range-slider__bound');
    expect(bounds[0].textContent).toBe('5');
    expect(bounds[1].textContent).toBe('5');
  });
});

describe('RangeSlider — public setters', () => {
  it('setValues writes both inputs and refreshes labels', () => {
    const { slider } = makeSlider({ label: 'Range' });
    slider.setValues(0.3, 0.7);
    const { low, high } = getInputs();
    expect(low.value).toBe('0.3');
    expect(high.value).toBe('0.7');

    const valueLabels = host.querySelectorAll('.luxar-range-slider__value');
    expect(valueLabels[0].textContent).toBe('0.3');
    expect(valueLabels[1].textContent).toBe('0.7');
  });

  it('setBounds rewrites min/max on both inputs and updates the bound labels', () => {
    const { slider } = makeSlider();
    slider.setBounds(-2, 2);
    const { low, high } = getInputs();
    expect(low.min).toBe('-2');
    expect(low.max).toBe('2');
    expect(high.min).toBe('-2');
    expect(high.max).toBe('2');
    const bounds = host.querySelectorAll('.luxar-range-slider__bound');
    expect(bounds[0].textContent).toBe('-2');
    expect(bounds[1].textContent).toBe('2');
  });
});

describe('RangeSlider — bound wheel adjustment', () => {
  it('scrolling the low bound up increases min (no fine modifier)', () => {
    const { slider, onBoundsChange } = makeSlider({ min: 0, max: 10 });
    const lowBound = host.querySelectorAll('.luxar-range-slider__bound')[0];
    // computeWheelStep([0,10], false) = 1 — scroll up adds one step.
    const evt = new WheelEvent('wheel', { deltaY: -100, bubbles: true, cancelable: true });
    lowBound.dispatchEvent(evt);

    const { low } = getInputs();
    expect(parseFloat(low.min)).toBeCloseTo(1, 5);
    expect(onBoundsChange).toHaveBeenCalled();
    void slider; // keep ref
  });

  it('scrolling the high bound down decreases max', () => {
    const { onBoundsChange } = makeSlider({ min: 0, max: 10 });
    const highBound = host.querySelectorAll('.luxar-range-slider__bound')[1];
    const evt = new WheelEvent('wheel', { deltaY: 100, bubbles: true, cancelable: true });
    highBound.dispatchEvent(evt);
    const { low } = getInputs();
    expect(parseFloat(low.max)).toBeCloseTo(9, 5);
    expect(onBoundsChange).toHaveBeenCalled();
  });

  it('shift+scroll uses a 10× finer step', () => {
    makeSlider({ min: 0, max: 10 });
    const lowBound = host.querySelectorAll('.luxar-range-slider__bound')[0];
    const evt = new WheelEvent('wheel', {
      deltaY: -100,
      shiftKey: true,
      bubbles: true,
      cancelable: true,
    });
    lowBound.dispatchEvent(evt);
    const { low } = getInputs();
    // Fine = step/10 = 0.1
    expect(parseFloat(low.min)).toBeCloseTo(0.1, 5);
  });

  it('clamps min ≤ max when the bound is dragged past the other side', () => {
    const { slider, onBoundsChange } = makeSlider({ min: 0, max: 1 });
    slider.setValues(0, 1);
    const lowBound = host.querySelectorAll('.luxar-range-slider__bound')[0];
    // Repeatedly scroll up — eventually we hit the high bound and clamp.
    for (let i = 0; i < 50; i++) {
      const evt = new WheelEvent('wheel', { deltaY: -100, cancelable: true });
      lowBound.dispatchEvent(evt);
    }
    const { low } = getInputs();
    expect(parseFloat(low.min)).toBeLessThanOrEqual(parseFloat(low.max));
    expect(onBoundsChange).toHaveBeenCalled();
  });
});

describe('RangeSlider — bound click-to-edit', () => {
  it('clicking the low bound replaces it with a text input', () => {
    makeSlider();
    const lowBound = host.querySelectorAll('.luxar-range-slider__bound')[0] as HTMLElement;
    lowBound.click();
    const editor = host.querySelector('.luxar-range-slider__bound-input') as HTMLInputElement;
    expect(editor).toBeTruthy();
    expect(editor.value).toBe('0');
    expect(lowBound.style.display).toBe('none');
  });

  it('typing a value + Enter commits new bounds and fires onBoundsChange', () => {
    const { onBoundsChange } = makeSlider({ min: 0, max: 1 });
    const lowBound = host.querySelectorAll('.luxar-range-slider__bound')[0] as HTMLElement;
    lowBound.click();
    const editor = host.querySelector('.luxar-range-slider__bound-input') as HTMLInputElement;
    editor.value = '-0.5';
    editor.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    const { low } = getInputs();
    expect(parseFloat(low.min)).toBeCloseTo(-0.5, 5);
    expect(onBoundsChange).toHaveBeenCalledWith(-0.5, 1);
    // Editor is gone, label restored.
    expect(host.querySelector('.luxar-range-slider__bound-input')).toBeNull();
    expect(lowBound.style.display).toBe('');
  });

  it('Escape cancels the edit without firing onBoundsChange', () => {
    const { onBoundsChange } = makeSlider({ min: 0, max: 1 });
    const lowBound = host.querySelectorAll('.luxar-range-slider__bound')[0] as HTMLElement;
    lowBound.click();
    const editor = host.querySelector('.luxar-range-slider__bound-input') as HTMLInputElement;
    editor.value = '-99';
    editor.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    const { low } = getInputs();
    expect(parseFloat(low.min)).toBe(0);
    expect(onBoundsChange).not.toHaveBeenCalled();
  });

  it('committing a low value above the high bound clamps to the high bound', () => {
    const { onBoundsChange } = makeSlider({ min: 0, max: 1 });
    const lowBound = host.querySelectorAll('.luxar-range-slider__bound')[0] as HTMLElement;
    lowBound.click();
    const editor = host.querySelector('.luxar-range-slider__bound-input') as HTMLInputElement;
    editor.value = '5';
    editor.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    // Min got pinned to current max (1).
    expect(onBoundsChange).toHaveBeenCalledWith(1, 1);
  });
});

describe('RangeSlider — dispose', () => {
  it('removes the wrapper from the DOM', () => {
    const { slider } = makeSlider();
    expect(host.querySelector('.luxar-range-slider')).toBeTruthy();
    slider.dispose();
    expect(host.querySelector('.luxar-range-slider')).toBeNull();
  });

  it('detaches input + wheel + click listeners (no fires after dispose)', () => {
    const { slider, onChange, onBoundsChange } = makeSlider();
    const inputs = host.querySelectorAll('input[type="range"]');
    const low = inputs[0] as HTMLInputElement;
    const lowBound = host.querySelectorAll('.luxar-range-slider__bound')[0] as HTMLElement;

    slider.dispose();

    // After dispose the wrapper is gone, but the original element refs we
    // captured still receive events — the listener attached by the slider
    // should already have been detached.
    low.dispatchEvent(new Event('input'));
    lowBound.dispatchEvent(new WheelEvent('wheel', { deltaY: -100 }));
    lowBound.click();

    expect(onChange).not.toHaveBeenCalled();
    expect(onBoundsChange).not.toHaveBeenCalled();
  });
});
