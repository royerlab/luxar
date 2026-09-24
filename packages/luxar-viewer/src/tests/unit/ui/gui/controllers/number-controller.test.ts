// @vitest-environment jsdom
/**
 * NumberController Tests
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { NumberController } from '../../../../../ui/gui/controllers/number-controller';

describe('NumberController', () => {
  let object: Record<string, number>;
  let controller: NumberController;

  afterEach(() => {
    if (controller) {
      controller.dispose();
    }
  });

  describe('without range (input only)', () => {
    beforeEach(() => {
      object = { value: 42 };
      controller = new NumberController(object, 'value', {});
    });

    it('should create input element', () => {
      const input = controller.domElement.querySelector('.luxar-gui__input--number');

      expect(input).toBeInstanceOf(HTMLInputElement);
    });

    it('leaves the keyboard hint unset when the range is unbounded', () => {
      const input = controller.domElement.querySelector<HTMLInputElement>(
        '.luxar-gui__input--number'
      )!;
      expect(input.hasAttribute('inputmode')).toBe(false);
      expect(input.inputMode).toBe('');
    });

    it('leaves the keyboard hint unset for a range with a negative minimum', () => {
      const ranged = new NumberController({ value: 0 }, 'value', { min: -10, max: 10 });
      expect(ranged.$input!.hasAttribute('inputmode')).toBe(false);
      expect(ranged.$input!.inputMode).toBe('');
      ranged.dispose();
    });

    it('should not create slider without range', () => {
      const slider = controller.domElement.querySelector('.luxar-gui__slider');

      expect(slider).toBeNull();
    });

    it('should update object value when input changes', () => {
      const input = controller.domElement.querySelector(
        '.luxar-gui__input--number'
      ) as HTMLInputElement;

      input.value = '100';
      input.dispatchEvent(new Event('change'));

      expect(object.value).toBe(100);
    });

    it('should expose $input property', () => {
      expect(controller.$input).toBeInstanceOf(HTMLInputElement);
      expect(controller.$input).toBe(
        controller.domElement.querySelector('.luxar-gui__input--number')
      );
    });
  });

  describe('with range (slider + input)', () => {
    beforeEach(() => {
      object = { value: 50 };
      controller = new NumberController(object, 'value', { min: 0, max: 100, step: 1 });
    });

    it('asks touch keyboards for a decimal keypad when the range is non-negative', () => {
      expect(controller.$input!.inputMode).toBe('decimal');
    });

    it('wheel uses the declared step with the shared modifier tiers', () => {
      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;
      const wheel = (init: WheelEventInit) =>
        slider.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, ...init }));

      // Base: scroll up = +1×step.
      wheel({ deltaY: -120 });
      expect(object.value).toBeCloseTo(51, 9);
      // Shift = fine (÷10).
      wheel({ deltaY: -120, shiftKey: true });
      expect(object.value).toBeCloseTo(51.1, 9);
      // Ctrl = coarse (×10).
      wheel({ deltaY: 120, ctrlKey: true });
      expect(object.value).toBeCloseTo(41.1, 9);
      // Ctrl+Shift = extra-fine (÷100).
      wheel({ deltaY: -120, ctrlKey: true, shiftKey: true });
      expect(object.value).toBeCloseTo(41.11, 9);
    });

    it('widens the number readout for fine wheel values', () => {
      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;

      slider.dispatchEvent(
        new WheelEvent('wheel', {
          deltaY: -120,
          shiftKey: true,
          bubbles: true,
          cancelable: true,
        })
      );

      expect(object.value).toBeCloseTo(50.1, 9);
      expect(controller.$input?.value).toBe('50.10');
    });

    it('wheel reads the horizontal axis when Shift swaps it; zero delta is a no-op', () => {
      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;

      // Shift+wheel on a standard mouse arrives as deltaX with deltaY = 0.
      slider.dispatchEvent(
        new WheelEvent('wheel', {
          deltaY: 0,
          deltaX: 120,
          shiftKey: true,
          bubbles: true,
          cancelable: true,
        })
      );
      expect(object.value).toBeCloseTo(49.9, 9); // scrolled "down" → −0.1

      slider.dispatchEvent(
        new WheelEvent('wheel', { deltaY: 0, deltaX: 0, bubbles: true, cancelable: true })
      );
      expect(object.value).toBeCloseTo(49.9, 9); // unchanged
    });

    it('wheel bubbles to the window after the coarse tier is applied', () => {
      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;
      document.body.appendChild(controller.domElement);
      const windowSpy = vi.fn();
      window.addEventListener('wheel', windowSpy);
      slider.dispatchEvent(
        new WheelEvent('wheel', { deltaY: -120, ctrlKey: true, bubbles: true, cancelable: true })
      );
      window.removeEventListener('wheel', windowSpy);
      expect(windowSpy).toHaveBeenCalledTimes(1);
      expect(object.value).toBeCloseTo(60, 9);
    });

    it('should create both slider and input', () => {
      const slider = controller.domElement.querySelector('.luxar-gui__slider');
      const input = controller.domElement.querySelector('.luxar-gui__input--number');

      expect(slider).toBeInstanceOf(HTMLInputElement);
      expect(input).toBeInstanceOf(HTMLInputElement);
    });

    it('should set slider attributes', () => {
      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;

      expect(slider.min).toBe('0');
      expect(slider.max).toBe('100');
      expect(slider.step).toBe('0.01');
      expect(slider.dataset.baseStep).toBe('1');
    });

    it('should update object when slider changes', () => {
      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;

      slider.value = '75';
      slider.dispatchEvent(new Event('input'));

      expect(object.value).toBe(75);
    });

    it('should sync slider and input', () => {
      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;
      const input = controller.domElement.querySelector(
        '.luxar-gui__input--number'
      ) as HTMLInputElement;

      slider.value = '80';
      slider.dispatchEvent(new Event('input'));

      expect(input.value).toBe('80');
    });

    it('should clamp values to min/max', () => {
      const input = controller.domElement.querySelector(
        '.luxar-gui__input--number'
      ) as HTMLInputElement;

      input.value = '150'; // Above max
      input.dispatchEvent(new Event('change'));

      expect(object.value).toBe(100); // Clamped to max
    });

    // [ui.md/G — boundary cases on gui library components] Numeric boundary
    // cases. The controller is wired to user-driven HTML inputs which can
    // emit any string — defend against NaN, ±Infinity, and below-min.
    it('clamps values below min back to min', () => {
      const input = controller.domElement.querySelector(
        '.luxar-gui__input--number'
      ) as HTMLInputElement;
      input.value = '-50'; // below min (min=0)
      input.dispatchEvent(new Event('change'));
      expect(object.value).toBe(0);
    });

    it('does not mutate the object when the input is non-numeric (NaN guard)', () => {
      const input = controller.domElement.querySelector(
        '.luxar-gui__input--number'
      ) as HTMLInputElement;
      const before = object.value;
      input.value = 'not-a-number';
      input.dispatchEvent(new Event('change'));
      // Either kept the previous value or wrote a finite number — must
      // NOT have written NaN, which would silently corrupt every
      // downstream consumer of this.object[propertyName].
      expect(Number.isNaN(object.value)).toBe(false);
      expect(Number.isFinite(object.value)).toBe(true);
      // Conservative default behaviour: previous value retained.
      expect(object.value).toBe(before);
    });

    // [R11/D-C1+D-G2][P5/P7] Pin Infinity / very-large-finite behaviour at
    // two layers:
    //
    // 1. DOM-level defense: HTML5 <input type="number"> rejects the literal
    //    strings 'Infinity' / '-Infinity' (they are not valid float
    //    literals per the HTML float-parsing grammar). The browser/jsdom
    //    clears `.value` to '' and parseFloat('') = NaN — so the NaN guard
    //    catches it and the model retains its prior value. A regression
    //    that swapped <input type="number"> for type="text" (allowing
    //    'Infinity' through) would silently break this defense.
    // 2. setValue() bypasses the DOM: when an upstream caller programmatically
    //    invokes `setValue(Infinity)`, clamp() must protect the model.
    //    Pin both bounds — a mutation that drops `value > max` or
    //    `value < min` in clamp would fail the corresponding case.
    it('rejects "Infinity" string input via DOM number-input filtering', () => {
      const input = controller.domElement.querySelector(
        '.luxar-gui__input--number'
      ) as HTMLInputElement;
      const before = object.value;
      input.value = 'Infinity';
      input.dispatchEvent(new Event('change'));
      expect(Number.isFinite(object.value)).toBe(true);
      expect(object.value).toBe(before);
    });

    it('rejects "-Infinity" string input via DOM number-input filtering', () => {
      const input = controller.domElement.querySelector(
        '.luxar-gui__input--number'
      ) as HTMLInputElement;
      const before = object.value;
      input.value = '-Infinity';
      input.dispatchEvent(new Event('change'));
      expect(Number.isFinite(object.value)).toBe(true);
      expect(object.value).toBe(before);
    });

    // [R11/D-G2 — boundary in input-change path]
    // A value just-above-max through the change event must clamp. Use a
    // large finite value (1e6) — kills a clamp mutation that swaps
    // `value > max` for `value >= max` or drops the upper bound entirely.
    it('clamps very-large finite values from the change event to max', () => {
      const input = controller.domElement.querySelector(
        '.luxar-gui__input--number'
      ) as HTMLInputElement;
      input.value = '1000000'; // 1e6, valid HTML5 float literal
      input.dispatchEvent(new Event('change'));
      expect(object.value).toBe(100);
    });

    // [R11/D-G2 — boundary in slider input path]
    // The slider path uses `constrainValue` on parseFloat(slider.value).
    // jsdom's <input type="range"> clamps the value attribute internally to
    // [min, max], so 1000 → 100 even before parseFloat runs. Pin that.
    it('clamps slider values above max via the input event', () => {
      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;
      slider.value = '1000'; // valid float, above max=100
      slider.dispatchEvent(new Event('input'));
      expect(object.value).toBe(100);
    });

    // [R11-OOS FIX] NumberController.setValue() now overrides the base
    // controller's raw assignment to clamp into [min, max] and reject
    // NaN / ±Infinity. The 4 tests below pin the new contract; a
    // regression that re-introduced the raw super.setValue() call would
    // fail every one of them.
    it('clamps programmatic setValue(Infinity) to max', () => {
      controller.setValue(Infinity);
      expect(Number.isFinite(object.value)).toBe(true);
      expect(object.value).toBe(100);
    });

    it('clamps programmatic setValue(-Infinity) to min', () => {
      controller.setValue(-Infinity);
      expect(Number.isFinite(object.value)).toBe(true);
      expect(object.value).toBe(0);
    });

    it('clamps programmatic setValue(1e20) to max', () => {
      controller.setValue(1e20);
      expect(object.value).toBe(100);
    });

    it('rejects programmatic setValue(NaN) without mutating the model', () => {
      const before = object.value;
      controller.setValue(NaN);
      expect(Number.isFinite(object.value)).toBe(true);
      expect(object.value).toBe(before);
    });

    it('should trigger onChange callback on user input', () => {
      const onChange = vi.fn();
      controller.onChange(onChange);

      // Simulate user interaction (setValue does NOT trigger callbacks, matching lil-gui)
      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;
      slider.value = '60';
      slider.dispatchEvent(new Event('input'));

      expect(onChange).toHaveBeenCalledWith(60);
    });

    it('should NOT trigger onChange on setValue (matches lil-gui)', () => {
      const onChange = vi.fn();
      controller.onChange(onChange);

      controller.setValue(60);

      expect(onChange).not.toHaveBeenCalled();
    });

    it('should trigger onFinishChange on mouseup', () => {
      const onFinishChange = vi.fn();
      controller.onFinishChange(onFinishChange);

      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;
      slider.value = '70';
      slider.dispatchEvent(new Event('input'));
      slider.dispatchEvent(new Event('mouseup'));

      expect(onFinishChange).toHaveBeenCalledWith(70);
    });

    it('should use default step (1% of range) if not specified', () => {
      const controller2 = new NumberController(object, 'value', { min: 0, max: 100 });
      const slider = controller2.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;

      expect(slider.step).toBe('0.01'); // fine DOM grid for base step 1

      controller2.dispose();
    });

    it('uses step="any" when the range is degenerate', () => {
      const controller2 = new NumberController(object, 'value', { min: 50, max: 50 });
      const slider = controller2.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;
      const input = controller2.domElement.querySelector(
        '.luxar-gui__input--number'
      ) as HTMLInputElement;
      expect(slider.step).toBe('any');
      expect(input.step).toBe('any');
      controller2.dispose();
    });
  });

  describe('setValue() / getValue()', () => {
    beforeEach(() => {
      object = { value: 50 };
      controller = new NumberController(object, 'value', { min: 0, max: 100 });
    });

    it('should set value and update display', () => {
      controller.setValue(75);

      expect(object.value).toBe(75);
      expect(controller.getValue()).toBe(75);

      const input = controller.domElement.querySelector(
        '.luxar-gui__input--number'
      ) as HTMLInputElement;
      expect(input.value).toBe('75');
    });

    it('should NOT trigger onChange when setValue is called (matches lil-gui)', () => {
      const onChange = vi.fn();
      controller.onChange(onChange);

      controller.setValue(80);

      // setValue should NOT trigger callbacks (lil-gui behavior)
      expect(onChange).not.toHaveBeenCalled();
    });
  });

  describe('min() / max() / step()', () => {
    beforeEach(() => {
      object = { value: 50 };
      controller = new NumberController(object, 'value', { min: 0, max: 100 });
    });

    it('should update min value', () => {
      controller.min(10);

      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;
      expect(slider.min).toBe('10');
    });

    it('should update max value', () => {
      controller.max(200);

      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;
      expect(slider.max).toBe('200');
    });

    it('should update step value', () => {
      controller.step(5);

      const slider = controller.domElement.querySelector('.luxar-gui__slider') as HTMLInputElement;
      expect(slider.step).toBe('0.05');
      expect(slider.dataset.baseStep).toBe('5');
    });

    it('should return this for chaining', () => {
      const result = controller.min(0).max(100).step(1);

      expect(result).toBe(controller);
    });
  });

  describe('show() / hide()', () => {
    beforeEach(() => {
      object = { value: 50 };
      controller = new NumberController(object, 'value', {});
    });

    it('should show controller', () => {
      controller.hide();
      controller.show();

      expect(controller.domElement.style.display).toBe('');
    });

    it('should hide controller', () => {
      controller.show();
      controller.hide();

      expect(controller.domElement.style.display).toBe('none');
    });
  });

  describe('name()', () => {
    beforeEach(() => {
      object = { value: 50 };
      controller = new NumberController(object, 'value', {});
    });

    it('should set label text', () => {
      controller.name('Custom Label');

      const label = controller.domElement.querySelector('.luxar-gui__controller-name');
      expect(label?.textContent).toBe('Custom Label');
    });

    it('should return this for chaining', () => {
      const result = controller.name('FOV');

      expect(result).toBe(controller);
    });
  });

  describe('Custom updateDisplay (logarithmic pattern)', () => {
    beforeEach(() => {
      object = { log: 0 };
      controller = new NumberController(object, 'log', { min: -2, max: 2, step: 0.01 });
    });

    it('should support custom updateDisplay override', () => {
      let customCallCount = 0;
      const customDisplay = () => {
        customCallCount++;
        const actual = Math.pow(10, object.log);
        if (controller.$input) {
          (controller.$input as HTMLInputElement).value = actual.toFixed(2);
        }
      };

      (controller as any).setCustomUpdateDisplay(customDisplay);

      controller.updateDisplay();

      expect(customCallCount).toBe(1);
      expect((controller.$input as HTMLInputElement).value).toBe('1.00'); // 10^0 = 1
    });

    it('should use custom display when setValue is called', () => {
      const customDisplay = vi.fn(() => {
        if (controller.$input) {
          const actual = Math.pow(10, object.log);
          (controller.$input as HTMLInputElement).value = actual.toFixed(2);
        }
      });

      (controller as any).setCustomUpdateDisplay(customDisplay);

      controller.setValue(1); // log = 1 → 10^1 = 10

      expect(customDisplay).toHaveBeenCalled();
      expect((controller.$input as HTMLInputElement).value).toBe('10.00');
    });
  });

  describe('dispose()', () => {
    beforeEach(() => {
      object = { value: 50 };
      controller = new NumberController(object, 'value', { min: 0, max: 100 });
    });

    it('should remove event listeners', () => {
      const eventManagerSpy = vi.spyOn(controller['eventManager'], 'removeAll');

      controller.dispose();

      expect(eventManagerSpy).toHaveBeenCalled();
    });

    it('should remove from DOM', () => {
      const parent = document.createElement('div');
      parent.appendChild(controller.domElement);

      controller.dispose();

      expect(parent.contains(controller.domElement)).toBe(false);
    });

    it('should clear callbacks', () => {
      const onChange = vi.fn();
      controller.onChange(onChange);

      controller.dispose();

      controller.setValue(75);

      // onChange should not fire after dispose
      expect(onChange).not.toHaveBeenCalled();
    });
  });
});
