/**
 * BooleanController Tests
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { BooleanController } from '../../../../../ui/gui/controllers/boolean-controller';

describe('BooleanController', () => {
  let object: { enabled: boolean };
  let controller: BooleanController;

  afterEach(() => {
    if (controller) {
      controller.dispose();
    }
  });

  describe('constructor', () => {
    // ui.md O6 / Phase E33: P9 rename — describes input class (BooleanController
    // construction) + expected behavior (a checkbox input element is mounted).
    it('mounts an <input type=checkbox> under .luxar-gui__checkbox when constructed', () => {
      object = { enabled: true };
      controller = new BooleanController(object, 'enabled');

      const checkbox = controller.domElement.querySelector('.luxar-gui__checkbox');
      expect(checkbox).toBeInstanceOf(HTMLInputElement);
    });

    // ui.md O6 / Phase E33: P9 rename — pins that the checkbox.checked
    // mirror reflects the initial value passed to the constructor.
    it('checkbox.checked mirrors the initial property value (object.enabled=true → checkbox.checked=true)', () => {
      object = { enabled: true };
      controller = new BooleanController(object, 'enabled');

      const checkbox = controller.domElement.querySelector(
        '.luxar-gui__checkbox'
      ) as HTMLInputElement;
      expect(checkbox.checked).toBe(true);
    });

    // ui.md O6 / Phase E33: P9 rename — the input must be nested inside
    // the .luxar-gui__controller-name <label> so clicking the label
    // toggles the checkbox (HTML label association without `for=`).
    it('wraps the checkbox in a <label class="luxar-gui__controller-name"> for click-to-toggle association', () => {
      object = { enabled: false };
      controller = new BooleanController(object, 'enabled');

      const label = controller.domElement.querySelector('.luxar-gui__controller-name');
      expect(label).toBeInstanceOf(HTMLLabelElement);
      expect(label?.contains(controller.$input!)).toBe(true);
    });
  });

  describe('setValue() / getValue()', () => {
    beforeEach(() => {
      object = { enabled: false };
      controller = new BooleanController(object, 'enabled');
    });

    // ui.md O6 / Phase E38: P9 rename — pins that setValue(v) writes
    // through to the backing property AND surfaces via getValue().
    it('setValue(true) writes to object.enabled AND mirrors via getValue()', () => {
      controller.setValue(true);

      expect(object.enabled).toBe(true);
      expect(controller.getValue()).toBe(true);
    });

    // ui.md O6 / Phase E38: P9 rename — pins that setValue() also
    // syncs the DOM checkbox's `.checked` mirror.
    it('setValue(true) updates the DOM checkbox.checked mirror in lockstep', () => {
      controller.setValue(true);

      const checkbox = controller.domElement.querySelector(
        '.luxar-gui__checkbox'
      ) as HTMLInputElement;
      expect(checkbox.checked).toBe(true);
    });
  });

  describe('updateDisplay()', () => {
    beforeEach(() => {
      object = { enabled: false };
      controller = new BooleanController(object, 'enabled');
    });

    // ui.md O6 / Phase E38: P9 rename — pins that updateDisplay()
    // re-reads object.enabled (mutated externally) and syncs the
    // checkbox.checked mirror to match.
    it('updateDisplay() re-reads object.enabled after external mutation and updates checkbox.checked', () => {
      object.enabled = true;
      controller.updateDisplay();

      const checkbox = controller.domElement.querySelector(
        '.luxar-gui__checkbox'
      ) as HTMLInputElement;
      expect(checkbox.checked).toBe(true);
    });
  });

  describe('onChange() / onFinishChange()', () => {
    beforeEach(() => {
      object = { enabled: false };
      controller = new BooleanController(object, 'enabled');
    });

    it('should trigger onChange when checkbox changes', () => {
      const onChange = vi.fn();
      controller.onChange(onChange);

      const checkbox = controller.domElement.querySelector(
        '.luxar-gui__checkbox'
      ) as HTMLInputElement;
      checkbox.checked = true;
      checkbox.dispatchEvent(new Event('change'));

      expect(onChange).toHaveBeenCalledWith(true);
    });

    it('should trigger onFinishChange when checkbox changes', () => {
      const onFinishChange = vi.fn();
      controller.onFinishChange(onFinishChange);

      const checkbox = controller.domElement.querySelector(
        '.luxar-gui__checkbox'
      ) as HTMLInputElement;
      checkbox.checked = true;
      checkbox.dispatchEvent(new Event('change'));

      expect(onFinishChange).toHaveBeenCalledWith(true);
    });

    it('should NOT trigger onChange when setValue is called (matches lil-gui)', () => {
      const onChange = vi.fn();
      controller.onChange(onChange);

      controller.setValue(true);

      // setValue should NOT trigger callbacks (lil-gui behavior)
      expect(onChange).not.toHaveBeenCalled();
    });
  });

  describe('name()', () => {
    beforeEach(() => {
      object = { enabled: false };
      controller = new BooleanController(object, 'enabled');
    });

    it('should set label text', () => {
      controller.name('Auto Rotate');

      const label = controller.domElement.querySelector('.luxar-gui__controller-name');
      // Text should be after checkbox
      expect(label?.textContent).toContain('Auto Rotate');
    });
  });

  describe('show() / hide()', () => {
    beforeEach(() => {
      object = { enabled: false };
      controller = new BooleanController(object, 'enabled');
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

  describe('dispose()', () => {
    beforeEach(() => {
      object = { enabled: false };
      controller = new BooleanController(object, 'enabled');
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
  });
});
