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
    it('should create checkbox element', () => {
      object = { enabled: true };
      controller = new BooleanController(object, 'enabled');

      const checkbox = controller.domElement.querySelector('.luxar-gui__checkbox');
      expect(checkbox).toBeInstanceOf(HTMLInputElement);
    });

    it('should set initial checked state', () => {
      object = { enabled: true };
      controller = new BooleanController(object, 'enabled');

      const checkbox = controller.domElement.querySelector(
        '.luxar-gui__checkbox'
      ) as HTMLInputElement;
      expect(checkbox.checked).toBe(true);
    });

    it('should wrap checkbox in label', () => {
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

    it('should set value', () => {
      controller.setValue(true);

      expect(object.enabled).toBe(true);
      expect(controller.getValue()).toBe(true);
    });

    it('should update checkbox display', () => {
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

    it('should sync checkbox with object value', () => {
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
