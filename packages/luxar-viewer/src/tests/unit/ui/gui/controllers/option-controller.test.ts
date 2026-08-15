// @vitest-environment jsdom
/**
 * OptionController Tests
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OptionController } from '../../../../../ui/gui/controllers/option-controller';

describe('OptionController', () => {
  let object: { mode: string };
  let controller: OptionController;

  afterEach(() => {
    if (controller) {
      controller.dispose();
    }
  });

  describe('with array options', () => {
    beforeEach(() => {
      object = { mode: 'orbit' };
      controller = new OptionController(object, 'mode', {
        options: ['orbit', 'ortho', 'fly'],
      });
    });

    it('should create select element', () => {
      const select = controller.domElement.querySelector('.luxar-gui__select');

      expect(select).toBeInstanceOf(HTMLSelectElement);
    });

    it('should create option elements', () => {
      const options = controller.domElement.querySelectorAll('option');

      expect(options.length).toBe(3);
      expect(options[0].textContent).toBe('orbit');
      expect(options[1].textContent).toBe('ortho');
      expect(options[2].textContent).toBe('fly');
    });

    it('should set initial selected value', () => {
      const select = controller.domElement.querySelector('.luxar-gui__select') as HTMLSelectElement;

      expect(select.value).toBe('orbit');
    });

    it('should update object when selection changes', () => {
      const select = controller.domElement.querySelector('.luxar-gui__select') as HTMLSelectElement;

      select.value = 'fly';
      select.dispatchEvent(new Event('change'));

      expect(object.mode).toBe('fly');
    });
  });

  describe('with object options (labeled)', () => {
    beforeEach(() => {
      object = { mode: 'orbit' };
      controller = new OptionController(object, 'mode', {
        options: {
          'Orbit Camera': 'orbit',
          'Ortho Camera': 'ortho',
          'Fly Camera': 'fly',
        },
      });
    });

    it('should create options with labels', () => {
      const options = controller.domElement.querySelectorAll('option');

      expect(options.length).toBe(3);
      expect(options[0].textContent).toBe('Orbit Camera');
      expect(options[1].textContent).toBe('Ortho Camera');
      expect(options[2].textContent).toBe('Fly Camera');
    });

    it('should map labels to values', () => {
      const select = controller.domElement.querySelector('.luxar-gui__select') as HTMLSelectElement;

      select.value = 'Fly Camera';
      select.dispatchEvent(new Event('change'));

      expect(object.mode).toBe('fly');
    });

    it('should select correct label for current value', () => {
      object.mode = 'ortho';
      controller.updateDisplay();

      const select = controller.domElement.querySelector('.luxar-gui__select') as HTMLSelectElement;
      expect(select.value).toBe('Ortho Camera');
    });
  });

  // [R11/D-G6][P5] Empty options array / empty options dict — the
  // controller must not throw, must produce a select element, and must
  // produce zero <option> children. A regression that called
  // `options[0]` without a length check would crash on construction.
  describe('with empty options', () => {
    it('does not throw and produces a select with zero options when options is []', () => {
      const obj: { mode: string } = { mode: 'orbit' };
      const c = new OptionController(obj, 'mode', { options: [] });
      try {
        const select = c.domElement.querySelector('.luxar-gui__select');
        expect(select).toBeInstanceOf(HTMLSelectElement);
        expect(c.domElement.querySelectorAll('option')).toHaveLength(0);
      } finally {
        c.dispose();
      }
    });

    it('does not throw and produces a select with zero options when options is {}', () => {
      const obj: { mode: string } = { mode: 'orbit' };
      const c = new OptionController(obj, 'mode', { options: {} });
      try {
        const select = c.domElement.querySelector('.luxar-gui__select');
        expect(select).toBeInstanceOf(HTMLSelectElement);
        expect(c.domElement.querySelectorAll('option')).toHaveLength(0);
      } finally {
        c.dispose();
      }
    });
  });

  describe('setValue() / getValue()', () => {
    beforeEach(() => {
      object = { mode: 'orbit' };
      controller = new OptionController(object, 'mode', {
        options: ['orbit', 'ortho', 'fly'],
      });
    });

    it('should set value and update display', () => {
      controller.setValue('fly');

      expect(object.mode).toBe('fly');
      expect(controller.getValue()).toBe('fly');

      const select = controller.domElement.querySelector('.luxar-gui__select') as HTMLSelectElement;
      expect(select.value).toBe('fly');
    });
  });

  describe('onChange() / onFinishChange()', () => {
    beforeEach(() => {
      object = { mode: 'orbit' };
      controller = new OptionController(object, 'mode', {
        options: ['orbit', 'ortho', 'fly'],
      });
    });

    it('should trigger onChange when selection changes', () => {
      const onChange = vi.fn();
      controller.onChange(onChange);

      const select = controller.domElement.querySelector('.luxar-gui__select') as HTMLSelectElement;
      select.value = 'ortho';
      select.dispatchEvent(new Event('change'));

      expect(onChange).toHaveBeenCalledWith('ortho');
    });

    it('should trigger onFinishChange when selection changes', () => {
      const onFinishChange = vi.fn();
      controller.onFinishChange(onFinishChange);

      const select = controller.domElement.querySelector('.luxar-gui__select') as HTMLSelectElement;
      select.value = 'fly';
      select.dispatchEvent(new Event('change'));

      expect(onFinishChange).toHaveBeenCalledWith('fly');
    });
  });

  describe('dispose()', () => {
    beforeEach(() => {
      object = { mode: 'orbit' };
      controller = new OptionController(object, 'mode', { options: ['orbit', 'fly'] });
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
