/**
 * StringController Tests
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { StringController } from '../../../../../ui/gui/controllers/string-controller';

describe('StringController', () => {
  let object: { name: string };
  let controller: StringController;

  afterEach(() => {
    if (controller) {
      controller.dispose();
    }
  });

  describe('constructor', () => {
    beforeEach(() => {
      object = { name: 'test' };
      controller = new StringController(object, 'name');
    });

    it('should create text input element', () => {
      const input = controller.domElement.querySelector('.luxar-gui__input--string');

      expect(input).toBeInstanceOf(HTMLInputElement);
      expect((input as HTMLInputElement).type).toBe('text');
    });

    it('should set initial value', () => {
      const input = controller.domElement.querySelector(
        '.luxar-gui__input--string'
      ) as HTMLInputElement;

      expect(input.value).toBe('test');
    });

    it('should expose $input property', () => {
      expect(controller.$input).toBeInstanceOf(HTMLInputElement);
    });
  });

  describe('setValue() / getValue()', () => {
    beforeEach(() => {
      object = { name: 'initial' };
      controller = new StringController(object, 'name');
    });

    it('should set value', () => {
      controller.setValue('updated');

      expect(object.name).toBe('updated');
      expect(controller.getValue()).toBe('updated');
    });

    it('should update input display', () => {
      controller.setValue('new value');

      const input = controller.domElement.querySelector(
        '.luxar-gui__input--string'
      ) as HTMLInputElement;
      expect(input.value).toBe('new value');
    });
  });

  describe('updateDisplay()', () => {
    beforeEach(() => {
      object = { name: 'test' };
      controller = new StringController(object, 'name');
    });

    it('should sync input with object value', () => {
      object.name = 'changed';
      controller.updateDisplay();

      const input = controller.domElement.querySelector(
        '.luxar-gui__input--string'
      ) as HTMLInputElement;
      expect(input.value).toBe('changed');
    });
  });

  describe('onChange()', () => {
    beforeEach(() => {
      object = { name: 'test' };
      controller = new StringController(object, 'name');
    });

    it('should trigger onChange on input', () => {
      const onChange = vi.fn();
      controller.onChange(onChange);

      const input = controller.domElement.querySelector(
        '.luxar-gui__input--string'
      ) as HTMLInputElement;
      input.value = 'typing...';
      input.dispatchEvent(new Event('input'));

      expect(onChange).toHaveBeenCalledWith('typing...');
    });

    it('should trigger onChange on change event', () => {
      const onChange = vi.fn();
      controller.onChange(onChange);

      const input = controller.domElement.querySelector(
        '.luxar-gui__input--string'
      ) as HTMLInputElement;
      input.value = 'final';
      input.dispatchEvent(new Event('change'));

      expect(onChange).toHaveBeenCalledWith('final');
    });

    it('should update object value on input', () => {
      const input = controller.domElement.querySelector(
        '.luxar-gui__input--string'
      ) as HTMLInputElement;

      input.value = 'live update';
      input.dispatchEvent(new Event('input'));

      expect(object.name).toBe('live update');
    });
  });

  describe('onFinishChange()', () => {
    beforeEach(() => {
      object = { name: 'test' };
      controller = new StringController(object, 'name');
    });

    it('should trigger onFinishChange on change event', () => {
      const onFinishChange = vi.fn();
      controller.onFinishChange(onFinishChange);

      const input = controller.domElement.querySelector(
        '.luxar-gui__input--string'
      ) as HTMLInputElement;
      input.value = 'final';
      input.dispatchEvent(new Event('change'));

      expect(onFinishChange).toHaveBeenCalledWith('final');
    });
  });

  describe('name()', () => {
    beforeEach(() => {
      object = { name: 'test' };
      controller = new StringController(object, 'name');
    });

    it('should set label text', () => {
      controller.name('User Name');

      const label = controller.domElement.querySelector('.luxar-gui__controller-name');
      expect(label?.textContent).toBe('User Name');
    });

    it('should return this for chaining', () => {
      const result = controller.name('Label');

      expect(result).toBe(controller);
    });
  });

  describe('dispose()', () => {
    beforeEach(() => {
      object = { name: 'test' };
      controller = new StringController(object, 'name');
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
