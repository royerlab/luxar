// @vitest-environment jsdom
/**
 * FunctionController Tests
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { FunctionController } from '../../../../../ui/gui/controllers/function-controller';

describe('FunctionController', () => {
  let object: { reset: () => void; count: number };
  let controller: FunctionController;

  afterEach(() => {
    if (controller) {
      controller.dispose();
    }
  });

  describe('constructor', () => {
    beforeEach(() => {
      object = { reset: vi.fn(), count: 0 };
      controller = new FunctionController(object, 'reset');
    });

    it('should create button element', () => {
      const button = controller.domElement.querySelector('.luxar-gui__button');

      expect(button).toBeInstanceOf(HTMLButtonElement);
    });

    it('should use property name as button text', () => {
      const button = controller.domElement.querySelector('.luxar-gui__button');

      expect(button?.textContent).toBe('reset');
    });
  });

  describe('button click', () => {
    beforeEach(() => {
      object = {
        count: 0,
        reset() {
          this.count = 0;
        },
      };
      controller = new FunctionController(object, 'reset');
    });

    it('should call function when button clicked', () => {
      const button = controller.domElement.querySelector('.luxar-gui__button') as HTMLButtonElement;

      object.count = 10;
      button.click();

      expect(object.count).toBe(0);
    });

    it('should call function with correct context', () => {
      const spy = vi.fn(function (this: typeof object) {
        expect(this).toBe(object);
      });

      object.reset = spy;
      controller = new FunctionController(object, 'reset');

      const button = controller.domElement.querySelector('.luxar-gui__button') as HTMLButtonElement;
      button.click();

      expect(spy).toHaveBeenCalled();

      controller.dispose();
    });
  });

  describe('onChange() / onFinishChange()', () => {
    beforeEach(() => {
      object = { reset: vi.fn(), count: 0 };
      controller = new FunctionController(object, 'reset');
    });

    it('should trigger onChange when button clicked', () => {
      const onChange = vi.fn();
      controller.onChange(onChange);

      const button = controller.domElement.querySelector('.luxar-gui__button') as HTMLButtonElement;
      button.click();

      expect(onChange).toHaveBeenCalled();
    });

    it('should trigger onFinishChange when button clicked', () => {
      const onFinishChange = vi.fn();
      controller.onFinishChange(onFinishChange);

      const button = controller.domElement.querySelector('.luxar-gui__button') as HTMLButtonElement;
      button.click();

      expect(onFinishChange).toHaveBeenCalled();
    });
  });

  describe('name()', () => {
    beforeEach(() => {
      object = { reset: vi.fn(), count: 0 };
      controller = new FunctionController(object, 'reset');
    });

    it('should update button text', () => {
      controller.name('Reset to Defaults');

      const button = controller.domElement.querySelector('.luxar-gui__button');
      expect(button?.textContent).toBe('Reset to Defaults');
    });

    it('should return this for chaining', () => {
      const result = controller.name('Reset');

      expect(result).toBe(controller);
    });
  });

  describe('dispose()', () => {
    beforeEach(() => {
      object = { reset: vi.fn(), count: 0 };
      controller = new FunctionController(object, 'reset');
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
