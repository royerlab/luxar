/**
 * GUI Tests - Main GUI class functionality
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { GUI } from '../../../../../ui/gui/gui';

describe('GUI', () => {
  let gui: GUI;

  afterEach(() => {
    if (gui) {
      gui.destroy();
    }
  });

  describe('constructor', () => {
    it('should create GUI with default options', () => {
      gui = new GUI();

      expect(gui.domElement).toBeInstanceOf(HTMLElement);
      // Carries the base class + the glass-surface marker (shared theme hook).
      expect(gui.domElement.classList.contains('luxar-gui')).toBe(true);
      expect(gui.domElement.classList.contains('luxar-glass-surface')).toBe(true);
    });

    it('should apply title option', () => {
      gui = new GUI({ title: 'Test GUI' });

      const title = gui.domElement.querySelector('.luxar-gui__title');
      expect(title?.textContent).toBe('Test GUI');
    });

    it('should apply width option', () => {
      gui = new GUI({ width: 350 });

      expect(gui.domElement.style.width).toBe('350px');
    });

    it('should append to container', () => {
      const container = document.createElement('div');
      gui = new GUI({ container });

      expect(container.contains(gui.domElement)).toBe(true);
    });

    it('should append to document.body by default', () => {
      gui = new GUI();

      expect(document.body.contains(gui.domElement)).toBe(true);
    });

    it('should start hidden', () => {
      gui = new GUI();

      expect(gui.domElement.style.display).toBe('none');
    });
  });

  describe('show()', () => {
    beforeEach(() => {
      gui = new GUI();
    });

    it('should show the GUI', () => {
      gui.show();

      expect(gui.domElement.style.display).toBe('');
    });

    it('should return this for chaining', () => {
      const result = gui.show();

      expect(result).toBe(gui);
    });
  });

  describe('hide()', () => {
    beforeEach(() => {
      gui = new GUI();
      gui.show();
    });

    it('should hide the GUI', () => {
      gui.hide();

      expect(gui.domElement.style.display).toBe('none');
    });

    it('should return this for chaining', () => {
      const result = gui.hide();

      expect(result).toBe(gui);
    });
  });

  describe('add()', () => {
    beforeEach(() => {
      gui = new GUI();
    });

    it('should create number controller with range', () => {
      const obj = { value: 50 };
      const controller = gui.add(obj, 'value', 0, 100);

      expect(controller).toBeDefined();
      expect(controller.domElement).toBeInstanceOf(HTMLElement);
      expect(gui.domElement.contains(controller.domElement)).toBe(true);
    });

    it('should create boolean controller', () => {
      const obj = { enabled: true };
      const controller = gui.add(obj, 'enabled');

      expect(controller).toBeDefined();
      const checkbox = controller.domElement.querySelector('.luxar-gui__checkbox');
      expect(checkbox).toBeInstanceOf(HTMLInputElement);
    });

    it('should create string controller', () => {
      const obj = { name: 'test' };
      const controller = gui.add(obj, 'name');

      expect(controller).toBeDefined();
      const input = controller.domElement.querySelector('.luxar-gui__input--string');
      expect(input).toBeInstanceOf(HTMLInputElement);
    });

    it('should create option controller from array', () => {
      const obj = { mode: 'orbit' };
      const controller = gui.add(obj, 'mode', ['orbit', 'ortho', 'fly']);

      expect(controller).toBeDefined();
      const select = controller.domElement.querySelector('.luxar-gui__select');
      expect(select).toBeInstanceOf(HTMLSelectElement);
    });

    it('should create function controller', () => {
      const obj = { reset: () => {} };
      const controller = gui.add(obj, 'reset');

      expect(controller).toBeDefined();
      const button = controller.domElement.querySelector('.luxar-gui__button');
      expect(button).toBeInstanceOf(HTMLButtonElement);
    });
  });

  describe('addFolder()', () => {
    beforeEach(() => {
      gui = new GUI();
    });

    it('should create a folder', () => {
      const folder = gui.addFolder('Test Folder');

      expect(folder).toBeDefined();
      expect(folder.domElement).toBeDefined();
    });

    it('should add folder to GUI', () => {
      gui.addFolder('Test Folder');

      const folderElement = gui.domElement.querySelector('.luxar-gui__folder');
      expect(folderElement).toBeTruthy();
    });

    it('should support nested controllers', () => {
      const folder = gui.addFolder('Test Folder');
      const obj = { value: 10 };
      const controller = folder.add(obj, 'value');

      expect(controller).toBeDefined();
      expect(folder.domElement?.contains(controller.domElement)).toBe(true);
    });
  });

  describe('controllersRecursive()', () => {
    beforeEach(() => {
      gui = new GUI();
    });

    it('should return empty array initially', () => {
      const controllers = gui.controllersRecursive();

      expect(controllers).toEqual([]);
    });

    it('should return all controllers', () => {
      const obj = { a: 1, b: 2, c: 3 };
      gui.add(obj, 'a');
      gui.add(obj, 'b');
      gui.add(obj, 'c');

      const controllers = gui.controllersRecursive();

      expect(controllers.length).toBe(3);
    });

    it('should include controllers from nested folders', () => {
      const obj = { a: 1, b: 2, c: 3 };
      gui.add(obj, 'a');

      const folder = gui.addFolder('Folder');
      folder.add(obj, 'b');
      folder.add(obj, 'c');

      const controllers = gui.controllersRecursive();

      expect(controllers.length).toBe(3);
    });
  });

  describe('destroy()', () => {
    it('should remove GUI from DOM', () => {
      gui = new GUI();
      const parent = gui.domElement.parentElement;

      gui.destroy();

      expect(parent?.contains(gui.domElement)).toBe(false);
    });

    it('should dispose all controllers', () => {
      gui = new GUI();
      const obj = { value: 10 };
      const controller = gui.add(obj, 'value');

      gui.destroy();

      // Controller should be removed from DOM
      expect(document.body.contains(controller.domElement)).toBe(false);
    });

    it('should be safe to call multiple times', () => {
      gui = new GUI();

      expect(() => {
        gui.destroy();
        gui.destroy();
      }).not.toThrow();
    });
  });

  describe('Integration', () => {
    beforeEach(() => {
      gui = new GUI({ title: 'Test GUI' });
    });

    it('should support chaining', () => {
      const result = gui.show().hide().show();

      expect(result).toBe(gui);
      expect(gui.domElement.style.display).toBe('');
    });

    it('should handle complex nested structure', () => {
      const settings = {
        enabled: true,
        value: 50,
        mode: 'test',
        action: () => {},
      };

      const folder1 = gui.addFolder('Folder 1');
      folder1.add(settings, 'enabled');
      folder1.add(settings, 'value', 0, 100);

      const folder2 = gui.addFolder('Folder 2');
      folder2.add(settings, 'mode', ['test', 'prod']);
      folder2.add(settings, 'action');

      const controllers = gui.controllersRecursive();

      expect(controllers.length).toBe(4);
    });
  });
});
