// @vitest-environment jsdom
/**
 * Folder Tests - Folder functionality and nesting
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { Folder } from '../../../../../ui/gui/folder';

describe('Folder', () => {
  let folder: Folder;

  describe('constructor', () => {
    it('should create folder with name', () => {
      folder = new Folder('Test Folder', null, false);

      expect(folder).toBeDefined();
    });

    it('should create children container', () => {
      folder = new Folder('Test Folder', null, false);

      expect(folder['childrenContainer']).toBeInstanceOf(HTMLElement);
      expect(folder['childrenContainer'].className).toBe('luxar-gui__children');
    });

    it('should not create folder element for root (parent = null)', () => {
      folder = new Folder('Root', null, false);

      expect(folder['domElement']).toBeUndefined();
    });

    it('should create folder element for non-root', () => {
      const parent = new Folder('Parent', null, false);
      folder = new Folder('Child', parent, false);

      expect(folder['domElement']).toBeDefined();
      expect(folder['domElement']?.className).toContain('luxar-gui__folder');
    });

    it('should start open by default', () => {
      const parent = new Folder('Parent', null, false);
      folder = new Folder('Child', parent, false);

      expect(folder['isOpen']).toBe(true);
      expect(folder['domElement']?.classList.contains('luxar-gui__folder--open')).toBe(true);
    });

    it('should start closed if defaultClosed is true', () => {
      const parent = new Folder('Parent', null, true);
      folder = new Folder('Child', parent, true);

      expect(folder['isOpen']).toBe(false);
      expect(folder['domElement']?.classList.contains('luxar-gui__folder--closed')).toBe(true);
    });
  });

  describe('add()', () => {
    beforeEach(() => {
      folder = new Folder('Test', null, false);
    });

    it('should create controller and add to children', () => {
      const obj = { value: 10 };
      const controller = folder.add(obj, 'value');

      expect(controller).toBeDefined();
      expect(folder['controllers'].length).toBe(1);
      expect(folder['childrenContainer'].contains(controller.domElement)).toBe(true);
    });

    it('should auto-detect number controller', () => {
      const obj = { num: 42 };
      const controller = folder.add(obj, 'num');

      expect(controller.domElement.className).toContain('luxar-gui__controller--number');
    });

    it('should auto-detect boolean controller', () => {
      const obj = { bool: true };
      const controller = folder.add(obj, 'bool');

      expect(controller.domElement.className).toContain('luxar-gui__controller--boolean');
    });

    it('should auto-detect string controller', () => {
      const obj = { str: 'test' };
      const controller = folder.add(obj, 'str');

      expect(controller.domElement.className).toContain('luxar-gui__controller--string');
    });

    it('should auto-detect option controller from array', () => {
      const obj = { mode: 'a' };
      const controller = folder.add(obj, 'mode', ['a', 'b', 'c']);

      expect(controller.domElement.className).toContain('luxar-gui__controller--option');
    });

    it('should auto-detect function controller', () => {
      const obj = { fn: () => {} };
      const controller = folder.add(obj, 'fn');

      expect(controller.domElement.className).toContain('luxar-gui__controller--function');
    });

    it('should create number controller with range', () => {
      const obj = { value: 50 };
      const controller = folder.add(obj, 'value', 0, 100, 1);

      expect(controller.domElement.className).toContain('luxar-gui__controller--number');
      const slider = controller.domElement.querySelector('.luxar-gui__slider');
      expect(slider).toBeTruthy();
    });

    it('should throw for unsupported type', () => {
      const obj = { symbol: Symbol('test') };

      expect(() => {
        folder.add(obj, 'symbol');
      }).toThrow('Unsupported value type');
    });
  });

  describe('addFolder()', () => {
    beforeEach(() => {
      folder = new Folder('Parent', null, false);
    });

    it('should create nested folder', () => {
      const child = folder.addFolder('Child');

      expect(child).toBeDefined();
      expect(folder['folders'].length).toBe(1);
    });

    it('should add nested folder to children container', () => {
      const child = folder.addFolder('Child');

      expect(folder['childrenContainer'].contains(child['domElement']!)).toBe(true);
    });

    it('should support multiple levels of nesting', () => {
      const level1 = folder.addFolder('Level 1');
      const level2 = level1.addFolder('Level 2');
      const level3 = level2.addFolder('Level 3');

      expect(level3).toBeDefined();
      expect(folder['folders'].length).toBe(1);
      expect(level1['folders'].length).toBe(1);
      expect(level2['folders'].length).toBe(1);
    });
  });

  describe('controllersRecursive()', () => {
    beforeEach(() => {
      folder = new Folder('Test', null, false);
    });

    it('should return empty array initially', () => {
      const controllers = folder.controllersRecursive();

      expect(controllers).toEqual([]);
    });

    it('should return direct controllers', () => {
      const obj = { a: 1, b: 2 };
      folder.add(obj, 'a');
      folder.add(obj, 'b');

      const controllers = folder.controllersRecursive();

      expect(controllers.length).toBe(2);
    });

    it('should return controllers from nested folders', () => {
      const obj = { a: 1, b: 2, c: 3 };
      folder.add(obj, 'a');

      const child = folder.addFolder('Child');
      child.add(obj, 'b');
      child.add(obj, 'c');

      const controllers = folder.controllersRecursive();

      expect(controllers.length).toBe(3);
    });

    it('should return controllers from deeply nested folders', () => {
      const obj = { a: 1, b: 2, c: 3, d: 4 };
      folder.add(obj, 'a');

      const level1 = folder.addFolder('Level 1');
      level1.add(obj, 'b');

      const level2 = level1.addFolder('Level 2');
      level2.add(obj, 'c');
      level2.add(obj, 'd');

      const controllers = folder.controllersRecursive();

      expect(controllers.length).toBe(4);
    });
  });

  describe('open() / close()', () => {
    let parent: Folder;

    beforeEach(() => {
      parent = new Folder('Parent', null, false);
      folder = new Folder('Child', parent, false);
    });

    it('should open folder', () => {
      folder.close();
      folder.open();

      expect(folder['isOpen']).toBe(true);
      expect(folder['childrenContainer'].style.display).toBe('');
    });

    it('should close folder', () => {
      folder.open();
      folder.close();

      expect(folder['isOpen']).toBe(false);
      expect(folder['childrenContainer'].style.display).toBe('none');
    });

    it('should update caret icon on open', () => {
      folder.open();

      const caret = folder['domElement']?.querySelector('.luxar-gui__folder-caret');
      expect(caret?.textContent).toBe('▼');
    });

    it('should update caret icon on close', () => {
      folder.close();

      const caret = folder['domElement']?.querySelector('.luxar-gui__folder-caret');
      expect(caret?.textContent).toBe('▶');
    });

    it('should return this for chaining', () => {
      expect(folder.open()).toBe(folder);
      expect(folder.close()).toBe(folder);
    });
  });

  describe('show() / hide()', () => {
    let parent: Folder;

    beforeEach(() => {
      parent = new Folder('Parent', null, false);
      folder = new Folder('Child', parent, false);
    });

    it('should show folder', () => {
      folder.hide();
      folder.show();

      expect(folder['domElement']?.style.display).toBe('');
    });

    it('should hide folder', () => {
      folder.show();
      folder.hide();

      expect(folder['domElement']?.style.display).toBe('none');
    });

    it('should return this for chaining', () => {
      expect(folder.show()).toBe(folder);
      expect(folder.hide()).toBe(folder);
    });
  });

  describe('dispose()', () => {
    beforeEach(() => {
      folder = new Folder('Test', null, false);
    });

    it('should dispose all controllers', () => {
      const obj = { a: 1, b: 2 };
      folder.add(obj, 'a');
      folder.add(obj, 'b');

      folder['dispose']();

      expect(folder['controllers'].length).toBe(0);
    });

    it('should dispose nested folders', () => {
      const child = folder.addFolder('Child');
      const obj = { value: 1 };
      child.add(obj, 'value');

      folder['dispose']();

      expect(folder['folders'].length).toBe(0);
      expect(child['controllers'].length).toBe(0);
    });
  });
});
