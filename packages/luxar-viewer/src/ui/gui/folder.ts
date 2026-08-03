/**
 * Folder class - collapsible container for controllers
 *
 * Responsibilities:
 * - Create and manage child controllers
 * - Support nested folders
 * - Handle open/close state
 * - Provide recursive controller access
 * - Auto-detect controller type from value
 */

import { Controller } from './controller';
import type { ControllerOptions } from './types';
import { NumberController } from './controllers/number-controller';
import { BooleanController } from './controllers/boolean-controller';
import { StringController } from './controllers/string-controller';
import { OptionController } from './controllers/option-controller';
import { FunctionController } from './controllers/function-controller';
import { EventManager } from './dom/event-manager';

export class Folder {
  /** Display name */
  protected name: string;

  /** Optional leading icon (inline SVG string, rail-style). */
  protected icon?: string;

  /** Parent folder (null for root GUI) */
  protected parent: Folder | null;

  /** Child controllers */
  protected controllers: Controller[] = [];

  /** Child folders */
  protected folders: Folder[] = [];

  /** DOM element for children */
  protected childrenContainer: HTMLElement;

  /** DOM element for folder (if not root) - can be overridden in subclasses */
  public domElement?: HTMLElement;

  /** Open/closed state */
  private isOpen: boolean;

  /** Default open/close state for new folders */
  private defaultClosed: boolean;

  /** Event manager for cleanup */
  protected eventManager: EventManager;

  /**
   * Create a folder
   *
   * @param name - Display name
   * @param parent - Parent folder (null for root GUI)
   * @param defaultClosed - Whether folders start closed
   * @param icon - Optional leading icon (inline SVG string, rail-style)
   */
  constructor(name: string, parent: Folder | null, defaultClosed: boolean = false, icon?: string) {
    this.name = name;
    this.icon = icon;
    this.parent = parent;
    this.defaultClosed = defaultClosed;
    this.isOpen = !defaultClosed;
    this.eventManager = new EventManager();

    this.childrenContainer = document.createElement('div');
    this.childrenContainer.className = 'luxar-gui__children';

    // If not root, create folder UI
    if (parent !== null) {
      this.domElement = this.createFolderElement();
    }
  }

  /**
   * Create folder DOM structure
   */
  private createFolderElement(): HTMLElement {
    const folder = document.createElement('div');
    folder.className = 'luxar-gui__folder';

    // Title bar (clickable, keyboard-accessible). Layout: caret → [icon] → label.
    // The label lives in its own span (not the title's textContent) so a leading
    // icon can sit between the caret and the text without either clobbering the
    // other. The caret stays the `.luxar-gui__folder-caret` element that
    // open/close looks up via querySelector.
    const title = document.createElement('div');
    title.className = 'luxar-gui__folder-title';
    title.setAttribute('role', 'button');
    title.setAttribute('tabindex', '0');
    title.setAttribute('aria-expanded', String(this.isOpen));

    // Caret icon
    const caret = document.createElement('span');
    caret.className = 'luxar-gui__folder-caret';
    caret.textContent = this.isOpen ? '▼' : '▶';
    title.appendChild(caret);

    // Optional leading icon (inline SVG, styled by CSS via currentColor).
    if (this.icon) {
      const iconSpan = document.createElement('span');
      iconSpan.className = 'luxar-gui__folder-icon';
      iconSpan.setAttribute('aria-hidden', 'true');
      iconSpan.innerHTML = this.icon;
      title.appendChild(iconSpan);
    }

    // Label text (accessible name for the role="button" title).
    const label = document.createElement('span');
    label.className = 'luxar-gui__folder-label';
    label.textContent = this.name;
    title.appendChild(label);

    const toggleFolder = () => {
      if (this.isOpen) {
        this.close();
      } else {
        this.open();
      }
    };

    // Toggle on click (tracked by EventManager for cleanup)
    this.eventManager.add(title, 'click', toggleFolder);

    // Toggle on Enter/Space for keyboard accessibility
    this.eventManager.add(title, 'keydown', (e: Event) => {
      const keyEvent = e as KeyboardEvent;
      if (keyEvent.key === 'Enter' || keyEvent.key === ' ') {
        keyEvent.preventDefault();
        toggleFolder();
      }
    });

    folder.appendChild(title);
    folder.appendChild(this.childrenContainer);

    // Set initial state (pass element directly since this.domElement not set yet)
    this.updateOpenStateForElement(folder);

    return folder;
  }

  /**
   * Update open state for a specific element (used during construction)
   */
  private updateOpenStateForElement(element: HTMLElement): void {
    const caret = element.querySelector('.luxar-gui__folder-caret');
    if (caret) {
      caret.textContent = this.isOpen ? '▼' : '▶';
    }

    const title = element.querySelector('.luxar-gui__folder-title');
    if (title) {
      title.setAttribute('aria-expanded', String(this.isOpen));
    }

    this.childrenContainer.style.display = this.isOpen ? '' : 'none';

    if (this.isOpen) {
      element.classList.add('luxar-gui__folder--open');
      element.classList.remove('luxar-gui__folder--closed');
    } else {
      element.classList.remove('luxar-gui__folder--open');
      element.classList.add('luxar-gui__folder--closed');
    }
  }

  /**
   * Add a controller for a property
   *
   * Auto-detects controller type based on value type and arguments
   *
   * @param object - Target object
   * @param property - Property name
   * @param arg1 - Min value OR options array/object
   * @param arg2 - Max value (if arg1 is min)
   * @param arg3 - Step value (if arg1 and arg2 are min/max)
   * @returns Created controller
   */
  // The DSL accepts arbitrary user-typed config objects (e.g., the
  // strongly-typed RenderingSettings interface) that lack an index
  // signature. Tightening to `Record<string, unknown>` would force every
  // typed caller to add `as any`. The looser `any` here keeps the public
  // surface ergonomic; internal Controller storage uses
  // Record<string, unknown> so the unsafe surface is just this entry point.
  /* eslint-disable @typescript-eslint/no-explicit-any */
  public add(
    object: Record<string, any>,
    property: string,
    arg1?: number | string[] | Record<string, any>,
    arg2?: number,
    arg3?: number
  ): Controller {
    /* eslint-enable @typescript-eslint/no-explicit-any */
    const value = object[property];
    const valueType = typeof value;

    let controller: Controller;

    // Type detection logic (order matters - most specific first)
    if (valueType === 'function') {
      // Function controller (button)
      controller = new FunctionController(object, property);
    } else if (valueType === 'boolean') {
      // Boolean controller (checkbox)
      controller = new BooleanController(object, property);
    } else if (valueType === 'string') {
      if (Array.isArray(arg1) || (arg1 && typeof arg1 === 'object')) {
        // Option controller (dropdown) with string value
        controller = new OptionController(object, property, { options: arg1 });
      } else {
        // String controller (text input)
        controller = new StringController(object, property);
      }
    } else if (valueType === 'number') {
      if (
        Array.isArray(arg1) ||
        (arg1 && typeof arg1 === 'object' && !Number.isFinite(arg1 as unknown as number))
      ) {
        // Option controller (dropdown) with number value
        controller = new OptionController(object, property, { options: arg1 });
      } else if (typeof arg1 === 'number') {
        // Number controller (slider + input) with range
        const options: ControllerOptions = {
          min: arg1,
          max: arg2,
          step: arg3,
        };
        controller = new NumberController(object, property, options);
      } else {
        // Number controller without range (input only)
        controller = new NumberController(object, property, {});
      }
    } else {
      throw new Error(`Unsupported value type: ${valueType} for property: ${property}`);
    }

    // Add to children
    this.controllers.push(controller);
    this.childrenContainer.appendChild(controller.domElement);

    return controller;
  }

  /**
   * Add a nested folder
   *
   * @param name - Folder name
   * @param icon - Optional leading icon (inline SVG string, rail-style)
   * @returns Created folder
   */
  public addFolder(name: string, icon?: string): Folder {
    const folder = new Folder(name, this, this.defaultClosed, icon);
    this.folders.push(folder);

    if (folder.domElement) {
      this.childrenContainer.appendChild(folder.domElement);
    }

    return folder;
  }

  /**
   * Get all controllers recursively (including nested folders)
   *
   * @returns Array of all controllers
   */
  public controllersRecursive(): Controller[] {
    const result = [...this.controllers];
    for (const folder of this.folders) {
      result.push(...folder.controllersRecursive());
    }
    return result;
  }

  /**
   * Open the folder
   *
   * @returns this (for chaining)
   */
  public open(): this {
    this.isOpen = true;
    this.updateOpenState();
    return this;
  }

  /**
   * Close the folder
   *
   * @returns this (for chaining)
   */
  public close(): this {
    this.isOpen = false;
    this.updateOpenState();
    return this;
  }

  /**
   * Show the folder
   *
   * @returns this (for chaining)
   */
  public show(): this {
    if (this.domElement) {
      this.domElement.style.display = '';
    }
    return this;
  }

  /**
   * Hide the folder
   *
   * @returns this (for chaining)
   */
  public hide(): this {
    if (this.domElement) {
      this.domElement.style.display = 'none';
    }
    return this;
  }

  /**
   * Update DOM to reflect open/closed state
   */
  private updateOpenState(): void {
    if (!this.domElement) return;

    const caret = this.domElement.querySelector('.luxar-gui__folder-caret');
    if (caret) {
      caret.textContent = this.isOpen ? '▼' : '▶';
    }

    const title = this.domElement.querySelector('.luxar-gui__folder-title');
    if (title) {
      title.setAttribute('aria-expanded', String(this.isOpen));
    }

    this.childrenContainer.style.display = this.isOpen ? '' : 'none';

    if (this.isOpen) {
      this.domElement.classList.add('luxar-gui__folder--open');
      this.domElement.classList.remove('luxar-gui__folder--closed');
    } else {
      this.domElement.classList.remove('luxar-gui__folder--open');
      this.domElement.classList.add('luxar-gui__folder--closed');
    }
  }

  /**
   * Dispose all controllers and folders
   */
  protected dispose(): void {
    // Clean up event listeners
    this.eventManager.removeAll();

    // Dispose all controllers
    for (const controller of this.controllers) {
      controller.dispose();
    }

    // Dispose all folders
    for (const folder of this.folders) {
      folder.dispose();
    }

    this.controllers = [];
    this.folders = [];
  }
}
