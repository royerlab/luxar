/**
 * Main GUI class - root container for all controls
 *
 * Extends Folder to inherit controller management functionality.
 *
 * Responsibilities:
 * - Create and manage the root DOM element
 * - Provide factory methods for creating controllers
 * - Manage visibility state
 * - Handle cleanup and disposal
 */

import { Folder } from './folder';
import type { GUIOptions } from './types';

/** Internal type with optional onClose */
interface InternalGUIOptions {
  title: string;
  width: number;
  closeFolders: boolean;
  container: HTMLElement;
  onClose?: () => void;
}

export class GUI extends Folder {
  /** Root DOM element - override parent optional type */
  declare public domElement: HTMLElement;

  /** GUI configuration */
  private guiOptions: InternalGUIOptions;

  /**
   * Create a new GUI instance
   *
   * @param options - GUI configuration options
   *
   * @example
   * ```typescript
   * const gui = new GUI({
   *   title: 'Rendering Controls',
   *   width: 300,
   *   closeFolders: false
   * });
   *
   * gui.add(settings, 'fov', 10, 170, 1).name('Field of View');
   * ```
   */
  constructor(options: GUIOptions = {}) {
    // Default options
    const fullOptions: InternalGUIOptions = {
      title: options.title ?? 'Controls',
      width: options.width ?? 300,
      closeFolders: options.closeFolders ?? false,
      container: options.container ?? document.body,
      onClose: options.onClose,
    };

    // Initialize Folder base class with null parent (this is root)
    super(fullOptions.title, null, fullOptions.closeFolders);

    this.guiOptions = fullOptions;

    // Create root DOM structure
    this.domElement = this.createRootElement();

    // Append to container
    fullOptions.container.appendChild(this.domElement);

    // Start hidden (matches lil-gui behavior)
    this.domElement.style.display = 'none';
  }

  /**
   * Create the root DOM element with header and title
   */
  private createRootElement(): HTMLElement {
    const root = document.createElement('div');
    root.className = 'luxar-gui';
    root.style.width = `${this.guiOptions.width}px`;

    // Header element (contains title and optional close button)
    const header = document.createElement('div');
    header.className = 'luxar-gui__header';

    // Title element
    const title = document.createElement('div');
    title.className = 'luxar-gui__title';
    title.textContent = this.guiOptions.title;
    header.appendChild(title);

    // Close button (only if onClose callback is provided)
    if (this.guiOptions.onClose) {
      const closeBtn = document.createElement('button');
      closeBtn.className = 'luxar-gui__close-btn';
      closeBtn.textContent = '×';
      closeBtn.addEventListener('click', () => {
        this.guiOptions.onClose?.();
      });
      header.appendChild(closeBtn);
    }

    root.appendChild(header);

    // Children container (from Folder base class)
    root.appendChild(this.childrenContainer);

    return root;
  }

  /**
   * Show the GUI panel
   *
   * @returns this (for chaining)
   */
  public show(): this {
    this.domElement.style.display = '';
    return this;
  }

  /**
   * Hide the GUI panel
   *
   * @returns this (for chaining)
   */
  public hide(): this {
    this.domElement.style.display = 'none';
    return this;
  }

  /**
   * Destroy the GUI and clean up all resources
   *
   * CRITICAL: Must remove ALL event listeners to prevent memory leaks
   */
  public destroy(): void {
    // Dispose all controllers and folders recursively (includes EventManager cleanup)
    this.dispose();

    // Remove from DOM
    if (this.domElement.parentElement) {
      this.domElement.parentElement.removeChild(this.domElement);
    }
  }
}
