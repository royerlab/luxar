/**
 * Base Controller class
 *
 * All specialized controllers (Number, Boolean, String, Option, Function)
 * extend this base class.
 *
 * Responsibilities:
 * - Manage target object and property binding
 * - Provide common API (name, show, hide, onChange, etc.)
 * - Track event listeners for cleanup via EventManager
 * - Create base DOM structure
 */

import type { ChangeCallback, FinishChangeCallback, ControllerType } from './types';
import { EventManager } from '../dom/event-manager';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export abstract class Controller<T = any> {
  /** Target object containing the property */
  protected object: Record<string, unknown>;

  /** Property name being controlled */
  protected property: string;

  /** Controller type (for internal discrimination) */
  protected abstract type: ControllerType;

  /** Display label (defaults to property name) */
  protected label: string;

  /** Root DOM element for this controller */
  public domElement!: HTMLElement;

  /** Input element (varies by controller type) */
  public $input?: HTMLInputElement | HTMLSelectElement;

  /** Change callbacks */
  protected changeCallbacks: ChangeCallback<T>[] = [];

  /** Finish change callbacks */
  protected finishChangeCallbacks: FinishChangeCallback<T>[] = [];

  /** Event manager for cleanup */
  protected eventManager: EventManager;

  /** Visibility state */
  protected isVisible: boolean = true;

  /**
   * Create a controller
   *
   * @param object - Target object
   * @param property - Property name
   */
  constructor(object: Record<string, unknown>, property: string) {
    this.object = object;
    this.property = property;
    this.label = property; // Default label
    this.eventManager = new EventManager();

    // NOTE: domElement is created by calling initializeDOMElement() from subclass
    // This is called AFTER the subclass constructor finishes initialization
  }

  /**
   * Initialize DOM element (must be called by subclass after all properties are set)
   */
  protected initializeDOMElement(): void {
    this.domElement = this.createDOMElement();
  }

  /**
   * Create the DOM structure for this controller
   *
   * Must be implemented by subclasses
   */
  protected abstract createDOMElement(): HTMLElement;

  /**
   * Set the display label
   *
   * @param label - Label text
   * @returns this (for chaining)
   */
  public name(label: string): this {
    this.label = label;
    const labelElement = this.domElement.querySelector('.luxar-gui__controller-name');
    if (labelElement) {
      labelElement.textContent = label;
    }
    return this;
  }

  /**
   * Get current value from target object
   *
   * @returns Current value
   */
  public getValue(): T {
    // The target object holds property values keyed by name; the
    // `T` is declared by the concrete subclass (NumberController,
    // StringController, etc.), so this cast bridges the structural
    // `unknown` of the bag to the controller's declared value type.
    return this.object[this.property] as T;
  }

  /**
   * Set value on target object and update display
   *
   * NOTE: Does NOT trigger onChange callbacks (matches lil-gui behavior).
   * Only user interaction triggers callbacks.
   *
   * @param value - New value
   * @returns this (for chaining)
   */
  public setValue(value: T): this {
    this.object[this.property] = value;
    this.updateDisplay();
    return this;
  }

  /**
   * Update display to reflect current value
   *
   * Must be implemented by subclasses
   *
   * @returns this (for chaining)
   */
  public abstract updateDisplay(): this;

  /**
   * Register a change callback (fired on every change)
   *
   * @param callback - Change handler
   * @returns this (for chaining)
   */
  public onChange(callback: ChangeCallback<T>): this {
    this.changeCallbacks.push(callback);
    return this;
  }

  /**
   * Register a finish change callback (fired on blur/mouseup)
   *
   * @param callback - Finish change handler
   * @returns this (for chaining)
   */
  public onFinishChange(callback: FinishChangeCallback<T>): this {
    this.finishChangeCallbacks.push(callback);
    return this;
  }

  /**
   * Trigger all change callbacks
   */
  protected triggerChange(): void {
    const value = this.getValue();
    for (const callback of this.changeCallbacks) {
      callback(value);
    }
  }

  /**
   * Trigger all finish change callbacks
   */
  protected triggerFinishChange(): void {
    const value = this.getValue();
    for (const callback of this.finishChangeCallbacks) {
      callback(value);
    }
  }

  /**
   * Show the controller
   *
   * @returns this (for chaining)
   */
  public show(): this {
    this.isVisible = true;
    this.domElement.style.display = '';
    return this;
  }

  /**
   * Hide the controller
   *
   * @returns this (for chaining)
   */
  public hide(): this {
    this.isVisible = false;
    this.domElement.style.display = 'none';
    return this;
  }

  /**
   * Set min value (number controllers only - overridden in NumberController)
   */
  public min(_value: number): this {
    return this;
  }

  /**
   * Set max value (number controllers only - overridden in NumberController)
   */
  public max(_value: number): this {
    return this;
  }

  /**
   * Set step value (number controllers only - overridden in NumberController)
   */
  public step(_value: number): this {
    return this;
  }

  /**
   * Dispose the controller and clean up resources
   *
   * CRITICAL: Must remove all event listeners to prevent memory leaks
   */
  public dispose(): void {
    // Remove all event listeners
    this.eventManager.removeAll();

    // Clear callbacks
    this.changeCallbacks = [];
    this.finishChangeCallbacks = [];

    // Remove from DOM
    if (this.domElement.parentElement) {
      this.domElement.parentElement.removeChild(this.domElement);
    }
  }

  /**
   * Create base DOM structure common to all controllers
   *
   * @returns Base controller element
   */
  protected createBaseElement(): HTMLElement {
    const container = document.createElement('div');
    container.className = `luxar-gui__controller luxar-gui__controller--${this.type}`;

    // Label
    const label = document.createElement('label');
    label.className = 'luxar-gui__controller-name';
    label.textContent = this.label;
    container.appendChild(label);

    // Widget container (for input elements)
    const widget = document.createElement('div');
    widget.className = 'luxar-gui__controller-widget';
    container.appendChild(widget);

    return container;
  }
}
