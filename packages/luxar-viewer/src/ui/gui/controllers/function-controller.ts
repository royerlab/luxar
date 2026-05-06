/**
 * FunctionController - Button for function calls
 */

import { Controller } from '../core/controller';
import { ControllerType } from '../core/types';

export class FunctionController extends Controller<Function> {
  protected type = ControllerType.FUNCTION;

  private button!: HTMLButtonElement;

  constructor(object: Record<string, unknown>, property: string) {
    super(object, property);

    // Initialize DOM
    this.initializeDOMElement();
  }

  protected createDOMElement(): HTMLElement {
    const container = document.createElement('div');
    container.className = `luxar-gui__controller luxar-gui__controller--${this.type}`;

    // Button
    this.button = document.createElement('button');
    this.button.className = 'luxar-gui__button';
    this.button.textContent = this.label;

    // Call function on click
    this.eventManager.add(this.button, 'click', () => {
      const fn = this.object[this.property];
      if (typeof fn === 'function') {
        fn.call(this.object);
        this.triggerChange();
        this.triggerFinishChange();
      }
    });

    container.appendChild(this.button);

    return container;
  }

  public updateDisplay(): this {
    // Function controllers don't need to update display
    return this;
  }

  // Override name() to update button text
  public name(label: string): this {
    this.label = label;
    this.button.textContent = label;
    return this;
  }
}
