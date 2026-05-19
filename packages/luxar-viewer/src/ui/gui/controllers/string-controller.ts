/**
 * StringController - Text input for string values
 */

import { Controller } from '../controller';
import { ControllerType } from '../types';
import { applyAutoBlur } from '../format/auto-blur';

export class StringController extends Controller<string> {
  protected type = ControllerType.STRING;

  private input!: HTMLInputElement;

  constructor(object: Record<string, unknown>, property: string) {
    super(object, property);

    // Initialize DOM
    this.initializeDOMElement();
  }

  protected createDOMElement(): HTMLElement {
    const container = this.createBaseElement();
    const widget = container.querySelector('.luxar-gui__controller-widget')!;

    // Text input
    this.input = document.createElement('input');
    this.input.type = 'text';
    this.input.className = 'luxar-gui__input luxar-gui__input--string';
    this.input.value = this.getValue();

    // Update on change
    this.eventManager.add(this.input, 'change', () => {
      if (!this.input) return; // Safety check
      this.object[this.property] = this.input.value;
      this.triggerChange();
      this.triggerFinishChange();
    });

    // Update live on input
    this.eventManager.add(this.input, 'input', () => {
      if (!this.input) return; // Safety check
      this.object[this.property] = this.input.value;
      this.triggerChange();
    });

    // Auto-blur after interaction
    applyAutoBlur(this.input, this.eventManager);

    widget.appendChild(this.input);

    // Expose input element
    this.$input = this.input;

    return container;
  }

  public updateDisplay(): this {
    if (!this.input) return this; // Safety check
    this.input.value = this.getValue();
    return this;
  }
}
