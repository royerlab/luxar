/**
 * BooleanController - Checkbox for boolean values
 */

import { Controller } from '../core/controller';
import { ControllerType } from '../core/types';
import { applyAutoBlur } from '../utils/auto-blur';

export class BooleanController extends Controller<boolean> {
  protected type = ControllerType.BOOLEAN;

  private checkbox!: HTMLInputElement;

  constructor(object: Record<string, unknown>, property: string) {
    super(object, property);

    // Initialize DOM
    this.initializeDOMElement();
  }

  protected createDOMElement(): HTMLElement {
    const container = document.createElement('div');
    container.className = `luxar-gui__controller luxar-gui__controller--${this.type}`;

    // Label wraps checkbox for better click area
    const label = document.createElement('label');
    label.className = 'luxar-gui__controller-name';

    // Checkbox
    this.checkbox = document.createElement('input');
    this.checkbox.type = 'checkbox';
    this.checkbox.className = 'luxar-gui__checkbox';
    this.checkbox.checked = this.getValue();

    // Update on change
    this.eventManager.add(this.checkbox, 'change', () => {
      if (!this.checkbox) return; // Safety check
      this.object[this.property] = this.checkbox.checked;
      this.triggerChange();
      this.triggerFinishChange();
    });

    // Auto-blur after interaction
    applyAutoBlur(this.checkbox, this.eventManager);

    label.appendChild(this.checkbox);
    const labelText = document.createTextNode(this.label);
    label.appendChild(labelText);

    container.appendChild(label);

    // Expose checkbox as $input
    this.$input = this.checkbox;

    return container;
  }

  public updateDisplay(): this {
    if (!this.checkbox) return this; // Safety check
    this.checkbox.checked = this.getValue();
    return this;
  }

  // Override name() to update text node
  public name(label: string): this {
    this.label = label;
    const labelElement = this.domElement.querySelector('.luxar-gui__controller-name');
    if (labelElement) {
      // Update text node (after checkbox)
      const textNode = labelElement.childNodes[1];
      if (textNode) {
        textNode.textContent = label;
      }
    }
    return this;
  }
}
