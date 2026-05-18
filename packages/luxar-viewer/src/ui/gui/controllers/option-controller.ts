/**
 * OptionController - Dropdown select for enumerated values
 */

import { Controller } from '../core/controller';
import { ControllerType, type ControllerOptions } from '../core/types';
import { applyAutoBlur } from '../utils/auto-blur';

export class OptionController extends Controller<unknown> {
  protected type = ControllerType.OPTION;

  private select!: HTMLSelectElement;
  private optionsMap: Map<string, unknown> = new Map();

  constructor(object: Record<string, unknown>, property: string, options: ControllerOptions) {
    super(object, property);

    if (!options.options) {
      throw new Error('OptionController requires options parameter');
    }

    // Build options map after super (now we can use this)
    if (Array.isArray(options.options)) {
      for (const value of options.options) {
        this.optionsMap.set(String(value), value);
      }
    } else {
      for (const [label, value] of Object.entries(options.options)) {
        this.optionsMap.set(label, value);
      }
    }

    // NOW initialize DOM after map is built
    this.initializeDOMElement();
  }

  protected createDOMElement(): HTMLElement {
    const container = this.createBaseElement();
    const widget = container.querySelector('.luxar-gui__controller-widget')!;

    // Select dropdown
    this.select = document.createElement('select');
    this.select.className = 'luxar-gui__select';

    // Add options
    for (const [label] of this.optionsMap.entries()) {
      const option = document.createElement('option');
      option.value = label;
      option.textContent = label;
      this.select.appendChild(option);
    }

    // Set initial value
    this.updateDisplay();

    // Update on change
    this.eventManager.add(this.select, 'change', () => {
      if (!this.select) return; // Safety check
      const label = this.select.value;
      const value = this.optionsMap.get(label);
      this.object[this.property] = value;
      this.triggerChange();
      this.triggerFinishChange();
    });

    // Auto-blur after interaction
    applyAutoBlur(this.select, this.eventManager);

    widget.appendChild(this.select);

    // Expose select element
    this.$input = this.select;

    return container;
  }

  public updateDisplay(): this {
    if (!this.select) return this; // Safety check

    const currentValue = this.getValue();

    // Find the label for current value
    for (const [label, value] of this.optionsMap.entries()) {
      if (value === currentValue) {
        this.select.value = label;
        break;
      }
    }

    return this;
  }
}
