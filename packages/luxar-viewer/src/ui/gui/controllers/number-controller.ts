/**
 * NumberController - Slider + text input for numeric values
 *
 * Features:
 * - Range slider (if min/max provided)
 * - Text input for precise values
 * - Min/max/step constraints
 * - Custom display override support (for logarithmic sliders)
 */

import { Controller } from '../core/controller';
import { ControllerType, type ControllerOptions } from '../core/types';
import { clamp, formatNumber } from '../utils/value-formatting';
import { applyAutoBlur } from '../utils/auto-blur';

export class NumberController extends Controller<number> {
  protected type = ControllerType.NUMBER;

  private minValue?: number;
  private maxValue?: number;
  private stepValue?: number;

  private slider?: HTMLInputElement;
  private input!: HTMLInputElement;

  /** Custom display override (for logarithmic patterns) */
  private customUpdateDisplay?: () => void;

  constructor(object: Record<string, any>, property: string, options: ControllerOptions) {
    super(object, property);

    this.minValue = options.min;
    this.maxValue = options.max;
    this.stepValue = options.step ?? (this.hasRange() ? this.getDefaultStep() : undefined);

    // Initialize DOM after properties are set
    this.initializeDOMElement();
  }

  protected createDOMElement(): HTMLElement {
    const container = this.createBaseElement();
    const widget = container.querySelector('.luxar-gui__controller-widget')!;

    // Create slider if range is specified
    if (this.hasRange()) {
      this.slider = document.createElement('input');
      this.slider.type = 'range';
      this.slider.className = 'luxar-gui__slider';
      this.slider.min = String(this.minValue);
      this.slider.max = String(this.maxValue);
      this.slider.step = String(this.stepValue);

      // Update on input (real-time)
      this.eventManager.add(this.slider, 'input', () => {
        if (!this.slider) return; // Safety check
        const value = this.constrainValue(parseFloat(this.slider.value));
        this.object[this.property] = value;
        this.updateDisplay();
        this.triggerChange();
      });

      // Trigger finish change on mouseup/touchend
      this.eventManager.add(this.slider, 'mouseup', () => {
        this.triggerFinishChange();
      });

      this.eventManager.add(this.slider, 'touchend', () => {
        this.triggerFinishChange();
      });

      // Auto-blur after interaction
      applyAutoBlur(this.slider, this.eventManager);

      widget.appendChild(this.slider);
    }

    // Create text input
    this.input = document.createElement('input');
    this.input.type = 'number';
    this.input.className = 'luxar-gui__input luxar-gui__input--number';

    if (this.minValue !== undefined) this.input.min = String(this.minValue);
    if (this.maxValue !== undefined) this.input.max = String(this.maxValue);
    if (this.stepValue !== undefined) this.input.step = String(this.stepValue);

    // Update on change (when user commits value)
    this.eventManager.add(this.input, 'change', () => {
      if (!this.input) return; // Safety check
      const parsed = parseFloat(this.input.value);
      if (isNaN(parsed)) {
        // Revert to current value on invalid input
        this.updateDisplay();
        return;
      }
      const value = this.constrainValue(parsed);
      this.object[this.property] = value;
      this.updateDisplay();
      this.triggerChange();
      this.triggerFinishChange();
    });

    // Update live on input
    this.eventManager.add(this.input, 'input', () => {
      if (!this.input) return; // Safety check
      const value = parseFloat(this.input.value);
      if (!isNaN(value)) {
        this.object[this.property] = this.constrainValue(value);
        if (this.slider) {
          this.slider.value = String(this.object[this.property]);
        }
        this.triggerChange();
      }
    });

    // Blur on Enter key
    this.eventManager.add(this.input, 'keydown', (e: Event) => {
      const keyEvent = e as KeyboardEvent;
      if (keyEvent.key === 'Enter') {
        this.input.blur();
      }
    });

    // Auto-blur after interaction
    applyAutoBlur(this.input, this.eventManager);

    widget.appendChild(this.input);

    // Expose input element for custom display logic
    this.$input = this.input;

    // Initialize display
    this.updateDisplay();

    return container;
  }

  public updateDisplay(): this {
    // Allow custom override (for logarithmic patterns)
    if (this.customUpdateDisplay) {
      this.customUpdateDisplay();
      return this;
    }

    if (!this.input) return this; // Safety check

    const value = this.getValue();

    if (this.slider) {
      this.slider.value = String(value);
    }

    this.input.value = formatNumber(value, this.stepValue);

    return this;
  }

  /**
   * Override updateDisplay method (for logarithmic sliders)
   *
   * This allows rendering-controls to implement custom display logic
   *
   * @param fn - Custom update function
   */
  public setCustomUpdateDisplay(fn: () => void): void {
    this.customUpdateDisplay = fn;
  }

  public min(value: number): this {
    this.minValue = value;
    if (this.slider) this.slider.min = String(value);
    if (this.input) this.input.min = String(value);
    return this;
  }

  public max(value: number): this {
    this.maxValue = value;
    if (this.slider) this.slider.max = String(value);
    if (this.input) this.input.max = String(value);
    return this;
  }

  public step(value: number): this {
    this.stepValue = value;
    if (this.slider) this.slider.step = String(value);
    if (this.input) this.input.step = String(value);
    return this;
  }

  private hasRange(): boolean {
    return this.minValue !== undefined && this.maxValue !== undefined;
  }

  private getDefaultStep(): number {
    const range = this.maxValue! - this.minValue!;
    return range / 100; // 1% of range
  }

  private constrainValue(value: number): number {
    return clamp(value, this.minValue, this.maxValue);
  }
}
