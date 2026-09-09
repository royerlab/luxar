/**
 * NumberController - Slider + text input for numeric values
 *
 * Features:
 * - Range slider (if min/max provided)
 * - Text input for precise values
 * - Min/max/step constraints
 * - Custom display override support (for logarithmic sliders)
 */

import { Controller } from '../controller';
import { ControllerType, type ControllerOptions } from '../types';
import { clamp, formatNumber } from '../format/value-formatting';
import { applyAutoBlur } from '../format/auto-blur';
import { normalizeWheelDeltaWithAxisFallback } from '../../../utils/wheel-delta';

export class NumberController extends Controller<number> {
  protected type = ControllerType.NUMBER;

  private minValue?: number;
  private maxValue?: number;
  private stepValue?: number;

  private slider?: HTMLInputElement;
  private input!: HTMLInputElement;

  /** Debounce timer for wheel finishChange */
  private wheelFinishTimer?: ReturnType<typeof setTimeout>;

  /** Initial value at construction time (for double-click reset) */
  private initialValue: number;

  /** Custom display override (for logarithmic patterns) */
  private customUpdateDisplay?: () => void;

  constructor(object: Record<string, unknown>, property: string, options: ControllerOptions) {
    super(object, property);

    this.minValue = options.min;
    this.maxValue = options.max;
    this.stepValue = options.step ?? (this.hasRange() ? this.getDefaultStep() : undefined);
    this.initialValue = object[property] as number;

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

      // Mousewheel fine-tuning: scroll = 1/10th step. Modifiers follow the
      // app-wide slider convention (dimension sliders, layers range
      // sliders): Shift = finer (÷10), Ctrl = coarse (×10), Ctrl+Shift =
      // finest (÷100) — relative to the 1/10th-step base.
      this.eventManager.add(
        this.slider,
        'wheel',
        (e: Event) => {
          const wheelEvent = e as WheelEvent;
          wheelEvent.preventDefault();
          if (!this.slider || !this.stepValue) return;
          let multiplier = 0.1;
          if (wheelEvent.shiftKey && wheelEvent.ctrlKey) multiplier = 0.001;
          else if (wheelEvent.shiftKey) multiplier = 0.01;
          else if (wheelEvent.ctrlKey) multiplier = 1;
          const delta = this.stepValue * multiplier;
          const wheelDelta = normalizeWheelDeltaWithAxisFallback(wheelEvent);
          if (wheelDelta === 0) return;
          const direction = wheelDelta < 0 ? 1 : -1;
          const value = this.constrainValue(this.getValue() + direction * delta);
          this.object[this.property] = value;
          this.updateDisplay();
          this.triggerChange();

          // Debounced finishChange — fires once scrolling stops (like mouseup for dragging)
          clearTimeout(this.wheelFinishTimer);
          this.wheelFinishTimer = setTimeout(() => this.triggerFinishChange(), 150);
        },
        { passive: false }
      );

      // Double-click to reset to initial value
      this.eventManager.add(this.slider, 'dblclick', () => {
        this.object[this.property] = this.constrainValue(this.initialValue);
        this.updateDisplay();
        this.triggerChange();
        this.triggerFinishChange();
      });

      // Alt+click to focus the number input for direct keyboard entry
      this.eventManager.add(this.slider, 'click', (e: Event) => {
        if ((e as MouseEvent).altKey) {
          e.preventDefault();
          this.input.focus();
          this.input.select();
        }
      });

      // Cursor hint and tooltip
      this.slider.title = 'Scroll: fine-tune · ⇧: finer · ⌃: coarse · Double-click: reset';

      // Auto-blur after interaction
      applyAutoBlur(this.slider, this.eventManager);

      widget.appendChild(this.slider);
    }

    // Create text input
    this.input = document.createElement('input');
    this.input.type = 'number';
    this.input.className = 'luxar-gui__input luxar-gui__input--number';
    // A numeric keypad (with a decimal point) on touch keyboards; inert on desktop.
    this.input.inputMode = 'decimal';

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

    // Auto-blur after interaction (handles Enter + Escape blur)
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

  /**
   * Override of `Controller.setValue` that:
   *  - rejects NaN (would silently corrupt downstream comparisons), and
   *  - clamps every other input — including ±Infinity and finite values
   *    outside `[min, max]` — into the configured range via the same
   *    `constrainValue` helper the DOM event paths use.
   *
   * Without this override the inherited base `setValue` performs a raw
   * assignment, so a programmatic caller writing `setValue(NaN)` or
   * `setValue(1e20)` would silently corrupt the model. Documented as a
   * round-11 audit OOS finding.
   */
  public override setValue(value: number): this {
    if (Number.isNaN(value)) {
      // Refresh the display in case it drifted, but do NOT write NaN
      // into the model — every comparison against it would be false.
      this.updateDisplay();
      return this;
    }
    return super.setValue(this.constrainValue(value));
  }

  public override dispose(): void {
    clearTimeout(this.wheelFinishTimer);
    super.dispose();
  }
}
