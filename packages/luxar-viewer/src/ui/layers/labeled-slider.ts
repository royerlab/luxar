/**
 * Single-thumb slider with a label row + value readout.
 *
 * Used for the simple per-layer scalars (gamma, opacity) where the layers
 * panel needs a `<input type="range">` paired with a value label. The
 * dual-thumb [low, high] pattern lives in `range-slider.ts`; the heavily
 * customised dimension navigation slider lives in `dimension-sliders.ts`.
 *
 * Programmatic `setValue` updates the input + readout silently — no
 * `onChange` callback fires. This mirrors the call sites in `LayersPanel`
 * which sync the slider to the primary selected layer without re-applying.
 */
export interface LabeledSliderOptions {
  container: HTMLElement;
  label: string;
  min: number;
  max: number;
  step: number;
  initialValue: number;
  /** Format the readout text. Defaults to `value.toFixed(2)`. */
  format?: (value: number) => string;
  /** Constrain the parsed value before invoking onChange. */
  constrain?: (value: number) => number;
  /** Fired with the constrained value on each `input` event. */
  onChange: (value: number) => void;
}

export class LabeledSlider {
  private wrapper: HTMLElement;
  private input: HTMLInputElement;
  private valueEl: HTMLElement;
  private options: LabeledSliderOptions;
  private inputHandler: () => void;

  constructor(options: LabeledSliderOptions) {
    this.options = options;

    this.wrapper = document.createElement('div');
    this.wrapper.className = 'luxar-layers-panel__control-group';

    const labelEl = document.createElement('div');
    labelEl.className = 'luxar-layers-panel__control-label';

    const labelText = document.createElement('span');
    labelText.textContent = options.label;

    this.valueEl = document.createElement('span');
    this.valueEl.className = 'luxar-layers-panel__control-value';
    this.valueEl.textContent = this.formatValue(options.initialValue);

    labelEl.appendChild(labelText);
    labelEl.appendChild(this.valueEl);

    this.input = document.createElement('input');
    this.input.type = 'range';
    this.input.min = String(options.min);
    this.input.max = String(options.max);
    this.input.step = String(options.step);
    this.input.value = String(options.initialValue);
    this.input.className = 'luxar-layers-panel__slider';

    this.inputHandler = (): void => {
      const raw = parseFloat(this.input.value);
      const value = options.constrain ? options.constrain(raw) : raw;
      this.valueEl.textContent = this.formatValue(value);
      options.onChange(value);
    };
    this.input.addEventListener('input', this.inputHandler);

    this.wrapper.appendChild(labelEl);
    this.wrapper.appendChild(this.input);
    options.container.appendChild(this.wrapper);
  }

  private formatValue(v: number): string {
    return this.options.format ? this.options.format(v) : v.toFixed(2);
  }

  /** Programmatically set the slider value + readout. Does not fire onChange. */
  setValue(value: number): void {
    this.input.value = String(value);
    this.valueEl.textContent = this.formatValue(value);
  }

  /**
   * Show/hide the whole control group (label + slider + readout) — for
   * mode-conditional controls like the volumetric Absorption slider
   * (same pattern as the colormap select's display toggle).
   */
  setVisible(visible: boolean): void {
    this.wrapper.style.display = visible ? '' : 'none';
  }

  /** Remove from DOM and detach listeners. */
  dispose(): void {
    this.input.removeEventListener('input', this.inputHandler);
    this.wrapper.remove();
  }
}
