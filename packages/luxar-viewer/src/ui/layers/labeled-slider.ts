/**
 * Single-thumb slider with a label row + value readout.
 *
 * Used for the simple per-layer scalars (gamma, opacity, absorption) where
 * the layers panel needs a `<input type="range">` paired with a value
 * label. The dual-thumb [low, high] pattern lives in `range-slider.ts`; the
 * heavily customised dimension navigation slider lives in
 * `dimension-sliders.ts`.
 *
 * Two value scales are supported. `linear` (the default) drives the DOM
 * input directly in value space. `log` drives it in NORMALISED position
 * space [0, 1] and maps geometrically onto `[min, max]`, with position 0
 * reserved for an exact 0 — for knobs whose useful magnitude spans decades
 * (the volumetric absorption κ, whose scale goes as 1/geometry-thickness;
 * see `absorption-range.ts`).
 *
 * Programmatic `setValue` updates the input + readout silently — no
 * `onChange` callback fires. This mirrors the call sites in `LayersPanel`
 * which sync the slider to the primary selected layer without re-applying.
 */

/** Value scale of the slider track — see the module note. */
export type SliderScale = 'linear' | 'log';

/** Number of `input` steps across a `log`-scale track. */
const LOG_STEPS = 1000;

export interface LabeledSliderOptions {
  container: HTMLElement;
  label: string;
  min: number;
  max: number;
  step: number;
  initialValue: number;
  /**
   * Value scale. `log` requires `min > 0` (position 0 still yields exactly
   * 0, so a κ=0 "no absorption" setting stays reachable) and ignores
   * `step`, which is fixed to the normalised track resolution.
   */
  scale?: SliderScale;
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
  private scale: SliderScale;
  private min: number;
  private max: number;
  /**
   * Last value in VALUE space. `setRange` re-derives the thumb position
   * from it, so re-scaling the track never silently moves the value.
   */
  private lastValue: number;

  constructor(options: LabeledSliderOptions) {
    this.options = options;
    this.scale = options.scale ?? 'linear';
    this.min = options.min;
    this.max = options.max;
    this.lastValue = options.initialValue;

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
    if (this.scale === 'log') {
      this.input.min = '0';
      this.input.max = '1';
      this.input.step = String(1 / LOG_STEPS);
    } else {
      this.input.min = String(options.min);
      this.input.max = String(options.max);
      this.input.step = String(options.step);
    }
    this.input.value = String(this.toPosition(options.initialValue));
    this.input.className = 'luxar-layers-panel__slider';

    this.inputHandler = (): void => {
      const raw = this.toValue(parseFloat(this.input.value));
      const value = options.constrain ? options.constrain(raw) : raw;
      this.lastValue = value;
      this.valueEl.textContent = this.formatValue(value);
      options.onChange(value);
    };
    this.input.addEventListener('input', this.inputHandler);

    this.wrapper.appendChild(labelEl);
    this.wrapper.appendChild(this.input);
    options.container.appendChild(this.wrapper);
  }

  /** Thumb position (DOM input space) for a value. */
  private toPosition(value: number): number {
    if (this.scale !== 'log') return value;
    if (!(value > 0) || !(this.min > 0) || !(this.max > this.min)) return 0;
    const t = Math.log(value / this.min) / Math.log(this.max / this.min);
    return Math.min(1, Math.max(0, t));
  }

  /** Value for a thumb position (DOM input space). */
  private toValue(position: number): number {
    if (this.scale !== 'log') return position;
    // Position 0 is the exact-zero stop, not `min` — κ=0 must stay
    // reachable (it is the additive limit of the volumetric mode).
    if (!(position > 0) || !(this.min > 0) || !(this.max > this.min)) return 0;
    return this.min * Math.pow(this.max / this.min, Math.min(1, position));
  }

  private formatValue(v: number): string {
    return this.options.format ? this.options.format(v) : v.toFixed(2);
  }

  /** Programmatically set the slider value + readout. Does not fire onChange. */
  setValue(value: number): void {
    this.lastValue = value;
    this.input.value = String(this.toPosition(value));
    this.valueEl.textContent = this.formatValue(value);
  }

  /**
   * Replace the value-space bounds, keeping the current value put (the
   * thumb position is re-derived, so a `log` track re-scales around the
   * same value rather than jumping). Does not fire onChange.
   */
  setRange(min: number, max: number): void {
    this.min = min;
    this.max = max;
    if (this.scale !== 'log') {
      this.input.min = String(min);
      this.input.max = String(max);
    }
    this.setValue(this.lastValue);
  }

  /** Current value-space bounds — `[min, max]`. */
  getRange(): [number, number] {
    return [this.min, this.max];
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
