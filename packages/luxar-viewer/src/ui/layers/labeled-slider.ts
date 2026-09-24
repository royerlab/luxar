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

/**
 * Position of the `log` track's DEDICATED zero stop, one step below the
 * geometric range: position 0 is exactly 0, and `[LOG_ZERO_GAP, 1]` maps
 * geometrically onto `[min, max]`.
 *
 * The zero stop must not be shared with `min`. Overloading position 0 for
 * both made a value AT the track floor round (via the input's step
 * snapping) onto the zero stop, so merely touching the slider collapsed
 * that value to 0 instead of leaving it put.
 */
const LOG_ZERO_GAP = 1 / LOG_STEPS;

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
  private disposeInteractions: () => void;
  private disposeInlineEdit: () => void;
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
    // `writeReadout` also needs `this.input`, which does not exist yet — the
    // aria mirror is applied right after the input is created below.

    labelEl.appendChild(labelText);
    labelEl.appendChild(this.valueEl);

    this.input = document.createElement('input');
    this.input.type = 'range';
    if (this.scale === 'log') {
      this.input.min = '0';
      this.input.max = '1';
      this.input.step = fineTrackStep(1 / LOG_STEPS);
      this.input.dataset.baseStep = String(1 / LOG_STEPS);
    } else {
      this.input.min = String(options.min);
      this.input.max = String(options.max);
      this.input.step = fineTrackStep(options.step);
      this.input.dataset.baseStep = String(options.step);
    }
    this.input.value = String(this.toPosition(options.initialValue));
    this.input.className = 'luxar-layers-panel__slider';
    this.input.setAttribute('aria-label', `${options.label} slider`);
    this.syncAriaValueText(options.initialValue);

    this.inputHandler = (): void => {
      const position =
        this.scale === 'log'
          ? parseFloat(this.input.value)
          : snapToGrid(parseFloat(this.input.value), this.min, options.step);
      this.applyPosition(position);
    };
    this.input.addEventListener('input', this.inputHandler);
    const baseStep = this.scale === 'log' ? 1 / LOG_STEPS : options.step;
    this.disposeInteractions = attachSliderInteractions({
      input: this.input,
      baseStep,
      initialValue: this.toPosition(options.initialValue),
      getValue: () => parseFloat(this.input.value),
      setValue: (position) => this.applyPosition(position),
    });
    this.disposeInlineEdit = attachInlineNumberEdit(this.valueEl, {
      ariaLabel: `${options.label} value`,
      getValue: () => this.lastValue,
      formatValue: (value) => String(value),
      onCommit: (parsed) => this.commitTypedValue(parsed),
    });

    this.wrapper.appendChild(labelEl);
    this.wrapper.appendChild(this.input);
    options.container.appendChild(this.wrapper);
  }

  private applyPosition(position: number): void {
    const clampedPosition = Math.min(
      parseFloat(this.input.max),
      Math.max(parseFloat(this.input.min), position)
    );
    const raw = this.toValue(clampedPosition);
    const value = this.options.constrain ? this.options.constrain(raw) : raw;
    this.lastValue = value;
    this.input.value = String(this.toPosition(value));
    this.valueEl.textContent = this.formatValue(value);
    this.syncAriaValueText(value);
    this.options.onChange(value);
  }

  private commitTypedValue(parsed: number): void {
    const value = this.options.constrain ? this.options.constrain(parsed) : parsed;
    if (value === parsed) this.setRange(Math.min(this.min, value), Math.max(this.max, value));
    this.lastValue = value;
    this.input.value = String(this.toPosition(value));
    this.valueEl.textContent = this.formatValue(value);
    this.syncAriaValueText(value);
    this.options.onChange(value);
  }

  /** Thumb position (DOM input space) for a value. */
  private toPosition(value: number): number {
    if (this.scale !== 'log') return value;
    if (!(value > 0) || !(this.min > 0) || !(this.max > this.min)) return 0;
    const t = Math.log(value / this.min) / Math.log(this.max / this.min);
    return LOG_ZERO_GAP + Math.min(1, Math.max(0, t)) * (1 - LOG_ZERO_GAP);
  }

  /** Value for a thumb position (DOM input space). */
  private toValue(position: number): number {
    if (this.scale !== 'log') return position;
    // Below the gap is the exact-zero stop — κ=0 must stay reachable (it is
    // the additive limit of the volumetric mode). `min` itself lives at
    // LOG_ZERO_GAP, one step in, so no in-range value shares this stop.
    if (!(position >= LOG_ZERO_GAP) || !(this.min > 0) || !(this.max > this.min)) return 0;
    const t = (Math.min(1, position) - LOG_ZERO_GAP) / (1 - LOG_ZERO_GAP);
    return this.min * Math.pow(this.max / this.min, t);
  }

  /**
   * Mirror the visible readout into `aria-valuetext` on a `log` track.
   *
   * On a linear track the input's native `value` IS the value, so assistive
   * tech reads it correctly and no override is needed. A log track drives the
   * input in NORMALISED position space, so AT would otherwise announce the
   * position ("0.75") instead of the value the readout shows ("1.00").
   */
  private syncAriaValueText(value: number): void {
    if (this.scale !== 'log') return;
    this.input.setAttribute('aria-valuetext', this.formatValue(value));
  }

  private formatValue(v: number): string {
    return this.options.format
      ? this.options.format(v)
      : this.scale === 'log'
        ? v.toFixed(2)
        : formatSliderValue(v, this.options.step, this.min);
  }

  /** Programmatically set the slider value + readout. Does not fire onChange. */
  setValue(value: number): void {
    this.lastValue = value;
    this.input.value = String(this.toPosition(value));
    this.valueEl.textContent = this.formatValue(value);
    this.syncAriaValueText(value);
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

  /**
   * Mark the slider INERT — a knob that, in the current state of its siblings,
   * changes nothing on screen (a clearcoat roughness with no clearcoat, a
   * transmission on a metal). The input is disabled, the group dims, and the
   * reason is the hover text, so a dead-feeling control explains itself instead
   * of reading as broken. `null` restores the live state.
   */
  setInert(reason: string | null): void {
    this.input.disabled = reason !== null;
    this.wrapper.classList.toggle('luxar-layers-panel__control-group--inert', reason !== null);
    this.wrapper.title = reason ?? '';
  }

  /** The hover text of the whole group (the inert reason, or empty). */
  getInertReason(): string | null {
    return this.input.disabled ? this.wrapper.title || null : null;
  }

  /** Remove from DOM and detach listeners. */
  dispose(): void {
    this.input.removeEventListener('input', this.inputHandler);
    this.disposeInteractions();
    this.disposeInlineEdit();
    this.wrapper.remove();
  }
}
import {
  attachInlineNumberEdit,
  attachSliderInteractions,
  fineTrackStep,
  formatSliderValue,
  snapToGrid,
} from '../slider-kit';
