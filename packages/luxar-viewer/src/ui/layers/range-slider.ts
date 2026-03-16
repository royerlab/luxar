/**
 * Dual-thumb range slider component for [min, max] display range control.
 *
 * Renders two overlapping <input type="range"> elements over a shared track.
 * CSS styling uses the theme's CSS variables for consistent appearance.
 */

export interface RangeSliderOptions {
  container: HTMLElement;
  min: number;
  max: number;
  valueLow: number;
  valueHigh: number;
  step?: number;
  label?: string;
  onChange: (low: number, high: number) => void;
}

export class RangeSlider {
  private wrapper: HTMLElement;
  private lowInput: HTMLInputElement;
  private highInput: HTMLInputElement;
  private lowLabel: HTMLElement;
  private highLabel: HTMLElement;
  private trackFill: HTMLElement;
  private options: RangeSliderOptions;

  constructor(options: RangeSliderOptions) {
    this.options = options;

    // Create wrapper
    this.wrapper = document.createElement('div');
    this.wrapper.className = 'luxar-range-slider';

    // Label row
    if (options.label) {
      const labelRow = document.createElement('div');
      labelRow.className = 'luxar-range-slider__label-row';
      const labelEl = document.createElement('span');
      labelEl.className = 'luxar-range-slider__label';
      labelEl.textContent = options.label;

      this.lowLabel = document.createElement('span');
      this.lowLabel.className = 'luxar-range-slider__value';

      this.highLabel = document.createElement('span');
      this.highLabel.className = 'luxar-range-slider__value';

      const valuesEl = document.createElement('span');
      valuesEl.className = 'luxar-range-slider__values';
      valuesEl.appendChild(this.lowLabel);
      valuesEl.appendChild(document.createTextNode(' – '));
      valuesEl.appendChild(this.highLabel);

      labelRow.appendChild(labelEl);
      labelRow.appendChild(valuesEl);
      this.wrapper.appendChild(labelRow);
    } else {
      this.lowLabel = document.createElement('span');
      this.highLabel = document.createElement('span');
    }

    // Track container
    const trackContainer = document.createElement('div');
    trackContainer.className = 'luxar-range-slider__track-container';

    // Track background
    const track = document.createElement('div');
    track.className = 'luxar-range-slider__track';

    // Track fill (highlighted region between thumbs)
    this.trackFill = document.createElement('div');
    this.trackFill.className = 'luxar-range-slider__track-fill';
    track.appendChild(this.trackFill);

    // Low thumb
    this.lowInput = document.createElement('input');
    this.lowInput.type = 'range';
    this.lowInput.className = 'luxar-range-slider__input luxar-range-slider__input--low';
    this.lowInput.min = String(options.min);
    this.lowInput.max = String(options.max);
    this.lowInput.step = String(options.step ?? 0.001);
    this.lowInput.value = String(options.valueLow);

    // High thumb
    this.highInput = document.createElement('input');
    this.highInput.type = 'range';
    this.highInput.className = 'luxar-range-slider__input luxar-range-slider__input--high';
    this.highInput.min = String(options.min);
    this.highInput.max = String(options.max);
    this.highInput.step = String(options.step ?? 0.001);
    this.highInput.value = String(options.valueHigh);

    trackContainer.appendChild(track);
    trackContainer.appendChild(this.lowInput);
    trackContainer.appendChild(this.highInput);
    this.wrapper.appendChild(trackContainer);

    // Event listeners
    this.lowInput.addEventListener('input', this.onLowChange);
    this.highInput.addEventListener('input', this.onHighChange);

    options.container.appendChild(this.wrapper);
    this.updateLabels();
    this.updateTrackFill();
  }

  private onLowChange = (): void => {
    let low = parseFloat(this.lowInput.value);
    const high = parseFloat(this.highInput.value);
    // Enforce low <= high
    if (low > high) {
      low = high;
      this.lowInput.value = String(low);
    }
    this.updateLabels();
    this.updateTrackFill();
    this.options.onChange(low, high);
  };

  private onHighChange = (): void => {
    const low = parseFloat(this.lowInput.value);
    let high = parseFloat(this.highInput.value);
    // Enforce high >= low
    if (high < low) {
      high = low;
      this.highInput.value = String(high);
    }
    this.updateLabels();
    this.updateTrackFill();
    this.options.onChange(low, high);
  };

  private updateLabels(): void {
    const low = parseFloat(this.lowInput.value);
    const high = parseFloat(this.highInput.value);
    this.lowLabel.textContent = this.formatValue(low);
    this.highLabel.textContent = this.formatValue(high);
  }

  private updateTrackFill(): void {
    const min = parseFloat(this.lowInput.min);
    const max = parseFloat(this.lowInput.max);
    const range = max - min;
    if (range <= 0) return;

    const low = parseFloat(this.lowInput.value);
    const high = parseFloat(this.highInput.value);
    const leftPct = ((low - min) / range) * 100;
    const rightPct = ((high - min) / range) * 100;
    this.trackFill.style.left = `${leftPct}%`;
    this.trackFill.style.width = `${rightPct - leftPct}%`;
  }

  private formatValue(v: number): string {
    // Show up to 3 decimal places, strip trailing zeros
    return v.toFixed(3).replace(/\.?0+$/, '');
  }

  /** Programmatically set both values */
  setValues(low: number, high: number): void {
    this.lowInput.value = String(low);
    this.highInput.value = String(high);
    this.updateLabels();
    this.updateTrackFill();
  }

  /** Update the slider bounds */
  setBounds(min: number, max: number): void {
    this.lowInput.min = String(min);
    this.lowInput.max = String(max);
    this.highInput.min = String(min);
    this.highInput.max = String(max);
    this.updateTrackFill();
  }

  /** Remove from DOM and detach listeners */
  dispose(): void {
    this.lowInput.removeEventListener('input', this.onLowChange);
    this.highInput.removeEventListener('input', this.onHighChange);
    this.wrapper.remove();
  }
}
