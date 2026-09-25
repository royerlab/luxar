/**
 * Dual-thumb range slider component for [min, max] display range control.
 *
 * Renders two overlapping <input type="range"> elements over a shared track.
 * CSS styling uses the theme's CSS variables for consistent appearance.
 *
 * The bound labels (left/right of the track) are click-to-edit: clicking on
 * them opens a tiny text input so the user can type a custom slider limit,
 * napari-style.
 *
 * The overlapping range thumbs retain native drag and arrow behavior. Their
 * adaptive wheel stepping lives on the editable bound labels, so they are not
 * fine-grid members of the single-thumb slider interaction contract.
 */

import {
  attachInlineNumberEdit,
  INLINE_NUMBER_EDIT_HINT,
  WHEEL_INTERACTION_HINT,
} from '../slider-kit';
import { normalizeWheelDeltaWithAxisFallback } from '../../utils/wheel-delta';
import { applyModifierTier } from '../../utils/cross-layer/modifier-tiers';
import { clamp } from '../../utils/clamp';

export interface RangeSliderOptions {
  container: HTMLElement;
  min: number;
  max: number;
  valueLow: number;
  valueHigh: number;
  step?: number;
  label?: string;
  tooltip?: string;
  onChange: (low: number, high: number) => void;
  /** Called when the user edits the slider bounds (click-to-edit on limits). */
  onBoundsChange?: (min: number, max: number) => void;
}

/**
 * Compute an adaptive scroll step based on the current slider range.
 *
 * Strategy: step = 10^(floor(log10(range)) - 1), giving ~10–100 clean
 * power-of-10 increments across the full range.
 * The caller applies the shared Shift/Ctrl modifier ladder.
 */
function computeWheelStep(min: number, max: number): number {
  const range = Math.abs(max - min);
  if (range < 1e-10) return 0.1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(range)));
  return magnitude / 10; // ~10 steps per order of magnitude
}

export class RangeSlider {
  private wrapper: HTMLElement;
  private lowInput: HTMLInputElement;
  private highInput: HTMLInputElement;
  private labelEl: HTMLElement | null;
  private lowLabel: HTMLElement;
  private highLabel: HTMLElement;
  private boundsLowLabel: HTMLElement;
  private boundsHighLabel: HTMLElement;
  private trackFill: HTMLElement;
  private options: RangeSliderOptions;

  // Bound wheel handlers (stored for removeEventListener in dispose)
  private onWheelLow: (e: WheelEvent) => void;
  private onWheelHigh: (e: WheelEvent) => void;

  // Bound click handlers (stored for removeEventListener in dispose)
  private onClickLow: () => void;
  private onClickHigh: () => void;
  private disposeLowValueEdit: () => void = () => {};
  private disposeHighValueEdit: () => void = () => {};

  constructor(options: RangeSliderOptions) {
    this.options = options;

    // Create wrapper
    this.wrapper = document.createElement('div');
    this.wrapper.className = 'luxar-range-slider';

    // Label row (title + current values)
    if (options.label) {
      const labelRow = document.createElement('div');
      labelRow.className = 'luxar-range-slider__label-row';
      this.labelEl = document.createElement('span');
      this.labelEl.className = 'luxar-range-slider__label';

      this.lowLabel = document.createElement('span');
      this.lowLabel.className = 'luxar-range-slider__value';

      this.highLabel = document.createElement('span');
      this.highLabel.className = 'luxar-range-slider__value';

      const valuesEl = document.createElement('span');
      valuesEl.className = 'luxar-range-slider__values';
      valuesEl.appendChild(this.lowLabel);
      valuesEl.appendChild(document.createTextNode(' \u2013 '));
      valuesEl.appendChild(this.highLabel);

      labelRow.appendChild(this.labelEl);
      labelRow.appendChild(valuesEl);
      this.wrapper.appendChild(labelRow);
    } else {
      this.labelEl = null;
      this.lowLabel = document.createElement('span');
      this.highLabel = document.createElement('span');
    }
    if (options.label) this.setLabel(options.label, options.tooltip);

    // Track container (bounds labels + track + sliders)
    const trackRow = document.createElement('div');
    trackRow.className = 'luxar-range-slider__track-row';

    // Editable bounds label — low (left of track)
    this.boundsLowLabel = document.createElement('span');
    this.boundsLowLabel.className = 'luxar-range-slider__bound';
    this.boundsLowLabel.setAttribute('role', 'button');
    this.boundsLowLabel.tabIndex = 0;
    this.boundsLowLabel.setAttribute('aria-label', `${options.label ?? 'Range'} lower bound`);
    this.boundsLowLabel.title = `${INLINE_NUMBER_EDIT_HINT} · ${WHEEL_INTERACTION_HINT}`;
    this.boundsLowLabel.textContent = this.formatValue(options.min);
    this.onClickLow = () => this.editBound('low');
    this.boundsLowLabel.addEventListener('click', this.onClickLow);

    // Editable bounds label — high (right of track)
    this.boundsHighLabel = document.createElement('span');
    this.boundsHighLabel.className = 'luxar-range-slider__bound';
    this.boundsHighLabel.setAttribute('role', 'button');
    this.boundsHighLabel.tabIndex = 0;
    this.boundsHighLabel.setAttribute('aria-label', `${options.label ?? 'Range'} upper bound`);
    this.boundsHighLabel.title = `${INLINE_NUMBER_EDIT_HINT} · ${WHEEL_INTERACTION_HINT}`;
    this.boundsHighLabel.textContent = this.formatValue(options.max);
    this.onClickHigh = () => this.editBound('high');
    this.boundsHighLabel.addEventListener('click', this.onClickHigh);
    this.boundsLowLabel.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        event.stopPropagation();
        this.editBound('low');
      }
    });
    this.boundsHighLabel.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        event.stopPropagation();
        this.editBound('high');
      }
    });

    // Mousewheel adjustment on bound labels
    this.onWheelLow = (e: WheelEvent) => this.handleBoundWheel(e, 'low');
    this.onWheelHigh = (e: WheelEvent) => this.handleBoundWheel(e, 'high');
    this.boundsLowLabel.addEventListener('wheel', this.onWheelLow, { passive: false });
    this.boundsHighLabel.addEventListener('wheel', this.onWheelHigh, { passive: false });

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
    this.updateAccessibleNames();

    trackContainer.appendChild(track);
    trackContainer.appendChild(this.lowInput);
    trackContainer.appendChild(this.highInput);

    trackRow.appendChild(this.boundsLowLabel);
    trackRow.appendChild(trackContainer);
    trackRow.appendChild(this.boundsHighLabel);
    this.wrapper.appendChild(trackRow);

    // Event listeners
    this.lowInput.addEventListener('input', this.onLowChange);
    this.highInput.addEventListener('input', this.onHighChange);

    this.disposeLowValueEdit = attachInlineNumberEdit(this.lowLabel, {
      ariaLabel: `${options.label ?? 'Range'} minimum value`,
      getValue: () => parseFloat(this.lowInput.value),
      formatValue: (value) => String(value),
      onCommit: (value) => this.commitValue('low', value),
    });
    this.disposeHighValueEdit = attachInlineNumberEdit(this.highLabel, {
      ariaLabel: `${options.label ?? 'Range'} maximum value`,
      getValue: () => parseFloat(this.highInput.value),
      formatValue: (value) => String(value),
      onCommit: (value) => this.commitValue('high', value),
    });

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

  private updateAccessibleNames(): void {
    if (!this.lowInput || !this.highInput) return;
    const label = this.options.label ?? 'Range';
    this.lowInput.setAttribute('aria-label', `${label} minimum slider`);
    this.highInput.setAttribute('aria-label', `${label} maximum slider`);
    this.lowLabel.setAttribute('aria-label', `${label} minimum value`);
    this.highLabel.setAttribute('aria-label', `${label} maximum value`);
    this.boundsLowLabel?.setAttribute('aria-label', `${label} lower bound`);
    this.boundsHighLabel?.setAttribute('aria-label', `${label} upper bound`);
  }

  private commitValue(which: 'low' | 'high', parsed: number): void {
    let min = parseFloat(this.lowInput.min);
    let max = parseFloat(this.lowInput.max);
    const low = parseFloat(this.lowInput.value);
    const high = parseFloat(this.highInput.value);
    const value = which === 'low' ? Math.min(parsed, high) : Math.max(parsed, low);
    if (value === parsed) {
      min = Math.min(min, value);
      max = Math.max(max, value);
      if (min !== parseFloat(this.lowInput.min) || max !== parseFloat(this.lowInput.max)) {
        this.setBounds(min, max);
        this.options.onBoundsChange?.(min, max);
      }
    }
    if (which === 'low') this.lowInput.value = String(value);
    else this.highInput.value = String(value);
    this.updateLabels();
    this.updateTrackFill();
    this.options.onChange(parseFloat(this.lowInput.value), parseFloat(this.highInput.value));
  }

  private updateTrackFill(): void {
    const min = parseFloat(this.lowInput.min);
    const max = parseFloat(this.lowInput.max);
    const range = max - min;
    // MED-30 (audit-ack, false-positive): the audit warned that a
    // zero-width range would produce `style.width: 'NaN%'`. The guard
    // below short-circuits BEFORE the `(value - min) / range` division
    // would yield NaN, so the failure mode the audit described cannot
    // occur. Regression test:
    //   tests/unit/ui/range-slider.test.ts §"handles a zero-width
    //   range gracefully" asserts the fill's left/width never contain
    //   'NaN' after min===max construction.
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

  /**
   * Replace a bound label with a text input for inline editing.
   * On Enter or blur, parse the value and update the slider bounds.
   */
  private editBound(which: 'low' | 'high'): void {
    const label = which === 'low' ? this.boundsLowLabel : this.boundsHighLabel;
    const currentValue =
      which === 'low' ? parseFloat(this.lowInput.min) : parseFloat(this.lowInput.max);

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'luxar-range-slider__bound-input';
    input.value = this.formatValue(currentValue);
    input.style.width = `${Math.max(label.offsetWidth, 28)}px`;

    // Replace label with input
    label.style.display = 'none';
    label.parentElement!.insertBefore(input, label.nextSibling);
    input.focus();
    input.select();

    const commit = (): void => {
      const parsed = parseFloat(input.value);
      if (!isNaN(parsed)) {
        let newMin = parseFloat(this.lowInput.min);
        let newMax = parseFloat(this.lowInput.max);

        if (which === 'low') {
          newMin = Math.min(parsed, newMax); // Don't let min exceed max
        } else {
          newMax = Math.max(parsed, newMin); // Don't let max go below min
        }

        this.setBounds(newMin, newMax);

        // Clamp current values into new bounds
        let low = parseFloat(this.lowInput.value);
        let high = parseFloat(this.highInput.value);
        low = clamp(low, newMin, newMax);
        high = clamp(high, newMin, newMax);
        this.lowInput.value = String(low);
        this.highInput.value = String(high);
        this.updateLabels();
        this.updateTrackFill();

        this.options.onBoundsChange?.(newMin, newMax);
        this.options.onChange(low, high);
      }

      // Restore label
      input.remove();
      label.style.display = '';
    };

    let committed = false;
    input.addEventListener('keydown', (e) => {
      // Stop propagation so viewer keyboard shortcuts (dimension navigation etc.)
      // don't fire while the user is typing a number.
      e.stopPropagation();
      if (e.key === 'Enter') {
        committed = true;
        commit();
      } else if (e.key === 'Escape') {
        committed = true;
        input.remove();
        label.style.display = '';
      }
    });
    input.addEventListener('blur', () => {
      if (!committed) commit();
    });
  }

  /**
   * Adjust a bound via mousewheel.
   * Scroll up → increase value, scroll down → decrease.
   * Shift key gives 10× finer increments.
   */
  private handleBoundWheel(e: WheelEvent, which: 'low' | 'high'): void {
    e.preventDefault();

    const curMin = parseFloat(this.lowInput.min);
    const curMax = parseFloat(this.lowInput.max);
    const step = applyModifierTier(computeWheelStep(curMin, curMax), e);
    // Scroll up → increase, scroll down → decrease.
    const wheelDelta = normalizeWheelDeltaWithAxisFallback(e);
    if (wheelDelta === 0) return;
    const direction = wheelDelta < 0 ? 1 : -1;
    const delta = step * direction;

    let newMin = curMin;
    let newMax = curMax;

    if (which === 'low') {
      newMin = curMin + delta;
      if (newMin > curMax) newMin = curMax;
    } else {
      newMax = curMax + delta;
      if (newMax < curMin) newMax = curMin;
    }

    this.setBounds(newMin, newMax);

    // Clamp current thumb values into new bounds
    let low = parseFloat(this.lowInput.value);
    let high = parseFloat(this.highInput.value);
    low = clamp(low, newMin, newMax);
    high = clamp(high, newMin, newMax);
    this.lowInput.value = String(low);
    this.highInput.value = String(high);
    this.updateLabels();
    this.updateTrackFill();

    this.options.onBoundsChange?.(newMin, newMax);
    this.options.onChange(low, high);
  }

  /** Programmatically set both values */
  setValues(low: number, high: number): void {
    this.lowInput.value = String(low);
    this.highInput.value = String(high);
    this.updateLabels();
    this.updateTrackFill();
  }

  /** Update the visible label and its explanatory tooltip. */
  setLabel(label: string, tooltip?: string): void {
    this.options.label = label;
    this.updateAccessibleNames();
    if (!this.labelEl) return;
    this.labelEl.textContent = label;
    this.labelEl.classList.toggle('luxar-range-slider__label--with-tooltip', Boolean(tooltip));
    if (tooltip) this.labelEl.title = tooltip;
    else this.labelEl.removeAttribute('title');
  }

  /** Update the slider bounds */
  setBounds(min: number, max: number): void {
    this.lowInput.min = String(min);
    this.lowInput.max = String(max);
    this.highInput.min = String(min);
    this.highInput.max = String(max);
    this.boundsLowLabel.textContent = this.formatValue(min);
    this.boundsHighLabel.textContent = this.formatValue(max);
    this.updateTrackFill();
  }

  /** Remove from DOM and detach listeners */
  dispose(): void {
    this.lowInput.removeEventListener('input', this.onLowChange);
    this.highInput.removeEventListener('input', this.onHighChange);
    this.boundsLowLabel.removeEventListener('wheel', this.onWheelLow);
    this.boundsHighLabel.removeEventListener('wheel', this.onWheelHigh);
    this.boundsLowLabel.removeEventListener('click', this.onClickLow);
    this.boundsHighLabel.removeEventListener('click', this.onClickHigh);
    this.disposeLowValueEdit();
    this.disposeHighValueEdit();
    this.wrapper.remove();
  }
}
