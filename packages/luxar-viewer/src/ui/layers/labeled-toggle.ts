/**
 * A labelled on/off control for the Layers panel — the boolean sibling of
 * `LabeledSlider`, with the same DOM shape (a control group whose label row holds the
 * text in a `span`), so the panel's control-group lookups find it by label like every
 * other control. The checkbox sits where a slider's readout would.
 *
 * First use: the physical block's "Refract data" switch (spec
 * MESH_PHYSICAL_MATERIALS §3.4 Phase 3). Like the sliders it can be marked INERT with a
 * reason, for a state in which flipping it changes nothing on screen.
 *
 * @module ui/layers/labeled-toggle
 */

export interface LabeledToggleOptions {
  container: HTMLElement;
  label: string;
  initialChecked: boolean;
  /** Fired with the new state on each user change (not on `setChecked`). */
  onChange: (checked: boolean) => void;
}

export class LabeledToggle {
  private readonly wrapper: HTMLElement;
  private readonly input: HTMLInputElement;
  private readonly changeHandler: () => void;

  constructor(options: LabeledToggleOptions) {
    this.wrapper = document.createElement('div');
    this.wrapper.className = 'luxar-layers-panel__control-group';

    const labelEl = document.createElement('label');
    labelEl.className = 'luxar-layers-panel__control-label';

    const labelText = document.createElement('span');
    labelText.textContent = options.label;

    this.input = document.createElement('input');
    this.input.type = 'checkbox';
    this.input.className = 'luxar-layers-panel__toggle';
    this.input.checked = options.initialChecked;
    this.input.setAttribute('aria-label', options.label);

    labelEl.appendChild(labelText);
    labelEl.appendChild(this.input);
    this.wrapper.appendChild(labelEl);
    options.container.appendChild(this.wrapper);

    this.changeHandler = (): void => options.onChange(this.input.checked);
    this.input.addEventListener('change', this.changeHandler);
  }

  /** Programmatically set the state. Does not fire onChange. */
  setChecked(checked: boolean): void {
    this.input.checked = checked;
  }

  /** The current state. */
  isChecked(): boolean {
    return this.input.checked;
  }

  /** Show/hide the whole control group. */
  setVisible(visible: boolean): void {
    this.wrapper.style.display = visible ? '' : 'none';
  }

  /**
   * Mark the toggle INERT — flipping it changes nothing in the current state of its
   * siblings (refracting data needs a transmitting surface). Disabled, dimmed, with
   * the reason as hover text; `null` restores the live state. Same contract as
   * `LabeledSlider.setInert`.
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
    this.input.removeEventListener('change', this.changeHandler);
    this.wrapper.remove();
  }
}
