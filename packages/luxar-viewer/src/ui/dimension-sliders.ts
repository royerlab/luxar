import { SimpleDims } from '../types/dims';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import type { DimensionAnimationManager } from '../scene/animation/dimension-animation-manager';
import { config } from '../config';
import { log, Modules } from '../utils/log';
import { EventGroup } from '../utils/cross-layer/event-group';
import {
  clampWithCyclicWrap,
  valueToFraction,
  fractionToValue,
  fractionToThumbLeft,
  clampInteger,
} from './dimension-sliders/slider-math';

/**
 * Configuration interface for initializing dimension sliders.
 *
 * @interface SliderConfig
 */
export interface SliderConfig {
  /** DOM container where slider UI will be mounted */
  container: HTMLElement;

  /** Current dimension state from scene manager */
  dims: SimpleDims;

  /** Navigable bounds for each dimension */
  dimensionRanges: Array<[number, number]>;

  /** Human-readable names for each dimension */
  dimensionNames: string[];

  /** Optional physical units for each dimension */
  dimensionUnits?: string[];
}

/**
 * Interactive UI component providing sliders for navigating through non-displayed dimensions.
 *
 * This component creates a sophisticated slider interface that allows users to navigate
 * through nD datasets by adjusting positions in dimensions not currently displayed in 3D.
 * It provides both mouse and keyboard interaction with visual feedback and status display.
 *
 * Key features:
 * - Custom-styled sliders with progress bars and thumb indicators
 * - Automatic handling of discrete vs continuous dimensions
 * - Real-time synchronization with the scene dimension manager
 * - Keyboard navigation with fine/coarse stepping
 * - Status bar showing current slice position
 * - Responsive layout that adapts to available screen space
 *
 * Design philosophy:
 * - Mimics napari-style slider aesthetics for scientific familiarity
 * - Only shows sliders for non-displayed dimensions to avoid confusion
 * - Provides immediate visual feedback during navigation
 * - Handles edge cases gracefully (empty slices, discrete quantization)
 *
 * Slider synchronization:
 * 1. User moves slider → triggers sceneDimsManager.setDimensionValue()
 * 2. sceneDimsManager notifies all listeners → triggers update()
 * 3. update() refreshes slider visuals and status display
 * 4. Points re-slice automatically via their own listeners
 *
 * @class DimensionSliders
 */
export class DimensionSliders {
  /** Root DOM container for the slider UI */
  private container: HTMLElement;

  /** Container for all individual dimension sliders */
  private slidersContainer: HTMLElement;

  /** Status bar displaying current slice position */
  private statusBar: HTMLElement;

  /** Status text element in the title bar */
  private statusText: HTMLElement | null = null;

  /** Current dimension state (reference to scene manager state) */
  private dims: SimpleDims;

  /** Navigable bounds for each dimension */
  private dimensionRanges: Array<[number, number]>;

  /** Human-readable dimension names for UI labeling */
  private dimensionNames: string[];

  /** Physical units for each dimension */
  private dimensionUnits: string[];

  /** Map of dimension indices to their corresponding HTML slider elements */
  private sliders: Map<number, HTMLInputElement> = new Map();

  /** Map of dimension indices to their corresponding dropdown select elements */
  private dropdowns: Map<number, HTMLSelectElement> = new Map();

  /** Map of dimension indices to their corresponding toggle elements (binary categoricals) */
  private toggles: Map<number, HTMLElement> = new Map();

  /**
   * Cleanup group for all per-slider DOM listeners (input, keydown, change,
   * toggle click, play-button click + contextmenu). The group is rebuilt on
   * every `createSliders()` call so that disposing it removes every listener
   * from the previous render in one shot — no per-handler bookkeeping.
   *
   * Hover/focus visual states are handled entirely by CSS `:hover` and
   * `:focus` pseudo-classes — no JS listeners are attached for those.
   */
  private sliderEvents: EventGroup = new EventGroup();

  /** Animation manager for dimension playback (set by InputHandler) */
  private animationManager?: DimensionAnimationManager;

  /** Map of dimension indices to play button elements */
  private playButtons: Map<number, HTMLButtonElement> = new Map();

  /**
   * Cached DOM refs for the per-slider visual children. Filled in
   * `createSlider`, read by `updateSliderVisuals` to avoid three
   * `document.getElementById` lookups per call (this method runs at
   * up to 60 fps during animation playback).
   */
  private sliderElements: Map<
    number,
    { valueLabel: HTMLElement; progressBar: HTMLElement; thumb: HTMLElement }
  > = new Map();

  /** Active context menu element (only one can be open at a time) */
  private activeContextMenu: HTMLElement | null = null;

  /** Stored event handlers for context menu cleanup */
  private contextMenuCleanup: {
    clickOutside?: (e: MouseEvent) => void;
    escape?: (e: KeyboardEvent) => void;
    /**
     * Pending setTimeout that will install the click-outside handler.
     * Tracked so closeContextMenu() can cancel it if the menu is closed
     * before the deferred handler is attached — without this, the
     * listener gets attached but never removed, leaking on every
     * Escape-cancel.
     */
    clickOutsideTimeout?: ReturnType<typeof setTimeout>;
  } = {};

  /**
   * Cleanup group for animation-manager listeners. Rebuilt every time
   * `setAnimationManager()` is called so that re-binding to a new manager
   * (or detaching from the old one) is a single dispose.
   */
  private animationManagerEvents: EventGroup = new EventGroup();

  /**
   * Create and initialize the dimension slider UI component.
   *
   * Builds the complete slider interface including styled containers, individual
   * sliders for each non-displayed dimension, and status bar showing current
   * slice positions. The UI follows napari-style aesthetics for scientific
   * data visualization familiarity.
   *
   * The sliders are automatically synchronized with sceneDimsManager - moving
   * a slider triggers dimension changes which update all nD objects in the scene.
   *
   * @param config - Configuration object for slider initialization
   * @param config.container - DOM element to append sliders to (typically document.body)
   * @param config.dims - Current dimension state from sceneDimsManager
   * @param config.dimensionRanges - [min, max] bounds for each dimension
   * @param config.dimensionNames - Human-readable names (e.g., ['X', 'Y', 'Z', 'Time'])
   * @param config.dimensionUnits - Optional units (e.g., ['μm', 'μm', 'μm', 's'])
   *
   * @example
   * ```typescript
   * // After scene loads and dims are initialized
   * const dims = sceneDimsManager.getDims();
   * const ranges = sceneDimsManager.getDimensionRanges();
   * const names = sceneDimsManager.getDimensionNames();
   *
   * const sliders = new DimensionSliders({
   *   container: document.body,
   *   dims,
   *   dimensionRanges: ranges,
   *   dimensionNames: names,
   *   dimensionUnits: ['μm', 'μm', 'μm', 's', '']
   * });
   *
   * // Sliders now appear at bottom of viewport
   * // User can drag sliders or use arrow keys to navigate
   * ```
   */
  constructor(config: SliderConfig) {
    this.container = config.container;
    this.dims = config.dims;
    this.dimensionRanges = config.dimensionRanges;
    this.dimensionNames = config.dimensionNames;
    this.dimensionUnits = config.dimensionUnits || [];

    // Build the UI hierarchy
    this.slidersContainer = this.createSlidersContainer();
    // Status bar removed - status now shown in title
    this.statusBar = document.createElement('div'); // Keep for compatibility but hidden

    // Populate with actual sliders and initialize display
    this.createSliders();
    this.updateStatusBar();
  }

  /**
   * Set animation manager and create animation controls for existing sliders
   * Called by InputHandler after both DimensionSliders and DimensionAnimationManager are initialized
   *
   * @param manager - The animation manager instance
   */
  public setAnimationManager(manager: DimensionAnimationManager): void {
    // Tear down listeners from any previous manager.
    this.animationManagerEvents.dispose();
    this.animationManagerEvents = new EventGroup();

    this.animationManager = manager;

    // Add animation controls to existing sliders
    this.addAnimationControlsToSliders();

    const playHandler = (e: { dimIndex: number }): void =>
      this.updatePlayButtonState(e.dimIndex, true);
    const pauseHandler = (e: { dimIndex: number }): void =>
      this.updatePlayButtonState(e.dimIndex, false);

    manager.addEventListener('play', playHandler);
    manager.addEventListener('pause', pauseHandler);
    this.animationManagerEvents.add(() => manager.removeEventListener('play', playHandler));
    this.animationManagerEvents.add(() => manager.removeEventListener('pause', pauseHandler));
  }

  /**
   * Create the main container element for sliders with napari-style styling.
   *
   * Builds a fixed-position panel at bottom-center of viewport with:
   * - Semi-transparent dark background with blur
   * - Rounded corners and subtle shadow
   * - Responsive width (80% of viewport, max 800px, min 400px)
   * - Scroll support if many dimensions
   *
   * @returns Container element ready to receive slider controls
   * @private
   */
  private createSlidersContainer(): HTMLElement {
    const container = document.createElement('div');
    container.id = 'luxar-dimension-sliders';
    container.className = 'luxar-dimension-sliders';

    this.container.appendChild(container);
    return container;
  }

  // Status bar method removed - status now shown in title

  /**
   * Creates individual slider controls for all non-displayed dimensions.
   *
   * This method rebuilds the entire slider interface, creating a separate
   * control for each dimension that isn't currently being displayed in the 3D scene.
   * The logic ensures that users only see controls for dimensions they can
   * actually navigate through.
   *
   * UI structure:
   * - Title header with visual separator
   * - Individual sliders for each non-displayed dimension
   * - Fallback message if all dimensions are displayed
   *
   * @private
   */
  private createSliders(): void {
    // Tear down all listeners from the previous render in one shot.
    this.sliderEvents.dispose();
    this.sliderEvents = new EventGroup();

    // Clear any existing slider UI to prevent duplicates
    this.slidersContainer.innerHTML = '';
    this.sliders.clear();
    this.dropdowns.clear();
    this.toggles.clear();
    this.sliderElements.clear();

    // Add title section with status text
    const titleContainer = document.createElement('div');
    titleContainer.className = 'luxar-dimension-sliders__header';

    const title = document.createElement('div');
    title.className = 'luxar-dimension-sliders__title';
    title.textContent = 'Dimension Navigation';

    this.statusText = document.createElement('div');
    this.statusText.className = 'luxar-dimension-sliders__status';

    titleContainer.appendChild(title);
    titleContainer.appendChild(this.statusText);
    this.slidersContainer.appendChild(titleContainer);

    // Separate dimensions by type: toggles (binary categorical) vs dropdowns
    // (3-9 categories) vs sliders (everything else, incl. all discrete-numeric)
    const sliderDims: number[] = [];
    const toggleDims: number[] = [];
    const dropdownDims: number[] = [];

    for (let i = 0; i < this.dims.ndim; i++) {
      if (!this.dims.displayed.includes(i)) {
        const dimMeta = this.dims.metadata?.[i];
        const categories = dimMeta?.categories;

        // Toggle/dropdown pickers are reserved for TRUE categorical dimensions
        // (those with explicit `categories` labels). Discrete-but-numeric
        // dimensions such as `time`, frame index, or an unlabeled channel index
        // are ordinal — they get a scrubbing slider (which supports discrete
        // stepping) regardless of how few values they have. Treating a
        // small-count discrete-numeric dim as a categorical pick-one control was
        // wrong: e.g. a 6-frame `time` dimension would render as a dropdown.

        // Binary categorical (exactly 2 categories) → toggle button
        if (categories && categories.length === 2) {
          toggleDims.push(i);
        }
        // Categorical with 3-9 categories → dropdown
        else if (categories && categories.length < 10) {
          dropdownDims.push(i);
        }
        // Everything else (discrete-numeric, continuous, or categorical with
        // many categories) → slider (at top)
        else {
          sliderDims.push(i);
        }
      }
    }

    // Create sliders first (continuous dimensions + categorical with many categories)
    for (const dimIndex of sliderDims) {
      this.createSlider(dimIndex);
    }

    // Create toggles and dropdowns in a grid at the bottom (max 3 per row)
    const gridDims = [...toggleDims, ...dropdownDims];
    if (gridDims.length > 0) {
      const dropdownGrid = document.createElement('div');
      dropdownGrid.className =
        `luxar-dimension-dropdown-grid ${gridDims.length >= 3 ? 'luxar-dimension-dropdown-grid--three-cols' : ''} ${sliderDims.length > 0 ? 'luxar-dimension-dropdown-grid--with-spacing' : ''}`.trim();

      for (const dimIndex of toggleDims) {
        this.createToggleInGrid(dimIndex, dropdownGrid);
      }

      for (const dimIndex of dropdownDims) {
        this.createDropdownInGrid(dimIndex, dropdownGrid);
      }

      this.slidersContainer.appendChild(dropdownGrid);
    }

    // Handle edge case: no dimensions are navigable
    if (this.sliders.size === 0 && this.dropdowns.size === 0 && this.toggles.size === 0) {
      const message = document.createElement('div');
      message.className = 'luxar-dimension-sliders__empty';
      message.textContent = 'All dimensions are displayed';
      this.slidersContainer.appendChild(message);
    }
  }

  /**
   * Create a dropdown control optimized for grid layout.
   *
   * Compact dropdown designed to fit in a responsive grid (max 3 per row).
   * Used for categorical dimensions with < 10 categories.
   *
   * Provides:
   * - Compact layout with dimension name prefix
   * - Category labels in dropdown options
   * - Keyboard navigation (arrow keys, [ / ] keys)
   * - Cyclic wrapping if dimension.cyclic = true
   * - Tooltips matching Luxar UI style
   *
   * @param dimIndex - Zero-based index of dimension to create dropdown for
   * @param gridContainer - Grid container to append dropdown to
   * @private
   */
  private createDropdownInGrid(dimIndex: number, gridContainer: HTMLElement): void {
    const dimMeta = this.dims.metadata?.[dimIndex];
    const categories = dimMeta?.categories;
    const range = this.dimensionRanges[dimIndex];
    const step = dimMeta?.step ?? 1;
    const isDiscrete = dimMeta?.discrete || false;

    // Must have either categories or be a discrete dimension with valid range
    if (!categories && !(isDiscrete && range)) return;

    // Container for this dropdown (will be a grid item)
    const dropdownItem = document.createElement('div');
    dropdownItem.className = 'luxar-dimension-dropdown';

    // Compact label with dimension name (smaller, consistent with sliders)
    const label = document.createElement('div');
    label.className = dimMeta?.description
      ? 'luxar-dimension-dropdown__label luxar-dimension-dropdown__label--with-tooltip'
      : 'luxar-dimension-dropdown__label';

    const name = this.dimensionNames[dimIndex] || `Dim ${dimIndex}`;
    label.textContent = name;

    // Add tooltip with description if available
    if (dimMeta?.description) {
      label.title = dimMeta.description;
    }

    // Create dropdown matching Luxar UI style
    const dropdown = document.createElement('select');
    dropdown.id = `luxar-dim-dropdown-${dimIndex}`;
    dropdown.className = 'luxar-dimension-dropdown__select';

    // Note: Hover/focus states now handled by CSS :hover and :focus pseudo-classes
    // No need for JavaScript event handlers for styling!

    // Populate dropdown with categories or generate from range
    if (categories) {
      // Use explicit category labels
      categories.forEach((category, index) => {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = category;
        option.title = `${category} (index: ${index})`;
        dropdown.appendChild(option);
      });
    } else {
      // Generate numeric options from range for discrete dimensions
      const [min, max] = range;
      for (let value = min; value <= max; value += step) {
        const option = document.createElement('option');
        option.value = String(value);
        option.textContent = String(Math.round(value)); // Round for display
        option.title = `Value: ${value}`;
        dropdown.appendChild(option);
      }
    }

    // Set initial value
    const currentValue = Math.round(this.dims.currentStep[dimIndex]);
    dropdown.value = String(currentValue);

    // Create bound handlers for cleanup
    const changeHandler = () => {
      const value = parseInt(dropdown.value);
      sceneDimsManager.setDimensionValue(dimIndex, value);
    };

    // Add keyboard navigation (arrow keys and [ / ] keys) with cyclic support
    const keydownHandler = (event: KeyboardEvent) => {
      const [min, max] = this.dimensionRanges[dimIndex];
      const currentVal = parseInt(dropdown.value);
      const isCyclic = dimMeta?.cyclic || false;

      let newVal: number | null = null;

      if (event.key === 'ArrowUp' || event.key === 'ArrowLeft' || event.key === '[') {
        event.preventDefault();
        newVal = clampWithCyclicWrap(currentVal - 1, min, max, isCyclic);
      } else if (event.key === 'ArrowDown' || event.key === 'ArrowRight' || event.key === ']') {
        event.preventDefault();
        newVal = clampWithCyclicWrap(currentVal + 1, min, max, isCyclic);
      }

      if (newVal !== null) {
        dropdown.value = String(newVal);
        dropdown.dispatchEvent(new Event('change'));
      }
    };

    // Cleanup handled centrally by sliderEvents.dispose() in createSliders().
    this.sliderEvents.on(dropdown, 'change', changeHandler);
    this.sliderEvents.on(dropdown, 'keydown', keydownHandler);

    dropdownItem.appendChild(label);
    dropdownItem.appendChild(dropdown);
    gridContainer.appendChild(dropdownItem);

    this.dropdowns.set(dimIndex, dropdown);
  }

  /**
   * Create a binary toggle control for a dimension with exactly 2 values.
   *
   * Segmented toggle button that shows both labels side by side with the
   * active value highlighted. Single click toggles between the two values.
   *
   * Provides:
   * - Compact segmented layout: [ValueA | ValueB]
   * - One-click toggle interaction (vs 2-click dropdown)
   * - Keyboard navigation (arrow keys, [ / ] keys, Space, Enter)
   * - Tooltips matching Luxar UI style
   *
   * @param dimIndex - Zero-based index of dimension to create toggle for
   * @param gridContainer - Grid container to append toggle to
   * @private
   */
  private createToggleInGrid(dimIndex: number, gridContainer: HTMLElement): void {
    const dimMeta = this.dims.metadata?.[dimIndex];
    const categories = dimMeta?.categories;
    const range = this.dimensionRanges[dimIndex];

    // Determine the two labels
    let label0: string;
    let label1: string;
    if (categories && categories.length === 2) {
      label0 = categories[0];
      label1 = categories[1];
    } else {
      // Discrete non-categorical: use numeric labels
      label0 = String(Math.round(range[0]));
      label1 = String(Math.round(range[1]));
    }

    // Container (grid item) — reuses dropdown wrapper class for consistent grid layout
    const toggleItem = document.createElement('div');
    toggleItem.className = 'luxar-dimension-dropdown';

    // Label — reuses dropdown label class for consistent styling
    const label = document.createElement('div');
    label.className = dimMeta?.description
      ? 'luxar-dimension-dropdown__label luxar-dimension-dropdown__label--with-tooltip'
      : 'luxar-dimension-dropdown__label';

    const name = this.dimensionNames[dimIndex] || `Dim ${dimIndex}`;
    label.textContent = name;

    if (dimMeta?.description) {
      label.title = dimMeta.description;
    }

    // Single button toggle — shows current value, click swaps to other
    const toggle = document.createElement('div');
    toggle.className = 'luxar-dimension-toggle';
    toggle.id = `luxar-dim-toggle-${dimIndex}`;
    toggle.setAttribute('role', 'switch');
    toggle.setAttribute('aria-label', `${name}: click to toggle`);
    toggle.tabIndex = 0;

    // Store both labels as data attributes for text swapping
    toggle.dataset.label0 = label0;
    toggle.dataset.label1 = label1;

    // Set initial state
    const currentValue = Math.round(this.dims.currentStep[dimIndex]);
    this.updateToggleState(toggle, currentValue);

    // Click handler — clicking toggles to other value
    const clickHandler = () => {
      const current = Math.round(this.dims.currentStep[dimIndex]);
      const newValue = current === 0 ? 1 : 0;
      sceneDimsManager.setDimensionValue(dimIndex, newValue);
    };

    // Keyboard handler — same keys as dropdowns for consistency
    const keydownHandler = (event: KeyboardEvent) => {
      if (
        event.key === 'ArrowUp' ||
        event.key === 'ArrowLeft' ||
        event.key === 'ArrowDown' ||
        event.key === 'ArrowRight' ||
        event.key === '[' ||
        event.key === ']' ||
        event.key === ' ' ||
        event.key === 'Enter'
      ) {
        event.preventDefault();
        event.stopPropagation();
        const current = Math.round(this.dims.currentStep[dimIndex]);
        const newValue = current === 0 ? 1 : 0;
        sceneDimsManager.setDimensionValue(dimIndex, newValue);
      }
    };

    this.sliderEvents.on(toggle, 'click', clickHandler);
    this.sliderEvents.on(toggle, 'keydown', keydownHandler);

    toggleItem.appendChild(label);
    toggleItem.appendChild(toggle);
    gridContainer.appendChild(toggleItem);

    this.toggles.set(dimIndex, toggle);
  }

  /**
   * Update a toggle element's text and visual state.
   *
   * Shows the current value's label. When value is 1 (second option),
   * applies the --on modifier for highlighted styling.
   *
   * @param toggle - The toggle button element
   * @param activeValue - The currently active value (0 or 1)
   * @private
   */
  private updateToggleState(toggle: HTMLElement, activeValue: number): void {
    const label0 = toggle.dataset.label0 || '0';
    const label1 = toggle.dataset.label1 || '1';

    if (activeValue === 1) {
      toggle.textContent = label1;
      toggle.classList.add('luxar-dimension-toggle--on');
      toggle.title = `${label1} (click to switch to ${label0})`;
      toggle.setAttribute('aria-checked', 'true');
    } else {
      toggle.textContent = label0;
      toggle.classList.remove('luxar-dimension-toggle--on');
      toggle.title = `${label0} (click to switch to ${label1})`;
      toggle.setAttribute('aria-checked', 'false');
    }
  }

  /**
   * Create an individual slider control for a specific dimension.
   *
   * Builds a complete slider UI with:
   * - Dimension name label and current value display
   * - Custom-styled range input with visual progress bar
   * - Animated thumb indicator
   * - Keyboard navigation support (arrow keys with Shift for fine control)
   *
   * Handles both discrete (frame-based) and continuous (time-based) dimensions
   * with appropriate step sizes and value formatting.
   * For categorical dimensions with many categories (≥10), displays category labels.
   *
   * @param dimIndex - Zero-based index of dimension to create slider for
   * @private
   */
  private createSlider(dimIndex: number): void {
    const sliderGroup = document.createElement('div');
    sliderGroup.className = 'luxar-dimension-slider';

    // Get dimension metadata
    const dimMeta = this.dims.metadata?.[dimIndex];
    const isDiscrete = dimMeta?.discrete || false;
    const step = dimMeta?.step || 1.0;

    // Label with dimension name and current value
    const label = document.createElement('div');
    label.className = 'luxar-dimension-slider__label';

    const dimName = document.createElement('span');
    dimName.className = dimMeta?.description
      ? 'luxar-dimension-slider__name luxar-dimension-slider__name--with-tooltip'
      : 'luxar-dimension-slider__name';
    const name = this.dimensionNames[dimIndex] || `Dim ${dimIndex}`;
    dimName.textContent = name;

    // Add tooltip with description if available
    if (dimMeta?.description) {
      dimName.title = dimMeta.description;
    }

    const valueLabel = document.createElement('span');
    valueLabel.id = `luxar-dim-value-${dimIndex}`;
    valueLabel.className = 'luxar-dimension-slider__value';

    label.appendChild(dimName);
    label.appendChild(valueLabel);

    // Create slider container with napari-like styling
    const sliderContainer = document.createElement('div');
    sliderContainer.className = 'luxar-dimension-slider__track';

    // Progress bar background
    const progressBar = document.createElement('div');
    progressBar.id = `luxar-dim-progress-${dimIndex}`;
    progressBar.className = 'luxar-dimension-slider__progress';

    // Create range input
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.id = `luxar-dim-slider-${dimIndex}`;
    slider.className = 'luxar-dimension-slider__input';

    // Configure slider based on discrete/continuous
    const [min, max] = this.dimensionRanges[dimIndex];
    if (isDiscrete) {
      // For discrete dimensions, use actual range values
      slider.min = String(min);
      slider.max = String(max);
      slider.step = String(step);
    } else {
      // For continuous dimensions, use high resolution
      slider.min = '0';
      slider.max = '1000';
      slider.step = '1';
    }

    // Custom thumb indicator
    const thumb = document.createElement('div');
    thumb.id = `luxar-dim-thumb-${dimIndex}`;
    thumb.className = 'luxar-dimension-slider__thumb';

    // Cache the per-slider visual children for fast lookup in updateSliderVisuals.
    this.sliderElements.set(dimIndex, { valueLabel, progressBar, thumb });

    // Set initial value
    const currentValue = this.dims.currentStep[dimIndex];
    if (isDiscrete) {
      slider.value = String(currentValue);
    } else {
      slider.value = String(Math.round(valueToFraction(currentValue, min, max) * 1000));
    }

    this.updateSliderVisuals(dimIndex, currentValue, isDiscrete);

    // Create bound handlers for cleanup
    const inputHandler = () => {
      let value: number;
      if (isDiscrete) {
        value = parseFloat(slider.value);
      } else {
        const fraction = parseInt(slider.value) / 1000;
        const [min, max] = this.dimensionRanges[dimIndex];
        value = fractionToValue(fraction, min, max);
      }

      sceneDimsManager.setDimensionValue(dimIndex, value);
      // Visual update will happen via listener callback
    };

    // Add keyboard navigation with cyclic wrapping support
    const keydownHandler = (event: KeyboardEvent) => {
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        const isCyclic = dimMeta?.cyclic || false;

        if (isDiscrete) {
          const currentVal = parseFloat(slider.value);
          const sliderStep = parseFloat(slider.step);
          const candidate =
            event.key === 'ArrowLeft' ? currentVal - sliderStep : currentVal + sliderStep;

          slider.value = String(clampWithCyclicWrap(candidate, min, max, isCyclic));
        } else {
          const stepSize = event.shiftKey ? 10 : 1;
          const currentVal = parseInt(slider.value);
          const candidate =
            event.key === 'ArrowLeft' ? currentVal - stepSize : currentVal + stepSize;

          // Continuous dimensions use the 0–1000 internal slider range and
          // never wrap (cyclic is not meaningful for continuous values here).
          slider.value = String(clampInteger(candidate, 0, 1000));
        }
        slider.dispatchEvent(new Event('input'));
      }
    };

    this.sliderEvents.on(slider, 'input', inputHandler);
    this.sliderEvents.on(slider, 'keydown', keydownHandler);

    sliderContainer.appendChild(progressBar);
    sliderContainer.appendChild(slider);
    sliderContainer.appendChild(thumb);

    sliderGroup.appendChild(label);
    sliderGroup.appendChild(sliderContainer);

    this.slidersContainer.appendChild(sliderGroup);
    this.sliders.set(dimIndex, slider);
  }

  /**
   * Update visual elements of a slider to reflect current value.
   *
   * Synchronizes all visual components:
   * - Value label text (formatted with units, or category label for categorical dimensions)
   * - Progress bar width (fraction of full range)
   * - Thumb position (aligned with progress bar)
   * - Tooltip with detailed information
   *
   * Called during slider creation and whenever dimension value changes
   * (from keyboard navigation or programmatic updates).
   *
   * @param dimIndex - Dimension index to update visuals for
   * @param value - Current dimension value to display
   * @param isDiscrete - If true, rounds value to integer for display
   * @private
   */
  private updateSliderVisuals(dimIndex: number, value: number, isDiscrete: boolean): void {
    const dimMeta = this.dims.metadata?.[dimIndex];
    const unit = this.dimensionUnits[dimIndex] || '';

    // Prefer the cached refs created in createSlider — this method runs every
    // animation frame so avoiding three getElementById calls per slider per
    // frame matters. Fall back to a getElementById lookup for the rare case
    // where a slider was reused without going through createSlider.
    const cached = this.sliderElements.get(dimIndex);
    const valueLabel = cached?.valueLabel ?? document.getElementById(`luxar-dim-value-${dimIndex}`);

    if (valueLabel) {
      const categories = dimMeta?.categories;

      if (categories) {
        // Categorical dimension: show category label
        const index = Math.round(value);
        const label = categories[index];

        if (label !== undefined) {
          valueLabel.textContent = label;
          // Tooltip with more info: "DAPI (index: 0)"
          valueLabel.title = `${label} (index: ${index})`;
        } else {
          // Handle invalid index gracefully
          valueLabel.textContent = `Invalid (${index})`;
          valueLabel.title = `Index ${index} is out of range [0, ${categories.length - 1}]`;
          log.warning(
            Modules.UI,
            `Invalid category index ${index} for dimension ${dimMeta.name}, ` +
              `valid range: [0, ${categories.length - 1}]`
          );
        }
      } else if (isDiscrete) {
        // Discrete (numeric): show rounded value
        const roundedValue = Math.round(value);
        valueLabel.textContent = `${roundedValue}${unit ? ' ' + unit : ''}`;
        valueLabel.title = `Value: ${roundedValue}${unit ? ' ' + unit : ''}`;
      } else {
        // Continuous: show decimal value
        valueLabel.textContent = `${value.toFixed(2)}${unit ? ' ' + unit : ''}`;
        valueLabel.title = `Value: ${value.toFixed(4)}${unit ? ' ' + unit : ''}`;
      }
    }

    // Calculate fraction for visual position
    const [min, max] = this.dimensionRanges[dimIndex];
    const fraction = valueToFraction(value, min, max);

    // Update progress bar
    const progressBar =
      cached?.progressBar ?? document.getElementById(`luxar-dim-progress-${dimIndex}`);
    if (progressBar) {
      progressBar.style.width = `${fraction * 100}%`;
    }

    // Update thumb position
    const thumb = cached?.thumb ?? document.getElementById(`luxar-dim-thumb-${dimIndex}`);
    if (thumb) {
      const containerWidth = thumb.parentElement?.offsetWidth || 300;
      const thumbWidth = 16;
      thumb.style.left = `${fractionToThumbLeft(fraction, containerWidth, thumbWidth)}px`;
    }
  }

  /**
   * Toggle dimension sliders visibility on/off.
   *
   * Switches between visible and hidden states. Triggered by N key.
   * Does not destroy the sliders - they remain in DOM but hidden.
   *
   * @example
   * ```typescript
   * // User presses 'N' key
   * dimensionSliders.toggle();
   * // Sliders disappear if visible, appear if hidden
   * ```
   */
  public toggle(): void {
    const isVisible = this.slidersContainer.style.display !== 'none';
    this.slidersContainer.style.display = isVisible ? 'none' : 'block';
  }

  /**
   * Get current visibility state of sliders.
   *
   * @returns true if sliders are currently visible, false if hidden
   */
  public getIsVisible(): boolean {
    return this.slidersContainer.style.display !== 'none';
  }

  /**
   * Hide the dimension sliders panel.
   *
   * Sets display to 'none'. Sliders remain in DOM for fast re-showing.
   * Use when temporarily hiding UI or when dataset has no non-displayed dimensions.
   */
  public hide(): void {
    this.slidersContainer.style.display = 'none';
  }

  /**
   * Updates the status bar text to reflect the current dimensional state.
   *
   * The status bar provides a concise overview of the current navigation state,
   * showing both which dimensions are being displayed in 3D and the current
   * slice positions in all non-displayed dimensions.
   *
   * Format:
   * - Categorical: "Display: X, Y, Z | Channel: DAPI | Time: 5.20s"
   * - Numeric: "Display: X, Y, Z | Time: 5.20s | Index: 2"
   *
   * @public
   */
  public updateStatusBar(): void {
    const parts: string[] = [];

    // Show which dimensions are currently displayed in 3D
    const displayedNames = this.dims.displayed
      .map((idx) => this.dimensionNames[idx] || `Dim ${idx}`)
      .join(', ');
    parts.push(`Display: ${displayedNames}`);

    // Show current slice position for each non-displayed dimension
    for (let i = 0; i < this.dims.ndim; i++) {
      if (!this.dims.displayed.includes(i)) {
        const name = this.dimensionNames[i] || `Dim ${i}`;
        const dimMeta = this.dims.metadata?.[i];
        const categories = dimMeta?.categories;

        let valueStr: string;
        if (categories) {
          // Categorical: show category label
          const index = Math.round(this.dims.currentStep[i]);
          const label = categories[index];
          valueStr = label !== undefined ? label : `Invalid(${index})`;
        } else {
          // Numeric: show value with unit
          const value = this.dims.currentStep[i].toFixed(2);
          const unit = this.dimensionUnits[i] || '';
          valueStr = `${value}${unit ? ' ' + unit : ''}`;
        }

        parts.push(`${name}: ${valueStr}`);
      }
    }

    const statusContent = parts.join(' | ');
    this.statusBar.textContent = statusContent;

    // Also update the status text in title if it exists
    if (this.statusText) {
      this.statusText.textContent = statusContent;
    }
  }

  /**
   * Synchronizes all controls (sliders and dropdowns) with the current dimension state.
   *
   * This method is called by the scene dimension manager's observer system
   * whenever dimensions change. It ensures the UI accurately reflects the
   * current slice positions by updating control positions, value labels,
   * and the status bar.
   *
   * Critical for maintaining UI consistency during:
   * - Keyboard navigation
   * - Programmatic dimension changes
   * - Camera centering operations that adjust displayed dimension positions
   *
   * @public
   */
  public update(): void {
    // Update each slider's position and visual indicators
    for (const [dimIndex, slider] of this.sliders) {
      const dimMeta = this.dims.metadata?.[dimIndex];
      const isDiscrete = dimMeta?.discrete || false;
      const currentValue = this.dims.currentStep[dimIndex];

      // Update slider input value based on dimension type
      if (isDiscrete) {
        slider.value = String(currentValue);
      } else {
        const [min, max] = this.dimensionRanges[dimIndex];
        const range = max - min;
        const fraction = range === 0 ? 0.5 : (currentValue - min) / range;
        slider.value = String(Math.round(fraction * 1000));
      }

      // Update visual elements (progress bar, thumb, value label)
      this.updateSliderVisuals(dimIndex, currentValue, isDiscrete);
    }

    // Update each dropdown's selected value
    for (const [dimIndex, dropdown] of this.dropdowns) {
      const currentValue = Math.round(this.dims.currentStep[dimIndex]);
      dropdown.value = String(currentValue);
    }

    // Update each toggle's text and state
    for (const [dimIndex, toggle] of this.toggles) {
      const currentValue = Math.round(this.dims.currentStep[dimIndex]);
      this.updateToggleState(toggle, currentValue);
    }

    // Refresh the status bar to show current state
    this.updateStatusBar();
  }

  /**
   * Add animation controls to all existing sliders
   * @private
   */
  private addAnimationControlsToSliders(): void {
    // Add controls to each slider
    for (const [dimIndex, slider] of this.sliders) {
      const sliderGroup = slider.closest('.luxar-dimension-slider');
      if (sliderGroup) {
        this.addAnimationControlsToSlider(dimIndex, sliderGroup as HTMLElement);
      }
    }
  }

  /**
   * Add animation controls to a specific slider (compact layout with context menu)
   * @private
   */
  private addAnimationControlsToSlider(dimIndex: number, sliderGroup: HTMLElement): void {
    if (!this.animationManager) return;

    // Skip if controls already exist for this dimension
    if (this.playButtons.has(dimIndex)) return;

    // Create compact play/pause button
    const playButton = document.createElement('button');
    playButton.className = 'luxar-dimension-slider__play-btn';
    playButton.setAttribute('aria-label', 'Play/Pause animation (right-click for settings)');
    playButton.setAttribute('title', 'Play/Pause (right-click for settings)');
    playButton.textContent = '▶'; // Play icon

    // Click handler for play/pause
    const playClickHandler = () => {
      if (!this.animationManager) return;

      const isPlaying = this.animationManager.isAnimating(dimIndex);
      if (isPlaying) {
        this.animationManager.pause(dimIndex);
      } else {
        // Get current settings from animation state or use defaults
        const state = this.animationManager.getState(dimIndex);
        const fps = state?.targetFPS ?? config.dimensionAnimation.defaults.targetFPS;
        const loopMode = state?.loopMode ?? config.dimensionAnimation.defaults.loop;
        this.animationManager.play(dimIndex, { targetFPS: fps, loopMode });
      }
    };

    // Context menu handler for settings
    const contextMenuHandler = (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      this.showContextMenu(dimIndex, e.clientX, e.clientY);
    };

    // Cleanup handled centrally by sliderEvents.dispose() in createSliders().
    this.sliderEvents.on(playButton, 'click', playClickHandler);
    this.sliderEvents.on(playButton, 'contextmenu', contextMenuHandler);
    this.playButtons.set(dimIndex, playButton);

    // Create a wrapper to hold play button and slider track horizontally
    const sliderTrack = sliderGroup.querySelector('.luxar-dimension-slider__track');
    if (sliderTrack) {
      // Create wrapper container for horizontal layout
      const controlsWrapper = document.createElement('div');
      controlsWrapper.className = 'luxar-dimension-slider__controls-wrapper';

      // Replace slider track with wrapper containing button + track
      sliderGroup.replaceChild(controlsWrapper, sliderTrack);
      controlsWrapper.appendChild(playButton);
      controlsWrapper.appendChild(sliderTrack);
    }
  }

  /**
   * Show context menu for animation settings (Napari-style)
   * @private
   */
  private showContextMenu(dimIndex: number, x: number, y: number): void {
    // Close any existing context menu
    this.closeContextMenu();

    // Get current animation state
    const state = this.animationManager?.getState(dimIndex);
    const currentFPS = state?.targetFPS ?? config.dimensionAnimation.defaults.targetFPS;
    const currentLoopMode = state?.loopMode ?? config.dimensionAnimation.defaults.loop;

    // Create context menu
    const menu = document.createElement('div');
    menu.className = 'luxar-dimension-slider__context-menu';

    // Speed section
    const speedSection = document.createElement('div');
    speedSection.className = 'luxar-dimension-slider__context-section';

    const speedHeader = document.createElement('div');
    speedHeader.className = 'luxar-dimension-slider__context-header';
    speedHeader.textContent = 'Speed';
    speedSection.appendChild(speedHeader);

    const fpsPresets = config.dimensionAnimation.presets.fps;
    fpsPresets.forEach((fps) => {
      const item = document.createElement('div');
      item.className = 'luxar-dimension-slider__context-item';
      if (fps === currentFPS) {
        item.classList.add('luxar-dimension-slider__context-item--selected');
      }

      const radio = document.createElement('span');
      radio.className = 'luxar-dimension-slider__context-radio';
      radio.textContent = fps === currentFPS ? '●' : '○';

      const label = document.createElement('span');
      label.textContent = `${fps} FPS`;

      item.appendChild(radio);
      item.appendChild(label);

      item.addEventListener('click', () => {
        if (this.animationManager) {
          this.animationManager.setTargetFPS(dimIndex, fps);
          this.closeContextMenu();
        }
      });

      speedSection.appendChild(item);
    });

    menu.appendChild(speedSection);

    // Loop mode section
    const loopSection = document.createElement('div');
    loopSection.className = 'luxar-dimension-slider__context-section';

    const loopHeader = document.createElement('div');
    loopHeader.className = 'luxar-dimension-slider__context-header';
    loopHeader.textContent = 'Loop Mode';
    loopSection.appendChild(loopHeader);

    const loopModes: Array<{ value: 'once' | 'loop' | 'bounce'; label: string }> = [
      { value: 'once', label: 'Once' },
      { value: 'loop', label: 'Loop' },
      { value: 'bounce', label: 'Bounce' },
    ];

    loopModes.forEach((mode) => {
      const item = document.createElement('div');
      item.className = 'luxar-dimension-slider__context-item';
      if (mode.value === currentLoopMode) {
        item.classList.add('luxar-dimension-slider__context-item--selected');
      }

      const radio = document.createElement('span');
      radio.className = 'luxar-dimension-slider__context-radio';
      radio.textContent = mode.value === currentLoopMode ? '●' : '○';

      const label = document.createElement('span');
      label.textContent = mode.label;

      item.appendChild(radio);
      item.appendChild(label);

      item.addEventListener('click', () => {
        if (this.animationManager) {
          this.animationManager.setLoopMode(dimIndex, mode.value);
          this.closeContextMenu();
        }
      });

      loopSection.appendChild(item);
    });

    menu.appendChild(loopSection);

    // Add to document first (needed to measure height)
    document.body.appendChild(menu);
    this.activeContextMenu = menu;

    // Position menu above the cursor (Napari-style) with bounds checking
    const menuHeight = menu.offsetHeight;
    const menuWidth = menu.offsetWidth;

    // Ensure menu stays within viewport bounds
    let menuX = x;
    let menuY = y - menuHeight;

    // Check right edge
    if (menuX + menuWidth > window.innerWidth) {
      menuX = window.innerWidth - menuWidth - 10; // 10px padding from edge
    }

    // Check left edge
    if (menuX < 10) {
      menuX = 10; // 10px padding from edge
    }

    // Check top edge - if menu would go above viewport, show below cursor instead
    if (menuY < 10) {
      menuY = y + 10; // Show below cursor with 10px gap
    }

    menu.style.left = `${menuX}px`;
    menu.style.top = `${menuY}px`;

    // Close on click outside - store handler for cleanup
    const closeOnClickOutside = (e: MouseEvent) => {
      if (!menu.contains(e.target as Node)) {
        this.closeContextMenu();
      }
    };
    this.contextMenuCleanup.clickOutside = closeOnClickOutside;
    // Defer attaching the click-outside listener by one tick so the
    // contextmenu's own click event doesn't immediately trigger close.
    // Track the timeout so closeContextMenu() can cancel it if the menu
    // is closed (e.g. via Escape) before the listener gets attached.
    this.contextMenuCleanup.clickOutsideTimeout = setTimeout(() => {
      this.contextMenuCleanup.clickOutsideTimeout = undefined;
      // Only attach if the menu is still open — otherwise we'd add a
      // listener for a menu that's already gone.
      if (this.activeContextMenu) {
        document.addEventListener('click', closeOnClickOutside);
      }
    }, 0);

    // Close on escape - store handler for cleanup
    const closeOnEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        this.closeContextMenu();
      }
    };
    this.contextMenuCleanup.escape = closeOnEscape;
    document.addEventListener('keydown', closeOnEscape);
  }

  /**
   * Close active context menu and clean up event listeners
   * @private
   */
  private closeContextMenu(): void {
    if (this.activeContextMenu) {
      this.activeContextMenu.remove();
      this.activeContextMenu = null;
    }

    // Cancel a pending click-outside attach if the menu was closed before
    // the deferred attach fired. Otherwise the listener would be added
    // for a menu that no longer exists and never removed.
    if (this.contextMenuCleanup.clickOutsideTimeout !== undefined) {
      clearTimeout(this.contextMenuCleanup.clickOutsideTimeout);
      this.contextMenuCleanup.clickOutsideTimeout = undefined;
    }

    // Remove document-level event listeners
    if (this.contextMenuCleanup.clickOutside) {
      document.removeEventListener('click', this.contextMenuCleanup.clickOutside);
      this.contextMenuCleanup.clickOutside = undefined;
    }
    if (this.contextMenuCleanup.escape) {
      document.removeEventListener('keydown', this.contextMenuCleanup.escape);
      this.contextMenuCleanup.escape = undefined;
    }
  }

  /**
   * Update play button appearance based on animation state
   * @private
   */
  private updatePlayButtonState(dimIndex: number, isPlaying: boolean): void {
    const playButton = this.playButtons.get(dimIndex);
    if (!playButton) return;

    if (isPlaying) {
      playButton.textContent = '⏸'; // Pause icon
      playButton.classList.add('luxar-dimension-slider__play-btn--playing');
    } else {
      playButton.textContent = '▶'; // Play icon
      playButton.classList.remove('luxar-dimension-slider__play-btn--playing');
    }
  }

  /**
   * Set visibility of slider interface (show or hide).
   *
   * Used by main application to control slider display based on dataset
   * characteristics (nD vs 3D) or user preferences. Affects both slider
   * container and status bar.
   *
   * @param visible - true to show sliders, false to hide them
   *
   * @example
   * ```typescript
   * // Show sliders only if dataset has non-displayed dimensions
   * const hasNonDisplayed = sceneDimsManager.hasNonDisplayedDimensions();
   * dimensionSliders.setVisible(hasNonDisplayed);
   * ```
   */
  public setVisible(visible: boolean): void {
    this.slidersContainer.style.display = visible ? 'block' : 'none';
    this.statusBar.style.display = visible ? 'block' : 'none';
  }

  /**
   * Clean up slider UI and release resources.
   *
   * Removes all DOM elements and clears internal state. Important for
   * preventing memory leaks when visualization is destroyed or reinitialized
   * with different dataset.
   *
   * After calling dispose(), the DimensionSliders instance cannot be reused.
   * Create a new instance if sliders are needed again.
   *
   * @example
   * ```typescript
   * // Before loading new scene
   * dimensionSliders.dispose();
   * dimensionSliders = null;
   *
   * // After new scene loads
   * dimensionSliders = new DimensionSliders(newConfig);
   * ```
   */
  public dispose(): void {
    // Tear down DOM listeners + animation-manager listeners in one shot each.
    this.sliderEvents.dispose();
    this.animationManagerEvents.dispose();

    // Close any open context menu
    this.closeContextMenu();

    // Clear all maps
    this.sliders.clear();
    this.dropdowns.clear();
    this.toggles.clear();
    this.playButtons.clear();
    this.sliderElements.clear();

    // Remove DOM elements
    this.slidersContainer.remove();
    this.statusBar.remove();
  }
}
