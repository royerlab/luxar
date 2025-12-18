import { SimpleDims } from '../types/dims';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import type { DimensionAnimationManager } from '../scene/dimension-animation-manager';
import { config } from '../config';

/**
 * Configuration interface for initializing dimension sliders.
 *
 * @interface SliderConfig
 */
interface SliderConfig {
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

  /** Map to store bound event handlers for cleanup */
  private eventHandlers: Map<
    number,
    {
      input?: () => void;
      keydown?: (e: KeyboardEvent) => void;
      change?: () => void;
      mouseenter?: () => void;
      mouseleave?: () => void;
      focus?: () => void;
      blur?: () => void;
      playClick?: () => void;
      fpsChange?: () => void;
      loopChange?: () => void;
    }
  > = new Map();

  /** Animation manager for dimension playback (set by InputHandler) */
  private animationManager?: DimensionAnimationManager;

  /** Map of dimension indices to play button elements */
  private playButtons: Map<number, HTMLButtonElement> = new Map();

  /** Map of dimension indices to FPS selector elements */
  private fpsSelectors: Map<number, HTMLSelectElement> = new Map();

  /** Map of dimension indices to loop mode selector elements */
  private loopSelectors: Map<number, HTMLSelectElement> = new Map();

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
    this.animationManager = manager;

    // Add animation controls to existing sliders
    this.addAnimationControlsToSliders();

    // Listen for animation events to update UI
    this.animationManager.addEventListener('play', (e) => {
      this.updatePlayButtonState(e.dimIndex, true);
    });

    this.animationManager.addEventListener('pause', (e) => {
      this.updatePlayButtonState(e.dimIndex, false);
    });

    this.animationManager.addEventListener('speedChange', (e) => {
      this.updateFPSDisplay(e.dimIndex, e.fps);
    });
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
    container.id = 'dimension-sliders';
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
    // Remove event listeners from existing sliders before clearing
    for (const [dimIndex, slider] of this.sliders) {
      const handlers = this.eventHandlers.get(dimIndex);
      if (handlers) {
        if (handlers.input) {
          slider.removeEventListener('input', handlers.input);
        }
        if (handlers.keydown) {
          slider.removeEventListener('keydown', handlers.keydown);
        }
      }
    }

    // Remove event listeners from existing dropdowns
    for (const [dimIndex, dropdown] of this.dropdowns) {
      const handlers = this.eventHandlers.get(dimIndex);
      if (handlers) {
        if (handlers.change) dropdown.removeEventListener('change', handlers.change);
        if (handlers.keydown) dropdown.removeEventListener('keydown', handlers.keydown);
        // Hover/focus handlers removed - now handled by CSS
      }
    }

    // Clear any existing slider UI to prevent duplicates
    this.slidersContainer.innerHTML = '';
    this.sliders.clear();
    this.dropdowns.clear();
    this.eventHandlers.clear();

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

    // Separate dimensions by type: sliders (continuous + many categorical) vs dropdowns (few categorical)
    const sliderDims: number[] = [];
    const dropdownDims: number[] = [];

    for (let i = 0; i < this.dims.ndim; i++) {
      if (!this.dims.displayed.includes(i)) {
        const dimMeta = this.dims.metadata?.[i];
        const categories = dimMeta?.categories;

        // Categorical with < 10 categories → dropdown (at bottom)
        if (categories && categories.length < 10) {
          dropdownDims.push(i);
        }
        // Otherwise → slider (at top)
        else {
          sliderDims.push(i);
        }
      }
    }

    // Create sliders first (continuous dimensions + categorical with many categories)
    for (const dimIndex of sliderDims) {
      this.createSlider(dimIndex);
    }

    // Create categorical dropdowns in a grid at the bottom (max 3 per row)
    if (dropdownDims.length > 0) {
      const dropdownGrid = document.createElement('div');
      dropdownGrid.className =
        `luxar-dimension-dropdown-grid ${dropdownDims.length >= 3 ? 'luxar-dimension-dropdown-grid--three-cols' : ''} ${sliderDims.length > 0 ? 'luxar-dimension-dropdown-grid--with-spacing' : ''}`.trim();

      for (const dimIndex of dropdownDims) {
        this.createDropdownInGrid(dimIndex, dropdownGrid);
      }

      this.slidersContainer.appendChild(dropdownGrid);
    }

    // Handle edge case: no dimensions are navigable
    if (this.sliders.size === 0 && this.dropdowns.size === 0) {
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
    if (!categories) return;

    // Container for this dropdown (will be a grid item)
    const dropdownItem = document.createElement('div');
    dropdownItem.className = 'luxar-dimension-dropdown';

    // Compact label with dimension name (smaller, consistent with sliders)
    const label = document.createElement('div');
    label.className = dimMeta.description
      ? 'luxar-dimension-dropdown__label luxar-dimension-dropdown__label--with-tooltip'
      : 'luxar-dimension-dropdown__label';

    const name = this.dimensionNames[dimIndex] || `Dim ${dimIndex}`;
    label.textContent = name;

    // Add tooltip with description if available
    if (dimMeta.description) {
      label.title = dimMeta.description;
    }

    // Create dropdown matching Luxar UI style
    const dropdown = document.createElement('select');
    dropdown.id = `dim-dropdown-${dimIndex}`;
    dropdown.className = 'luxar-dimension-dropdown__select';

    // Note: Hover/focus states now handled by CSS :hover and :focus pseudo-classes
    // No need for JavaScript event handlers for styling!

    // Populate dropdown with categories
    categories.forEach((category, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = category;
      option.title = `${category} (index: ${index})`;
      dropdown.appendChild(option);
    });

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
        newVal = currentVal - 1;
        if (newVal < min) {
          newVal = isCyclic ? max : min; // Wrap if cyclic
        }
      } else if (event.key === 'ArrowDown' || event.key === 'ArrowRight' || event.key === ']') {
        event.preventDefault();
        newVal = currentVal + 1;
        if (newVal > max) {
          newVal = isCyclic ? min : max; // Wrap if cyclic
        }
      }

      if (newVal !== null) {
        dropdown.value = String(newVal);
        dropdown.dispatchEvent(new Event('change'));
      }
    };

    // Add event listeners with bound handlers
    dropdown.addEventListener('change', changeHandler);
    dropdown.addEventListener('keydown', keydownHandler);

    // Store handlers for cleanup (hover/focus handled by CSS now)
    this.eventHandlers.set(dimIndex, {
      change: changeHandler,
      keydown: keydownHandler,
    });

    dropdownItem.appendChild(label);
    dropdownItem.appendChild(dropdown);
    gridContainer.appendChild(dropdownItem);

    this.dropdowns.set(dimIndex, dropdown);
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
    valueLabel.id = `dim-value-${dimIndex}`;
    valueLabel.className = 'luxar-dimension-slider__value';

    label.appendChild(dimName);
    label.appendChild(valueLabel);

    // Create slider container with napari-like styling
    const sliderContainer = document.createElement('div');
    sliderContainer.className = 'luxar-dimension-slider__track';

    // Progress bar background
    const progressBar = document.createElement('div');
    progressBar.id = `progress-${dimIndex}`;
    progressBar.className = 'luxar-dimension-slider__progress';

    // Create range input
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.id = `dim-slider-${dimIndex}`;
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
    thumb.id = `thumb-${dimIndex}`;
    thumb.className = 'luxar-dimension-slider__thumb';

    // Set initial value
    const currentValue = this.dims.currentStep[dimIndex];
    if (isDiscrete) {
      slider.value = String(currentValue);
    } else {
      const range = max - min;
      const fraction = range === 0 ? 0.5 : (currentValue - min) / range;
      slider.value = String(Math.round(fraction * 1000));
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
        const range = max - min;
        // Handle edge case: dimension with no range (single value)
        value = range === 0 ? min : min + fraction * range;
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
          let newVal =
            event.key === 'ArrowLeft' ? currentVal - sliderStep : currentVal + sliderStep;

          // Apply cyclic wrapping or clamping
          if (newVal < min) {
            newVal = isCyclic ? max : min;
          } else if (newVal > max) {
            newVal = isCyclic ? min : max;
          }

          slider.value = String(newVal);
        } else {
          const stepSize = event.shiftKey ? 10 : 1;
          const currentVal = parseInt(slider.value);
          let newVal = event.key === 'ArrowLeft' ? currentVal - stepSize : currentVal + stepSize;

          // Clamp to range (continuous dimensions don't typically use cyclic)
          newVal = Math.max(0, Math.min(1000, newVal));
          slider.value = String(newVal);
        }
        slider.dispatchEvent(new Event('input'));
      }
    };

    // Add event listeners with bound handlers
    slider.addEventListener('input', inputHandler);
    slider.addEventListener('keydown', keydownHandler);

    // Store handlers for cleanup
    this.eventHandlers.set(dimIndex, { input: inputHandler, keydown: keydownHandler });

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
    const valueLabel = document.getElementById(`dim-value-${dimIndex}`);

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
          console.warn(
            `Invalid category index ${index} for dimension ${dimMeta.name}, valid range: [0, ${categories.length - 1}]`
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
    const range = max - min;
    // Handle edge case: dimension with no range (single value)
    const fraction = range === 0 ? 0.5 : (value - min) / range;

    // Update progress bar
    const progressBar = document.getElementById(`progress-${dimIndex}`);
    if (progressBar) {
      progressBar.style.width = `${fraction * 100}%`;
    }

    // Update thumb position
    const thumb = document.getElementById(`thumb-${dimIndex}`);
    if (thumb) {
      const containerWidth = thumb.parentElement?.offsetWidth || 300;
      const thumbWidth = 16;
      const left = fraction * (containerWidth - thumbWidth);
      thumb.style.left = `${left}px`;
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
   * Add animation controls to a specific slider
   * @private
   */
  private addAnimationControlsToSlider(dimIndex: number, sliderGroup: HTMLElement): void {
    if (!this.animationManager) return;

    // Create controls container
    const controlsContainer = document.createElement('div');
    controlsContainer.className = 'luxar-dimension-slider__controls';

    // Play/Pause button
    const playButton = document.createElement('button');
    playButton.className = 'luxar-dimension-slider__play-btn';
    playButton.setAttribute('aria-label', 'Play/Pause animation');
    playButton.textContent = '▶'; // Play icon

    // FPS selector
    const fpsContainer = document.createElement('div');
    fpsContainer.className = 'luxar-dimension-slider__speed';

    const fpsLabel = document.createElement('label');
    fpsLabel.textContent = 'FPS:';
    fpsLabel.className = 'luxar-dimension-slider__speed-label';

    const fpsSelect = document.createElement('select');
    fpsSelect.className = 'luxar-dimension-slider__speed-select';

    // Add FPS presets
    const presets = config.dimensionAnimation.presets.fps;
    presets.forEach((fps) => {
      const option = document.createElement('option');
      option.value = String(fps);
      option.textContent = String(fps);
      fpsSelect.appendChild(option);
    });

    // Set default
    fpsSelect.value = String(config.dimensionAnimation.defaults.targetFPS);

    fpsContainer.appendChild(fpsLabel);
    fpsContainer.appendChild(fpsSelect);

    // Loop mode selector
    const loopContainer = document.createElement('div');
    loopContainer.className = 'luxar-dimension-slider__loop';

    const loopLabel = document.createElement('label');
    loopLabel.textContent = 'Loop:';
    loopLabel.className = 'luxar-dimension-slider__loop-label';

    const loopSelect = document.createElement('select');
    loopSelect.className = 'luxar-dimension-slider__loop-select';

    const loopModes: Array<{ value: string; label: string }> = [
      { value: 'loop', label: 'Loop' },
      { value: 'once', label: 'Once' },
      { value: 'bounce', label: 'Bounce' },
    ];

    loopModes.forEach((mode) => {
      const option = document.createElement('option');
      option.value = mode.value;
      option.textContent = mode.label;
      loopSelect.appendChild(option);
    });

    loopSelect.value = config.dimensionAnimation.defaults.loop;

    loopContainer.appendChild(loopLabel);
    loopContainer.appendChild(loopSelect);

    // Create bound event handlers
    const playClickHandler = () => {
      if (!this.animationManager) return;

      const isPlaying = this.animationManager.isAnimating(dimIndex);
      if (isPlaying) {
        this.animationManager.pause(dimIndex);
      } else {
        const fps = parseInt(fpsSelect.value);
        const loopMode = loopSelect.value as 'once' | 'loop' | 'bounce';
        this.animationManager.play(dimIndex, { targetFPS: fps, loopMode });
      }
    };

    const fpsChangeHandler = () => {
      if (!this.animationManager) return;
      const fps = parseInt(fpsSelect.value);
      this.animationManager.setTargetFPS(dimIndex, fps);
    };

    const loopChangeHandler = () => {
      if (!this.animationManager) return;
      const loopMode = loopSelect.value as 'once' | 'loop' | 'bounce';
      this.animationManager.setLoopMode(dimIndex, loopMode);
    };

    // Add event listeners
    playButton.addEventListener('click', playClickHandler);
    fpsSelect.addEventListener('change', fpsChangeHandler);
    loopSelect.addEventListener('change', loopChangeHandler);

    // Store handlers for cleanup
    const handlers = this.eventHandlers.get(dimIndex) || {};
    handlers.playClick = playClickHandler;
    handlers.fpsChange = fpsChangeHandler;
    handlers.loopChange = loopChangeHandler;
    this.eventHandlers.set(dimIndex, handlers);

    // Store element references
    this.playButtons.set(dimIndex, playButton);
    this.fpsSelectors.set(dimIndex, fpsSelect);
    this.loopSelectors.set(dimIndex, loopSelect);

    // Assemble controls
    controlsContainer.appendChild(playButton);
    controlsContainer.appendChild(fpsContainer);
    controlsContainer.appendChild(loopContainer);

    // Insert controls after the slider track
    const sliderTrack = sliderGroup.querySelector('.luxar-dimension-slider__track');
    if (sliderTrack && sliderTrack.nextSibling) {
      sliderGroup.insertBefore(controlsContainer, sliderTrack.nextSibling);
    } else {
      sliderGroup.appendChild(controlsContainer);
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
   * Update FPS selector display
   * @private
   */
  private updateFPSDisplay(dimIndex: number, fps: number): void {
    const fpsSelector = this.fpsSelectors.get(dimIndex);
    if (!fpsSelector) return;

    fpsSelector.value = String(fps);
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
    // Remove all event listeners from sliders
    for (const [dimIndex, slider] of this.sliders) {
      const handlers = this.eventHandlers.get(dimIndex);
      if (handlers) {
        if (handlers.input) {
          slider.removeEventListener('input', handlers.input);
        }
        if (handlers.keydown) {
          slider.removeEventListener('keydown', handlers.keydown);
        }
      }
    }

    // Remove all event listeners from dropdowns
    for (const [dimIndex, dropdown] of this.dropdowns) {
      const handlers = this.eventHandlers.get(dimIndex);
      if (handlers) {
        if (handlers.change) dropdown.removeEventListener('change', handlers.change);
        if (handlers.keydown) dropdown.removeEventListener('keydown', handlers.keydown);
        // Hover/focus handlers removed - now handled by CSS
      }
    }

    // Remove all event listeners from animation controls
    for (const [dimIndex, playButton] of this.playButtons) {
      const handlers = this.eventHandlers.get(dimIndex);
      if (handlers?.playClick) {
        playButton.removeEventListener('click', handlers.playClick);
      }
    }

    for (const [dimIndex, fpsSelector] of this.fpsSelectors) {
      const handlers = this.eventHandlers.get(dimIndex);
      if (handlers?.fpsChange) {
        fpsSelector.removeEventListener('change', handlers.fpsChange);
      }
    }

    for (const [dimIndex, loopSelector] of this.loopSelectors) {
      const handlers = this.eventHandlers.get(dimIndex);
      if (handlers?.loopChange) {
        loopSelector.removeEventListener('change', handlers.loopChange);
      }
    }

    // Clear all maps
    this.eventHandlers.clear();
    this.sliders.clear();
    this.dropdowns.clear();
    this.playButtons.clear();
    this.fpsSelectors.clear();
    this.loopSelectors.clear();

    // Remove DOM elements
    this.slidersContainer.remove();
    this.statusBar.remove();
  }
}
