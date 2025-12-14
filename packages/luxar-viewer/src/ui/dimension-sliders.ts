import { SimpleDims } from '../types/dims';
import { sceneDimsManager } from '../scene/scene-dims-manager';
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
    container.style.position = 'fixed';
    container.style.left = '50%';
    container.style.transform = 'translateX(-50%)';
    container.style.bottom = '20px'; // Lower since no status bar
    container.style.backgroundColor = 'rgba(30, 30, 30, 0.9)';
    container.style.borderRadius = '8px';
    container.style.padding = '15px'; // Reduced from 20px
    container.style.width = '80%';
    container.style.maxWidth = '800px';
    container.style.minWidth = '400px';
    container.style.maxHeight = '240px'; // Add max height for compression
    container.style.overflowY = 'auto'; // Allow scrolling if needed
    container.style.fontFamily =
      '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif';
    container.style.fontSize = '12px'; // Slightly smaller
    container.style.color = '#e0e0e0';
    container.style.backdropFilter = 'blur(10px)';
    container.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.3)';
    container.style.zIndex = String(config.ui.zIndex.dimensionSliders);
    container.style.userSelect = 'none';

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
    // Clear any existing slider UI to prevent duplicates
    this.slidersContainer.innerHTML = '';
    this.sliders.clear();

    // Add title section with status text
    const titleContainer = document.createElement('div');
    titleContainer.style.display = 'flex';
    titleContainer.style.justifyContent = 'space-between';
    titleContainer.style.alignItems = 'center';
    titleContainer.style.marginBottom = '10px'; // Reduced from 12px
    titleContainer.style.borderBottom = '1px solid rgba(255, 255, 255, 0.2)';
    titleContainer.style.paddingBottom = '6px'; // Reduced from 8px

    const title = document.createElement('div');
    title.textContent = 'Dimension Navigation';
    title.style.fontWeight = 'bold';
    title.style.fontSize = '14px';

    this.statusText = document.createElement('div');
    this.statusText.style.fontFamily = 'monospace';
    this.statusText.style.fontSize = '12px';
    this.statusText.style.color = 'rgba(255, 255, 255, 0.8)';

    titleContainer.appendChild(title);
    titleContainer.appendChild(this.statusText);
    this.slidersContainer.appendChild(titleContainer);

    // Create individual slider for each non-displayed dimension
    for (let i = 0; i < this.dims.ndim; i++) {
      if (!this.dims.displayed.includes(i)) {
        this.createSlider(i);
      }
    }

    // Handle edge case: no dimensions are navigable
    if (this.sliders.size === 0) {
      const message = document.createElement('div');
      message.textContent = 'All dimensions are displayed';
      message.style.color = '#888';
      message.style.fontStyle = 'italic';
      message.style.textAlign = 'center';
      message.style.padding = '10px';
      this.slidersContainer.appendChild(message);
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
   *
   * @param dimIndex - Zero-based index of dimension to create slider for
   * @private
   */
  private createSlider(dimIndex: number): void {
    const sliderGroup = document.createElement('div');
    sliderGroup.style.marginBottom = '12px'; // Reduced from 15px

    // Get dimension metadata
    const dimMeta = this.dims.metadata?.[dimIndex];
    const isDiscrete = dimMeta?.discrete || false;
    const step = dimMeta?.step || 1.0;

    // Label with dimension name and current value
    const label = document.createElement('div');
    label.style.marginBottom = '4px'; // Reduced from 5px
    label.style.display = 'flex';
    label.style.justifyContent = 'space-between';
    label.style.alignItems = 'center';

    const dimName = document.createElement('span');
    const name = this.dimensionNames[dimIndex] || `Dim ${dimIndex}`;
    dimName.textContent = name;
    dimName.style.fontWeight = '500';

    const valueLabel = document.createElement('span');
    valueLabel.id = `dim-value-${dimIndex}`;
    valueLabel.style.fontFamily = 'monospace';
    valueLabel.style.fontSize = '12px';
    valueLabel.style.color = '#4CAF50';

    label.appendChild(dimName);
    label.appendChild(valueLabel);

    // Create slider container with napari-like styling
    const sliderContainer = document.createElement('div');
    sliderContainer.style.position = 'relative';
    sliderContainer.style.height = '20px';
    sliderContainer.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
    sliderContainer.style.borderRadius = '10px';
    sliderContainer.style.overflow = 'hidden';

    // Progress bar background
    const progressBar = document.createElement('div');
    progressBar.id = `progress-${dimIndex}`;
    progressBar.style.position = 'absolute';
    progressBar.style.left = '0';
    progressBar.style.top = '0';
    progressBar.style.height = '100%';
    progressBar.style.backgroundColor = 'rgba(76, 175, 80, 0.3)';
    progressBar.style.transition = 'width 0.003s ease-out';
    progressBar.style.pointerEvents = 'none';

    // Create range input
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.id = `dim-slider-${dimIndex}`;

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

    // Style the slider to be invisible but functional
    slider.style.position = 'absolute';
    slider.style.width = '100%';
    slider.style.height = '100%';
    slider.style.margin = '0';
    slider.style.padding = '0';
    slider.style.opacity = '0';
    slider.style.cursor = 'pointer';
    slider.style.zIndex = '10';

    // Custom thumb indicator
    const thumb = document.createElement('div');
    thumb.id = `thumb-${dimIndex}`;
    thumb.style.position = 'absolute';
    thumb.style.width = '16px';
    thumb.style.height = '16px';
    thumb.style.backgroundColor = '#4CAF50';
    thumb.style.borderRadius = '50%';
    thumb.style.top = '50%';
    thumb.style.transform = 'translateY(-50%)';
    thumb.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.3)';
    thumb.style.pointerEvents = 'none';
    thumb.style.transition = 'left 0.003s ease-out';

    // Set initial value
    const currentValue = this.dims.currentStep[dimIndex];
    if (isDiscrete) {
      slider.value = String(currentValue);
    } else {
      const fraction = (currentValue - min) / (max - min);
      slider.value = String(Math.round(fraction * 1000));
    }

    this.updateSliderVisuals(dimIndex, currentValue, isDiscrete);

    // Add event listeners
    slider.addEventListener('input', () => {
      let value: number;
      if (isDiscrete) {
        value = parseFloat(slider.value);
      } else {
        const fraction = parseInt(slider.value) / 1000;
        const [min, max] = this.dimensionRanges[dimIndex];
        value = min + fraction * (max - min);
      }

      sceneDimsManager.setDimensionValue(dimIndex, value);
      // Visual update will happen via listener callback
    });

    // Add keyboard navigation
    slider.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        if (isDiscrete) {
          const currentVal = parseFloat(slider.value);
          const sliderStep = parseFloat(slider.step);
          const newVal =
            event.key === 'ArrowLeft'
              ? Math.max(min, currentVal - sliderStep)
              : Math.min(max, currentVal + sliderStep);
          slider.value = String(newVal);
        } else {
          const stepSize = event.shiftKey ? 10 : 1;
          const currentVal = parseInt(slider.value);
          const newVal =
            event.key === 'ArrowLeft'
              ? Math.max(0, currentVal - stepSize)
              : Math.min(1000, currentVal + stepSize);
          slider.value = String(newVal);
        }
        slider.dispatchEvent(new Event('input'));
      }
    });

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
   * - Value label text (formatted with units and appropriate precision)
   * - Progress bar width (fraction of full range)
   * - Thumb position (aligned with progress bar)
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
    // Update value label
    const unit = this.dimensionUnits[dimIndex] || '';
    const valueLabel = document.getElementById(`dim-value-${dimIndex}`);
    if (valueLabel) {
      if (isDiscrete) {
        valueLabel.textContent = `${Math.round(value)}${unit ? ' ' + unit : ''}`;
      } else {
        valueLabel.textContent = `${value.toFixed(2)}${unit ? ' ' + unit : ''}`;
      }
    }

    // Calculate fraction for visual position
    const [min, max] = this.dimensionRanges[dimIndex];
    const fraction = (value - min) / (max - min);

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
   * Format: "Display: X, Y, Z | Time: 5.20s | Channel: 2"
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
        const value = this.dims.currentStep[i].toFixed(2);
        const unit = this.dimensionUnits[i] || '';
        parts.push(`${name}: ${value}${unit ? ' ' + unit : ''}`);
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
   * Synchronizes all slider visuals with the current dimension state.
   *
   * This method is called by the scene dimension manager's observer system
   * whenever dimensions change. It ensures the UI accurately reflects the
   * current slice positions by updating slider positions, value labels,
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
        const fraction = (currentValue - min) / (max - min);
        slider.value = String(Math.round(fraction * 1000));
      }

      // Update visual elements (progress bar, thumb, value label)
      this.updateSliderVisuals(dimIndex, currentValue, isDiscrete);
    }

    // Refresh the status bar to show current state
    this.updateStatusBar();
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
    this.slidersContainer.remove();
    this.statusBar.remove();
    this.sliders.clear();
  }
}
