import { SimpleDims } from '../types/dims';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import { calculateNextPosition, calculateStepSize } from '../scene/dims/step-math';
import { getNonDisplayedDimensions } from '../scene/dims/selection';
import { getViewerContainer } from '../utils/viewer-container';
import { getInputProfile } from '../utils/input-capabilities';
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

  /** Zero-based position in the non-displayed dimension list selected for [ / ]. */
  selectedDimension?: number;

  /**
   * Fired when the panel selects the [ / ] target itself: under a coarse
   * pointer each dimension's name is a tappable chip, the finger's stand-in
   * for the 1–9 keys. Receives the position in the non-displayed list.
   */
  onSelectDimension?: (navigableIndex: number) => void;
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

  /** Glass-surface root of the panel (positioning, sizing, visibility) */
  private slidersContainer: HTMLElement;

  /**
   * Inner scroll wrapper that holds all panel content. Scrolling must not
   * live on the glass root (its glass layers paint at `inset: 0` behind it
   * and a scroll container clips them) — see UI_DESIGN_GUIDE §5.1.2/§7.4.
   * Created once, so clearing the content never destroys it.
   */
  private scrollBody: HTMLElement;

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

  /** Zero-based position in the non-displayed dimension list selected for [ / ]. */
  private selectedDimension: number;
  private readonly onSelectDimension?: (navigableIndex: number) => void;

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
   * @param config.container - DOM element to append sliders to (the viewer
   *   container — `getViewerContainer()`, i.e. the embedder's `container` or
   *   `document.body` by default)
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
   *   container: getViewerContainer(),
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
    this.selectedDimension = config.selectedDimension ?? 0;
    this.onSelectDimension = config.onSelectDimension;

    // Build the UI hierarchy
    const { root, scroll } = this.createSlidersContainer();
    this.slidersContainer = root;
    this.scrollBody = scroll;

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
   * - Scroll support if many dimensions, delegated to an inner wrapper so the
   *   glass root can stay `overflow: visible` (§5.1.2/§7.4)
   *
   * @returns The glass root and the inner scroll wrapper that receives content
   * @private
   */
  private createSlidersContainer(): { root: HTMLElement; scroll: HTMLElement } {
    const container = document.createElement('div');
    container.id = 'luxar-dimension-sliders';
    container.className = 'luxar-dimension-sliders luxar-glass-surface';

    const scroll = document.createElement('div');
    scroll.className = 'luxar-dimension-sliders__scroll';
    container.appendChild(scroll);

    this.container.appendChild(container);
    return { root: container, scroll };
  }

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

    // Defensive clear against duplicate slider UI (there is one call site
    // today, from the constructor). It must target the scroll wrapper rather
    // than the root so that it can never destroy the wrapper itself — nor
    // anything the theme injects into the root, such as the
    // `.luxar-glass-refraction` layer.
    this.scrollBody.innerHTML = '';
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
    this.statusText.setAttribute('aria-live', 'polite');

    titleContainer.appendChild(title);
    titleContainer.appendChild(this.statusText);
    this.scrollBody.appendChild(titleContainer);

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

      this.scrollBody.appendChild(dropdownGrid);
    }

    // Handle edge case: no dimensions are navigable
    if (this.sliders.size === 0 && this.dropdowns.size === 0 && this.toggles.size === 0) {
      const message = document.createElement('div');
      message.className = 'luxar-dimension-sliders__empty';
      message.textContent = 'All dimensions are displayed';
      this.scrollBody.appendChild(message);
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
    } else {
      label.title = name;
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
        option.textContent = String(Math.round(value));
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
    } else {
      label.title = name;
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

    // Resolved ONCE, because the underline affordance and the tooltip text are
    // two decisions off the same fact and must not disagree. Compilers write an
    // EMPTY description rather than omitting the key (the shipped
    // `dimension_sliders_5d_example` does), so this has to be a truthiness test:
    // reading it as `dimMeta?.description ?? name` left the real viewer with a
    // `title=""` — the class ternary correctly saw no description while the
    // tooltip used it anyway, and the full name became unrecoverable exactly
    // where the CSS had just started truncating it.
    const description = dimMeta?.description ? dimMeta.description : undefined;

    const dimName = document.createElement('span');
    dimName.className = description
      ? 'luxar-dimension-slider__name luxar-dimension-slider__name--with-tooltip'
      : 'luxar-dimension-slider__name';
    const name = this.dimensionNames[dimIndex] || `Dim ${dimIndex}`;
    dimName.textContent = name;

    // The name is ellipsised by CSS once it would claim the value's reserved
    // width, so it always needs a tooltip to stay recoverable — falling back to
    // the bare name, exactly as the dropdown and toggle labels already do
    // (`createDropdownInGrid` / `createToggleInGrid`). The dotted underline
    // stays tied to an authored description: a tooltip that only repeats
    // visible text should not advertise itself.
    dimName.title = description ?? name;

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

    // Mouse wheel steps the value by the dimension's BASE step (authored
    // step, else 1% of range); Shift = fine (÷10), Ctrl = coarse (×10),
    // Ctrl+Shift = extra-fine (÷100) — the same modifier tiers as [/].
    // Requiring Shift alongside Ctrl for extra-fine keeps a macOS trackpad
    // pinch (which arrives as a ctrlKey-only wheel event) off that tier.
    // Deliberately DECOUPLED from the animation menu's Step override — hand
    // stepping stays on the dimension's own grid (user decision).
    // setDimensionValue is the authoritative clamp + discrete snap; no
    // cyclic wrap on wheel.
    const wheelHandler = (event: WheelEvent) => {
      // preventDefault: don't scroll the page, and don't let Ctrl+wheel
      // zoom it (requires { passive: false }).
      event.preventDefault();
      // Shift+wheel on a standard mouse arrives as a HORIZONTAL scroll
      // (the browser swaps the axis, leaving deltaY = 0) — read whichever
      // axis carries the motion.
      const delta = event.deltaY !== 0 ? event.deltaY : event.deltaX;
      if (delta === 0) return;
      const wheelStep = calculateStepSize(dimIndex, this.dims, {
        shift: event.shiftKey,
        ctrl: event.ctrlKey,
      });
      const direction = delta < 0 ? 1 : -1; // scroll up = increase
      // Read the live value, not slider.value — the continuous slider's
      // 0–1000 integer scale would quantize and drift under fine steps.
      const live = this.dims.currentStep[dimIndex];
      sceneDimsManager.setDimensionValue(dimIndex, live + direction * wheelStep);
    };

    this.sliderEvents.on(slider, 'input', inputHandler);
    this.sliderEvents.on(slider, 'keydown', keydownHandler);
    this.sliderEvents.on(sliderContainer, 'wheel', wheelHandler, { passive: false });

    // Discoverability, matching the layers range-slider's affordance.
    slider.title = 'Scroll to step (Shift = fine, Ctrl = coarse, Ctrl+Shift = extra-fine)';

    sliderContainer.appendChild(progressBar);
    sliderContainer.appendChild(slider);
    sliderContainer.appendChild(thumb);

    sliderGroup.appendChild(label);
    if (getInputProfile().coarsePointer) {
      sliderGroup.appendChild(this.wrapWithStepButtons(dimIndex, name, sliderContainer));
      this.makeNameChip(dimIndex, dimName);
    } else {
      sliderGroup.appendChild(sliderContainer);
    }

    this.scrollBody.appendChild(sliderGroup);
    this.sliders.set(dimIndex, slider);
  }

  /**
   * Coarse pointers only: `‹ track ›`, one dimension step per tap (the
   * authored step, else 1 % of the range — the same base step as the [ / ]
   * keys), in the same wrapper the play button later joins.
   * A finger cannot scroll a slider by a single step, and a phone has no
   * bracket keys; the buttons are the missing precise input.
   */
  private wrapWithStepButtons(
    dimIndex: number,
    name: string,
    sliderContainer: HTMLElement
  ): HTMLElement {
    const wrapper = document.createElement('div');
    wrapper.className = 'luxar-dimension-slider__controls-wrapper';
    const makeStep = (direction: -1 | 1): HTMLButtonElement => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'luxar-dimension-slider__step';
      btn.textContent = direction < 0 ? '\u2039' : '\u203A';
      btn.setAttribute('aria-label', `${direction < 0 ? 'Previous' : 'Next'} ${name}`);
      this.sliderEvents.on(btn, 'click', () => {
        const dimMeta = this.dims.metadata?.[dimIndex];
        const step = calculateStepSize(
          dimIndex,
          this.dims,
          {},
          undefined,
          this.animationManager?.getStepSize(dimIndex) ?? null
        );
        const next = calculateNextPosition(
          this.dims.currentStep[dimIndex],
          direction,
          step,
          this.dimensionRanges[dimIndex],
          dimMeta?.discrete,
          dimMeta?.cyclic,
          dimMeta?.step
        );
        sceneDimsManager.setDimensionValue(dimIndex, next);
      });
      return btn;
    };
    wrapper.appendChild(makeStep(-1));
    wrapper.appendChild(sliderContainer);
    wrapper.appendChild(makeStep(1));
    return wrapper;
  }

  /** Coarse pointers only: the dimension's name selects it as the [ / ] target. */
  private makeNameChip(dimIndex: number, dimName: HTMLElement): void {
    dimName.classList.add('luxar-dimension-slider__name--chip');
    dimName.setAttribute('role', 'button');
    dimName.tabIndex = 0;
    const select = (): void => {
      const navigableIndex = getNonDisplayedDimensions(this.dims).indexOf(dimIndex);
      if (navigableIndex < 0) return;
      this.setSelectedDimension(navigableIndex);
      this.onSelectDimension?.(navigableIndex);
    };
    this.sliderEvents.on(dimName, 'click', select);
    this.sliderEvents.on(dimName, 'keydown', (e) => {
      const key = (e as KeyboardEvent).key;
      if (key === 'Enter' || key === ' ') {
        e.preventDefault();
        select();
      }
    });
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
      const thumbWidth = thumb.offsetWidth || 16;
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
    // Show by CLEARING the inline value, not writing 'block': the stylesheet
    // makes the root a flex column so the __scroll wrapper's `flex: 1;
    // min-height: 0` can cap content at the panel's max-height. An inline
    // 'block' would override that and let content spill out of the panel.
    this.slidersContainer.style.display = isVisible ? 'none' : '';
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
   * The status bar shows the current keyboard-navigation target and the
   * dimensions displayed in 3D. Per-dimension values remain visible on their
   * own controls.
   *
   * Format:
   * - Available target: "[/]: 1 · Channel · Display: X, Y, Z"
   * - No target: "[/]: unavailable · Display: X, Y, Z"
   *
   * @public
   */
  public updateStatusBar(): void {
    const parts: string[] = [];

    const navigableDims = getNonDisplayedDimensions(this.dims);
    const selectedDim = navigableDims[this.selectedDimension];
    if (selectedDim !== undefined) {
      const selectedName = this.dimensionNames[selectedDim] || `Dim ${selectedDim}`;
      parts.push(`[/]: ${this.selectedDimension + 1} · ${selectedName}`);
    } else {
      parts.push('[/]: unavailable');
    }

    // Show which dimensions are currently displayed in 3D
    const displayedNames = this.dims.displayed
      .map((idx) => this.dimensionNames[idx] || `Dim ${idx}`)
      .join(', ');
    parts.push(`Display: ${displayedNames}`);

    const statusContent = parts.join(' · ');

    if (this.statusText && this.statusText.textContent !== statusContent) {
      this.statusText.textContent = statusContent;
      this.statusText.title = statusContent;
    }
  }

  /** Update the dimension targeted by the global [ / ] keyboard shortcuts. */
  public setSelectedDimension(selectedDimension: number): void {
    this.selectedDimension = selectedDimension;
    this.updateStatusBar();
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
    playButton.textContent = '▶';

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

    // Create a wrapper to hold play button and slider track horizontally.
    // Under a coarse pointer the track already sits in a wrapper with the
    // step buttons (see wrapWithStepButtons); the play button joins it.
    const sliderTrack = sliderGroup.querySelector('.luxar-dimension-slider__track');
    if (sliderTrack) {
      const existing = sliderTrack.parentElement;
      if (existing?.classList.contains('luxar-dimension-slider__controls-wrapper')) {
        existing.appendChild(playButton);
        return;
      }
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

    // Compact chip layout (UI Design Guide §6/§8): each section is one
    // micro-header row (title + an optional right-aligned muted readout)
    // over one WRAPPING row of selectable chips — the sanctioned
    // "active chip/segment" idiom — instead of a tall radio list. Header
    // TEXT stays exactly 'Speed' / 'Loop Mode' / 'Step' (E2E-pinned); the
    // uppercase rendering comes from CSS.
    const makeSection = (title: string, aside?: string): HTMLDivElement => {
      const section = document.createElement('div');
      section.className = 'luxar-dimension-slider__context-section';
      const headerRow = document.createElement('div');
      headerRow.className = 'luxar-dimension-slider__context-header-row';
      const header = document.createElement('div');
      header.className = 'luxar-dimension-slider__context-header';
      header.textContent = title;
      headerRow.appendChild(header);
      if (aside) {
        const asideEl = document.createElement('span');
        asideEl.className = 'luxar-dimension-slider__context-aside';
        asideEl.textContent = aside;
        headerRow.appendChild(asideEl);
      }
      section.appendChild(headerRow);
      const chips = document.createElement('div');
      chips.className = 'luxar-dimension-slider__context-chips';
      chips.setAttribute('role', 'radiogroup');
      chips.setAttribute('aria-label', title);
      section.appendChild(chips);
      menu.appendChild(section);
      return chips;
    };

    const makeChip = (
      labelText: string,
      selected: boolean,
      onPick: () => void,
      opts?: { tooltip?: string; mono?: boolean; grow?: boolean }
    ): HTMLButtonElement => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'luxar-dimension-slider__context-item';
      if (selected) chip.classList.add('luxar-dimension-slider__context-item--selected');
      if (opts?.mono) chip.classList.add('luxar-dimension-slider__context-item--mono');
      if (opts?.grow) chip.classList.add('luxar-dimension-slider__context-item--grow');
      chip.setAttribute('role', 'radio');
      chip.setAttribute('aria-checked', String(selected));
      if (opts?.tooltip) chip.title = opts.tooltip;
      chip.textContent = labelText;
      chip.addEventListener('click', onPick);
      return chip;
    };

    // Speed: FPS presets as mono numeral chips; the unit rides the header
    // as a muted aside so the chips don't each repeat 'FPS'. Sub-1 rates
    // read as fractions ('1/2', one frame every 2 s), not decimals.
    const speedChips = makeSection('Speed', 'fps');
    config.dimensionAnimation.presets.fps.forEach((fps) => {
      const label = fps >= 1 ? String(fps) : `1/${Math.round(1 / fps)}`;
      const tooltip = fps >= 1 ? `${fps} FPS` : `1 frame every ${Math.round(1 / fps)} s`;
      speedChips.appendChild(
        makeChip(
          label,
          fps === currentFPS,
          () => {
            if (this.animationManager) {
              this.animationManager.setTargetFPS(dimIndex, fps);
              this.closeContextMenu();
            }
          },
          { tooltip, mono: true }
        )
      );
    });

    // Loop Mode: three equal segments.
    const loopChips = makeSection('Loop Mode');
    const loopModes: Array<{ value: 'once' | 'loop' | 'bounce'; label: string }> = [
      { value: 'once', label: 'Once' },
      { value: 'loop', label: 'Loop' },
      { value: 'bounce', label: 'Bounce' },
    ];
    loopModes.forEach((mode) => {
      loopChips.appendChild(
        makeChip(
          mode.label,
          mode.value === currentLoopMode,
          () => {
            if (this.animationManager) {
              this.animationManager.setLoopMode(dimIndex, mode.value);
              this.closeContextMenu();
            }
          },
          { grow: true }
        )
      );
    });

    // Step section: the per-tick quantum for animation AND the [ / ] keys.
    // Presets are multipliers of the dimension's BASE step (authored step,
    // else 1% of the range); Auto restores the historical behavior
    // (continuous: fps-derived range/10s traversal; discrete: authored
    // step). The slider wheel/drag deliberately do NOT follow this override
    // — hand stepping stays on the base step.
    const currentStepOverride = this.animationManager?.getStepSize(dimIndex) ?? null;
    const meta = this.dims.metadata?.[dimIndex];
    const [rangeMin, rangeMax] = this.dimensionRanges[dimIndex];
    const baseStep = meta?.step ?? (rangeMax - rangeMin) * 0.01;
    const unit = this.dimensionUnits[dimIndex] || '';
    const formatStep = (v: number): string =>
      Number.isInteger(v) ? String(v) : Number(v.toPrecision(3)).toString();

    // Per-preset computed values live in chip tooltips; the header's muted
    // readout shows the ACTIVE quantum ('auto' when no override is set).
    const stepAside =
      currentStepOverride === null
        ? 'auto'
        : `${formatStep(currentStepOverride)}${unit ? ` ${unit}` : ''}`;
    const stepChips = makeSection('Step', stepAside);

    stepChips.appendChild(
      makeChip('Auto', currentStepOverride === null, () => {
        this.animationManager?.setStepSize(dimIndex, null);
        this.closeContextMenu();
      })
    );

    let presetMatched = currentStepOverride === null;
    // A discrete dimension cannot honour a quantum below one grid cell —
    // the animation step is quantized to the authored grid with a one-cell
    // floor (#1520) — so only offer presets whose VALUE reaches a cell.
    // Compare values, not multipliers: with no authored step the base falls
    // back to 1% of the range, so a sub-1 multiplier can still be several
    // grid cells (the epsilon absorbs the float product).
    const gridStep = meta?.step && meta.step > 0 ? meta.step : 1;
    const stepMultipliers = config.dimensionAnimation.presets.stepMultipliers.filter(
      (m) => !meta?.discrete || baseStep * m >= gridStep * (1 - 1e-9)
    );
    for (const m of stepMultipliers) {
      const value = baseStep * m;
      const selected =
        currentStepOverride !== null && Math.abs(currentStepOverride - value) <= value * 1e-6;
      if (selected) presetMatched = true;
      stepChips.appendChild(
        makeChip(
          `×${m}`,
          selected,
          () => {
            this.animationManager?.setStepSize(dimIndex, value);
            this.closeContextMenu();
          },
          { tooltip: `×${m} = ${formatStep(value)}${unit ? ` ${unit}` : ''}`, mono: true }
        )
      );
    }

    // Custom value input, on its own row directly under the chips — NOT
    // inside them: the chips row is a radiogroup, whose owned children must
    // all be radios, and a number input is a spinbutton (assistive tech
    // would report a broken radio-group ownership). MUST be a number input,
    // never type=range (the E2E slider helper indexes input[type=range]
    // inside the slider group), and number inputs are covered by
    // isTypingInInput, so global shortcuts ([ ] k …) stay suppressed while
    // typing.
    const customInput = document.createElement('input');
    customInput.type = 'number';
    customInput.className = 'luxar-dimension-slider__context-step-input';
    customInput.min = '0';
    customInput.step = 'any';
    customInput.placeholder = 'custom';
    customInput.setAttribute('aria-label', 'Custom step size');
    if (!presetMatched && currentStepOverride !== null) {
      customInput.value = formatStep(currentStepOverride);
    }
    // Commit once, on Enter or on blur (blur also fires when the
    // click-outside close removes the menu, so a typed value is not lost).
    // Routed Escape closes the menu without committing via PanelCoordinator.
    // Only an actually-edited value commits: the input is seeded with the
    // 3-significant-digit display form, so committing it untouched would
    // silently truncate a full-precision override (0.123456 → 0.123).
    let committed = false;
    let edited = false;
    customInput.addEventListener('input', () => {
      edited = true;
    });
    const commitCustom = (): void => {
      if (committed || !edited) return;
      const v = parseFloat(customInput.value);
      if (Number.isFinite(v) && v > 0) {
        committed = true;
        this.animationManager?.setStepSize(dimIndex, v);
      }
    };
    customInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        commitCustom();
        this.closeContextMenu();
      }
    });
    customInput.addEventListener('blur', commitCustom);
    const customRow = document.createElement('div');
    customRow.className = 'luxar-dimension-slider__context-step-custom';
    customRow.appendChild(customInput);
    // makeSection() appended the chips row to its section, so this lands the
    // field as the chips row's sibling — inside the Step section, outside the
    // radiogroup.
    stepChips.parentElement?.appendChild(customRow);

    // Add to document first (needed to measure height)
    getViewerContainer().appendChild(menu);
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
  }

  /**
   * Close active context menu and clean up event listeners
   */
  public closeContextMenu(): void {
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
      playButton.textContent = '▶';
      playButton.classList.remove('luxar-dimension-slider__play-btn--playing');
    }
  }

  /**
   * Set visibility of slider interface (show or hide).
   *
   * Used by main application to control slider display based on dataset
   * characteristics (nD vs 3D) or user preferences.
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
    // '' rather than 'block' when shown — the stylesheet's `display: flex`
    // must win (see toggle()).
    this.slidersContainer.style.display = visible ? '' : 'none';
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
  }
}
