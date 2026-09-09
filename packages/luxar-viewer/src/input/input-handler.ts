/**
 * Comprehensive input handling system for nD navigation and scene interaction.
 *
 * This class provides the complete user interface layer for Luxar, handling:
 * - Keyboard navigation through nD dimensions
 * - Mouse and touch interaction coordination
 * - Fullscreen and FOV control
 * - UI element state management
 * - Dimension slider integration
 * - Performance monitoring controls
 *
 * The input handler bridges user interactions with the underlying nD visualization
 * system, translating keyboard/mouse events into dimension changes that trigger
 * coordinated updates across all scene objects.
 *
 * Key interaction patterns:
 * - [ ] keys navigate through dimensions with adaptive step sizes
 * - Number keys (1-9) select which non-displayed dimension to control
 * - Space bar toggles fullscreen mode
 * - Ctrl+wheel adjusts field of view, Shift+wheel rolls the view axis
 * - P key toggles performance statistics
 * - M key cycles data loading monitor
 * - R key toggles rendering controls
 * - H key shows/hides help overlay
 * - Ctrl+L toggles debug console
 *
 * The system maintains careful separation between:
 * - Camera controls (handled by THREE.js OrbitControls)
 * - Dimension navigation (handled by scene dimension manager)
 * - UI state management (handled by various UI components)
 *
 * All navigation events are coordinated through the scene dimension manager
 * to ensure consistent state across all nD objects in the scene.
 */

import { SceneManager } from '../scene/scene-manager';
import { AnimationController } from '../scene/animation/animation-controller';
import type { DimensionAnimationManager } from '../scene/animation/dimension-animation-manager';
import { notifier } from '../utils/cross-layer/notifier';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import {
  InputContext,
  InputContextManager,
  type ContextConfig,
  type InputContextId,
  type KeyBinding,
} from './input-handler/context-manager';
import {
  computeDimensionStep,
  resolveSelectedDimension,
} from './input-handler/dimension-navigation/compute-step';
import { describeNavigableKeys } from '../scene/dims/selection';
import { PanelCoordinator } from './input-handler/commands/panel-coordinator';
import { WindowEventHandler } from './input-handler/window-events/window-event-handler';
import { registerAllKeyBindings } from './input-handler/key-bindings/register-all';
import type { RegisteredShortcutBindings } from '../types/shortcut-help';
import { KeyAction, type KeyActionId } from './input-handler/key-bindings/actions';
import type {
  KeyBindingsCommands,
  KeyBindingsPanelGetters,
} from './input-handler/key-bindings/register-all';
import { isTypingInInput, isFocusOnSceneCanvas } from '../utils/dom/focus';
import {
  toggleControlMode,
  setControlMode,
  toggleInertialMode,
  type ControlModeCtx,
  type ControlType,
} from './input-handler/commands/control-mode';
import {
  clearDimensionUI,
  initDimensionSliders,
  type DimNavSetupCtx,
} from './input-handler/dimension-navigation/setup';
import { toggleFullscreen } from './input-handler/window-events/fullscreen-toggle';
import { cycleDataMonitor } from './input-handler/commands/data-monitor-cycle';
import {
  exportViewerState,
  type ViewerStateExportCtx,
} from './input-handler/commands/viewer-state-export';
import { log, Modules } from '../utils/log';
import type {
  ControlRailHandle,
  DebugConsoleHandle,
  DimensionSlidersFactory,
  DimensionSlidersHandle,
  LayersPanelHandle,
  PerformanceMonitorHandle,
  RecordingPanelHandle,
  RenderingControlsHandle,
  ToggleableHandle,
} from './input-handler/panel-capabilities';

// Re-exported through the package facade (`input/index.ts`).
export { KeyAction, type KeyActionId };
export type {
  ControlRailHandle,
  DimensionSlidersFactory,
} from './input-handler/panel-capabilities';

/**
 * Central coordinator for all user input events and nD navigation.
 *
 * @class InputHandler
 */
export class InputHandler {
  /** Cleanup functions for all registered event listeners */
  private eventListeners: (() => void)[] = [];

  // Idempotency guard. `init()` is one-shot — calling it twice would
  // double-bind keydown/keyup, controls start/change, and canvas
  // mousedown/touchstart listeners (each is a fresh bound function, so
  // removeEventListener can't dedupe). The guard prevents an HMR
  // re-init / context-restore / test re-setup from silently doubling
  // input event volume.
  private _initialized = false;

  /** Optional reference to advanced rendering controls */
  private renderingControls?: RenderingControlsHandle;

  /** Optional reference to scale bar overlay */
  private scaleBar?: ToggleableHandle;

  /** Optional reference to colormap legend overlay */
  private colormapLegend?: ToggleableHandle;

  /** Optional reference to recording panel */
  private recordingPanel?: RecordingPanelHandle;

  /** Optional reference to layers panel */
  private layersPanel?: LayersPanelHandle;

  /**
   * The command + panel surface shared with the keyboard bindings.
   * Built in {@link registerAllKeyBindings}; exposed via {@link getUiActions}.
   */
  private uiActions?: { commands: KeyBindingsCommands; panels: KeyBindingsPanelGetters };

  /** Optional reference to overlay manager */
  private overlayManager?: ToggleableHandle;

  /** Index of currently selected dimension for keyboard navigation */
  private selectedDimension: number = 0;

  /** UI component for interactive dimension sliders */
  private dimensionSliders?: DimensionSlidersHandle;

  /** Animation manager for dimension playback */
  private animationManager?: DimensionAnimationManager;

  /**
   * sceneDimsManager listener. Stored so dispose / clearDimensionUI
   * can remove it — without this, app dispose (without a subsequent
   * dataset switch) leaves the listener attached to the singleton
   * and retains a disposed InputHandler.
   */
  private sceneDimsListener?: () => Promise<void>;

  /** Debug console for capturing browser console output */
  private debugConsole: DebugConsoleHandle;

  /** Input context manager for handling keyboard conflicts */
  private contextManager: InputContextManager;

  /**
   * Panel-coordination concern: owns the priority-ordered "close all
   * panels" flow used by Escape. Constructed in the InputHandler ctor
   * once `debugConsole` and `animationController` are available.
   */
  private panelCoordinator: PanelCoordinator;
  private controlRail?: ControlRailHandle;

  /**
   * Window-event concern: owns resize / wheel / fullscreenchange.
   * Keyboard listeners stay in InputHandler — they're a separate
   * concern coordinating with InputContextManager.
   */
  private windowEvents: WindowEventHandler;

  /**
   * Create a new input handler for nD visualization interaction.
   *
   * Sets up the complete input handling infrastructure including context
   * management and debug console. Does not register event listeners until
   * init() is called.
   *
   * The input handler coordinates keyboard, mouse, and touch input across
   * the entire application, managing conflicts between different UI contexts
   * (navigation, typing, fly controls, etc.).
   *
   * @param sceneManager - Scene management system providing access to THREE.js
   *                       scene, camera, controls, and renderer
   * @param animationController - Animation loop coordinator for triggering
   *                              re-renders after input events
   *
   * @example
   * ```typescript
   * const sceneManager = new SceneManager(canvas);
   * const animController = new AnimationController(sceneManager);
   * const inputHandler = new InputHandler(sceneManager, animController);
   *
   * // Initialize event listeners
   * inputHandler.init();
   *
   * // Later, when scene loads, initialize dimension navigation
   * await sceneManager.loadSceneData(url);
   * inputHandler.initDimensionSliders();
   * ```
   */
  /**
   * Optional factory injected by `core/app.ts` to construct
   * `DimensionSliders` lazily once a scene is loaded. When omitted
   * (e.g. tests, embedders without nD navigation),
   * `initDimensionSliders()` becomes a no-op rather than reaching into
   * the ui layer directly.
   */
  private dimensionSlidersFactory?: DimensionSlidersFactory;

  constructor(
    private sceneManager: SceneManager,
    private animationController: AnimationController,
    private performanceMonitor: PerformanceMonitorHandle,
    debugConsole: DebugConsoleHandle,
    dimensionSlidersFactory?: DimensionSlidersFactory
  ) {
    this.dimensionSlidersFactory = dimensionSlidersFactory;
    // DebugConsole is constructed at the app level and passed in here,
    // so InputHandler doesn't need to import the class — keeps the
    // input → ui layer-cruiser rule clean.
    this.debugConsole = debugConsole;

    // Initialize input context manager
    this.contextManager = new InputContextManager();

    // Wire the panel coordinator with the always-present panels.
    // Optional panels (renderingControls, dimensionSliders, recordingPanel)
    // are pushed in via setRenderingControls / initDimensionSliders /
    // setRecordingPanel as they're created.
    this.panelCoordinator = new PanelCoordinator({
      debugConsole: this.debugConsole,
      performanceStats: this.performanceMonitor,
    });

    this.windowEvents = new WindowEventHandler(this.sceneManager, this.animationController);
  }

  /**
   * Associate rendering controls for post-processing and visual effects.
   *
   * Enables keyboard shortcuts (R, C) to toggle rendering controls panel
   * and cinematic mode. Should be called after rendering controls are
   * created. Optional - rendering controls integration is not required
   * for basic functionality.
   *
   * @param controls - Rendering controls UI component providing access to
   *                   bloom, HDR, noise, and other post-processing effects
   *
   * @example
   * ```typescript
   * const renderingControls = new RenderingControls(sceneManager);
   * inputHandler.setRenderingControls(renderingControls);
   *
   * // Now 'R' key toggles rendering controls panel
   * // Now 'C' key toggles cinematic mode
   * ```
   */
  setRenderingControls(controls: RenderingControlsHandle): void {
    this.renderingControls = controls;
    this.panelCoordinator.setRenderingControls(controls);
    this.windowEvents.setRenderingControls(controls);
  }

  setScaleBar(scaleBar: ToggleableHandle): void {
    this.scaleBar = scaleBar;
  }

  setColormapLegend(legend: ToggleableHandle): void {
    this.colormapLegend = legend;
  }

  setRecordingPanel(panel: RecordingPanelHandle): void {
    this.recordingPanel = panel;
    this.panelCoordinator.setRecordingPanel(panel);
  }

  /**
   * Forward a dataset-browser close handle (or `undefined` to clear it)
   * to PanelCoordinator so the Escape path closes via the panel's own
   * `close()` method — which fires `onClose` and clears the owner's
   * `LuxarApp.datasetBrowser` reference. The `O` shortcut needs that
   * reference cleared in order to reopen the panel.
   */
  setDatasetBrowser(browser: { close(): void } | undefined): void {
    this.panelCoordinator.setDatasetBrowser(browser);
  }

  setOverlayManager(manager: ToggleableHandle): void {
    this.overlayManager = manager;
  }

  setLayersPanel(panel: LayersPanelHandle): void {
    this.layersPanel = panel;
    // Forward to PanelCoordinator so Escape (the shortcut the panel's
    // close button advertises via aria-keyshortcuts) actually closes
    // the panel. Without this, Escape only flows through key-bindings
    // for the `L` shortcut and never reaches LayersPanel.hide().
    this.panelCoordinator.setLayersPanel(panel);
  }

  /**
   * Adopt the control rail (or `undefined` to clear it on dispose, the
   * contract the app's teardown relies on). Two duties: the handler keeps the
   * rail so a keydown the router reports as handled can notify it
   * (`handleRoutedKeyDown` — dismisses the first-run hint, refreshes
   * active-state), and forwards it to PanelCoordinator so the Escape flow
   * closes the rail's flyout/popover in the right priority order.
   */
  setControlRail(rail: ControlRailHandle | undefined): void {
    this.controlRail = rail;
    this.panelCoordinator.setControlRail(rail);
  }

  /**
   * Initialize all event listeners for user interaction.
   *
   * Sets up the complete input handling system including:
   * - Window events (resize, wheel, keyboard, fullscreen)
   * - Control events (orbit/fly control integration)
   * - User interaction events (mousedown, touchstart)
   * - Context-specific key bindings
   *
   * Called once during application initialization, after scene manager
   * is created but before scene loading. Re-entry is guarded: a second
   * call logs a warning and returns without re-binding listeners (so
   * HMR / context-restore / test re-setup can't silently double event
   * volume). Event listeners are automatically cleaned up when
   * dispose() is called.
   *
   * @example
   * ```typescript
   * const app = new LuxarApp();
   * const inputHandler = new InputHandler(sceneManager, animController);
   *
   * // Initialize input system
   * inputHandler.init();
   *
   * // Input handlers are now active
   * // User can press H for help, P for performance, etc.
   * ```
   */
  init(): void {
    if (this._initialized) {
      log.warning(Modules.INPUT, 'InputHandler.init() called twice; ignoring re-entry');
      return;
    }
    this._initialized = true;
    this.setupWindowEvents();
    this.setupControlEvents();
    this.setupUserInteractionEvents();
  }

  /**
   * Clear dimension UI and reset dimension manager to initial state.
   *
   * Disposes of dimension sliders and resets the scene dimension manager.
   * Used when loading a new scene to ensure clean state. The dimension
   * manager is reset to allow it to be reinitialized with new scene metadata.
   *
   * This is called automatically before loading a new scene. You typically
   * don't need to call this manually unless implementing custom scene
   * switching logic.
   *
   * @example
   * ```typescript
   * // Before loading new scene
   * inputHandler.clearDimensionUI();
   * await sceneManager.loadSceneData(newUrl);
   * inputHandler.initDimensionSliders();
   * ```
   */
  clearDimensionUI(): void {
    clearDimensionUI(this.makeDimNavSetupCtx());
  }

  /**
   * Initialize dimension navigation UI after scene loading completes.
   *
   * This method must be called after the scene is fully loaded and dimension
   * metadata is available. It sets up:
   * - Scene dimension manager integration with scene metadata
   * - Interactive dimension sliders UI for non-displayed dimensions
   * - Reactive update system for all nD objects (points, lines, splats)
   * - Keyboard navigation targets ([/] keys and number keys 1-9)
   *
   * The initialization process ensures all nD objects share the same
   * dimensional coordinate system and respond consistently to navigation.
   * If no nD objects are found in the scene, initialization is skipped
   * gracefully (3D-only scene).
   *
   * @example
   * ```typescript
   * // After scene loads
   * await sceneManager.loadSceneData(url);
   *
   * // Initialize dimension navigation
   * inputHandler.initDimensionSliders();
   *
   * // Now users can:
   * // - Press 1-9 to select a non-displayed dimension
   * // - Press [ ] to navigate selected dimension
   * // - Use sliders to navigate visually
   * ```
   *
   * @example
   * ```typescript
   * // Check if dimension sliders were created
   * inputHandler.initDimensionSliders();
   *
   * const dims = sceneDimsManager.getDims();
   * if (!dims) {
   *   console.log('No nD objects in scene (3D only)');
   * } else {
   *   console.log(`Navigating ${dims.ndim}D dataset`);
   * }
   * ```
   */
  initDimensionSliders(): void {
    initDimensionSliders(this.makeDimNavSetupCtx());
  }

  /**
   * Show the dimension sliders panel (if it exists).
   * Called from viewer_config application.
   */
  showDimensionSliders(): void {
    this.dimensionSliders?.setVisible(true);
  }

  /**
   * The dimension animation manager, once a scene with an animatable
   * dimension has been loaded. Undefined for a purely 3D scene, which never
   * builds one — callers must handle that rather than assume it exists.
   */
  getAnimationManager(): DimensionAnimationManager | undefined {
    return this.animationManager;
  }

  private makeDimNavSetupCtx(): DimNavSetupCtx {
    return {
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      dimensionSlidersFactory: this.dimensionSlidersFactory,
      panelCoordinator: this.panelCoordinator,
      recordingPanel: this.recordingPanel,
      getSelectedDimension: () => this.selectedDimension,
      setSelectedDimension: (value) => {
        this.selectedDimension = value;
      },
      getAnimationManager: () => this.animationManager,
      setAnimationManager: (manager) => {
        this.animationManager = manager;
      },
      getDimensionSliders: () => this.dimensionSliders,
      setDimensionSliders: (sliders) => {
        this.dimensionSliders = sliders;
      },
      getSceneDimsListener: () => this.sceneDimsListener,
      setSceneDimsListener: (listener) => {
        this.sceneDimsListener = listener;
      },
    };
  }

  /**
   * Set up window-level event listeners for global input handling.
   *
   * Registers listeners for:
   * - Window resize: Updates canvas size and camera aspect ratio
   * - Mouse wheel: Zoom and FOV control (Ctrl+wheel for FOV, Shift+wheel for roll)
   * - Keyboard: All keyboard shortcuts and navigation
   * - Fullscreen changes: Adjusts canvas styling for fullscreen mode
   *
   * All listeners are bound to class instance and stored for cleanup.
   * Called once during init().
   *
   * @private
   */
  private setupWindowEvents(): void {
    // Window-level resize / wheel / fullscreenchange — owned by
    // WindowEventHandler. Keyboard stays here because it has to
    // coordinate with InputContextManager and the registered
    // key-binding table.
    this.windowEvents.attach(this.eventListeners);

    const onKeyDown = this.onKeyDown.bind(this);
    const onKeyUp = this.onKeyUp.bind(this);
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    // Register all key bindings with context manager
    this.registerAllKeyBindings();

    this.eventListeners.push(
      () => window.removeEventListener('keydown', onKeyDown),
      () => window.removeEventListener('keyup', onKeyUp)
    );
  }

  /**
   * Set up event listeners for THREE.js orbit controls.
   *
   * Registers listeners on the controls object to trigger animation
   * when user interacts with camera controls (orbit, pan, zoom).
   * Ensures smooth rendering during camera manipulation.
   *
   * @private
   */
  private setupControlEvents(): void {
    // Use arrow functions so the `startAnimation` lookup is late-bound
    // on each event — if AnimationController swaps its `startAnimation`
    // method (HMR / test injection), the new method runs. Capturing
    // the bare method reference at construction time would freeze
    // the listener to the original closure.
    const startAnimation = (): void => this.animationController.startAnimation();
    const syncInputContext = (event?: { controlType?: ControlType }): void => {
      const { controlType } = event ?? {};
      if (!controlType) return;
      const context = controlType === 'fly' ? InputContext.FLY_CONTROLS : InputContext.NAVIGATION;
      if (this.contextManager.getContext() !== context) {
        this.contextManager.setContext(context);
      }
    };

    this.sceneManager.controls.addEventListener('start', startAnimation);
    this.sceneManager.controls.addEventListener('change', startAnimation);
    this.sceneManager.controls.addEventListener('change', syncInputContext);

    this.eventListeners.push(
      () => this.sceneManager.controls.removeEventListener('start', startAnimation),
      () => this.sceneManager.controls.removeEventListener('change', startAnimation),
      () => this.sceneManager.controls.removeEventListener('change', syncInputContext)
    );
  }

  /**
   * Set up user interaction event listeners for canvas.
   *
   * Registers listeners for mousedown and touchstart on the canvas
   * to trigger animation when user begins interaction. Provides
   * visual feedback that system is responding to input.
   *
   * @private
   */
  private setupUserInteractionEvents(): void {
    // Late-bind the startAnimation lookup via arrow function — see
    // setupControlEvents() for the rationale. A reference captured at
    // construction would survive any controller-method swap and call
    // the stale closure.
    const startAnimation = (): void => this.animationController.startAnimation();
    const canvas = this.sceneManager.renderer.domElement;

    canvas.addEventListener('mousedown', startAnimation);
    canvas.addEventListener('touchstart', startAnimation);

    this.eventListeners.push(
      () => canvas.removeEventListener('mousedown', startAnimation),
      () => canvas.removeEventListener('touchstart', startAnimation)
    );
  }

  /**
   * Handle cycling the data loading monitor (hidden → mini → expanded → hidden).
   *
   * Extracted to a method to support binding registration.
   *
   * @private
   */
  private handleDataMonitorCycle(): void {
    cycleDataMonitor();
  }

  /**
   * Register all key bindings with the context manager.
   *
   * This method registers all keyboard shortcuts using the binding registration
   * system. Bindings are organized by input context:
   * - NAVIGATION: Default orbit mode shortcuts
   * - FLY_CONTROLS: WASD movement keys for fly mode
   * - Explicit fallback contexts: shared shortcuts remain reachable where intended
   *
   * Called during init() to set up the complete keyboard interface.
   *
   * @private
   */
  private registerAllKeyBindings(): void {
    const panels: KeyBindingsPanelGetters = {
      getScaleBar: () => this.scaleBar,
      getColormapLegend: () => this.colormapLegend,
      getOverlayManager: () => this.overlayManager,
      getRecordingPanel: () => this.recordingPanel,
      getLayersPanel: () => this.layersPanel,
    };
    const commands: KeyBindingsCommands = {
      navigateDimension: (direction) => this.handleDimensionNavigation(direction),
      selectDimension: (index) => this.selectDimension(index),
      toggleHelp: () => this.toggleHelp(),
      toggleDimensionSliders: () => this.toggleDimensionSliders(),
      toggleDatasetBrowser: () => window.dispatchEvent(new CustomEvent('open-dataset-browser')),
      openElementMenu: (event) => {
        event.preventDefault();
        window.dispatchEvent(new CustomEvent('luxar-open-element-menu'));
      },
      togglePerformanceStats: () => this.togglePerformanceStats(),
      toggleRenderingControls: () => this.toggleRenderingControls(),
      toggleControlMode: () => this.toggleControlMode(),
      setControlMode: (type) => this.setControlMode(type),
      toggleInertialMode: () => this.toggleInertialMode(),
      toggleCinematicMode: () => this.toggleCinematicMode(),
      toggleFullscreen: () => this.toggleFullscreen(),
      cycleDataMonitor: () => this.handleDataMonitorCycle(),
      recenterCamera: () => this.recenterCamera(),
      exportViewerState: () => this.exportViewerState(),
      closeAllPanels: () => this.panelCoordinator.closeAll(),
      handleEscape: () => this.handleEscapeKey(),
      shouldHandleSpaceKey: () => this.shouldHandleSpaceKey(),
    };
    // Cache the same surface so on-screen affordances (the control rail)
    // can trigger identical actions without duplicating logic.
    this.uiActions = { commands, panels };
    registerAllKeyBindings({
      contextManager: this.contextManager,
      sceneManager: this.sceneManager,
      debugConsole: this.debugConsole,
      panels,
      commands,
      animationShortcuts: {
        getSelectedDimension: () => this.selectedDimension,
        getAnimationManager: () => this.animationManager,
      },
    });
  }

  /**
   * The command + panel surface the keyboard bindings dispatch into,
   * exposed so on-screen affordances (e.g. the {@link ControlRail}) can
   * trigger the exact same actions. Available after {@link init}.
   */
  getUiActions(): { commands: KeyBindingsCommands; panels: KeyBindingsPanelGetters } {
    if (!this.uiActions) {
      throw new Error('InputHandler.getUiActions() called before init()');
    }
    return this.uiActions;
  }

  /**
   * Handle key down events.
   *
   * Routes ALL keyboard events through the context manager's binding system.
   * No special cases - everything uses the unified binding system.
   */
  private onKeyDown(event: KeyboardEvent): void {
    // Escape always reaches the context manager so it can close panels
    // even when focus is inside a text input — e.g. the dataset-browser
    // manual-path field, the debug-console filter input. The typing-
    // context Escape path dispatches through NAVIGATION bindings (see
    // InputContextManager.dispatchEscapeFromTypingContext); this
    // exception is what routes Escape into the panel-close flow when
    // focus is inside an input.
    if (this.isTypingInInput() && event.key !== 'Escape') {
      return;
    }

    // Route ALL keys through context manager (including Shift, fly controls, etc.)
    const handled = this.contextManager.handleKeyEvent(event, 'down');
    if (handled) this.controlRail?.handleRoutedKeyDown();
  }

  /**
   * Handle key up events.
   *
   * Routes through context manager for keys that have keyupHandler registered.
   * Keys without keyupHandler (most toggle actions) are ignored on keyup.
   */
  private onKeyUp(event: KeyboardEvent): void {
    // Route through context manager
    // Only bindings with keyupHandler will execute (Shift, fly controls)
    // Toggle actions (C, R, etc.) don't have keyupHandler so they're ignored
    this.contextManager.handleKeyEvent(event, 'up');
  }

  /**
   * Toggle help overlay visibility on/off.
   *
   * Shows or hides the keyboard shortcuts help overlay. Triggered by H key.
   * The help overlay displays all available keyboard shortcuts organized
   * by category (navigation, view controls, panels, etc.).
   *
   * @private
   */
  private toggleHelp(): void {
    const helpOverlay = document.getElementById('luxar-help-overlay');
    if (helpOverlay) {
      notifier.hideHelp();
    } else {
      notifier.showHelp(this.contextManager.getRegisteredShortcutBindings());
    }
  }

  /** Registered bindings used to build the keyboard-shortcut overlay. */
  public getRegisteredShortcutBindings(): RegisteredShortcutBindings {
    return this.contextManager.getRegisteredShortcutBindings();
  }

  /** Register a custom keyboard-routing context. */
  public registerContext(context: InputContextId, config: ContextConfig): void {
    this.contextManager.registerContext(context, config);
  }

  /** Remove a custom keyboard-routing context and its bindings. */
  public unregisterContext(context: InputContextId): void {
    this.contextManager.unregisterContext(context);
  }

  /** Register a binding in a built-in or custom context. */
  public registerBinding(context: InputContextId, binding: KeyBinding): void {
    this.contextManager.registerBinding(context, binding);
  }

  /** Remove a binding from a built-in or custom context. */
  public unregisterBinding(
    context: InputContextId,
    key: string,
    modifiers?: KeyBinding['modifiers']
  ): void {
    this.contextManager.unregisterBinding(context, key, modifiers);
  }

  /** Activate a nested input context until {@link popContext} is called. */
  public pushContext(context: InputContextId): void {
    this.contextManager.pushContext(context);
  }

  /** Restore the context active before the latest {@link pushContext}. */
  public popContext(): void {
    this.contextManager.popContext();
  }

  /** Resolve the active binding label for a registered action. */
  public getShortcutLabel(actionId: string): string | undefined {
    return this.contextManager.getShortcutLabel(actionId);
  }

  /** Enable or disable all routed keyboard input. */
  public setEnabled(enabled: boolean): void {
    this.contextManager.setEnabled(enabled);
  }

  /**
   * Toggle dimension sliders panel visibility.
   *
   * Shows or hides the nD dimension navigation sliders. Triggered by N key.
   * Only functional if dimension sliders have been initialized (nD dataset loaded).
   * Has no effect for 3D-only datasets.
   *
   * @private
   */
  private toggleDimensionSliders(): void {
    if (this.dimensionSliders) {
      this.dimensionSliders.toggle();
    }
  }

  /**
   * Toggle performance statistics (FPS, memory) display.
   *
   * Shows or hides the stats.js performance monitor in top-left corner.
   * Triggered by P key. Displays:
   * - FPS (frames per second)
   * - Frame time in milliseconds
   * - Memory usage (if available)
   *
   * @private
   */
  private togglePerformanceStats(): void {
    this.performanceMonitor.toggle();
  }

  /**
   * Toggle fullscreen mode on/off.
   *
   * Requests fullscreen for the document element (true fullscreen including
   * browser chrome). Triggered by Space key (when not focused on UI element)
   * and the View-options fullscreen chip.
   *
   * Fullscreen exit is also possible via browser's native ESC key handling.
   *
   * @private
   */
  private toggleFullscreen(): void {
    toggleFullscreen();
  }

  /**
   * Toggle advanced rendering controls panel visibility.
   *
   * Shows or hides the rendering controls UI providing access to:
   * - Post-processing effects (bloom, HDR, vignette, chromatic aberration)
   * - Camera settings (FOV presets)
   * - Control mode selection (orbit, fly, ortho)
   * - Point rendering parameters
   *
   * Triggered by R key. Only functional if rendering controls have been
   * associated via setRenderingControls().
   *
   * @private
   */
  private toggleRenderingControls(): void {
    this.renderingControls?.toggle();
  }

  /**
   * Toggle cinematic mode (film-like visual effects).
   *
   * Enables or disables a preset combination of effects:
   * - Film grain noise
   * - Vignette (darkened corners)
   * - Chromatic aberration (color fringing)
   * - Lens distortion
   *
   * Triggered by C key. Provides quick access to cinematic aesthetics without
   * manually adjusting individual effects. Only functional if rendering controls
   * have been associated.
   *
   * @private
   */
  private toggleCinematicMode(): void {
    this.renderingControls?.toggleCinematicMode();
  }

  /**
   * Export the complete viewer state as JSON to the clipboard.
   *
   * Triggered by Ctrl+Shift+S. Captures all rendering settings, camera state,
   * dimensions, theme, etc. and copies the JSON to the clipboard. The JSON
   * can be loaded in Python with `luxar.ViewerConfig.from_json()`.
   *
   * @private
   */
  private exportViewerState(): void {
    exportViewerState(this.makeViewerStateExportCtx());
  }

  private makeViewerStateExportCtx(): ViewerStateExportCtx {
    return {
      sceneManager: this.sceneManager,
      renderingControls: this.renderingControls,
      animationManager: this.animationManager,
    };
  }

  /**
   * Cycle through camera control modes: Orbit → Fly → Ortho → Orbit.
   *
   * Triggered by V key. Control modes provide different camera interaction styles:
   * - Orbit: Quaternion-based rotation with no gimbal lock (drag to rotate around target)
   * - Fly: First-person WASD movement (for exploring inside datasets)
   * - Ortho: Orthographic pan + zoom (for 2D viewing)
   *
   * Updates input context when switching to fly mode to enable WASD keys.
   * Syncs rendering controls display if active.
   *
   * @private
   */
  private toggleControlMode(): void {
    toggleControlMode(this.makeControlModeCtx());
  }

  /**
   * Switch directly to a specific camera control mode (orbit / fly / ortho).
   * Triggered by the control rail's Navigation popover mode selector; reuses
   * the same context/sync wiring as the V-key cycle.
   *
   * @private
   */
  private setControlMode(type: ControlType): void {
    setControlMode(this.makeControlModeCtx(), type);
  }

  /**
   * Toggle inertial mode for fly controls (momentum-based movement).
   *
   * Triggered by I key. Only functional when in fly control mode.
   *
   * Inertial mode adds physics-based momentum:
   * - ON: Movement continues after releasing keys (space-like float)
   * - OFF: Movement stops immediately when keys released (FPS-like control)
   *
   * Syncs rendering controls display if active. Logs info message if called
   * while not in fly mode.
   *
   * @private
   */
  private toggleInertialMode(): void {
    toggleInertialMode(this.makeControlModeCtx());
  }

  private makeControlModeCtx(): ControlModeCtx {
    return {
      sceneManager: this.sceneManager,
      contextManager: this.contextManager,
      renderingControls: this.renderingControls,
    };
  }

  /**
   * Check if Space key should trigger fullscreen toggle.
   *
   * Returns true only if focus is on document body or canvas, preventing
   * fullscreen toggle when user is interacting with UI elements (buttons,
   * inputs, etc.) where Space might have other meanings (submit, type space).
   *
   * @returns true if Space key should toggle fullscreen, false otherwise
   * @private
   */
  private shouldHandleSpaceKey(): boolean {
    return isFocusOnSceneCanvas(document.activeElement, this.sceneManager.renderer.domElement);
  }

  /**
   * Check if user is currently typing in a text input field. Thin wrapper
   * around the pure {@link isTypingInInput} helper so callers in this file
   * keep their compact `this.isTypingInInput()` shape.
   * @private
   */
  private isTypingInInput(): boolean {
    return isTypingInInput(document.activeElement);
  }

  /**
   * Handles keyboard navigation through nD dimensions using [ and ] keys.
   *
   * This implements intelligent dimension navigation with adaptive step sizes:
   * - The animation menu's per-dimension Step override wins when set
   *   (quantized to the authored grid for discrete dims)
   * - Otherwise discrete dimensions step by their defined increment
   * - Otherwise continuous dimensions step by 1% of their total range
   * - Steps are clamped to dimension bounds
   * - Only updates if the value actually changes
   *
   * The navigation respects the currently selected dimension (set by number keys)
   * and provides smooth, predictable movement through nD space.
   *
   * @param direction - Direction to navigate: -1 for backward, 1 for forward
   * @private
   */
  private handleDimensionNavigation(direction: -1 | 1): void {
    const dims = sceneDimsManager.getDims();
    const ranges = sceneDimsManager.getDimensionRanges();
    const step = computeDimensionStep(
      direction,
      this.selectedDimension,
      dims,
      ranges,
      // The animation menu's per-dimension Step override also drives [ / ]
      // (user decision: one quantum for animation + keyboard; the slider
      // wheel/drag deliberately stay on the dimension's own base step).
      (d) => this.animationManager?.getStepSize(d) ?? null
    );
    if (!step) return;
    if (!step.changed) {
      const range = ranges?.[step.targetDim];
      const bound = range?.[direction > 0 ? 1 : 0];
      const current = dims?.currentStep[step.targetDim];
      const isCyclic = dims?.metadata?.[step.targetDim]?.cyclic || false;
      if (!isCyclic && bound !== undefined && current === bound) {
        const name =
          sceneDimsManager.getDimensionNames()[step.targetDim] || `Dim ${step.targetDim}`;
        const categories = dims?.metadata?.[step.targetDim]?.categories;
        const category = categories?.[Math.round(bound)];
        const boundLabel = category !== undefined ? category : bound;
        const edge = direction > 0 ? 'maximum' : 'minimum';
        notifier.toast(`${name} is already at its ${edge} (${boundLabel}).`);
      }
      return;
    }

    sceneDimsManager.setDimensionValue(step.targetDim, step.newValue);
    this.animationController.startAnimation();
  }

  /**
   * Selects which dimension to control with keyboard navigation.
   *
   * Number keys (1-9) map to navigable dimensions, allowing users to
   * switch between controlling different non-displayed dimensions with
   * the [ and ] navigation keys.
   *
   * @param index - Zero-based position in the non-displayed dimension list
   * @private
   */
  private selectDimension(index: number): void {
    const dims = sceneDimsManager.getDims();
    const result = resolveSelectedDimension(index, dims);
    if (result.selectedDimension !== null) {
      this.selectedDimension = result.selectedDimension;
      this.dimensionSliders?.setSelectedDimension(result.selectedDimension);
    } else if ('navigableCount' in result) {
      const message = describeNavigableKeys(index, dims, sceneDimsManager.getDimensionNames());
      log.info(
        Modules.INPUT,
        `Dimension ${index + 1} not available (only ${result.navigableCount} non-displayed dimensions)`
      );
      notifier.toast(message);
    }
  }

  /**
   * Handle Escape key with context-aware behavior.
   *
   * Behavior depends on fullscreen state:
   * - If IN fullscreen: Does nothing (browser handles fullscreen exit natively)
   * - If NOT in fullscreen: Closes all open panels (help, controls, monitor, etc.)
   *
   * This ensures Escape behaves predictably - fullscreen exit takes priority,
   * then panel closing.
   *
   * @private
   */
  /**
   * Escape-key dispatch. Delegates to PanelCoordinator which owns the
   * recording-priority and fullscreen-defer rules.
   */
  private handleEscapeKey(): void {
    this.panelCoordinator.handleEscape();
  }

  /**
   * Frame camera to fit the entire scene.
   *
   * Triggered by F key. Computes bounding box of all visible geometry
   * and repositions camera at the optimal distance to see everything.
   * Works for both perspective (distance) and orthographic (zoom) cameras.
   *
   * @private
   */
  private recenterCamera(): void {
    // Use the scene manager's centerCameraOnScene() which properly computes
    // optimal camera distance AND orbit target (not just the pivot point).
    // This ensures F key actually zooms to fit the whole scene, not just
    // re-centers the orbit pivot at the same distance.
    this.sceneManager.centerCameraOnScene();

    // Ensure a render happens after reframing
    this.animationController.startAnimation();
  }

  /**
   * Clean up all event listeners and dispose of managed resources.
   *
   * Removes all registered event listeners from window, document, and canvas
   * to prevent memory leaks. Disposes of dimension sliders and debug console.
   * Should be called when the input handler is no longer needed (e.g., when
   * destroying the application).
   *
   * After calling dispose(), the input handler cannot be reused - create a
   * new instance if needed.
   *
   * @example
   * ```typescript
   * // During application teardown
   * inputHandler.dispose();
   * sceneManager.dispose();
   * animationController.dispose();
   * ```
   */
  dispose(): void {
    // Dispose dimension sliders
    if (this.dimensionSliders) {
      this.dimensionSliders.dispose();
      this.dimensionSliders = undefined;
    }

    // Dispose animation manager
    if (this.animationManager) {
      this.animationManager.dispose();
      this.animationManager = undefined;
    }

    // Remove the sceneDimsManager listener. The dataset-switch path
    // also does this via clearDimensionUI, but app-dispose without
    // a subsequent switch would otherwise leak the listener on the
    // singleton, retaining this disposed InputHandler.
    if (this.sceneDimsListener) {
      sceneDimsManager.removeListener(this.sceneDimsListener);
      this.sceneDimsListener = undefined;
    }

    // Dispose debug console
    this.debugConsole.dispose();

    // Clean up event listeners
    this.eventListeners.forEach((cleanup) => cleanup());
    this.eventListeners = [];
  }
}
