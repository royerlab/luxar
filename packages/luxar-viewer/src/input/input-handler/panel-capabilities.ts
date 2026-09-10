/**
 * Narrow UI capabilities consumed by the input package.
 *
 * These contracts keep input independent of concrete panel classes while
 * documenting the exact surface used by shortcuts and Escape handling.
 *
 * @module input/input-handler/panel-capabilities
 */

import type { RenderingSettings } from '../../config/sections/rendering-controls/types';
import type { DimensionAnimationManager } from '../../scene/animation/dimension-animation-manager';
import type { SimpleDims } from '../../types/dims';

/** Shared toggle surface implemented by several viewer panels. */
export interface ToggleableHandle {
  toggle(): void;
}

/** Rendering Controls surface used by shortcuts, state export, and FOV updates. */
export interface RenderingControlsHandle extends ToggleableHandle {
  /** The reference is readonly; input may still update fields on the settings object. */
  readonly settings: RenderingSettings;
  hide(): void;
  isVisible(): boolean;
  saveSettings(): void;
  syncCurrentState(): void;
  toggleCinematicMode(): void;
}

/** Recording panel surface used by capture shortcuts and Escape handling. */
export interface RecordingPanelHandle extends ToggleableHandle {
  captureScreenshot(): Promise<void>;
  hide(): void;
  isCurrentlyRecording(): boolean;
  isVisible(): boolean;
  setAnimationManager(manager: DimensionAnimationManager): void;
  stopVideoRecording(): void;
}

/** Layers panel surface used by its shortcut and Escape handling. */
export interface LayersPanelHandle extends ToggleableHandle {
  hide(): void;
  isVisible(): boolean;
}

/** Constructor inputs shared with the concrete `ui/dimension-sliders` panel. */
export interface DimensionSlidersConfig {
  container: HTMLElement;
  dims: SimpleDims;
  dimensionRanges: Array<[number, number]>;
  dimensionNames: string[];
  dimensionUnits?: string[];
  selectedDimension?: number;
  /**
   * Fired when the panel itself selects the dimension the global [ / ] keys
   * target (a tap on a dimension's name chip under a coarse pointer), so the
   * input layer's own selection stays in step. Position in the non-displayed
   * dimension list, as `selectedDimension`.
   */
  onSelectDimension?: (navigableIndex: number) => void;
}

/** Dimension sliders surface used by navigation lifecycle and shortcuts. */
export interface DimensionSlidersHandle extends ToggleableHandle {
  closeContextMenu(): void;
  dispose(): void;
  getIsVisible(): boolean;
  hide(): void;
  setAnimationManager(manager: DimensionAnimationManager): void;
  setSelectedDimension(index: number): void;
  setVisible(visible: boolean): void;
  update(): void;
}

/**
 * Factory injected from `core/app.ts` so input can construct dimension sliders
 * without importing the concrete UI class.
 */
export type DimensionSlidersFactory = (config: DimensionSlidersConfig) => DimensionSlidersHandle;

/** Debug console surface used by its shortcut and Escape handling. */
export interface DebugConsoleHandle extends ToggleableHandle {
  dispose(): void;
  getIsVisible(): boolean;
  hide(): void;
}

/** Performance monitor surface used by its shortcut and Escape handling. */
export interface PerformanceMonitorHandle extends ToggleableHandle {
  readonly visible: boolean;
  hide(): void;
}

/** Performance monitor subset used only by the Escape coordinator. */
export interface PerformanceStatsHandle {
  readonly visible: boolean;
  hide(): void;
}

/** Dataset browser surface needed by Escape handling. */
export interface CloseableHandle {
  close(): void;
}

/** Visible panel subset used only by the Escape coordinator. */
export interface VisiblyHideableHandle {
  isVisible(): boolean;
  hide(): void;
}

/** Minimal handle for transient overlays owned by a larger UI component. */
export interface OverlayCloseHandle {
  closeOverlay(): void;
}

/** Control-rail hooks used by routed keyboard handling. */
export interface ControlRailHandle extends OverlayCloseHandle {
  handleRoutedKeyDown(): void;
}
