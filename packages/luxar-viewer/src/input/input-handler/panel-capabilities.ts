import type { RenderingSettings } from '../../config/sections/rendering-controls/types';
import type { DimensionAnimationManager } from '../../scene/animation/dimension-animation-manager';
import type { SimpleDims } from '../../types/dims';

export interface ToggleableHandle {
  toggle(): void;
}

export interface RenderingControlsHandle extends ToggleableHandle {
  readonly settings: RenderingSettings;
  hide(): void;
  isVisible(): boolean;
  saveSettings(): void;
  syncCurrentState(): void;
  toggleCinematicMode(): void;
}

export interface RecordingPanelHandle extends ToggleableHandle {
  captureScreenshot(): Promise<void>;
  hide(): void;
  isCurrentlyRecording(): boolean;
  isVisible(): boolean;
  setAnimationManager(manager: DimensionAnimationManager): void;
  stopVideoRecording(): void;
}

export interface LayersPanelHandle extends ToggleableHandle {
  hide(): void;
  isVisible(): boolean;
}

export interface DimensionSlidersConfig {
  container: HTMLElement;
  dims: SimpleDims;
  dimensionRanges: Array<[number, number]>;
  dimensionNames: string[];
  dimensionUnits?: string[];
  selectedDimension?: number;
}

export interface DimensionSlidersHandle extends ToggleableHandle {
  dispose(): void;
  getIsVisible(): boolean;
  hide(): void;
  setAnimationManager(manager: DimensionAnimationManager): void;
  setSelectedDimension(index: number): void;
  setVisible(visible: boolean): void;
  update(): void;
}

export type DimensionSlidersFactory = (config: DimensionSlidersConfig) => DimensionSlidersHandle;

export interface DebugConsoleHandle extends ToggleableHandle {
  dispose(): void;
  getIsVisible(): boolean;
  hide(): void;
}

export interface PerformanceMonitorHandle extends ToggleableHandle {
  readonly visible: boolean;
  hide(): void;
}
