/**
 * Core type definitions for the custom GUI library
 */

/**
 * Base configuration for GUI initialization
 */
export interface GUIOptions {
  /** Title displayed at the top of the GUI panel */
  title?: string;
  /** Width of the GUI panel in pixels */
  width?: number;
  /** Whether folders should start closed */
  closeFolders?: boolean;
  /** Parent container element (defaults to document.body) */
  container?: HTMLElement;
  /** Callback when close button is clicked (if provided, shows close button) */
  onClose?: () => void;
}

/**
 * Base controller configuration
 */
export interface ControllerOptions {
  /** Minimum value (number controllers only) */
  min?: number;
  /** Maximum value (number controllers only) */
  max?: number;
  /** Step size (number controllers only) */
  step?: number;
  /** Options array or object (option controllers only) */
  options?: string[] | Record<string, any>;
}

/**
 * Change callback signature
 */
export type ChangeCallback<T = any> = (value: T) => void;

/**
 * Finish change callback signature (fired on blur/mouseup)
 */
export type FinishChangeCallback<T = any> = (value: T) => void;

/**
 * Controller types for internal discrimination
 */
export enum ControllerType {
  NUMBER = 'number',
  BOOLEAN = 'boolean',
  STRING = 'string',
  OPTION = 'option',
  FUNCTION = 'function',
}

/**
 * Internal event listener tracking for memory leak prevention
 */
export interface EventListenerRecord {
  element: HTMLElement | Window | Document;
  event: string;
  handler: EventListener;
  options?: AddEventListenerOptions;
}
