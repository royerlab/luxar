/**
 * Public facade for viewer-wide input handling.
 *
 * @module input
 */
export { InputHandler, KeyAction } from './input-handler';
export type { ControlRailHandle, DimensionSlidersFactory, KeyActionId } from './input-handler';
export { InputContext } from './input-handler/context-manager';
export type { ContextConfig, InputContextId, KeyBinding } from './input-handler/context-manager';
export type {
  RegisteredShortcutBinding,
  RegisteredShortcutBindings,
  ShortcutHelpMetadata,
  ShortcutHelpSectionId,
} from '../types/shortcut-help';
