/**
 * Luxar Theme System - Main Entry Point
 *
 * Exports all theme-related functionality for easy imports.
 */

// Export type definitions
export type { Theme, ThemeChangeHandler } from './types';

// Export ThemeManager
export { ThemeManager } from './theme-manager';

// Export built-in themes
export { darkTheme } from './themes/dark.theme';
