/**
 * Configuration constants for the Debug Console
 */

export const DEBUG_CONSOLE_CONFIG = {
  // Panel dimensions
  panel: {
    defaultWidth: 600,
    defaultHeight: 400,
    minWidth: 400,
    maxWidth: 1200,
    minHeight: 200,
    maxHeight: 800,
    bottomOffset: 20,
    leftOffset: 20, // Changed from rightOffset to avoid conflict with monitoring panel
  },

  // Console interceptor settings
  interceptor: {
    // Ring buffer size for console messages
    maxBufferSize: 10000,
  },

  // Resize handle dimensions
  resize: {
    borderWidth: 4,
  },

  // Styling
  style: {
    backgroundColor: 'rgba(30, 30, 30, 0.95)', // Standardized background
    borderColor: 'rgba(255, 255, 255, 0.1)', // Lighter border for removal
    borderRadius: 8,
    backdropBlur: 10,
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)', // Standardized shadow
  },
};
