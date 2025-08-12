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
    rightOffset: 20,
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
    backgroundColor: 'rgba(20, 20, 20, 0.95)',
    borderColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 8,
    backdropBlur: 10,
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
  },
};
