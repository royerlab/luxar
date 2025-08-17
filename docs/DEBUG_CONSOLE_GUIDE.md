# Debug Console Panel Guide

## Overview

The Luxar viewer now includes an in-app debug console that captures and displays all browser console output. This is especially useful for:
- Debugging without opening DevTools
- Monitoring console output in fullscreen mode
- Capturing errors and warnings
- Sharing debug logs with others

## How to Use

### Opening the Debug Console

Press **Ctrl+L** (or **Cmd+L** on Mac) to toggle the debug console panel.

The console appears as a draggable, resizable panel in the bottom-right corner of the screen.

## Features

### 1. **Real-time Console Capture**
- Automatically captures all `console.log`, `console.warn`, `console.error`, `console.info`, and `console.debug` calls
- Messages appear instantly in the panel
- Original console still works normally

### 2. **Message Types with Color Coding**
- 📝 **Log** (white) - Standard console.log messages
- ℹ️ **Info** (green) - Informational messages
- ⚠️ **Warning** (yellow) - Warning messages with yellow background
- ❌ **Error** (red) - Error messages with red background and stack traces
- 🔍 **Debug** (gray italic) - Debug messages

### 3. **Timestamps**
Each message shows a precise timestamp (HH:MM:SS.mmm format)

### 4. **Interactive Controls**

#### Filter
- Type in the filter box to search messages
- Filters in real-time as you type
- Case-insensitive search

#### Clear
- Clears all messages from the console
- Useful for starting fresh debug sessions

#### Copy
- Copies all messages to clipboard
- Formatted with timestamps and message types
- Great for sharing debug logs

#### Auto-scroll
- When checked, console automatically scrolls to newest messages
- Uncheck to manually review older messages

### 5. **Draggable & Resizable**
- **Drag**: Click and hold the header to move the panel
- **Resize**: Hover over top or left edge and drag to resize
- Panel remembers position during session

### 6. **Smart Formatting**
- **Objects**: Displayed as formatted JSON
- **Arrays**: Pretty-printed with proper indentation
- **Strings**: Shown in green with quotes
- **Numbers**: Displayed in orange
- **Booleans**: Shown in pink
- **Undefined/null**: Gray italic text

### 7. **Stack Traces**
Error messages automatically include stack traces when available

### 8. **Message Buffer**
- Stores up to 1000 messages
- Older messages automatically removed to prevent memory issues

## Implementation Details

### How It Works

1. **Console Interception**
   ```javascript
   // Saves original console methods
   const originalLog = console.log;
   
   // Overrides with wrapper
   console.log = (...args) => {
     captureMessage('log', args);
     originalLog(...args); // Still calls original
   };
   ```

2. **Message Capture**
   - All console output is captured before display
   - Messages stored with metadata (timestamp, type, stack)
   - Original console behavior preserved

3. **UI Updates**
   - Messages rendered to DOM when panel is visible
   - Hidden panel still captures messages in background
   - Opening panel shows all buffered messages

4. **Performance**
   - Minimal overhead when panel is hidden
   - Efficient DOM updates using document fragments
   - Limited message buffer prevents memory leaks

## Use Cases

### 1. **HDR Debugging**
When testing HDR support, the console shows:
```
🎨 HDR Display Capabilities
✅ P3 Wide Gamut
✅ High Dynamic Range
✅ Float Textures
```

### 2. **Performance Monitoring**
Track render performance without DevTools:
```
✓ Frame time: 16.67ms (60 FPS)
✓ Points rendered: 1,234,567
```

### 3. **Error Tracking**
Capture errors with full stack traces:
```
[ERROR] Failed to load texture
  at loadTexture (loader.js:123)
  at Scene.init (scene.js:45)
```

### 4. **Scene Loading**
Monitor scene loading progress:
```
Loading scene from test.zarr
✓ Metadata loaded
✓ Positions loaded: 10000 points
✓ Colors loaded
✓ Scene ready
```

## Keyboard Shortcuts Summary

- **Ctrl+L** (Cmd+L on Mac): Toggle debug console
- **Shift+T**: Toggle HDR test pattern (useful with console)
- **R**: Rendering controls
- **H**: Help overlay
- **C**: Toggle centering mode

## Tips

1. **Keep console open during development** to catch errors immediately
2. **Use filter** to focus on specific messages
3. **Copy logs** before reporting issues
4. **Clear console** before starting new tests
5. **Disable auto-scroll** when reviewing error stack traces

## Technical Notes

### CSS Classes for Customization
```css
.debug-console-panel        /* Main panel */
.debug-console-header       /* Header with controls */
.debug-console-content      /* Message area */
.console-message-log        /* Log messages */
.console-message-warn       /* Warnings */
.console-message-error      /* Errors */
```

### Programmatic Access
```javascript
// From browser console
window.app.inputHandler.debugConsole.toggle();
window.app.inputHandler.debugConsole.clear();
```

### Message Format
```
[timestamp] message content
[10:23:45.123] Scene loaded successfully
```

## Benefits Over Browser DevTools

1. **No DevTools Required**: Works without opening F12
2. **Fullscreen Compatible**: Stays visible in fullscreen mode
3. **Integrated UI**: Matches Luxar's visual style
4. **Focused Output**: Shows only app-related messages
5. **Easy Sharing**: Copy button for quick log sharing
6. **Persistent Position**: Stays where you put it

## Future Enhancements

Potential improvements:
- Export logs to file
- Log level filtering
- Message grouping/collapsing
- Network request logging
- Performance metrics graphs
- Custom message categories

## Troubleshooting

**Console not appearing?**
- Make sure to press Shift+C (not just C)
- Check if it's behind other panels
- Try refreshing the page

**Messages not showing?**
- Check filter isn't hiding messages
- Ensure auto-scroll is enabled for new messages
- Clear console if buffer is full

**Performance issues?**
- Clear old messages regularly
- Reduce message buffer size if needed
- Disable in production builds