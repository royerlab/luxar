# Console Output Style Guide

## Overview

This document defines the consistent console output style for the Luxar project. All console messages follow a structured format that makes logs easy to read, filter, and understand at a glance.

## Core Principles

1. **Structured Format**: All messages follow the pattern `[emoji] [Module] message`
2. **Semantic Emojis**: Each emoji conveys the type or category of the message
3. **Module Identification**: Every message identifies its source module
4. **Console Interceptor Compatible**: Works seamlessly with the debug console (Ctrl+L)
5. **Production vs Development**: Verbose debug logs only in development mode

## Message Format

```
[emoji] [Module] message content
```

### Examples
```
[🚀] [Luxar] Application starting...
[📥] [PointsSpatialIndexLoader] Loading positions for 1 ranges
[✅] [SceneLoader] Scene loaded successfully
[❌] [DataMonitor] Failed to load data: Network timeout
[🔄] [SceneManager] Updating view state
```

## Standard Emojis

### Status Indicators
- `🚀` **START** - Application or process initialization
- `✅` **SUCCESS** - Successful completion
- `❌` **ERROR** - Errors and failures
- `⚠️` **WARNING** - Warnings and potential issues
- `ℹ️` **INFO** - General information

### Action Indicators
- `📥` **LOAD** - Loading data or resources
- `💾` **SAVE/CACHE** - Saving data or cache operations
- `🔄` **UPDATE** - State updates or refreshes
- `🗑️` **DELETE** - Deletion or cleanup operations
- `🔍` **SEARCH/QUERY** - Search operations or spatial queries

### Domain-Specific
- `📊` **DATA** - Data processing or statistics
- `🎬` **SCENE** - Scene loading and management
- `🎨` **RENDER** - Rendering operations
- `🎮` **CONTROLS** - User input and controls
- `🔧` **DEBUG** - Debug console and tools

## Module Names

Use consistent module names for easy filtering:

### Core Modules
- `Luxar` - Main application
- `App` - Application lifecycle
- `Main` - Entry point

### Data Loading
- `SceneLoader` - Scene loading operations
- `PointsSpatialIndexLoader` - Point spatial index queries
- `DataMonitor` - Data loading monitoring
- `ZarrLoader` - Zarr file operations

### Rendering
- `Renderer` - Core rendering
- `PostProcessing` - Effects pipeline
- `HDR` - HDR rendering
- `SceneManager` - Scene management

### Controls & UI
- `Controls` - Control systems
- `Input` - Input handling
- `DebugConsole` - Debug console
- `UI` - User interface

## Implementation

### Using the Log Utility

Import the logging utility:
```typescript
import { log, Modules, LogEmoji } from '../utils/log';
```

### Basic Logging
```typescript
// Information
log.info(Modules.SCENE_LOADER, 'Loading scene from URL');

// Success
log.success(Modules.SCENE_LOADER, 'Scene loaded successfully');

// Error
log.error(Modules.DATA_MONITOR, 'Failed to load data:', error);

// Warning
log.warning(Modules.SPATIAL_INDEX_LOADER, 'Cache miss - loading from network');
```

### Action-Specific Logging
```typescript
// Loading operations
log.load(Modules.SPATIAL_INDEX_LOADER, 'Loading positions for 5 ranges');

// Updates
log.update(Modules.SCENE_MANAGER, 'Updating view state');

// Queries
log.query(Modules.SPATIAL_INDEX_LOADER, 'Querying spatial index');

// Data operations
log.data(Modules.DATA_MONITOR, 'Processing 100,000 points');
```

### Custom Emojis
```typescript
// Use specific emoji for special cases
log.custom('🌟', Modules.HDR, 'HDR display capabilities detected');
```

### Module-Specific Logger
```typescript
// Create a logger for repeated use in a module
const logger = createModuleLogger(Modules.SCENE_LOADER);

logger.info('Starting scene load');
logger.load('Loading points data');
logger.success('Scene ready');
```

## Best Practices

### 1. Appropriate Verbosity
- **Production**: Only log important events (start, success, errors)
- **Development**: Include detailed debugging information
- **Use Environment Checks**: Wrap verbose logs in `import.meta.env.DEV` (Vite)

### 2. Message Clarity
- Be concise but informative
- Include relevant data (counts, IDs, ranges)
- Use consistent terminology

### 3. Error Handling
```typescript
try {
  // operation
} catch (error) {
  log.error(Modules.MODULE_NAME, 'Operation failed:', error);
}
```

### 4. Progress Indication
For multi-step operations:
```typescript
log.info(Modules.SCENE_LOADER, 'Loading scene...');
log.load(Modules.SCENE_LOADER, 'Loading geometry');
log.load(Modules.SCENE_LOADER, 'Loading textures');
log.success(Modules.SCENE_LOADER, 'Scene loaded successfully');
```

### 5. Data Statistics
Include useful metrics:
```typescript
log.data(
  Modules.SPATIAL_INDEX_LOADER,
  `Query result: ${cells} cells → ${ranges} ranges → ${points.toLocaleString()} points`
);
```

## Console Interceptor Integration

The logging system works seamlessly with the console interceptor:

1. **All console output is captured** by `console-interceptor.ts`
2. **Debug console (Ctrl+L)** displays all captured messages
3. **Ring buffer** stores last 10,000 messages
4. **Message filtering** available in debug console

## Examples of Good vs Bad Logging

### ❌ Bad
```typescript
console.log('loaded');
console.log('Error!!!');
console.log('data:', data);
```

### ✅ Good
```typescript
log.success(Modules.SCENE_LOADER, 'Scene loaded successfully');
log.error(Modules.DATA_MONITOR, 'Failed to load spatial index:', error);
log.data(Modules.SPATIAL_INDEX_LOADER, `Loaded ${points.toLocaleString()} points`);
```

### ❌ Too Verbose
```typescript
log.info(Modules.SCENE_LOADER, 'Starting to load');
log.info(Modules.SCENE_LOADER, 'Checking cache');
log.info(Modules.SCENE_LOADER, 'Cache checked');
log.info(Modules.SCENE_LOADER, 'Opening file');
log.info(Modules.SCENE_LOADER, 'File opened');
```

### ✅ Appropriate Detail
```typescript
log.load(Modules.SCENE_LOADER, 'Loading data from cache');
// ... actual loading work ...
log.success(Modules.SCENE_LOADER, `Loaded ${size} bytes in ${time}ms`);
```

## Debugging Features

### Development-Only Logs
```typescript
if (import.meta.env.DEV) {
  log.info(Modules.SPATIAL_INDEX_LOADER, `  Query bounds: [${bounds}]`);
  log.info(Modules.SPATIAL_INDEX_LOADER, `  Cell sizes: [${sizes}]`);
}
```

### Performance Monitoring
```typescript
const start = performance.now();
// ... operation ...
const elapsed = performance.now() - start;
log.custom('⚡', Modules.PERFORMANCE, `Operation completed in ${elapsed.toFixed(1)}ms`);
```

## Updating Existing Code

When updating existing code:

1. **Add import**: `import { log, Modules, LogEmoji } from '../utils/log';`
2. **Replace console.log**: Use appropriate `log.*` function
3. **Add module name**: Use constants from `Modules` or create new ones
4. **Choose emoji**: Use `LogEmoji` constants or custom emojis
5. **Consider verbosity**: Wrap debug logs in environment checks

## Conclusion

Consistent logging improves:
- **Debugging** - Easy to trace issues
- **Monitoring** - Clear understanding of application state
- **User Experience** - Professional, polished output
- **Maintenance** - Quickly understand code behavior

Follow these guidelines to maintain high-quality, consistent console output throughout the Luxar application.
