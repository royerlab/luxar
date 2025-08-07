# Luxar TODO List

This file tracks known issues, bugs, and improvements needed in the Luxar project.
Last updated from comprehensive codebase scan.

## Critical Issues

### Anti-Aliasing Brightness Problems
- **MSAA Issue**: Scenes get brighter with more samples due to additive blending incompatibility
- **SSAA Issue**: Scenes get dimmer with higher multipliers due to downsampling averaging  
- **Location**: `packages/luxar-player/src/rendering/post-processing.ts`
- **Impact**: Visual inconsistency when changing anti-aliasing settings
- **Note**: Do NOT compensate in individual shaders - needs systematic solution

## Code Quality Issues

### Python Codebase

#### ✅ FIXED: API Improvements
- **Status**: COMPLETED - Python API now supports single values for radii, sharpness, and colors
- **Implementation**: Points class now accepts floats for radii/sharpness and tuples/lists for colors

#### ✅ FIXED: Missing Physical Units
- **Status**: COMPLETED - All units (nm, um, px, au, m) added to config.py
- **Implementation**: Created test suite to verify all units work correctly

#### ✅ FIXED: CLI ZipStore Support
- **Status**: COMPLETED - Removed ZipStore references from CLI
- **Implementation**: Cleaned up CLI to remove unsupported feature

### TypeScript Codebase

#### Type Safety Issues
- **Issue**: Some THREE.js extensions use `any` casting due to incomplete type definitions
- **Location**: Various files in `packages/luxar-player/src/`
- **Fix**: Add proper type definitions for THREE.js extensions

#### ✅ FIXED: Error Handling Gaps
- **Status**: COMPLETED - Improved error handling in zarr-loader.ts
- **Implementation**: Added try-catch blocks, better error messages, and graceful fallbacks

#### Race Conditions
- **Issue**: Fullscreen timing uses hardcoded 100ms delay
- **Location**: `packages/luxar-player/src/input/input-handler.ts`
- **Fix**: Use proper event-based synchronization

### Examples Directory

#### ✅ FIXED: Incomplete Examples
1. **`rendering_modes_example.py`** - COMPLETED
   - Now a proper educational example with blending mode comparisons
   - Shows normal, additive, and multiply modes with clear explanations

2. **`rendering_attributes_example.py`** - COMPLETED  
   - Full educational example demonstrating API usage
   - Shows property setting, modification, and method chaining

3. **`performance_benchmark_example.py`** - COMPLETED
   - Added comprehensive documentation and educational output
   - Includes performance analysis and optimization insights

#### ✅ FIXED: Inconsistencies
- **Import styles**: COMPLETED - All examples now use `from luxar import ...`
- **Print functions**: COMPLETED - All examples now use `from arbol import aprint`
- **Path handling**: COMPLETED - All examples use consistent path handling

#### ✅ FIXED: Path Issues
- **Status**: COMPLETED - No more `zarr_scenes/` directory usage
- **Implementation**: All examples write to current directory

## Missing Features

### ✅ FIXED: Essential Missing Examples
1. **Transform System Example** - COMPLETED (`transform_example.py`)
   - Comprehensive demonstration of all transform operations
   - Shows composition and hierarchical inheritance
   
2. **Scene Hierarchy Example** - COMPLETED (`hierarchy_example.py`)
   - 4-level deep hierarchy with property inheritance
   - Space-themed educational visualization
   
3. **Multiple Objects Example** - COMPLETED (`multiple_objects_example.py`)
   - Six distinct point cloud objects in one scene
   - ~31,000 points with varied rendering properties

### Still Missing Examples
1. **Error Handling Example** - No validation/error case examples
2. **Data Import Example** - No external data loading examples
3. **CLI Usage Example** - No demonstration of CLI tools

### Documentation Gaps
1. **Performance Guide** - Need comprehensive performance optimization guide
2. **Chunking Strategy Guide** - Best practices for Zarr chunking
3. **Memory Management Guide** - Handling large datasets efficiently
4. **Integration Guide** - How to embed Luxar in other applications

## Performance Improvements

### Python Performance
- Consider lazy loading for large scenes
- Optimize chunk sizes based on access patterns
- Add progress bars for long operations

### TypeScript Performance  
- Implement level-of-detail (LOD) for large point clouds
- Add frustum culling for off-screen points
- Consider WebGPU support for better performance

## Testing Coverage

### Python Tests Needed
- Integration tests for CLI commands
- Performance regression tests
- Cross-platform path handling tests

### TypeScript Tests Needed
- Full coverage for rendering pipeline
- nD slicing algorithm tests
- UI component interaction tests

## Recently Completed Features

### ✅ Dataset Navigation System (COMPLETED)
- **Server-side**: Modified CLI to serve entire directories with HTML listing
- **Client-side**: Created multi-strategy directory navigator service
- **UI**: Built interactive dataset browser panel with breadcrumb navigation
- **Detection strategies**: WebDAV, HTML parsing, index files, manual fallback
- **Keyboard shortcut**: Press 'O' to open dataset browser
- **Server-agnostic**: Works with any static file server (nginx, Apache, S3, etc.)

## Future Enhancements

### Data Format Extensions
- Support for meshes and lines
- Volume rendering support
- Material system with PBR shading
- Temporal interpolation for animations
- Hierarchical LOD support

### Viewer Enhancements
- Color maps and transfer functions
- Measurement tools
- Animation timeline controls
- Export capabilities (screenshots, videos)
- Collaborative viewing sessions

### Developer Experience
- Better error messages with suggested fixes
- Interactive documentation with live examples
- VS Code extension for .zarr file preview
- GitHub Actions for automated testing

## Maintenance Tasks

### Documentation Updates
- Sync all README files with current features
- Update API documentation
- Create video tutorials
- Add troubleshooting guide

### Code Cleanup
- Standardize import styles across examples
- Remove unused imports and variables
- Consistent error handling patterns
- Update deprecated dependencies

## Priority Levels

### High Priority
1. Fix anti-aliasing brightness issues
2. Complete incomplete examples
3. Add missing physical units
4. Improve error handling
5. API improvement for constant values (radii, sharpness, colors)

### Medium Priority
1. Add missing example categories
2. Improve TypeScript type safety
3. Create performance optimization guide
4. Add progress indicators

### Low Priority
1. Future format extensions
2. Advanced viewer features
3. Developer tooling improvements

## Notes

- This TODO list should be reviewed and updated regularly
- Issues marked as "Critical" should be addressed first
- Consider creating GitHub issues for tracking progress
- Update CLAUDE.md when implementing significant changes