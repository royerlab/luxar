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

#### API Improvements (from existing TODO)
- **Issue**: Should be able to pass radii, sharpness, color, etc. as constants (single float, or single color) instead of having to pass arrays every single time
- **Impact**: Better developer experience and cleaner code

#### Missing Physical Units
- **Issue**: `config.py` SUPPORTED_UNITS missing nm, um, px, au that are mentioned in CLAUDE.md and used in types.py
- **Location**: `packages/luxar/src/luxar/config.py:46-58`
- **Fix**: Add missing units to SUPPORTED_UNITS list

#### CLI ZipStore Support
- **Issue**: ZipStore support mentioned in CLI but not implemented
- **Location**: `packages/luxar/src/luxar/cli.py`
- **Fix**: Either implement ZipStore support or remove mention

### TypeScript Codebase

#### Type Safety Issues
- **Issue**: Some THREE.js extensions use `any` casting due to incomplete type definitions
- **Location**: Various files in `packages/luxar-player/src/`
- **Fix**: Add proper type definitions for THREE.js extensions

#### Error Handling Gaps
- **Issue**: Limited error boundaries in async operations
- **Location**: `packages/luxar-player/src/data/zarr-loader.ts`
- **Fix**: Add comprehensive error handling for data loading failures

#### Race Conditions
- **Issue**: Fullscreen timing uses hardcoded 100ms delay
- **Location**: `packages/luxar-player/src/input/input-handler.ts`
- **Fix**: Use proper event-based synchronization

### Examples Directory

#### Incomplete Examples
1. **`rendering_modes_example.py`**
   - Currently just a test script, not educational
   - Missing side-by-side comparisons
   - Needs complete rewrite as proper example

2. **`rendering_attributes_example.py`**
   - Simple test script lacking educational value
   - Missing scene finalization
   - Needs demonstration of inheritance patterns

3. **`performance_benchmark_example.py`**
   - Missing documentation on interpreting results
   - No guidance on performance optimization

#### Inconsistencies
- **Import styles**: Mix of `import luxar` vs `from luxar import Scene`
- **Print functions**: Some use `print()` instead of `arbol.aprint()`
- **Path handling**: Mix of string paths vs Path objects

#### Path Issues
- **`rainbow_sphere_spiral_example.py`**: Writes to `zarr_scenes/` which may not exist
- **Fix**: Use relative paths or ensure directories exist

## Missing Features

### Essential Missing Examples
1. **Transform System Example** - No demonstration of translate, rotate, scale, compose
2. **Scene Hierarchy Example** - No parent-child relationships demonstration
3. **Multiple Objects Example** - Limited multi-object scenes
4. **Error Handling Example** - No validation/error case examples
5. **Data Import Example** - No external data loading examples
6. **CLI Usage Example** - No demonstration of CLI tools

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