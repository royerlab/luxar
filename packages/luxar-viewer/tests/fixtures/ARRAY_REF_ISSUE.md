# Array Reference Deduplication Issue

## Status: INVESTIGATED BUT NOT WORKING

### What Works
- ✅ ArrayRefRegistry.check() correctly detects duplicates (9 tests pass)
- ✅ Direct encoder.encode() calls create array refs correctly
- ✅ Priority 2 logic is implemented in encoder.py

### What Doesn't Work
- ❌ LuxarZarrCompiler doesn't create array refs when same array passed twice
- ❌ Test fixtures don't generate array refs despite using shared colors

### Root Cause Investigation

**Hypothesis 1: Morton Ordering Creates New Arrays**
- Line compiler.py:312: `colors = colors[sort_order]` creates new array
- TESTED: Disabled spatial index (enable_spatial_index=False)
- RESULT: Still doesn't work

**Hypothesis 2: Quantization Happens Before Array Ref Check**  
- TESTED: Priority 2 (array ref) is before Priority 3+ (quantization) in code
- RESULT: Code order is correct

**Hypothesis 3: Different Parameters**
- Compiler passes: mode, n_elements, chunks, compressor
- Direct call passes: minimal parameters
- INVESTIGATION NEEDED: Does mode or chunks affect Priority 2?

### Working Example (Direct Encoder)

```python
encoder = ArrayEncoder()
encoder.encode(shared_colors, group, "colors1", SemanticType.COLOR, color_mode="sdr")  
encoder.encode(shared_colors, group, "colors2", SemanticType.COLOR, color_mode="sdr")
# Result: colors2 gets array_ref encoding ✅
```

### Not Working (Through Compiler)

```python
compiler.add_points("points1", positions, colors=shared_colors)
compiler.add_points("points2", positions, colors=shared_colors)  
# Result: Both get rgb_uint8 encoding ❌
```

### Next Steps for Future Investigation

1. Add debug logging to encoder Priority 2 check
2. Trace exact parameters passed by compiler vs direct call
3. Check if mode=MEMORY skips array ref check somehow  
4. Verify array identity is preserved through validation/preprocessing

### Workaround for Tests

Array ref tests are currently skipped. Feature works in principle (registry tests pass, direct encoding works) but not through the full compiler stack.
