# Quick Start Template

Use this template when adding Quick Start sections to package READMEs.

## Template Structure

```markdown
## Quick Start

[Brief description of what this example achieves - one sentence]

```[language]
from [package] import [classes/functions]
import [dependencies]

# 1. [First step - setup/initialization]
[code for step 1]

# 2. [Second step - main operation]
[code for step 2]

# 3. [Third step - verification/results]
[code for step 3]
```

**[What Just Happened / Key Benefits]**:
- [Bullet point explaining key feature 1]
- [Bullet point explaining key feature 2]
- [Bullet point explaining key feature 3]
```

## Guidelines

1. **Keep it minimal**: 3-5 steps maximum
2. **Make it runnable**: Example should work copy-paste
3. **Show value**: Demonstrate the package's core benefit
4. **Add context**: Explain what happened in plain language
5. **Use comments**: Number steps and explain each action

## Examples

### Example 1: Data Processing Package

```markdown
## Quick Start

Process and validate data in 3 steps:

```python
from luxar.validation import validate_positions_for_writing
import numpy as np

# 1. Create sample data
positions = np.random.randn(1000, 3).astype(np.float32)

# 2. Validate before processing
validated = validate_positions_for_writing(
    positions,
    expected_shape=(1000, 3),
    name="my_data"
)

# 3. Confirmed valid
print(f"✓ Validated {validated.shape[0]} positions")
```

**What Just Happened**:
- Shape validation - ensures correct dimensions
- Type checking - converts to float32 if needed
- Range validation - checks for NaN/Inf values
```

### Example 2: Configuration Package

```markdown
## Quick Start

Configure the viewer in 2 steps:

```typescript
import { createDefaultConfig } from './config/types';
import { App } from './core/app';

// 1. Create configuration
const config = createDefaultConfig({
  camera: {
    fov: 60,
    near: 0.1,
    far: 1000
  }
});

// 2. Initialize app
const app = new App(config);
await app.init();
```

**Key Benefits**:
- Type-safe configuration with TypeScript
- Sensible defaults for all settings
- Validation at initialization time
```

## Anti-Patterns (Avoid These)

### ❌ Too Complex
```markdown
## Quick Start

```python
# Don't show advanced features in quick start
from luxar.io import LuxarZarrCompiler
from luxar.encoding import EncodingMode, SemanticType
from luxar.validation import validate_all
from luxar.utils import create_demo_data
import numpy as np
import zarr

# Too many options confuse beginners
with LuxarZarrCompiler(
    'scene.luxar.zarr',
    encoding_mode=EncodingMode.AUTO,
    ordering_method="hilbert",
    enable_spatial_index=True,
    chunk_size_target=65536,
    compression_level=3
) as compiler:
    # ... complex setup
```
```

### ❌ No Explanation
```markdown
## Quick Start

```python
from luxar.io import LuxarZarrCompiler

with LuxarZarrCompiler('scene.luxar.zarr') as c:
    s = c.create_scene()
    s.add_points('p', pos, col, r)
```

**Missing**:
- `create_scene()` requires `dimensions` parameter
- What are `pos`, `col`, `r`? (show creation)
- What does this achieve?
- What happens next?
```

### ❌ Not Runnable
```markdown
## Quick Start

```python
from luxar.io import LuxarZarrCompiler

# Assumes you have 'positions' and 'colors' already
# User doesn't know where these come from!
compiler.write(positions, colors)
```
```

## Checklist

Before adding a Quick Start section, ensure:

- [ ] Code is complete and runnable copy-paste
- [ ] Has 3-5 clear steps with numbered comments
- [ ] Shows the most common/important use case
- [ ] Imports are explicit (no `import *`)
- [ ] Variables are created, not assumed
- [ ] Includes output/verification step
- [ ] Has "Key Benefits" or "What Just Happened" explanation
- [ ] Uses realistic example data (not foo/bar)
- [ ] Demonstrates package's core value proposition
- [ ] Is under 30 lines of code

## Testing Your Quick Start

1. Copy the example to a new file
2. Run it without modifications
3. Does it work? If not, fix it
4. Is it clear what happened? If not, add explanation
5. Could a beginner follow it? If not, simplify

## Location

Quick Start sections should appear:
- **Immediately after** package description
- **Before** detailed documentation sections
- **After** title and brief overview
- **At lines 5-40** typically

Good structure:
```
# Package Name
Brief one-line description

## Quick Start          ← Add here
[3-5 step example]

## Purpose              ← Then detailed docs
[Detailed explanation]
```
