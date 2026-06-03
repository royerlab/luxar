# Luxar Example Template

This document defines the standard structure and style for all Luxar example files.

## Purpose

Examples should be **educational**, **self-contained**, and **production-ready**. They demonstrate specific features or techniques to help users learn Luxar effectively.

---

## File Structure

### 1. Filename Convention
```
feature_description_example.py
```

**Rules:**
- Use snake_case
- Must end with `_example.py`
- Be descriptive but concise
- Examples: `single_point_example.py`, `hierarchy_example.py`, `dense_cubic_gradient_example.py`

### 2. Required Components (in order)

#### A. Shebang
```python
#!/usr/bin/env python3
```

#### B. Module Docstring
```python
"""Example Title - One-line description.

This example demonstrates:
- Feature 1 being demonstrated
- Feature 2 being demonstrated
- Feature 3 being demonstrated
- Why this example is useful/educational

Educational value:
- What users will learn from this example
- Key concepts illustrated
- Technical skills demonstrated
- When to use these techniques in practice
"""
```

**Docstring Requirements:**
- Start with title and one-line summary
- Blank line after title
- "This example demonstrates:" section with bullet points
- Optional "Educational value:" section for complex examples
- Be specific about what's shown
- Explain WHY, not just WHAT

#### C. Imports (Standard Order)
```python
import numpy as np
from _overlay_style import add_explainer
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler, transforms
from luxar.utils.paths import get_examples_output_dir
```

**Import Rules:**
- Standard library first (`pathlib`, etc.) — usually unneeded
- Third-party packages (`numpy`, `arbol`)
- The shared `add_explainer` helper from `_overlay_style` (see §E.1)
- Luxar imports last; pull `get_examples_output_dir` from `luxar.utils.paths`
- Alphabetical within each group
- Only import what you use

#### D. Helper Functions (if needed)
```python
def create_some_data(param: int) -> np.ndarray:
    """Create data for demonstration.

    Args:
        param: Description of parameter

    Returns:
        Description of return value
    """
    # Implementation with inline comments explaining key steps
    return result
```

**Helper Function Requirements:**
- Full docstrings with Args and Returns
- Type hints for parameters and return values
- Inline comments explaining non-obvious logic
- Keep functions focused (single responsibility)

#### E. Main Function
```python
def main():
    """Create [description of what this example creates]."""
    output_path = get_examples_output_dir() / "example_name_example.zarr"

    # Initial descriptive output using arbol
    aprint(f"Creating [example name] at {output_path}")
    aprint("This example demonstrates [key features]")
    aprint("")
    aprint("Scene features:")
    aprint("- Feature 1")
    aprint("- Feature 2")

    # Create scene using context manager
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Add scene metadata
        scene.attrs["description"] = """
Example Name
============

[Detailed description of what this scene contains]

Educational features:
- Point 1
- Point 2

Viewing tips:
- How to view this effectively
- What to look for
        """

        # Generate data with educational comments
        # Comment: Explain WHY you're doing this, not just WHAT
        positions = create_some_data()

        # Add to scene
        scene.add_points(
            "DescriptiveName",
            positions,
            colors=colors,
            radii=radii,
            # etc.
        )

        # Progress reporting
        aprint(f"✓ Added {len(positions):,} points")

    # Final instructions
    aprint("\n" + "=" * 60)
    aprint("VIEWING INSTRUCTIONS:")
    aprint("1. Run: luxar serve example_name_example.zarr")
    aprint("2. [Specific viewing instructions]")
    aprint("3. [What to look for]")
    aprint("=" * 60)


if __name__ == "__main__":
    main()
```

**Main Function Requirements:**
- Single docstring summarizing what gets created
- Output path uses `Path(__file__).parent` for portability
- Initial aprint statements describing the example
- Use `LuxarZarrCompiler` context manager (automatic finalization)
- Add scene.attrs["description"] with detailed metadata
- Educational inline comments explaining WHY
- Progress reporting with aprint
- Final viewing instructions

#### E.1 Explainer Overlay (house style — REQUIRED)

Every example that builds a viewable scene adds **exactly one** explainer card
per scene via the shared `add_explainer` helper (`_overlay_style.py`). This
gives the whole gallery a consistent, elegant overlay and tells the viewer what
the scene demonstrates and what to check. Add it **after** all geometry, while
the compiler/scene is still open:

```python
from _overlay_style import add_explainer

add_explainer(
    scene,
    title="Per-point Radii",            # concise feature name, Title case, no trailing period
    body=(                              # 1–2 sentences; <code>…</code> for params/APIs
        "Each point's on-screen size comes from the per-point "
        "<code>radii</code> array; three rows sweep small → large."
    ),
    observe=[                           # 2–4 short sentences of concrete, checkable things
        "Sizes increase smoothly from left to right.",
        "The smallest point is <code>0.1</code> units; the largest <code>1.0</code>.",
    ],
    observe_label="Verify",             # one of: "Look for" | "Verify" | "Observe" | "Notice"
)
```

**Rules:**
- The card reads top-to-bottom: **title → explanation → labelled look-for list**.
- One card per scene. If an example writes several scenes (e.g. `build_example`,
  `memory_optimization_example`), each scene gets its own tailored card.
- Default placement is top-left; override `anchor`/`position` only when the
  default would cover the geometry, or to avoid overlapping another overlay.
- For nD / time examples, mention keyboard navigation
  (`Press <code>1</code>–<code>9</code> then <code>[</code>/<code>]</code>`).
- Only the inline tags the sanitiser allows survive — `<code>`, `<strong>`,
  `<br>`, lists. The helper handles all styling; never hand-roll overlay CSS.
- An example that builds **no** scene (e.g. a pure fitting/benchmark script with
  no `LuxarZarrCompiler`) has nothing to annotate — skip the card.

#### F. Entry Point
```python
if __name__ == "__main__":
    main()
```

---

## Code Style Requirements

### Comments

**Do:**
- Explain WHY you're doing something, not just WHAT
- Add comments before complex calculations
- Document non-obvious choices
- Explain parameters and their effects
- Note educational points

**Don't:**
- Comment obvious code (`x = 5  # Set x to 5`)
- Write novels - be concise
- Leave outdated comments
- Comment everything - code should be self-explanatory where possible

**Examples:**
```python
# GOOD: Explains WHY
# Use golden ratio for even distribution on sphere
phi = np.pi * (1 + 5**0.5) * indices

# BAD: Just repeats the code
# Calculate phi
phi = np.pi * (1 + 5**0.5) * indices

# GOOD: Explains educational point
# KEY PRINCIPLE: spacing = 2 × radius makes spheres touch perfectly
spacing = 2 * radius

# GOOD: Explains non-obvious choice
# Use high sharpness (10.0) for sharp edges that clearly show grid structure
sharpness = 10.0
```

### arbol Usage

**Always use arbol for console output:**
```python
from arbol import aprint, asection

# Simple messages
aprint("Creating scene...")
aprint(f"Added {n_points:,} points")

# Hierarchical sections for complex operations
with asection("Data Generation"):
    aprint("Generating positions...")
    # ... work ...
    aprint(f"✓ Created {n_points:,} positions")
```

**Rules:**
- Use `aprint()` instead of `print()`
- Use `asection()` for logical blocks
- Format numbers with commas: `f"{n:,}"`
- Use checkmarks for success: `✓`
- Use section headers: `=====`

### Data Types

**Always specify dtypes explicitly:**
```python
# Positions - always float32
positions = np.array(data, dtype=np.float32)

# Colors - float32 for HDR support
colors = np.array(color_data, dtype=np.float32)

# Radii - float32
radii = np.array(radii_data, dtype=np.float32)

# Sharpness - float32
sharpness = np.full(n_points, 2.0, dtype=np.float32)
```

### Variable Naming

**Be descriptive:**
```python
# GOOD
n_points = 1000
sphere_radius = 0.5
grid_spacing = 2.0

# BAD
n = 1000
r = 0.5
s = 2.0
```

### Node Naming

Use **snake_case** for node names passed as the first argument to `add_points` /
`add_lines` / `add_gsplats` / `add_group`. These names become zarr group paths and
hover-tooltip labels; snake_case keeps them consistent with the rest of the
codebase and the viewer's path conventions.

```python
# GOOD
scene.add_points("rainbow_spiral", positions, …)
scene.add_lines("floor_grid", vertices, …)
parent = scene.add_group("solar_system")

# BAD (some legacy examples use this — do not propagate)
scene.add_points("RainbowSpiral", positions, …)
```

### Hierarchical Adds

Prefer the **method-on-parent** idiom over `parent=` kwarg:

```python
# GOOD — natural read order
sun_group = scene.add_group("sun_group")
sun_group.add_points("sun", positions, …)
earth_group = sun_group.add_group("earth_system")
earth_group.add_points("earth", positions, …)

# Discouraged — works but mixes hierarchy with a flat call
scene.add_points("sun", positions, parent=sun_group)
```

### Color Convention

Pick ONE color convention per example and stick to it:

- **float32 in [0.0, 1.0]** — the default; use `np.array([…], dtype=np.float32)`
  or a bare list of floats `[1.0, 0.31, 0.31]`.
- **uint8 in [0, 255]** — only when working with image-derived data; cast
  explicitly: `np.array([…], dtype=np.uint8)`.

Mixing within the same scene is a bug: Luxar auto-detects HDR when any value
exceeds 1.0, so a stray `[80, 255, 80]` (intended as uint8) silently becomes a
massively-overbright HDR triple.

### Size Cap

Examples should stay **≤ ~100,000 elements** (points / vertices / splats) and
**≤ ~250 lines of code**. Larger, more complex showcases belong in
`packages/luxar/src/luxar/demos/`.

### Examples Are Also E2E Fixtures

Most existing examples are referenced by the viewer's E2E suite (see
`packages/luxar-viewer/src/tests/e2e/`). The generated `.zarr` is the contract:

- **Do not** change colors, point counts, RNG seeds, sharpness math, or radii
  in an existing example without first checking which spec files reference its
  `.zarr` output (`grep -rn '<name>_example' packages/luxar-viewer/src/tests/e2e/`)
  and updating any visual-regression baselines.
- **Net-new** examples added per the conventions above have no such constraint.
- Cosmetic-only edits (aprint text, docstrings, dead-comment cleanup, control-flow
  restructuring that doesn't change written data) are safe.

---

## Example Categories

### Minimal Examples
- **Purpose**: Teach one specific concept
- **Size**: < 100 lines
- **Complexity**: Beginner-friendly
- **Examples**: `single_point_example.py`

### Comprehensive Examples
- **Purpose**: Show complete feature set
- **Size**: 100-300 lines
- **Complexity**: Intermediate
- **Examples**: `hierarchy_example.py`, `transform_example.py`

### Advanced Examples
- **Purpose**: Production techniques
- **Size**: 200-500 lines
- **Complexity**: Advanced
- **Examples**: `progressive_writing_example.py`, `memory_optimization_example.py`

### Stress Test Examples
- **Purpose**: Performance testing
- **Size**: Variable
- **Complexity**: High data volume
- **Examples**: `dense_cubic_gradient_example.py` (1M points)

---

## Quality Checklist

Before committing an example, verify:

### Code Quality
- [ ] Follows the template structure exactly
- [ ] Has comprehensive module docstring
- [ ] All functions have docstrings with Args/Returns
- [ ] Type hints on all function parameters
- [ ] Inline comments explain WHY, not WHAT
- [ ] Variable names are descriptive
- [ ] Uses arbol (aprint/asection) for all output
- [ ] Proper dtype specification (float32)

### Educational Value
- [ ] Demonstrates one clear concept or feature
- [ ] Includes educational comments explaining key principles
- [ ] Has viewing instructions
- [ ] Explains what to look for
- [ ] Has an `add_explainer` card per scene (title → explanation → look-for list)
- [ ] Mentions common pitfalls or tips

### Functionality
- [ ] Actually runs without errors
- [ ] Generates the described output
- [ ] Output path follows convention (`example_name_example.zarr`)
- [ ] Scene contains what the docstring claims
- [ ] Viewing instructions are accurate

### Documentation
- [ ] README.md includes this example
- [ ] Docstring matches README entry
- [ ] Example fits into the learning progression
- [ ] No misleading or outdated information

---

## Anti-Patterns (Avoid These)

### ❌ Test Code Disguised as Examples
```python
"""Test script to verify the loader works."""  # NO!
```
Test code belongs in `/tests/`, not `/examples/`.

### ❌ Overly Complex "Kitchen Sink" Examples
Don't try to demonstrate 10 features in one example. Focus!

### ❌ Undocumented Magic Numbers
```python
positions = np.random.randn(n_points, 3) * 47.3  # What's 47.3? Why?
```
Explain or use named constants.

### ❌ Missing Educational Context
```python
# Just shows code without explaining WHY or WHEN to use it
```
Every example should teach something specific.

### ❌ Obsolete or Deprecated APIs
If an API changes, update ALL examples immediately.

### ❌ Non-Standalone Examples
Every example must run independently without dependencies on other examples.

---

## Example Template (Copy-Paste Starting Point)

```python
#!/usr/bin/env python3
"""Example Title - Brief one-line description.

This example demonstrates:
- Key feature or concept 1
- Key feature or concept 2
- Key feature or concept 3

Educational value:
- What users will learn
- When to use this technique
- Important principles illustrated
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_example_data(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Create data for this example.

    Args:
        n_points: Number of points to generate

    Returns:
        Tuple of (positions, colors) arrays
    """
    # Generate positions with educational comment
    positions = np.random.randn(n_points, 3).astype(np.float32) * 10

    # Generate colors with educational comment
    colors = np.random.rand(n_points, 3).astype(np.float32)

    return positions, colors


def main():
    """Create an example demonstrating [specific feature]."""
    output_path = get_examples_output_dir() / "example_name_example.zarr"

    aprint(f"Creating example at {output_path}")
    aprint("This example demonstrates [key concept]")

    # Create scene
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Generate data
        n_points = 1000
        positions, colors = create_example_data(n_points)

        # Add to scene (snake_case node name)
        scene.add_points(
            "example_points",
            positions,
            colors=colors,
            opacity=1.0,
            blending_mode="additive",
        )

        aprint(f"✓ Added {n_points:,} points")

        # House-style explainer card (title → explanation → look-for list).
        add_explainer(
            scene,
            title="Example Feature",
            body="One or two sentences on what the scene shows and why.",
            observe=[
                "A concrete, checkable thing visible in the viewer.",
                "Another thing to verify.",
            ],
            observe_label="Look for",
        )

    # Viewing instructions
    aprint("\n" + "=" * 60)
    aprint("VIEWING INSTRUCTIONS:")
    aprint("1. Run: luxar serve example_name_example.zarr")
    aprint("2. [Specific viewing instructions]")
    aprint("3. [What to observe or test]")
    aprint("")
    aprint("EDUCATIONAL NOTES:")
    aprint("- Key principle or concept")
    aprint("- Important observation")
    aprint("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Maintenance

When APIs change:
1. Update ALL affected examples immediately
2. Test each example actually runs
3. Verify generated output matches description
4. Update README.md if example list changes

When adding new examples:
1. Follow this template exactly
2. Choose appropriate category (minimal/comprehensive/advanced/stress)
3. Add to README.md in the correct section
4. Ensure it teaches something new (no redundancy)
5. Run `make run-examples` to verify it works
