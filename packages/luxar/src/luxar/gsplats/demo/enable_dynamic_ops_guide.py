#!/usr/bin/env python3
"""
Guide: How to Enable Dynamic Operations in Existing Demos

This file shows the minimal changes needed to enable dynamic Gaussian splat operations
in any existing demo or script that uses fit_gaussian_splats().
"""

# === STEP 1: Import the dynamic operations configuration ===
from luxar.gsplats.dynamic_ops import DynamicOpsConfig


# === STEP 2: Create and configure dynamic operations ===
def create_dynamic_config_2d():
    """Recommended dynamic operations config for 2D data."""
    config = DynamicOpsConfig()
    config.step_every = 10          # Run every 10 iterations
    config.max_add_per_step = 40    # Allow moderate seeding
    config.max_merges_per_step = 30 # Allow moderate merging
    config.residual_quantile = 0.94 # Selective seeding (higher = more selective)
    config.merge_dist_vox = 2.0     # Merge distance threshold in voxels
    config.amp_abs_min = 1e-4       # Prune splats below this amplitude
    config.do_prune = True          # Enable pruning
    config.do_seed = True           # Enable seeding
    config.do_merge = True          # Enable merging
    config.do_split = True          # Enable splitting
    config.split_eig_thr = 2.5      # Split threshold (higher = less splitting)
    return config


def create_dynamic_config_3d():
    """Recommended dynamic operations config for 3D data."""
    config = DynamicOpsConfig()
    config.step_every = 15          # Less frequent for 3D (more expensive)
    config.max_add_per_step = 25    # Conservative seeding
    config.max_merges_per_step = 20 # Conservative merging
    config.residual_quantile = 0.95 # More selective seeding
    config.merge_dist_vox = 1.8     # 3D distance threshold
    config.amp_abs_min = 2e-4       # Slightly higher threshold for 3D
    config.do_prune = True
    config.do_seed = True
    config.do_merge = True
    config.do_split = True
    config.split_eig_thr = 2.2      # 3D split threshold
    return config


# === STEP 3: Update your fit_gaussian_splats call ===
def example_usage_2d():
    """Example showing how to modify an existing 2D fitting call."""
    # Your existing imports and data preparation...
    # from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    # V = your_2d_image
    # centers = your_candidate_centers

    # ADD: Configure dynamic operations
    create_dynamic_config_2d()

    # MODIFY: Add dynamic operations to your existing call
    # params_full, amps, stats = fit_gaussian_splats(
    #     V,
    #     centers_overcomplete=centers,
    #     # ... your existing parameters ...
    #     # ADD these two lines:
    #     enable_dynamic_ops=True,
    #     dynamic_config=dynamic_config,
    # )
    pass


def example_usage_3d():
    """Example showing how to modify an existing 3D fitting call."""
    # Your existing imports and data preparation...
    # from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    # V = your_3d_volume
    # centers = your_candidate_centers

    # ADD: Configure dynamic operations
    create_dynamic_config_3d()

    # MODIFY: Add dynamic operations to your existing call
    # params_full, amps, stats = fit_gaussian_splats(
    #     V,
    #     centers_overcomplete=centers,
    #     # ... your existing parameters ...
    #     # ADD these two lines:
    #     enable_dynamic_ops=True,
    #     dynamic_config=dynamic_config,
    # )
    pass


# === STEP 4: Optional - Add conditional enable/disable ===
def example_with_toggle():
    """Example showing how to add a toggle for dynamic operations."""

    # ADD: Toggle variable at top of your script
    USE_DYNAMIC_OPS = True  # Set to False to disable

    # Configuration
    create_dynamic_config_2d() if USE_DYNAMIC_OPS else None

    # Your fitting call with conditional dynamic ops
    # params_full, amps, stats = fit_gaussian_splats(
    #     V,
    #     centers_overcomplete=centers,
    #     # ... your existing parameters ...
    #     enable_dynamic_ops=USE_DYNAMIC_OPS,
    #     dynamic_config=dynamic_config,
    # )
    pass


# === Configuration Guidelines ===
"""
Key Parameters to Tune:

1. step_every: How often to run dynamic operations
   - Lower = more frequent operations, slower but potentially better quality
   - Higher = less frequent operations, faster but potentially lower quality
   - 2D: 8-15, 3D: 12-20

2. residual_quantile: How selective to be when seeding new splats
   - Higher = more selective (only seed at highest error regions)
   - Lower = less selective (seed more broadly)
   - Range: 0.85-0.98

3. merge_dist_vox: Distance threshold for considering splats for merging
   - Higher = merge splats further apart
   - Lower = only merge very close splats
   - 2D: 1.5-2.5, 3D: 1.0-2.0

4. amp_abs_min: Amplitude threshold for pruning weak splats
   - Higher = more aggressive pruning
   - Lower = keep more weak splats
   - Range: 1e-5 to 1e-3

5. split_eig_thr: Threshold for splitting large splats
   - Higher = less splitting (keep larger splats)
   - Lower = more splitting (create more detailed splats)
   - 2D: 2.0-3.0, 3D: 1.8-2.5

Performance vs Quality Trade-offs:
- More frequent operations (lower step_every): Better quality, slower
- More seeding/splitting: Better detail capture, more splats
- More merging/pruning: Fewer splats, faster inference, potential quality loss
"""

if __name__ == "__main__":
    print("📚 Dynamic Operations Integration Guide")
    print("=" * 50)
    print("This script shows how to enable dynamic operations in your demos.")
    print("See the function examples above for step-by-step instructions.")
    print("\n✅ Demos already updated:")
    print("  • demo_splats.py")
    print("  • demo_splats_3d_simple_example.py")
    print("\n📝 To update other demos, follow the 4-step pattern shown in this file.")
