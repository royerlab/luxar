#!/usr/bin/env python3
"""
Find optimal parameters for precision parameterization approach.
"""

import numpy as np
from arbol import aprint, asection

def test_parameter_sensitivity():
    """Test different parameter combinations for precision approach."""
    
    with asection("🔧 Parameter Sensitivity Analysis"):
        
        # Create test data (from the demo)
        from skimage import data, filters
        
        blobs = data.binary_blobs(
            length=128,  # Smaller for faster testing
            blob_size_fraction=0.06, 
            n_dim=2, 
            volume_fraction=0.18, 
            rng=42
        ).astype(float)
        V = filters.gaussian(blobs, 1.5)
        V = (V / V.max()).astype(np.float32)
        
        aprint(f"Test image - shape: {V.shape}")
        aprint(f"Test image - range: [{V.min():.3f}, {V.max():.3f}]")
        aprint(f"Test image - sum: {V.sum():.3f}")
        
        # Generate candidates
        from luxar.gsplats.candidates import find_candidates_overcomplete_nd
        centers = find_candidates_overcomplete_nd(
            V,
            spacing=(2, 2),
            peaks_per_scale=200,
            add_intensity_grid=False,
        )
        aprint(f"Candidate centers: {len(centers)}")
        
        # Test different parameter combinations
        test_configs = [
            {
                "name": "Demo defaults",
                "l1_amp": 0.001,
                "n_iters": 200,  # Reduced for testing
                "lr": 0.2,
                "init_sigma_vox": 1.6,
            },
            {
                "name": "Stronger sparsity",
                "l1_amp": 0.01,   # 10x stronger
                "n_iters": 200,
                "lr": 0.2,
                "init_sigma_vox": 1.6,
            },
            {
                "name": "Lower learning rate",
                "l1_amp": 0.005,
                "n_iters": 200,
                "lr": 0.1,      # Slower learning
                "init_sigma_vox": 1.6,
            },
            {
                "name": "Larger initial sigma",
                "l1_amp": 0.005,
                "n_iters": 200,
                "lr": 0.2,
                "init_sigma_vox": 2.5,  # Start with bigger splats
            },
            {
                "name": "Combined tweaks",
                "l1_amp": 0.01,     # Strong sparsity
                "n_iters": 200,
                "lr": 0.15,         # Moderate learning rate
                "init_sigma_vox": 2.2,  # Slightly larger initial
            }
        ]
        
        results = []
        
        for config in test_configs:
            with asection(f"Testing: {config['name']}"):
                try:
                    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
                    
                    params, amps, stats = fit_gaussian_splats(
                        V,
                        centers_overcomplete=centers,
                        init_sigma_vox=config["init_sigma_vox"],
                        n_iters=config["n_iters"],
                        lr=config["lr"],
                        l1_amp=config["l1_amp"],
                        use_precision_parameterization=True,  # Test precision approach
                        verbose=False,
                        early_stopping=True,
                    )
                    
                    # Render result
                    from luxar.gsplats.models.gsplats.gsplats_render import render_gaussians_full_numpy
                    result = render_gaussians_full_numpy(V.shape, params, amps, truncate=3.0)
                    
                    # Compute metrics
                    mse = np.mean((V - result) ** 2)
                    significant_splats = np.sum(amps > 0.1)
                    total_splats = len(amps)
                    sparsity = significant_splats / total_splats if total_splats > 0 else 0
                    
                    # Store results
                    result_data = {
                        "config": config,
                        "final_loss": stats["final_loss"],
                        "mse": mse,
                        "total_splats": total_splats,
                        "significant_splats": significant_splats,
                        "sparsity": sparsity,
                        "time": stats["time_seconds"],
                        "converged": stats["converged"],
                        "sum_error": abs(result.sum() - V.sum()),
                    }
                    results.append(result_data)
                    
                    aprint(f"  Final loss: {stats['final_loss']:.8f}")
                    aprint(f"  MSE: {mse:.8f}")  
                    aprint(f"  Significant splats: {significant_splats}/{total_splats} ({sparsity:.2%})")
                    aprint(f"  Sum error: {result_data['sum_error']:.6f}")
                    aprint(f"  Time: {stats['time_seconds']:.2f}s")
                    aprint(f"  Converged: {stats['converged']}")
                    
                except Exception as e:
                    aprint(f"  ❌ Failed: {e}")
        
        # Analysis
        if results:
            with asection("📊 Parameter Analysis"):
                
                # Find best configuration by different criteria
                best_mse = min(results, key=lambda x: x["mse"])
                best_sparsity = max(results, key=lambda x: x["sparsity"])
                best_loss = min(results, key=lambda x: x["final_loss"])
                
                aprint(f"Best MSE: {best_mse['config']['name']}")
                aprint(f"  MSE: {best_mse['mse']:.8f}")
                aprint(f"  Sparsity: {best_mse['sparsity']:.2%}")
                
                aprint(f"\nBest Sparsity: {best_sparsity['config']['name']}")
                aprint(f"  MSE: {best_sparsity['mse']:.8f}")
                aprint(f"  Sparsity: {best_sparsity['sparsity']:.2%}")
                
                aprint(f"\nBest Loss: {best_loss['config']['name']}")
                aprint(f"  Loss: {best_loss['final_loss']:.8f}")
                aprint(f"  Sparsity: {best_loss['sparsity']:.2%}")
                
                # Recommendations
                aprint(f"\n=== RECOMMENDATIONS ===")
                
                if best_sparsity['sparsity'] > 0.5:  # If we achieved good sparsity
                    aprint("✅ Found configuration with good sparsity")
                    rec_config = best_sparsity['config']
                else:
                    aprint("⚠️  Need stronger sparsity regularization")
                    rec_config = {"l1_amp": 0.02, "lr": 0.1, "init_sigma_vox": 2.5}
                
                aprint("Recommended parameters for precision approach:")
                for key, value in rec_config.items():
                    if key != "name":
                        aprint(f"  {key}: {value}")
                
                return rec_config

if __name__ == "__main__":
    recommendations = test_parameter_sensitivity()
    print(f"Recommended config: {recommendations}")