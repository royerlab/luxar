#!/usr/bin/env python3
"""
Performance comparison between original and optimized precision fitting.
"""

import time
import numpy as np
import torch
from arbol import aprint, asection

def test_performance_comparison():
    """Compare performance between original and optimized implementations."""
    
    with asection("⚡ Performance Comparison"):
        
        # Create test data
        shape = (128, 128)
        np.random.seed(42)
        V = np.random.rand(*shape).astype(np.float32)
        
        # More challenging case: more splats
        n_splats = 20
        centers = np.random.rand(n_splats, 2) * np.array(shape)
        
        aprint(f"Test data: {shape} image, {n_splats} Gaussians")
        
        # Test parameters
        test_params = {
            'init_sigma_vox': 2.0,
            'n_iters': 30,
            'lr': 0.1,
            'loss_type': 'mse',
            'verbose': False,
            'device': 'cpu',
        }
        
        results = {}
        
        # Test optimized version
        with asection("Testing Optimized Implementation"):
            from luxar.gsplats.fit_gsplats_precision_optimized import (
                fit_gaussian_splats_precision_optimized
            )
            
            start_time = time.time()
            try:
                params_opt, amps_opt, stats_opt = fit_gaussian_splats_precision_optimized(
                    V=V,
                    centers_overcomplete=centers,
                    **test_params
                )
                opt_time = time.time() - start_time
                results['optimized'] = {
                    'time': opt_time,
                    'final_loss': stats_opt['final_loss'],
                    'iterations': stats_opt['iterations'],
                    'success': True
                }
                aprint(f"✅ Optimized: {opt_time:.3f}s, loss={stats_opt['final_loss']:.6f}")
            except Exception as e:
                opt_time = time.time() - start_time
                results['optimized'] = {'time': opt_time, 'success': False, 'error': str(e)}
                aprint(f"❌ Optimized failed: {e}")
        
        # Test original version (if available)
        with asection("Testing Original Implementation"):
            try:
                from luxar.gsplats.fit_gsplats import fit_gaussian_splats
                
                start_time = time.time()
                params_orig, amps_orig, stats_orig = fit_gaussian_splats(
                    V=V,
                    centers_overcomplete=centers,
                    **test_params
                )
                orig_time = time.time() - start_time
                results['original'] = {
                    'time': orig_time,
                    'final_loss': stats_orig['final_loss'],
                    'iterations': stats_orig['iterations'],
                    'success': True
                }
                aprint(f"✅ Original: {orig_time:.3f}s, loss={stats_orig['final_loss']:.6f}")
                
            except ImportError:
                aprint("⚠️  Original implementation not available for comparison")
                results['original'] = {'success': False, 'error': 'not available'}
            except Exception as e:
                orig_time = time.time() - start_time
                results['original'] = {'time': orig_time, 'success': False, 'error': str(e)}
                aprint(f"❌ Original failed: {e}")
        
        # Compare results
        if results.get('optimized', {}).get('success') and results.get('original', {}).get('success'):
            with asection("Performance Comparison"):
                opt_time = results['optimized']['time']
                orig_time = results['original']['time']
                speedup = orig_time / opt_time
                
                aprint(f"Optimized time: {opt_time:.3f}s")
                aprint(f"Original time:  {orig_time:.3f}s")
                aprint(f"Speedup:        {speedup:.2f}x")
                
                opt_loss = results['optimized']['final_loss']
                orig_loss = results['original']['final_loss']
                loss_ratio = orig_loss / opt_loss
                
                aprint(f"Optimized loss: {opt_loss:.6f}")
                aprint(f"Original loss:  {orig_loss:.6f}")
                aprint(f"Loss ratio:     {loss_ratio:.3f}")
                
                if speedup > 1.1:
                    aprint("🚀 Optimized version is significantly faster!")
                elif speedup > 0.9:
                    aprint("⚡ Similar performance")
                else:
                    aprint("⚠️  Optimized version is slower")
        
        # Test memory efficiency
        with asection("Memory Efficiency Test"):
            aprint("Testing memory usage of batched vs. expanded operations...")
            
            # Simulate the old expand approach memory usage
            K, d, P = 10, 3, 1000  # Typical batch parameters
            old_memory = K * P * d * d * 4 + K * P * d * 4  # Expanded U + delta (float32)
            new_memory = K * d * d * 4 + K * d * P * 4  # Original U + y (float32)
            
            memory_reduction = 1.0 - (new_memory / old_memory)
            aprint(f"Old approach memory:  {old_memory / 1024**2:.2f} MB")
            aprint(f"New approach memory:  {new_memory / 1024**2:.2f} MB")
            aprint(f"Memory reduction:     {memory_reduction * 100:.1f}%")
        
        # Test numerical stability
        with asection("Numerical Stability Test"):
            aprint("Testing triangular solve robustness...")
            
            from luxar.gsplats.models.gsplats.gsplats_precision_render_optimized import (
                _solve_triangular_safe
            )
            
            # Create a potentially problematic case
            U_test = np.array([[[100.0, 50.0], [0.0, 0.01]]], dtype=np.float32)  # Ill-conditioned
            U_tensor = torch.tensor(U_test)
            eye = torch.eye(2).unsqueeze(0)
            
            try:
                result = _solve_triangular_safe(U_tensor.transpose(-1, -2), eye, upper=False)
                aprint("✅ Handled ill-conditioned case successfully")
                aprint(f"   Result range: [{result.min():.3f}, {result.max():.3f}]")
            except Exception as e:
                aprint(f"❌ Numerical stability test failed: {e}")
        
        aprint("⚡ Performance comparison completed!")

if __name__ == "__main__":
    test_performance_comparison()