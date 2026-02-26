"""
Fitting pipeline components for Gaussian splatting.

This package contains the refactored components of the Gaussian splat fitting pipeline:
- Configuration dataclasses
- Input validation and preprocessing
- Model and optimizer initialization
- Loss function creation
- Optimization loop logic
- Result finalization
- Visualization helpers
"""

from .config import (
    FitConfig,
    OptimizationResults,
    PreprocessedData,
)
from .initialization import initialize_optimization
from .losses import create_loss_function
from .optimization import run_optimization_loop
from .preprocessing import preprocess_data
from .results import finalize_results
from .validation import prepare_fit_config
from .visualization import display_compression_analysis, show_optimization_movie

__all__ = [
    "FitConfig",
    "PreprocessedData",
    "OptimizationResults",
    "prepare_fit_config",
    "preprocess_data",
    "initialize_optimization",
    "create_loss_function",
    "run_optimization_loop",
    "finalize_results",
    "display_compression_analysis",
    "show_optimization_movie",
]
