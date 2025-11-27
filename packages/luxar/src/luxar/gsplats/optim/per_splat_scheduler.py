"""
Per-Splat Learning Rate Schedulers for Gaussian Splatting

Specialized schedulers that work with PerSplatAdam to provide
individual learning rate schedules for each splat.
"""

from typing import Dict, Union

import torch

from .per_splat_adam import PerSplatAdam


class PerSplatReduceLROnPlateau:
    """
    Per-splat version of ReduceLROnPlateau scheduler.

    Tracks loss contribution per splat and reduces learning rate
    for individual splats when they plateau.
    """

    def __init__(
        self,
        optimizer: PerSplatAdam,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 1e-8,
        global_patience: int = 20,  # Global fallback
    ) -> None:
        """
        Initialize per-splat plateau scheduler.

        Args:
            optimizer: PerSplatAdam optimizer
            mode: 'min' for minimization (loss), 'max' for maximization
            factor: Factor by which to reduce LR (new_lr = lr * factor)
            patience: Number of steps without improvement before LR reduction
            threshold: Minimum change to qualify as improvement
            cooldown: Number of steps to wait after LR reduction
            min_lr: Minimum learning rate
            global_patience: Global patience for overall convergence
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.global_patience = global_patience

        # Per-splat state
        self.splat_scheduler_states: Dict[int, Dict] = {}

        # Global state
        self.global_best = None
        self.global_bad_epochs = 0
        self.global_cooldown_counter = 0
        self.last_epoch = 0

    def _init_splat_state(self, splat_idx: int) -> None:
        """Initialize scheduler state for a splat."""
        if self.mode == "min":
            best_loss = float("inf")
        else:
            best_loss = float("-inf")

        self.splat_scheduler_states[splat_idx] = {
            "best": best_loss,
            "num_bad_epochs": 0,
            "cooldown_counter": 0,
            "lr_reductions": 0,
        }

    def step(self, metrics: Union[float, torch.Tensor, Dict[int, float]]):
        """
        Update learning rates based on metrics.

        Args:
            metrics: Can be:
                - float: Global loss (affects all splats equally)
                - torch.Tensor: Per-splat losses (shape: [n_splats])
                - Dict[int, float]: Explicit per-splat metrics
        """
        if metrics is None:
            raise ValueError("Metrics cannot be None")

        self.last_epoch += 1

        try:
            # Convert metrics to per-splat format
            if isinstance(metrics, (float, int)):
                if not torch.isfinite(torch.tensor(metrics)):
                    raise ValueError(f"Invalid metric value: {metrics}")
                # Global metric - apply to all splats
                n_splats = self.optimizer.model.n_splats()
                splat_metrics = {i: float(metrics) for i in range(n_splats)}
            elif isinstance(metrics, torch.Tensor):
                if len(metrics.shape) == 0:
                    # Scalar tensor - convert to float and treat as global metric
                    if not torch.isfinite(metrics):
                        raise ValueError("Metric value must be finite")
                    n_splats = self.optimizer.model.n_splats()
                    splat_metrics = {i: float(metrics) for i in range(n_splats)}
                elif len(metrics.shape) == 1:
                    # Per-splat tensor
                    if not torch.all(torch.isfinite(metrics)):
                        raise ValueError("All metric values must be finite")
                    splat_metrics = {i: float(metrics[i]) for i in range(len(metrics))}
                else:
                    raise ValueError(
                        f"Metrics tensor must be scalar or 1D, got shape {metrics.shape}"
                    )
            elif isinstance(metrics, dict):
                # Validate dict values
                for k, v in metrics.items():
                    if not isinstance(k, int) or k < 0:
                        raise ValueError(
                            f"Dict keys must be non-negative integers, got {k}"
                        )
                    if not torch.isfinite(torch.tensor(v)):
                        raise ValueError(f"Invalid metric value for splat {k}: {v}")
                # Already per-splat dict
                splat_metrics = metrics
            else:
                raise TypeError(f"Unsupported metrics type: {type(metrics)}")
        except Exception as e:
            raise RuntimeError(f"Error processing metrics: {e}") from e

        # Update global state
        if isinstance(metrics, (float, int)) or (
            isinstance(metrics, torch.Tensor) and len(metrics.shape) == 0
        ):
            global_metric = (
                float(metrics) if isinstance(metrics, torch.Tensor) else metrics
            )
        elif isinstance(metrics, torch.Tensor):
            global_metric = float(torch.mean(metrics))
        else:
            global_metric = sum(splat_metrics.values()) / len(splat_metrics)

        self._update_global_state(global_metric)

        # Update each splat individually
        for splat_idx, metric in splat_metrics.items():
            self._update_splat_lr(splat_idx, metric)

    def _update_global_state(self, metric: float) -> None:
        """Update global scheduler state."""
        if self.global_best is None:
            self.global_best = metric

        if self.global_cooldown_counter > 0:
            self.global_cooldown_counter -= 1
            return

        if self._is_better(metric, self.global_best):
            self.global_best = metric
            self.global_bad_epochs = 0
        else:
            self.global_bad_epochs += 1

        # Global learning rate reduction fallback
        if self.global_bad_epochs >= self.global_patience:
            self._reduce_all_learning_rates()
            self.global_bad_epochs = 0
            self.global_cooldown_counter = self.cooldown

    def _update_splat_lr(self, splat_idx: int, metric: float) -> None:
        """Update learning rate for individual splat."""
        # Initialize state if needed
        if splat_idx not in self.splat_scheduler_states:
            self._init_splat_state(splat_idx)

        state = self.splat_scheduler_states[splat_idx]

        # Skip if in cooldown
        if state["cooldown_counter"] > 0:
            state["cooldown_counter"] -= 1
            return

        # Check for improvement
        if self._is_better(metric, state["best"]):
            state["best"] = metric
            state["num_bad_epochs"] = 0
        else:
            state["num_bad_epochs"] += 1

        # Reduce LR if plateau detected
        if state["num_bad_epochs"] >= self.patience:
            current_lr = self.optimizer.get_learning_rate(splat_idx)
            new_lr = max(current_lr * self.factor, self.min_lr)

            if new_lr < current_lr:
                self.optimizer.set_learning_rate(splat_idx, new_lr)
                state["lr_reductions"] += 1
                state["num_bad_epochs"] = 0
                state["cooldown_counter"] = self.cooldown

    def _reduce_all_learning_rates(self) -> None:
        """Global learning rate reduction for all splats."""
        for splat_idx in range(self.optimizer.model.n_splats()):
            current_lr = self.optimizer.get_learning_rate(splat_idx)
            new_lr = max(current_lr * self.factor, self.min_lr)
            if new_lr < current_lr:
                self.optimizer.set_learning_rate(splat_idx, new_lr)

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "min":
            return current < best - self.threshold
        else:
            return current > best + self.threshold

    def add_splats(self, n_new_splats: int):
        """Add scheduler state for new splats."""
        if n_new_splats < 0:
            raise ValueError(
                f"Number of new splats must be non-negative, got {n_new_splats}"
            )

        if n_new_splats == 0:
            return  # Nothing to do

        current_n = len(self.splat_scheduler_states)

        try:
            for i in range(n_new_splats):
                new_splat_idx = current_n + i
                self._init_splat_state(new_splat_idx)
        except Exception as e:
            raise RuntimeError(
                f"Failed to add scheduler state for {n_new_splats} new splats: {e}"
            ) from e

    def remove_splats(self, keep_mask: torch.Tensor):
        """Remove scheduler state for pruned splats."""
        if not isinstance(keep_mask, torch.Tensor):
            raise TypeError(f"keep_mask must be a torch.Tensor, got {type(keep_mask)}")

        if keep_mask.dtype != torch.bool:
            raise TypeError(f"keep_mask must be boolean tensor, got {keep_mask.dtype}")

        if len(keep_mask.shape) != 1:
            raise ValueError(
                f"keep_mask must be 1D tensor, got shape {keep_mask.shape}"
            )

        if len(keep_mask) != len(self.splat_scheduler_states):
            raise ValueError(
                f"keep_mask length {len(keep_mask)} doesn't match number of scheduler states {len(self.splat_scheduler_states)}"
            )

        try:
            new_states = {}
            keep_indices = torch.where(keep_mask)[0].cpu().numpy()

            for new_idx, old_idx in enumerate(keep_indices):
                old_idx_int = old_idx.item()
                if old_idx_int in self.splat_scheduler_states:
                    new_states[new_idx] = self.splat_scheduler_states[old_idx_int]

            self.splat_scheduler_states = new_states

        except Exception as e:
            raise RuntimeError(f"Failed to remove scheduler states: {e}") from e

    def get_lr_reduction_counts(self) -> torch.Tensor:
        """Get number of LR reductions per splat (for monitoring)."""
        n_splats = self.optimizer.model.n_splats()
        counts = torch.zeros(n_splats)

        for splat_idx in range(n_splats):
            if splat_idx in self.splat_scheduler_states:
                counts[splat_idx] = self.splat_scheduler_states[splat_idx][
                    "lr_reductions"
                ]

        return counts

    def state_dict(self) -> Dict:
        """Get scheduler state for serialization."""
        return {
            "splat_scheduler_states": self.splat_scheduler_states,
            "global_best": self.global_best,
            "global_bad_epochs": self.global_bad_epochs,
            "global_cooldown_counter": self.global_cooldown_counter,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load scheduler state from serialization."""
        self.splat_scheduler_states = state_dict["splat_scheduler_states"]
        self.global_best = state_dict["global_best"]
        self.global_bad_epochs = state_dict["global_bad_epochs"]
        self.global_cooldown_counter = state_dict["global_cooldown_counter"]
        self.last_epoch = state_dict["last_epoch"]


class PerSplatExponentialLR:
    """
    Per-splat exponential learning rate scheduler.

    Applies different decay rates to different splats based on their age.
    """

    def __init__(
        self, optimizer: PerSplatAdam, gamma: float = 0.95, age_based_decay: bool = True
    ) -> None:
        """
        Initialize per-splat exponential scheduler.

        Args:
            optimizer: PerSplatAdam optimizer
            gamma: Multiplicative factor of LR decay
            age_based_decay: If True, newer splats decay slower
        """
        self.optimizer = optimizer
        self.gamma = gamma
        self.age_based_decay = age_based_decay

        # Track splat ages (when they were added)
        self.splat_ages: Dict[int, int] = {}
        self.current_epoch = 0

        # Initialize ages for current splats
        for i in range(self.optimizer.model.n_splats()):
            self.splat_ages[i] = 0

    def step(self) -> None:
        """Apply exponential decay to all splats."""
        self.current_epoch += 1

        for splat_idx in range(self.optimizer.model.n_splats()):
            current_lr = self.optimizer.get_learning_rate(splat_idx)

            if self.age_based_decay:
                # Newer splats decay slower
                age = self.current_epoch - self.splat_ages.get(splat_idx, 0)
                age_factor = 1.0 / (1.0 + age * 0.1)  # Slower decay for newer splats
                gamma = self.gamma + (1 - self.gamma) * age_factor
            else:
                gamma = self.gamma

            new_lr = current_lr * gamma
            self.optimizer.set_learning_rate(splat_idx, new_lr)

    def add_splats(self, n_new_splats: int) -> None:
        """Add age tracking for new splats."""
        current_n = len(self.splat_ages)
        for i in range(n_new_splats):
            new_splat_idx = current_n + i
            self.splat_ages[new_splat_idx] = self.current_epoch  # Born now

    def remove_splats(self, keep_mask: torch.Tensor) -> None:
        """Remove age tracking for pruned splats."""
        new_ages = {}
        keep_indices = torch.where(keep_mask)[0].cpu().numpy()

        for new_idx, old_idx in enumerate(keep_indices):
            if old_idx.item() in self.splat_ages:
                new_ages[new_idx] = self.splat_ages[old_idx.item()]

        self.splat_ages = new_ages
