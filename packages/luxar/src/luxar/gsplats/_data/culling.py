"""Splat-culling mixin for ``GSplatData`` (heuristic + contribution-based)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import _GSplatDataOps

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData


class CullingMixin(_GSplatDataOps):
    """``cull`` and its heuristic helper — remove negligible splats."""

    def cull(
        self,
        target: np.ndarray | None = None,
        *,
        method: str = "auto",
        shape: tuple[int, ...] | None = None,
        truncate: float | None = None,
        # --- error_budget / redundancy params ---
        error_percentile: float = 99.0,
        error_tolerance: float = 1.0,
        redundancy_threshold: float = 0.01,
        max_binary_search_iters: int = 8,
        device: str | None = None,
        intensity_floor: float = 1e-5,
        # --- heuristic params ---
        retention: float = 0.95,
        amplitude_percentile: float = 5.0,
        volume_percentile: float = 95.0,
        verbose: bool = False,
    ) -> "GSplatData":
        """Cull splats that contribute negligibly to the reconstruction.

        This is the unified entry point for all splat removal strategies,
        from fast heuristics to principled contribution-based methods.
        The ``method`` parameter selects which strategy to use.

        Methods (ordered from cheapest to most principled)
        --------------------------------------------------

        **"cumulative"** — Keep the top splats that account for a target
        fraction of the total amplitude.  Fast (no rendering), but blind
        to spatial overlap: a low-amplitude splat covering a unique region
        will be removed even though it is the sole contributor there.

            >>> data.cull(method="cumulative", retention=0.95)

        **"amplitude_percentile"** — Remove splats in the bottom X
        percentile of amplitude.  Same limitation as cumulative: ignores
        spatial context.

            >>> data.cull(method="amplitude_percentile", amplitude_percentile=10)

        **"combined"** — Remove splats that have low amplitude OR unusually
        large volume (artifacts).  Useful as a quick cleanup pass.

            >>> data.cull(method="combined", amplitude_percentile=5, volume_percentile=95)

        **"redundancy"** — Render the full reconstruction and measure each
        splat's maximum *fractional contribution* ``g_j(x) / V_pred(x)``.
        If a splat never contributes more than ``redundancy_threshold`` of
        the local signal, it is redundant.  Does not need the target volume
        but requires GPU rendering.

            >>> data.cull(method="redundancy", shape=(128,128,128), redundancy_threshold=0.02)

        **"error_budget"** — The most principled mode.  Requires the
        original target volume.  Computes the residual ``R = target - V_pred``
        and derives an error budget from it.  A splat is safe to remove when
        the worst-case error *increase* from its removal is below the budget.
        Robust to pre-existing noise and accounts for spatial redundancy.

            >>> data.cull(target_volume, method="error_budget", error_percentile=99)

        **"auto"** (default) — Selects automatically:
        ``"error_budget"`` if *target* is provided, ``"redundancy"`` if
        *shape* is provided, ``"cumulative"`` otherwise.

        Joint compounding check (error_budget and redundancy only)
        ----------------------------------------------------------
        After identifying individual candidates, verifies that their
        *joint* removal does not exceed the budget.  If it does, a binary
        search tightens the per-splat threshold until the joint constraint
        holds, guaranteeing that the combined removal is safe.

        Args:
            target: Original target volume.  If provided and ``method="auto"``,
                selects error-budget mode.
            method: Culling strategy.  One of ``"auto"``, ``"error_budget"``,
                ``"redundancy"``, ``"cumulative"``, ``"amplitude_percentile"``,
                ``"combined"``.
            shape: Volume shape for rendering (error_budget / redundancy).
                Defaults to ``target.shape`` when target is provided.
            truncate: Truncation radius in standard deviations.
                Defaults to ``self.truncation_radius``.
            error_percentile: *error_budget only.*  Percentile of ``|residual|``
                for the budget (0--100).
            error_tolerance: *error_budget only.*  Multiplier on the budget.
            redundancy_threshold: *redundancy only.*  Max fractional
                contribution (0--1) below which a splat is redundant.
            max_binary_search_iters: *error_budget / redundancy only.*
                Max iterations for the joint compounding binary search.
            device: Device for GPU computation.  Auto-detected if None.
            intensity_floor: Min intensity threshold for AABB computation.
            retention: *cumulative only.*  Fraction of total amplitude to
                retain (0--1).
            amplitude_percentile: *amplitude_percentile / combined only.*
                Bottom percentile to remove (0--100).
            volume_percentile: *combined only.*  Remove splats above this
                volume percentile (0--100).
            verbose: Print progress information.

        Returns:
            New GSplatData with culled splats removed.  Stats include
            ``culled``, ``culling_method``, ``n_original``, ``n_culled``.
        """
        if truncate is None:
            truncate = self.truncation_radius

        # --- Resolve "auto" method ---
        if method == "auto":
            if target is not None:
                method = "error_budget"
            elif shape is not None:
                method = "redundancy"
            else:
                method = "cumulative"

        # Multi-substitutive: cull EVERY substitutive level and rebuild the
        # pyramid (decision 6) rather than collapsing to the default level via
        # the single-level mask that the strategies below feed to self.filter().
        # Each level is culled through the single-substitutive path (the same
        # target volume reconstructs every level). Mirrors filter_by().
        if self.n_substitutive > 1:
            out = self._map_substitutive(
                lambda lvl: lvl.cull(
                    target,
                    method=method,
                    shape=shape,
                    truncate=truncate,
                    error_percentile=error_percentile,
                    error_tolerance=error_tolerance,
                    redundancy_threshold=redundancy_threshold,
                    max_binary_search_iters=max_binary_search_iters,
                    device=device,
                    intensity_floor=intensity_floor,
                    retention=retention,
                    amplitude_percentile=amplitude_percentile,
                    volume_percentile=volume_percentile,
                    verbose=verbose,
                )
            )
            out.stats.update(
                {
                    "culled": True,
                    "culling_method": method,
                    "n_original": self.n_splats,
                    "n_culled": self.n_splats - out.n_splats,
                }
            )
            return out

        # =================================================================
        # Heuristic methods (no rendering, CPU-only, fast)
        # =================================================================
        if method in ("cumulative", "amplitude_percentile", "combined"):
            return self._cull_heuristic(
                method=method,
                retention=retention,
                amplitude_percentile=amplitude_percentile,
                volume_percentile=volume_percentile,
            )

        # =================================================================
        # Rendering-based methods (GPU, contribution-aware)
        # =================================================================
        if method not in ("error_budget", "redundancy"):
            raise ValueError(
                f"Unknown culling method: {method!r}. "
                "Choose from: 'auto', 'error_budget', 'redundancy', "
                "'cumulative', 'amplitude_percentile', 'combined'."
            )

        import torch

        from luxar.gsplats.culling import cull_by_contribution
        from luxar.gsplats.rendering.volume_rendering import auto_detect_device

        if target is not None and shape is None:
            shape = target.shape
        if shape is None:
            raise ValueError(
                "shape is required for error_budget/redundancy modes. "
                "Pass the volume shape, e.g. shape=(128, 128, 128), "
                "or provide a target volume."
            )

        if device is None:
            device = auto_detect_device()

        # Convert to GPU tensors
        centers_t = torch.from_numpy(self.centers.astype(np.float32)).to(device)
        amps_t = torch.from_numpy(self.amplitudes.astype(np.float32)).to(device)
        target_t = (
            torch.from_numpy(target.astype(np.float32)).to(device)
            if target is not None
            else None
        )

        # Unpack Cholesky factors: (N, d*(d+1)/2) -> (N, d, d) lower-triangular
        chol = self.cholesky_factors
        ndim = self.ndim
        chol_t = torch.from_numpy(chol).to(device)
        Ls_t = torch.zeros((len(chol), ndim, ndim), device=device, dtype=torch.float32)
        if ndim == 2:
            Ls_t[:, 0, 0] = chol_t[:, 0]
            Ls_t[:, 1, 0] = chol_t[:, 1]
            Ls_t[:, 1, 1] = chol_t[:, 2]
        elif ndim == 3:
            Ls_t[:, 0, 0] = chol_t[:, 0]
            Ls_t[:, 1, 0] = chol_t[:, 1]
            Ls_t[:, 1, 1] = chol_t[:, 2]
            Ls_t[:, 2, 0] = chol_t[:, 3]
            Ls_t[:, 2, 1] = chol_t[:, 4]
            Ls_t[:, 2, 2] = chol_t[:, 5]
        else:
            idx = 0
            for i in range(ndim):
                for j in range(i + 1):
                    Ls_t[:, i, j] = chol_t[:, idx]
                    idx += 1
        del chol_t

        result = cull_by_contribution(
            centers_t,
            Ls_t,
            amps_t,
            target_t,
            shape,
            truncate=truncate,
            error_percentile=error_percentile,
            error_tolerance=error_tolerance,
            redundancy_threshold=redundancy_threshold,
            max_binary_search_iters=max_binary_search_iters,
            intensity_floor=intensity_floor,
            verbose=verbose,
        )

        # Free GPU tensors used for culling
        del centers_t, amps_t, Ls_t
        if target_t is not None:
            del target_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        culled = self.filter(result.keep_mask)
        culled.stats.update(
            {
                "culled": True,
                "culling_method": result.mode,
                "n_original": self.n_splats,
                "n_culled": result.n_culled,
                "error_budget": result.error_budget,
                "phase1_candidates": result.phase1_candidates,
                "phase2_iterations": result.phase2_iterations,
                "max_joint_error": result.max_joint_error,
            }
        )
        return culled

    def _cull_heuristic(
        self,
        method: str,
        retention: float = 0.95,
        amplitude_percentile: float = 5.0,
        volume_percentile: float = 95.0,
    ) -> "GSplatData":
        """Heuristic culling methods (no rendering needed)."""
        N_original = self.n_splats

        if N_original == 0:
            result = self.filter(np.ones(0, dtype=bool))
            result.stats.update(
                {
                    "culled": True,
                    "culling_method": method,
                    "n_original": 0,
                    "n_culled": 0,
                }
            )
            return result

        mask = np.ones(N_original, dtype=bool)

        # Alpha-effective amplitudes (A·a when RGBA colors carry per-splat
        # opacity): every blending mode scales contribution by alpha, so
        # culling by raw A would misrank imported classical splats.
        from luxar.gsplats.utils.alpha import effective_amplitudes

        eff_amps = effective_amplitudes(self)

        if method == "cumulative":
            sorted_indices = np.argsort(eff_amps)[::-1]
            sorted_amps = eff_amps[sorted_indices]
            cumsum_amps = np.cumsum(sorted_amps)
            total_amp = cumsum_amps[-1]
            if total_amp == 0:
                mask = (
                    np.ones(N_original, dtype=bool)
                    if retention > 0
                    else np.zeros(N_original, dtype=bool)
                )
            else:
                cumsum_norm = cumsum_amps / total_amp
                n_keep = np.searchsorted(cumsum_norm, retention) + 1
                n_keep = min(n_keep, N_original)
                keep_indices = sorted_indices[:n_keep]
                mask = np.zeros(N_original, dtype=bool)
                mask[keep_indices] = True

        elif method == "amplitude_percentile":
            threshold = np.percentile(eff_amps, amplitude_percentile)
            mask = eff_amps >= threshold

        elif method == "combined":
            vols = self.volumes()
            amp_threshold = np.percentile(eff_amps, amplitude_percentile)
            vol_threshold = np.percentile(vols, volume_percentile)
            mask = (eff_amps >= amp_threshold) & (vols <= vol_threshold)

        else:
            raise ValueError(f"Unknown heuristic method: {method!r}")

        result = self.filter(mask)

        total_amp = np.sum(eff_amps)
        result.stats.update(
            {
                "culled": True,
                "culling_method": method,
                "n_original": N_original,
                "n_culled": N_original - result.n_splats,
                "amplitude_retention": (
                    float(np.sum(effective_amplitudes(result)) / total_amp)
                    if total_amp > 0
                    else 1.0
                ),
            }
        )
        return result
