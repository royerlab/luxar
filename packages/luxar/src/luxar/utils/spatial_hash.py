"""Spatial hash grids for fast nD proximity queries.

Two complementary classes for two access patterns:

- :class:`SpatialHashGrid` is **online**: points are inserted one at a
  time and each insert may depend on previous queries (e.g. Poisson-disk
  sampling, where a candidate is accepted iff no prior accept lies
  within ``min_distance``). CPU-only, dict-backed, $O(1)$ amortised
  ``has_neighbor_within``.

- :class:`BatchedSpatialHashGrid` is **batched**: built once from a
  fixed point set, then queried many times. NumPy and PyTorch backends
  share the same hash scheme (``floor(x * inv_cell_size)`` per axis,
  packed into a single ``int64`` via Müller-style primes); the NumPy
  backend uses ``np.argsort``/``np.searchsorted``, the PyTorch backend
  uses the equivalent tensor ops on CUDA/MPS/CPU. With
  ``device='auto'`` (the default) the GPU backend is preferred, with
  conservative fallback to NumPy on out-of-memory or device-unavailable
  conditions only — generic exceptions propagate so real bugs aren't
  masked.

Correctness guarantee (both classes): radius/distance queries find all
hits within the requested radius provided ``cell_size >= radius``. The
$3^D$ neighbour-cell scan is then exhaustive.
"""

from __future__ import annotations

import itertools
import math
from typing import Literal, Optional, Union

import numpy as np
import torch
from arbol import aprint

# Per-axis primes for the cell-key hash. Müller / Teschner-style primes
# scaled to fit in int64 even after multiplication by large cell
# indices. Repeats every 16 axes (well beyond any expected ndim).
_HASH_PRIMES = (
    73856093,
    19349669,
    83492791,
    1376312589,
    922683317,
    2654435761,
    340573321,
    1152609299,
    109987147,
    471470953,
    2935578901,
    1376914253,
    283823111,
    602216189,
    1873020821,
    1456993943,
)

# Maximum shell radius for k-NN expansion before we brute-force the
# whole stored set. With unit-density points and modest k, shell 1 is
# almost always enough; pathological inputs may need 2 or 3.
_KNN_MAX_SHELL = 3

# Maximum padded candidate matrix width for the GPU kNN path. Queries
# whose total shell-expanded candidate count exceeds this fall back to
# the NumPy per-query path (which has no padding overhead). Generous
# default — typical Lloyd workloads have <200 candidates per query.
_GPU_KNN_MAX_PAD = 4096


# ─────────────────────────────────────────────────────────────────────
# Online / incremental: SpatialHashGrid
# ─────────────────────────────────────────────────────────────────────


class SpatialHashGrid:
    """
    Spatial hash grid for O(1) amortized proximity queries on nD points.

    Points are hashed into cells of a given size. Proximity queries check
    only the 3^ndim neighboring cells, giving O(1) amortized cost per query
    (assuming bounded density per cell). This replaces KD-tree approaches
    that require O(M log M) rebuilds.

    Parameters
    ----------
    cell_size : float
        Size of each grid cell. Must be >= the query distance used in
        ``has_neighbor_within`` for correctness (the 3^ndim neighbor check
        only guarantees finding all points within ``cell_size``).
    ndim : int
        Number of spatial dimensions.

    Notes
    -----
    Correctness guarantee: if ``cell_size >= distance``, then for any query
    point p and stored point q with ||p - q|| < distance, q is guaranteed
    to be in the same cell or an adjacent cell (within 1 cell offset in
    each dimension). Proof: ||p - q|| < distance <= cell_size implies
    |p[d] - q[d]| < cell_size for each dimension d.
    """

    def __init__(self, cell_size: float, ndim: int):
        """Create an empty grid with ``cell_size``-sided cells over ``ndim`` axes.

        ``cell_size`` is floored at 1e-10 to keep ``1/cell_size`` finite; the
        backing point array starts at 64 rows and grows by doubling. The
        ``3^ndim`` neighbour-cell offsets used by :meth:`has_neighbor_within`
        are precomputed here once.
        """
        self._cell_size = max(cell_size, 1e-10)
        self._inv_cell_size = 1.0 / self._cell_size
        self._ndim = ndim
        self._grid: dict[tuple[int, ...], list[int]] = {}
        # Pre-allocate with doubling growth
        self._points = np.empty((64, ndim), dtype=np.float32)
        self._n_points = 0
        self._neighbor_offsets = list(itertools.product([-1, 0, 1], repeat=ndim))

    def _cell_key(self, point: np.ndarray) -> tuple[int, ...]:
        """Compute the grid cell key for a point.

        Uses ``math.floor`` (not ``int()``) so that negative coordinates
        are handled correctly.  ``int()`` truncates toward zero, which
        would map e.g. -0.5 and +0.5 to the same cell 0.
        """
        return tuple(math.floor(x) for x in (point * self._inv_cell_size))

    def insert(self, point: np.ndarray) -> int:
        """
        Insert a point into the grid.

        Parameters
        ----------
        point : np.ndarray
            Point coordinates, shape (ndim,).

        Returns
        -------
        int
            Index of the inserted point.
        """
        idx = self._n_points
        # Grow backing array if needed (doubling strategy)
        if idx >= len(self._points):
            new_size = len(self._points) * 2
            new_arr = np.empty((new_size, self._ndim), dtype=np.float32)
            new_arr[:idx] = self._points[:idx]
            self._points = new_arr
        self._points[idx] = point
        self._n_points += 1

        key = self._cell_key(point)
        if key in self._grid:
            self._grid[key].append(idx)
        else:
            self._grid[key] = [idx]
        return idx

    def has_neighbor_within(self, point: np.ndarray, distance: float) -> bool:
        """
        Check if any stored point is within the given distance.

        Parameters
        ----------
        point : np.ndarray
            Query point coordinates, shape (ndim,).
        distance : float
            Maximum distance threshold. Must be <= cell_size for the
            3^ndim neighbor check to be correct.

        Returns
        -------
        bool
            True if any stored point is strictly closer than ``distance``.
        """
        dist_sq = distance * distance
        cell_key = self._cell_key(point)
        for offset in self._neighbor_offsets:
            key = tuple(cell_key[d] + offset[d] for d in range(self._ndim))
            bucket = self._grid.get(key)
            if bucket is not None:
                for j in bucket:
                    diff = self._points[j] - point
                    if np.dot(diff, diff) < dist_sq:
                        return True
        return False

    @property
    def points(self) -> np.ndarray:
        """Return a copy of all stored points, shape (n_points, ndim)."""
        result: np.ndarray = self._points[: self._n_points].copy()
        return result

    def __len__(self) -> int:
        """Return the number of stored points."""
        return self._n_points


# ─────────────────────────────────────────────────────────────────────
# Helpers shared by the batched class
# ─────────────────────────────────────────────────────────────────────


def _is_oom_error(exc: BaseException) -> bool:
    """Return True iff ``exc`` is a CUDA out-of-memory condition.

    Catches both :class:`torch.cuda.OutOfMemoryError` and the older
    string-formatted ``RuntimeError`` raised by some PyTorch builds.
    Used by :class:`BatchedSpatialHashGrid` to decide whether to
    auto-fall-back to the NumPy backend; **other** ``RuntimeError``s
    propagate (those are real bugs, not transient resource issues).
    """
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def _mps_available() -> bool:
    """Return True when PyTorch's MPS backend is available."""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def _resolve_device(
    device: Union[str, torch.device, None],
) -> torch.device:
    """Resolve ``device`` to a concrete ``torch.device``.

    ``None`` and ``'auto'`` both mean "best available accelerator,
    fall back to CPU". Mirrors
    :func:`luxar.gsplats.utils.device.resolve_torch_device` but lives
    here so this module stays import-safe from any layer.
    """
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _hash_keys_numpy(cells: np.ndarray) -> np.ndarray:
    """Pack per-axis cell indices into a single int64 hash key.

    ``cells`` has shape ``(N, D)`` and integer dtype. Returns shape
    ``(N,)`` of int64. Each axis gets its own large prime; the products
    are summed and reinterpreted as int64. Negative cell indices are
    handled correctly because we sum signed products.
    """
    cells = cells.astype(np.int64, copy=False)
    D = cells.shape[1]
    primes = np.asarray(
        [_HASH_PRIMES[d % len(_HASH_PRIMES)] for d in range(D)],
        dtype=np.int64,
    )
    keys: np.ndarray = (cells * primes[None, :]).sum(axis=1)
    return keys


def _hash_keys_torch(cells: torch.Tensor) -> torch.Tensor:
    """PyTorch port of :func:`_hash_keys_numpy`."""
    cells = cells.to(dtype=torch.int64)
    D = cells.shape[-1]
    primes = torch.tensor(
        [_HASH_PRIMES[d % len(_HASH_PRIMES)] for d in range(D)],
        dtype=torch.int64,
        device=cells.device,
    )
    return (cells * primes).sum(dim=-1)


def _cell_offsets(ndim: int, radius_in_cells: int = 1) -> np.ndarray:
    """All ``(2 * radius_in_cells + 1)^ndim`` integer cell offsets within a shell."""
    rng = list(range(-radius_in_cells, radius_in_cells + 1))
    offsets = list(itertools.product(rng, repeat=ndim))
    return np.asarray(offsets, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────
# Batched: BatchedSpatialHashGrid (NumPy + PyTorch backends)
# ─────────────────────────────────────────────────────────────────────


class BatchedSpatialHashGrid:
    """Batched spatial hash for fast radius / k-NN queries on a fixed point set.

    Build once from ``N`` points; then issue many queries. Two backends:

    - ``backend="numpy"``: NumPy ``argsort`` + ``searchsorted``.
      Per-query NumPy loop with vectorised inner distance computation.
    - ``backend="torch"``: state lives on a torch device; k-NN queries
      use a GPU-batched pad-and-prune kernel that scales well to large
      query sets ($Q \\sim N$). Radius queries route through the NumPy
      path (jagged output is awkward to vectorise on GPU; the existing
      radius-query call sites are small).

    The hash scheme is the same in both backends (``floor(x * inv_cell)``
    per axis, packed by per-axis Müller primes) so results are
    deterministic across backends modulo floating-point rounding.

    Construct via :meth:`from_points`.
    """

    def __init__(
        self,
        *,
        backend: Literal["numpy", "torch"],
        device: torch.device,
        cell_size: float,
        ndim: int,
        n_points: int,
        # numpy state (always present, used as fallback ground truth)
        points_np: np.ndarray,
        sorted_indices_np: np.ndarray,
        unique_keys_np: np.ndarray,
        unique_starts_np: np.ndarray,
        # torch state (present when backend="torch")
        points_t: Optional[torch.Tensor] = None,
        sorted_indices_t: Optional[torch.Tensor] = None,
        unique_keys_t: Optional[torch.Tensor] = None,
        unique_starts_t: Optional[torch.Tensor] = None,
    ) -> None:
        """Store prebuilt hash state. Do not call directly — use :meth:`from_points`.

        Holds the always-present NumPy state (points, sorted point indices,
        unique cell keys, and their per-cell start offsets) and, when
        ``backend="torch"``, the mirrored torch tensors resident on ``device``.
        """
        self._backend: Literal["numpy", "torch"] = backend
        self._device = device
        self._cell_size = float(cell_size)
        self._inv_cell_size = 1.0 / self._cell_size
        self._ndim = int(ndim)
        self._n_points = int(n_points)

        self._points_np = points_np
        self._sorted_indices_np = sorted_indices_np
        self._unique_keys_np = unique_keys_np
        self._unique_starts_np = unique_starts_np

        self._points_t = points_t
        self._sorted_indices_t = sorted_indices_t
        self._unique_keys_t = unique_keys_t
        self._unique_starts_t = unique_starts_t

    # ── Construction ────────────────────────────────────────────────

    @classmethod
    def from_points(
        cls,
        points: Union[np.ndarray, torch.Tensor],
        cell_size: float,
        *,
        device: Union[str, torch.device, None] = "auto",
        fallback_to_cpu: bool = True,
    ) -> "BatchedSpatialHashGrid":
        """Build a batched hash grid over ``points``.

        Parameters
        ----------
        points
            Shape ``(N, D)``. NumPy or torch; copied/converted as needed.
        cell_size
            Cell side length. Must be strictly positive. Radius queries
            require ``radius <= cell_size`` for correctness; k-NN queries
            expand the search shell automatically until enough candidates
            are found, but tighter ``cell_size`` (~the typical query
            radius) gives the best performance.
        device
            ``"auto"`` (default), ``"cpu"``, ``"cuda"``, ``"mps"``, or a
            ``torch.device``. ``"cpu"`` selects the NumPy backend.
            Anything else uses the PyTorch backend.
        fallback_to_cpu
            If True (default), build failures on the requested non-CPU
            device caused by out-of-memory or device-unavailable
            conditions log an Arbol warning and rebuild on the NumPy
            backend. Generic ``RuntimeError``s propagate (they are
            bugs, not resource issues).
        """
        if cell_size <= 0.0:
            raise ValueError(f"cell_size must be > 0, got {cell_size}")

        # Always build the NumPy view (cheap, used by the radius path
        # and as a single source of truth for points).
        if isinstance(points, torch.Tensor):
            points_np = points.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            points_np = np.ascontiguousarray(np.asarray(points), dtype=np.float32)
        if points_np.ndim != 2:
            raise ValueError(f"points must be 2-D (N, D); got shape {points_np.shape}")
        N, D = points_np.shape
        np_state = cls._build_numpy_state(points_np, cell_size)

        target_device = _resolve_device(device)
        if target_device.type == "cpu":
            return cls(
                backend="numpy",
                device=target_device,
                cell_size=cell_size,
                ndim=D,
                n_points=N,
                points_np=np_state["points_np"],
                sorted_indices_np=np_state["sorted_indices_np"],
                unique_keys_np=np_state["unique_keys_np"],
                unique_starts_np=np_state["unique_starts_np"],
            )

        # Try the torch backend. Fall back only on resource issues.
        try:
            torch_state = cls._build_torch_state(points_np, cell_size, target_device)
        except Exception as exc:
            unavailable = (
                target_device.type == "cuda" and not torch.cuda.is_available()
            ) or (target_device.type == "mps" and not _mps_available())
            if fallback_to_cpu and (_is_oom_error(exc) or unavailable):
                aprint(
                    "[BatchedSpatialHashGrid] GPU build failed "
                    f"({type(exc).__name__}: {exc}); falling back to NumPy backend."
                )
                return cls(
                    backend="numpy",
                    device=torch.device("cpu"),
                    cell_size=cell_size,
                    ndim=D,
                    n_points=N,
                    points_np=np_state["points_np"],
                    sorted_indices_np=np_state["sorted_indices_np"],
                    unique_keys_np=np_state["unique_keys_np"],
                    unique_starts_np=np_state["unique_starts_np"],
                )
            # [Python-OOS-triage / D-W1] Non-OOM GPU failures (kernel
            # bugs, indexing overflow, unsupported dtype) used to
            # propagate as a bare traceback with no indication that
            # the GPU path was the offender — the user saw a generic
            # PyTorch error and couldn't tell whether their data
            # triggered it or the GPU backend itself was at fault.
            # Log a clear "GPU build failed (not recoverable)" line
            # BEFORE re-raising so the cause is unambiguous.
            aprint(
                "[BatchedSpatialHashGrid] GPU build failed and is NOT recoverable "
                f"({type(exc).__name__}: {exc}); re-raising. "
                "Pass fallback_to_cpu=True to allow OOM/unavailable-device fallback only."
            )
            raise

        return cls(
            backend="torch",
            device=target_device,
            cell_size=cell_size,
            ndim=D,
            n_points=N,
            points_np=np_state["points_np"],
            sorted_indices_np=np_state["sorted_indices_np"],
            unique_keys_np=np_state["unique_keys_np"],
            unique_starts_np=np_state["unique_starts_np"],
            points_t=torch_state["points_t"],
            sorted_indices_t=torch_state["sorted_indices_t"],
            unique_keys_t=torch_state["unique_keys_t"],
            unique_starts_t=torch_state["unique_starts_t"],
        )

    @staticmethod
    def _build_numpy_state(
        points_np: np.ndarray, cell_size: float
    ) -> dict[str, np.ndarray]:
        """Compute sorted-key + per-cell start arrays in NumPy."""
        N = points_np.shape[0]
        if N == 0:
            return dict(
                points_np=points_np,
                sorted_indices_np=np.empty(0, dtype=np.int64),
                unique_keys_np=np.empty(0, dtype=np.int64),
                unique_starts_np=np.zeros(1, dtype=np.int64),
            )
        cells_np = np.floor(points_np / cell_size).astype(np.int64)
        keys = _hash_keys_numpy(cells_np)
        order = np.argsort(keys, kind="stable")
        sorted_keys_np = keys[order]
        sorted_indices_np = order
        unique_keys_np, first_idx = np.unique(sorted_keys_np, return_index=True)
        # ``unique_starts_np`` has length ``len(unique_keys) + 1`` with
        # the final entry equal to N — convenient for slicing.
        unique_starts_np = np.concatenate(
            [first_idx.astype(np.int64), np.array([N], dtype=np.int64)]
        )
        return dict(
            points_np=points_np,
            sorted_indices_np=sorted_indices_np,
            unique_keys_np=unique_keys_np,
            unique_starts_np=unique_starts_np,
        )

    @staticmethod
    def _build_torch_state(
        points_np: np.ndarray, cell_size: float, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Compute the torch-backend state on ``device``."""
        points_t = torch.from_numpy(points_np).to(device=device)
        N = points_t.shape[0]
        if N == 0:
            empty_int = torch.empty(0, dtype=torch.int64, device=device)
            return dict(
                points_t=points_t,
                sorted_indices_t=empty_int,
                unique_keys_t=empty_int,
                unique_starts_t=torch.zeros(1, dtype=torch.int64, device=device),
            )
        cells_t = torch.floor(points_t / cell_size).to(dtype=torch.int64)
        keys = _hash_keys_torch(cells_t)
        sorted_keys_t, sorted_indices_t = torch.sort(keys, stable=True)
        if N == 1:
            is_new = torch.ones(1, dtype=torch.bool, device=device)
        else:
            is_new = torch.cat(
                [
                    torch.ones(1, dtype=torch.bool, device=device),
                    sorted_keys_t[1:] != sorted_keys_t[:-1],
                ]
            )
        first_idx = torch.nonzero(is_new, as_tuple=False).flatten()
        unique_keys_t = sorted_keys_t[first_idx]
        unique_starts_t = torch.cat(
            [
                first_idx.to(dtype=torch.int64),
                torch.tensor([N], dtype=torch.int64, device=device),
            ]
        )
        return dict(
            points_t=points_t,
            sorted_indices_t=sorted_indices_t,
            unique_keys_t=unique_keys_t,
            unique_starts_t=unique_starts_t,
        )

    # ── Properties ─────────────────────────────────────────────────

    @property
    def backend(self) -> Literal["numpy", "torch"]:
        return self._backend

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def cell_size(self) -> float:
        return self._cell_size

    def __len__(self) -> int:
        return self._n_points

    # ── Queries ────────────────────────────────────────────────────

    def query_radius(
        self,
        query: Union[np.ndarray, torch.Tensor],
        radius: float,
    ) -> list[np.ndarray]:
        """Per-query indices of stored points within ``radius``.

        Parameters
        ----------
        query
            Shape ``(Q, D)``.
        radius
            Must satisfy ``radius <= cell_size`` for the $3^D$ scan to
            be exhaustive.

        Returns
        -------
        list of np.ndarray
            Length-Q list; entry ``i`` is the int64 array of indices
            into the stored point set whose distance to ``query[i]`` is
            ``< radius``. Always returned as NumPy regardless of
            backend (jagged output is more naturally a Python list).
        """
        if radius < 0.0:
            raise ValueError(f"radius must be >= 0, got {radius}")
        if radius > self._cell_size + 1e-12:
            raise ValueError(
                f"radius={radius} exceeds cell_size={self._cell_size}; "
                "the 3^D neighbour scan is no longer exhaustive. "
                "Rebuild with a larger cell_size."
            )
        query_np = self._normalise_query(query)
        Q = query_np.shape[0]
        if self._n_points == 0 or Q == 0:
            return [np.empty(0, dtype=np.int64) for _ in range(Q)]
        return self._query_radius_numpy(query_np, radius)

    def query_knn(
        self,
        query: Union[np.ndarray, torch.Tensor],
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """k-nearest-neighbour query.

        Returns ``(distances, indices)`` arrays of shape ``(Q, k)``.
        Distances are Euclidean, returned as ``float32``; indices are
        int64 references into the stored point set. If the stored set
        has fewer than ``k`` points, the trailing columns are filled
        with ``-1`` (indices) and ``+inf`` (distances). Each row is
        sorted in ascending distance.

        The search expands the cell-shell radius until at least ``k``
        candidates are found (≤3 expansions in practice for uniform
        densities); beyond that we brute-force the whole stored set.
        Tight ``cell_size`` ≈ typical kNN distance gives the best
        performance.

        On the torch backend the heavy work runs as a single batched
        GPU kernel (pad-and-prune); the per-query Python loop is
        avoided.
        """
        if k <= 0:
            raise ValueError(f"k must be >= 1, got {k}")
        query_np = self._normalise_query(query)
        Q = query_np.shape[0]
        if self._n_points == 0 or Q == 0:
            distances = np.full((Q, k), np.inf, dtype=np.float32)
            indices = np.full((Q, k), -1, dtype=np.int64)
            return distances, indices
        if self._backend == "torch":
            return self._query_knn_torch(query_np, k)
        return self._query_knn_numpy(query_np, k)

    # ── Internal helpers ───────────────────────────────────────────

    def _normalise_query(self, query: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Coerce ``query`` to a ``(Q, ndim)`` float32 NumPy array.

        Accepts a NumPy array or torch tensor; raises ``ValueError`` if the
        shape is not 2-D with a trailing dimension matching the grid's ndim.
        """
        if isinstance(query, torch.Tensor):
            query_np = query.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            query_np = np.ascontiguousarray(np.asarray(query), dtype=np.float32)
        if query_np.ndim != 2 or query_np.shape[1] != self._ndim:
            raise ValueError(
                f"query must have shape (Q, {self._ndim}); got {query_np.shape}"
            )
        return query_np

    def _gather_candidates_for_query(
        self, q: np.ndarray, radius_in_cells: int
    ) -> np.ndarray:
        """Return a 1-D int64 array of candidate indices for query ``q``.

        Scans the $(2r+1)^D$ cell shell around ``q``. Returns an empty
        array if no cell in the shell is populated.
        """
        D = self._ndim
        cell = np.floor(q / self._cell_size).astype(np.int64)
        offsets = _cell_offsets(D, radius_in_cells=radius_in_cells)
        neighbour_cells = cell[None, :] + offsets
        neighbour_keys = _hash_keys_numpy(neighbour_cells)
        unique_keys = self._unique_keys_np
        if unique_keys.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        positions = np.searchsorted(unique_keys, neighbour_keys)
        # Bound positions before indexing into unique_keys.
        positions_clamped = np.minimum(positions, unique_keys.shape[0] - 1)
        valid = (positions < unique_keys.shape[0]) & (
            unique_keys[positions_clamped] == neighbour_keys
        )
        if not valid.any():
            return np.empty(0, dtype=np.int64)
        hit_positions = positions[valid]
        unique_starts = self._unique_starts_np
        sorted_indices = self._sorted_indices_np
        buckets = [
            sorted_indices[unique_starts[pos] : unique_starts[pos + 1]]
            for pos in hit_positions
        ]
        return np.concatenate(buckets) if buckets else np.empty(0, dtype=np.int64)

    def _query_radius_numpy(
        self, query_np: np.ndarray, radius: float
    ) -> list[np.ndarray]:
        """Per-query radius search over the NumPy state (used by both backends).

        For each query point, gathers the ``3^D`` neighbour-cell shell and
        keeps the candidate indices whose squared distance is ``< radius^2``.
        """
        radius_sq = radius * radius
        result: list[np.ndarray] = []
        points_np = self._points_np
        for q in query_np:
            cand = self._gather_candidates_for_query(q, radius_in_cells=1)
            if cand.size == 0:
                result.append(np.empty(0, dtype=np.int64))
                continue
            diff = points_np[cand] - q[None, :]
            dist_sq = np.einsum("ij,ij->i", diff, diff)
            mask = dist_sq < radius_sq
            result.append(cand[mask])
        return result

    def _query_knn_numpy(
        self, query_np: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-query k-NN over the NumPy state (the CPU-backend path).

        Expands the cell shell until the k-th neighbour is provably correct,
        then writes the ascending top-k distances / indices into each output
        row (trailing slots left as ``+inf`` / ``-1`` when fewer than ``k``).
        """
        Q = query_np.shape[0]
        N = self._n_points
        points_np = self._points_np

        out_dist = np.full((Q, k), np.inf, dtype=np.float32)
        out_idx = np.full((Q, k), -1, dtype=np.int64)

        for qi, q in enumerate(query_np):
            cand = self._gather_for_correct_knn(q, k, N)
            if cand.size == 0:
                continue
            diff = points_np[cand] - q[None, :]
            dist_sq = np.einsum("ij,ij->i", diff, diff)
            self._fill_topk_row(out_dist[qi], out_idx[qi], cand, dist_sq, k)
        return out_dist, out_idx

    def _gather_for_correct_knn(self, q: np.ndarray, k: int, N: int) -> np.ndarray:
        """Gather candidates with shell expansion sized for *correct* kNN.

        Shell radius ``r`` guarantees finding all points within Euclidean
        distance ``r * cell_size`` of the query (proof: a point outside
        the shell lies in a cell whose nearest edge is at L∞ distance
        ``≥ r * cell_size`` from ``q``; Euclidean ≥ L∞). We expand
        until the kth-nearest candidate's distance is within this
        guaranteed-coverage radius, otherwise farther-out true neighbours
        could be missed.
        """
        cs = self._cell_size
        for radius_in_cells in range(1, _KNN_MAX_SHELL + 1):
            cand = self._gather_candidates_for_query(q, radius_in_cells)
            if cand.size >= N:
                return cand
            if cand.size < k:
                continue
            diff = self._points_np[cand] - q[None, :]
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            kth = np.partition(dist, kth=k - 1)[k - 1]
            if kth <= radius_in_cells * cs:
                return cand
        # Past the cap: brute-force the whole set.
        return np.arange(N, dtype=np.int64)

    @staticmethod
    def _fill_topk_row(
        out_dist_row: np.ndarray,
        out_idx_row: np.ndarray,
        cand: np.ndarray,
        dist_sq: np.ndarray,
        k: int,
    ) -> None:
        """Write the smallest-``k`` distances/indices into the output row."""
        if cand.shape[0] <= k:
            order = np.argsort(dist_sq, kind="stable")
            m = cand.shape[0]
            out_dist_row[:m] = np.sqrt(dist_sq[order])
            out_idx_row[:m] = cand[order]
        else:
            topk = np.argpartition(dist_sq, kth=k - 1)[:k]
            order = topk[np.argsort(dist_sq[topk], kind="stable")]
            out_dist_row[:] = np.sqrt(dist_sq[order])
            out_idx_row[:] = cand[order]

    # ── Internal: GPU kNN (pad-and-prune) ──────────────────────────

    def _query_knn_torch(
        self, query_np: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched GPU kNN.

        Strategy: build candidate indices per query in NumPy (cheap;
        only index arithmetic, no distances), pad to a uniform width,
        ship to GPU, then compute distances + ``torch.topk`` in one
        batched kernel.

        For queries whose total candidate count exceeds
        :data:`_GPU_KNN_MAX_PAD`, fall back to the NumPy per-query path
        for that subset only — this caps padded-matrix memory at
        ``Q * _GPU_KNN_MAX_PAD * 8 B``. The threshold is generous; in
        practice substitutive Lloyd workloads stay well below it.
        """
        Q = query_np.shape[0]
        N = self._n_points
        assert self._points_t is not None  # narrow Optional for mypy

        # 1) Gather candidate index arrays per query (NumPy, fast).
        per_q_cand: list[np.ndarray] = []
        per_q_size = np.empty(Q, dtype=np.int64)
        for qi, q in enumerate(query_np):
            cand = self._gather_for_correct_knn(q, k, N)
            per_q_cand.append(cand)
            per_q_size[qi] = cand.size

        max_cand = int(per_q_size.max()) if Q > 0 else 0
        if max_cand == 0:
            return (
                np.full((Q, k), np.inf, dtype=np.float32),
                np.full((Q, k), -1, dtype=np.int64),
            )

        out_dist = np.full((Q, k), np.inf, dtype=np.float32)
        out_idx = np.full((Q, k), -1, dtype=np.int64)

        # 2) Split queries into "GPU-padded" and "fallback per-query".
        oversize = per_q_size > _GPU_KNN_MAX_PAD
        gpu_mask = ~oversize
        gpu_indices = np.where(gpu_mask)[0]
        cpu_indices = np.where(oversize)[0]

        if gpu_indices.size > 0:
            self._knn_torch_padded_block(
                query_np=query_np,
                cand_lists=[per_q_cand[i] for i in gpu_indices],
                cand_sizes=per_q_size[gpu_indices],
                global_query_indices=gpu_indices,
                k=k,
                max_cand=min(max_cand, _GPU_KNN_MAX_PAD),
                out_dist=out_dist,
                out_idx=out_idx,
            )

        if cpu_indices.size > 0:
            # Fallback path for the heaviest queries — tiny in practice.
            for qi in cpu_indices:
                cand = per_q_cand[qi]
                diff = self._points_np[cand] - query_np[qi][None, :]
                dist_sq = np.einsum("ij,ij->i", diff, diff)
                self._fill_topk_row(out_dist[qi], out_idx[qi], cand, dist_sq, k)

        return out_dist, out_idx

    def _knn_torch_padded_block(
        self,
        *,
        query_np: np.ndarray,
        cand_lists: list[np.ndarray],
        cand_sizes: np.ndarray,
        global_query_indices: np.ndarray,
        k: int,
        max_cand: int,
        out_dist: np.ndarray,
        out_idx: np.ndarray,
    ) -> None:
        """GPU kernel: distance + top-k on a padded ``(B, max_cand)`` block.

        ``out_dist`` and ``out_idx`` are mutated in-place at rows
        ``global_query_indices``.
        """
        device = self._device
        assert self._points_t is not None
        B = len(cand_lists)
        if B == 0:
            return

        # Pad candidate indices to (B, max_cand) with index 0; mask with
        # bool of shape (B, max_cand). Padded entries get distance +inf
        # via the mask, so they never enter the top-k.
        cand_padded = np.zeros((B, max_cand), dtype=np.int64)
        cand_mask = np.zeros((B, max_cand), dtype=bool)
        for bi, cand in enumerate(cand_lists):
            sz = cand.shape[0]
            cand_padded[bi, :sz] = cand
            cand_mask[bi, :sz] = True

        cand_t = torch.from_numpy(cand_padded).to(device=device)
        mask_t = torch.from_numpy(cand_mask).to(device=device)
        sub_query_np = query_np[global_query_indices]
        query_t = torch.from_numpy(sub_query_np).to(device=device)

        # Gather candidate point coords: (B, max_cand, D)
        cand_points = self._points_t[cand_t]
        diff = cand_points - query_t.unsqueeze(1)
        dist_sq = (diff * diff).sum(dim=-1)
        inf_t = torch.tensor(float("inf"), dtype=dist_sq.dtype, device=device)
        dist_sq = torch.where(mask_t, dist_sq, inf_t)

        topk_k = min(k, max_cand)
        top_d, top_local = torch.topk(dist_sq, k=topk_k, largest=False)
        # top_d already sorted ascending? torch.topk(largest=False) doesn't
        # guarantee sorted order on all backends. Sort explicitly.
        sorted_d, sort_perm = torch.sort(top_d, dim=1)
        sorted_local = torch.gather(top_local, 1, sort_perm)
        sorted_idx_t = torch.gather(cand_t, 1, sorted_local)
        invalid = sorted_d == float("inf")
        sorted_idx_t = torch.where(
            invalid,
            torch.full_like(sorted_idx_t, -1),
            sorted_idx_t,
        )
        sorted_dist_t = torch.where(
            invalid,
            sorted_d,
            torch.sqrt(sorted_d),
        )

        # Write back to the appropriate rows of out_*.
        sub_dist = sorted_dist_t.detach().cpu().numpy().astype(np.float32, copy=False)
        sub_idx = sorted_idx_t.detach().cpu().numpy().astype(np.int64, copy=False)
        # If topk_k < k, rest of row stays at default inf / -1.
        out_dist[global_query_indices, :topk_k] = sub_dist
        out_idx[global_query_indices, :topk_k] = sub_idx
