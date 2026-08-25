"""View-independent ambient occlusion baked from a point-sampled density field.

Luxar's Points, Lines and GSplats are *emissive*: nothing in their shaders knows
that neighbouring geometry exists, so a dense shell accumulates into a flat glow
and the eye loses the shape. Ambient occlusion restores it by darkening elements
that sit inside the mass and leaving exposed ones bright.

Why this belongs at authoring time
----------------------------------
Occlusion is a scalar function of the geometry alone — "how enclosed is this
element" — so it is identical from every camera. That is what makes it safe to
bake: the answer computed once offline is the answer for every frame. A
*directional* key light is deliberately NOT offered here. Baking one fixes it in
world space, so orbiting to the unlit side darkens the scene for no reason; a key
light has to follow the camera, which makes it a shader concern (see the mesh
material's view-space key) rather than an authoring one.

Normals are **optional**, and nothing here invents them. Averaged over the full
sphere, occlusion needs no surface orientation at all, which is the right reading
for genuinely volumetric data — a light-sheet fit, a cloud — and avoids
estimating normals by local PCA, which is meaningless on such data. But where the
subject really is a surface and the caller already holds normals (mesh normals,
marching-cubes gradients, a distance-field gradient), passing them switches to a
cosine-weighted hemisphere and roughly doubles the discrimination: on a thin shell
the full sphere is dominated by the in-plane material every element shares. Either
way the result stays view-independent — a normal describes the surface, not the
camera.

Method
------
A windowed Beer-Lambert column integral over a spherical direction set:

1. Splat each element's mass onto a regular grid aligned so its third axis points
   along the sampled direction.
2. Integrate density along that axis over a **finite** window of ``radius`` world
   units, exclusive of the element's own cell.
3. Transmittance along that direction is ``exp(-tau)``; the ambient term is the
   mean transmittance over all directions — cosine-weighted into the hemisphere a
   normal faces, when normals are supplied.

The finite window is what makes this occlusion rather than a depth map. An
unbounded integral (correct when you are chasing a real light, as
``demo_volumetric_cloud`` does for its sun) reports how deep an element sits
inside the whole object, so two elements at equal depth read identically however
differently shaped their surroundings are — exactly the flat-sheet-versus-crevice
confusion AO is supposed to resolve.

Two details differ from the same integral written for a single light, and both
are corrections rather than preferences. The cell size is fixed **once** from the
data extent and reused for every direction, so no direction integrates at a
coarser scale than another and the result carries no directional bias. And the
anti-banding blur clamps at the grid edge instead of wrapping, so mass on one
face of the object cannot leak onto the opposite one.

References
----------
The transmittance-toward-a-direction formulation follows Harris & Lastra (2001),
"Real-Time Cloud Rendering", computed on a grid rather than by rendering from the
light's point of view — which suits a caller that already holds every element as
an array.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "bake_ambient_occlusion",
    "directional_optical_depth",
    "sphere_directions",
]

#: Fraction of the bounding-box diagonal used when ``radius`` is not supplied.
#: The single most consequential knob in this module: it sets the scale of
#: structure AO responds to. Too small and only the tightest creases darken; too
#: large and the integral degenerates toward the depth map described above.
DEFAULT_RADIUS_FRACTION = 0.05

#: Grid cells across the longest axis of the data. The occlusion window is
#: quantized to whole cells, so this bounds how finely ``radius`` can be
#: resolved.
DEFAULT_GRID_CELLS = 64

#: Opposed direction pairs. 12 is enough that the mean transmittance is smooth;
#: the cost is linear in this, and so is the memory — ``4 * n_elements *
#: n_directions`` bytes for the column integrals, doubled when ``normals`` are
#: supplied and a weight matrix of the same shape is built alongside.
DEFAULT_N_DIRECTIONS = 12

#: ``extinction="auto"`` solves for the scale that puts the *median* element at
#: this transmittance, so the population lands mid-range whatever the mass units
#: and however densely the object was sampled.
AUTO_TARGET_TRANSMITTANCE = 0.5


def sphere_directions(n_directions: int) -> np.ndarray:
    """Return ``n_directions`` roughly equidistributed unit vectors on a sphere.

    Directions come in opposed pairs, because a single grid pass yields the
    column integral in both ``+d`` and ``-d`` for free — the forward and backward
    windowed sums along the same axis. So the number of grid builds, which is
    what the runtime actually is, is half the direction count.

    Args:
        n_directions: Requested direction count; rounded **up** to an even
            number so the pairing is exact.

    Returns:
        ``(n, 3)`` float64 unit vectors, ``n`` even and ``>= n_directions``.
    """
    if n_directions < 1:
        raise ValueError(f"n_directions must be >= 1, got {n_directions}")

    n_axes = (int(n_directions) + 1) // 2
    # Fibonacci lattice over a HEMISPHERE (z uniform in (0, 1) is area-uniform
    # there). Each axis is then mirrored, so the full set covers the sphere.
    index = np.arange(n_axes, dtype=np.float64) + 0.5
    z = index / n_axes
    radial = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    phi = np.pi * (1.0 + np.sqrt(5.0)) * index
    axes = np.stack([radial * np.cos(phi), radial * np.sin(phi), z], axis=1)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return np.concatenate([axes, -axes], axis=0)


def _direction_frame(direction: np.ndarray) -> np.ndarray:
    """Orthonormal basis whose THIRD column is ``direction``.

    Right-multiplying positions by this maps them into a frame where integrating
    along axis 2 integrates along ``direction``.
    """
    w = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(w))
    if norm < 1e-12:
        raise ValueError("direction must be a non-zero vector")
    w = w / norm
    # Any seed not parallel to w works; switching on the largest component keeps
    # the cross product well-conditioned.
    seed = np.array([0.0, 0.0, 1.0]) if abs(w[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(seed, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return np.stack([u, v, w], axis=1)


def _blur_transverse(grid: np.ndarray) -> np.ndarray:
    """Separable 3-tap blur across the two axes TRANSVERSE to the integration axis.

    Stops the nearest-cell read-back from showing the grid. Clamped rather than
    wrapped: ``np.roll`` would carry mass across the bounding box and light one
    face of the object with the opposite face's material.

    Deliberately skips axis 2, the integration axis, and that is a correctness
    matter rather than an optimization. Blurring along the ray smears an
    element's OWN mass into the adjacent slices, which the column integral then
    counts — so every element partly occludes itself and the whole field acquires
    a constant pedestal that has to be dialled back out with a magic strength and
    floor. Blurring only transversely keeps the exclusion of the element's own
    slice exact: transverse smearing moves an element's mass into columns it is
    never read back from, and moves neighbours' mass only within the slice that
    is excluded anyway. Nothing is lost by skipping it — the ray is being summed
    along that axis regardless.
    """
    for axis in range(2):
        if grid.shape[axis] < 2:
            continue
        lo = np.concatenate(
            [
                np.take(grid, [0], axis=axis),
                np.take(grid, range(0, grid.shape[axis] - 1), axis=axis),
            ],
            axis=axis,
        )
        hi = np.concatenate(
            [
                np.take(grid, range(1, grid.shape[axis]), axis=axis),
                np.take(grid, [grid.shape[axis] - 1], axis=axis),
            ],
            axis=axis,
        )
        grid = (grid + 0.5 * lo + 0.5 * hi) / 2.0
    return grid


def _mass_grid(
    local: np.ndarray, mass: np.ndarray, cell: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scatter ``mass`` onto a fixed-cell grid aligned to the integration axis.

    Returns:
        ``(grid, indices, reference_cell_mass)``. The reference is the mean mass
        of an *occupied* cell measured BEFORE the blur, and is what makes the
        result independent of mass units and of how densely the object was
        sampled. Measuring it after the blur would instead average in the smeared
        halo of empty cells and drift with the blur kernel.
    """
    lo = local.min(axis=0)
    hi = local.max(axis=0)
    dims = np.maximum(np.ceil((hi - lo) / cell).astype(np.int64) + 1, 1)

    idx = np.clip(((local - lo) / cell).astype(np.int64), 0, dims - 1)
    flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]

    grid = np.zeros(int(dims.prod()), dtype=np.float64)
    np.add.at(grid, flat, mass)
    occupied = grid[grid > 0.0]
    reference = float(occupied.mean()) if occupied.size else 0.0

    return _blur_transverse(grid.reshape(tuple(int(d) for d in dims))), idx, reference


def _windowed_columns(
    grid: np.ndarray, idx: np.ndarray, window_cells: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Windowed mass sums over ``window_cells`` in ``+axis2`` and ``-axis2``.

    Both are **exclusive** of the element's own slice: an element is not occluded
    by the material it is itself made of. That exclusion is only exact because
    :func:`_blur_transverse` leaves the integration axis alone.

    Returns:
        ``(forward, backward)``, each ``(N,)``, as raw mass sums. Callers apply
        their own normalization — occupancy for the ambient term, a physical
        density-length product for the directional one.
    """
    n_slices = grid.shape[2]

    # suffix[..., k] = sum over j >= k, with a zero sentinel at k == n_slices.
    suffix = np.zeros(grid.shape[:2] + (n_slices + 1,), dtype=np.float64)
    suffix[..., :n_slices] = np.cumsum(grid[..., ::-1], axis=2)[..., ::-1]
    # prefix[..., k] = sum over j < k, with prefix[..., 0] == 0.
    prefix = np.zeros_like(suffix)
    np.cumsum(grid, axis=2, out=prefix[..., 1:])

    k = idx[:, 2]
    i, j = idx[:, 0], idx[:, 1]
    upper = np.minimum(k + 1 + window_cells, n_slices)
    lower = np.maximum(k - window_cells, 0)

    forward = suffix[i, j, np.minimum(k + 1, n_slices)] - suffix[i, j, upper]
    backward = prefix[i, j, k] - prefix[i, j, lower]
    return forward, backward


def _spatial_positions(
    positions: np.ndarray, spatial_dims: Sequence[int]
) -> np.ndarray:
    """Extract the three occluding axes as a contiguous ``(N, 3)`` float64 array.

    ``spatial_dims`` is mandatory in spirit even though it has a default: an nD
    node's non-spatial axes (time, channel) must never take part, or timepoint 0
    occludes timepoint 40 and the whole sequence shades as one solid. This is the
    same hazard ``--coarsen-dims`` exists for on the LOD side.
    """
    dims = list(spatial_dims)
    if len(dims) != 3:
        raise ValueError(f"spatial_dims must name exactly 3 axes, got {dims}")
    n_dims = positions.shape[1]
    for axis in dims:
        if not -n_dims <= axis < n_dims:
            raise ValueError(
                f"spatial_dims entry {axis} is out of range for {n_dims}-D positions"
            )
    if len({axis % n_dims for axis in dims}) != 3:
        raise ValueError(f"spatial_dims must name three distinct axes, got {dims}")
    return np.ascontiguousarray(positions[:, dims], dtype=np.float64)


def directional_optical_depth(
    positions: np.ndarray,
    mass: np.ndarray,
    direction: Sequence[float],
    *,
    radius: Optional[float] = None,
    grid_cells: int = DEFAULT_GRID_CELLS,
    extinction: float = 1.0,
    spatial_dims: Sequence[int] = (0, 1, 2),
) -> np.ndarray:
    """Optical depth accumulated from each element toward ``direction``.

    The building block :func:`bake_ambient_occlusion` averages over a direction
    set. Exposed because it is the honest primitive, **not** because it is a
    recommended appearance path: a single baked direction is locked to world
    space, so it stops reading correctly the moment the camera moves. Use it to
    model a light that genuinely is part of the subject (a sun above a cloud),
    and use :func:`bake_ambient_occlusion` for everything else.

    Args:
        positions: ``(N, D)`` element positions.
        mass: ``(N,)`` occluding material per element — GSplat amplitudes, or a
            Points radius cubed, or ones for pure count density.
        direction: Vector pointing *from* the elements *toward* the source.
        radius: World-space integration window; ``None`` integrates all the way
            to the bounding box, which is the right choice for a real light.
        grid_cells: Cells across the longest data axis.
        extinction: Extinction per unit of mass column.
        spatial_dims: Which three ``positions`` columns are the occluding axes.

    Returns:
        ``(N,)`` float32 optical depth.
    """
    local_all = _spatial_positions(positions, spatial_dims)
    mass_arr = _validated_mass(mass, len(local_all))
    if len(local_all) == 0:
        return np.empty(0, dtype=np.float32)

    cell = _cell_size(local_all, grid_cells)
    if radius is None:
        # Integrate to the bounding box. A window wider than the grid clamps to
        # the grid, so any bound past the longest possible column works.
        window = 4 * grid_cells
    else:
        window = _window_cells(local_all, radius, cell)

    local = local_all @ _direction_frame(np.asarray(direction, dtype=np.float64))
    grid, idx, _reference = _mass_grid(local, mass_arr, cell)
    forward, _backward = _windowed_columns(grid, idx, window)
    # A physical column: density (mass / cell**3) integrated over a path of
    # `cell` per slice, so the mass sum carries a 1 / cell**2 factor. Unlike the
    # ambient term this stays in the caller's own units, because a real light's
    # extinction is a property of the medium rather than a look knob.
    return (forward * extinction / (cell * cell)).astype(np.float32)


def bake_ambient_occlusion(
    positions: np.ndarray,
    *,
    mass: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    n_directions: int = DEFAULT_N_DIRECTIONS,
    grid_cells: int = DEFAULT_GRID_CELLS,
    spatial_dims: Sequence[int] = (0, 1, 2),
    group_by: Optional[np.ndarray] = None,
    extinction: Union[float, str] = "auto",
    strength: float = 0.7,
    floor: float = 0.0,
) -> np.ndarray:
    """Bake a per-element ambient-occlusion factor from the element density.

    The result is a **multiplier**: ``1.0`` where an element is out in the open,
    lower where it is enclosed. Multiply it into linear-light colour, or carry it
    alongside as a scalar so the strength stays adjustable.

    AO always lowers the mean brightness, so authored exposure has to absorb it —
    ``result.mean()`` is the factor to compensate for. Judging a bake by whether
    the frame merely looks crisper is how a darkening gets mistaken for detail.

    Args:
        positions: ``(N, D)`` element positions.
        mass: ``(N,)`` occluding material per element. Defaults to ones, i.e.
            pure count density. GSplats should pass ``amplitudes``; Points a
            radius-cubed volume if radii vary widely.
        normals: Optional ``(N, 3)`` outward normals, in the ``spatial_dims``
            frame. When given, directions are cosine-weighted into the hemisphere
            each normal faces instead of averaged over the full sphere — still
            entirely view-independent, since this needs a surface orientation and
            not a camera.

            **Supply these whenever the data is surface-like and you have them**
            (mesh normals, marching-cubes gradients, a distance-field gradient).
            On a thin shell the full sphere is dominated by the in-plane material
            every element shares, which washes the signal out: measured against
            the Mandelbulb's own distance-estimator AO over its 27k surface
            points (``n_directions=24``, ``grid_cells=96``), correlation rises
            from ``+0.31`` to ``+0.61`` at a 0.05 radius and from ``+0.44`` to
            ``+0.68`` at 0.20.

            Left ``None``, no normals are needed or invented. That is the right
            default for genuinely volumetric data — a light-sheet fit, a cloud —
            where there is no surface to orient to, and it is deliberately not
            papered over with an estimate: normals from a local PCA of a
            volumetric point cloud are meaningless.
        radius: World-space occlusion radius — the scale of structure AO
            responds to. Defaults to :data:`DEFAULT_RADIUS_FRACTION` of the
            bounding-box diagonal, and is the first thing to tune.
        n_directions: Sphere directions to average; rounded up to even.
        grid_cells: Cells across the longest data axis.
        spatial_dims: Which three ``positions`` columns are the occluding axes.
            Everything else — time, channel — must be excluded, and is normally
            excluded via ``group_by`` as well.
        group_by: ``(N,)`` integer labels; each group is baked independently so
            occlusion never crosses a non-spatial axis. Pass the timepoint index
            for a timelapse.
        extinction: Occlusion per cell of typical material traversed — an O(1)
            quantity, unlike the directional function's extinction, because the
            column here is normalized by the reference density. Useful explicit
            values sit around ``0.3`` – ``1.0``; much above that the field
            saturates against ``1 - strength`` and stops discriminating. Or
            ``"auto"`` to solve for the value putting the median element at
            :data:`AUTO_TARGET_TRANSMITTANCE`, which is the default because it
            makes the bake independent of the mass units and of how densely the
            object was sampled.
        strength: Scales the darkening; ``0.0`` returns all ones.
        floor: Lower clamp, so fully enclosed elements keep some brightness.

    Returns:
        ``(N,)`` float32 in ``[floor, 1]``.
    """
    pos = np.asarray(positions)
    if pos.ndim != 2:
        raise ValueError(f"positions must have shape (N, D), got {pos.shape}")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    if strength < 0.0:
        raise ValueError(f"strength must be >= 0, got {strength}")
    if grid_cells < 4:
        raise ValueError(f"grid_cells must be >= 4, got {grid_cells}")
    if radius is not None and radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")

    n_elements = pos.shape[0]
    mass_arr = _validated_mass(mass, n_elements)
    normals_arr = _validated_normals(normals, n_elements)
    if n_elements == 0:
        return np.empty(0, dtype=np.float32)

    # Keywords are spelled out at both call sites rather than forwarded through a
    # shared dict: `**dict(...)` erases the argument types and mypy stops checking
    # them entirely.
    if group_by is None:
        return _bake_group(
            _spatial_positions(pos, spatial_dims),
            mass_arr,
            normals_arr,
            radius=radius,
            n_directions=n_directions,
            grid_cells=grid_cells,
            extinction=extinction,
            strength=strength,
            floor=floor,
        )

    groups = np.asarray(group_by)
    if groups.shape != (n_elements,):
        raise ValueError(
            f"group_by must have shape ({n_elements},), got {groups.shape}"
        )
    local_all = _spatial_positions(pos, spatial_dims)
    result = np.ones(n_elements, dtype=np.float32)
    for label in np.unique(groups):
        where = np.flatnonzero(groups == label)
        result[where] = _bake_group(
            local_all[where],
            mass_arr[where],
            None if normals_arr is None else normals_arr[where],
            radius=radius,
            n_directions=n_directions,
            grid_cells=grid_cells,
            extinction=extinction,
            strength=strength,
            floor=floor,
        )
    return result


def _bake_group(
    local_all: np.ndarray,
    mass: np.ndarray,
    normals: Optional[np.ndarray],
    *,
    radius: Optional[float],
    n_directions: int,
    grid_cells: int,
    extinction: Union[float, str],
    strength: float,
    floor: float,
) -> np.ndarray:
    """Bake one independent group. ``local_all`` is already ``(N, 3)`` float64."""
    n_elements = len(local_all)
    if n_elements == 0:
        return np.empty(0, dtype=np.float32)

    cell = _cell_size(local_all, grid_cells)
    window = _window_cells(local_all, radius, cell)
    directions = sphere_directions(n_directions)
    n_axes = len(directions) // 2

    # Column integrals for every direction, kept so the extinction calibration
    # can run before the exponential. Memory is 4 * N * n_directions bytes; that
    # is the price of a single pass and `n_directions` is the knob.
    columns = np.empty((n_elements, 2 * n_axes), dtype=np.float32)
    for axis_index in range(n_axes):
        local = local_all @ _direction_frame(directions[axis_index])
        grid, idx, reference = _mass_grid(local, mass, cell)
        forward, backward = _windowed_columns(grid, idx, window)
        # Express the column in units of "cells of typical material traversed"
        # rather than in the caller's mass units. That keeps `extinction` an O(1)
        # look knob instead of a per-dataset magic constant whose right value is
        # in the thousands and moves with the element count, and it makes the
        # result independent of both mass units and sampling density (the
        # reference tracks both).
        #
        # Deliberately a SUM over the window, NOT a mean. Dividing by the window
        # would make a THIN occluder — a single-cell-thick sheet, which is what a
        # surface-sampled point cloud is made of — block only 1/window of the
        # light, and block less the wider the radius. A one-cell wall must block
        # like a wall regardless of how far the window looks past it.
        scale = 1.0 / max(reference, 1e-12)
        columns[:, axis_index] = forward * scale
        columns[:, n_axes + axis_index] = backward * scale

    weights = _direction_weights(normals, directions)

    if isinstance(extinction, str):
        if extinction != "auto":
            raise ValueError(
                f"extinction must be a float or 'auto', got {extinction!r}"
            )
        resolved = _auto_extinction(columns, weights)
    else:
        if extinction < 0.0:
            raise ValueError(f"extinction must be >= 0, got {extinction}")
        resolved = float(extinction)

    transmittance = _combine(np.exp(-resolved * columns.astype(np.float64)), weights)
    return np.clip(1.0 - strength * (1.0 - transmittance), floor, 1.0).astype(
        np.float32
    )


def _direction_weights(
    normals: Optional[np.ndarray], directions: np.ndarray
) -> Optional[np.ndarray]:
    """Cosine-lobe weights over the hemisphere each normal faces, or ``None``.

    ``None`` means "average the full sphere equally", which is the volumetric
    reading of ambient occlusion. An element whose normal is degenerate (zero
    length, as a distance-field gradient can be at a critical point) gets a
    uniform row rather than an all-zero one, so it falls back to the full sphere
    instead of producing a division by zero.
    """
    if normals is None:
        return None

    weights = np.asarray(np.clip(normals @ directions.T, 0.0, None))
    total = weights.sum(axis=1)
    degenerate = total <= 1e-12
    if np.any(degenerate):
        weights[degenerate] = 1.0
    return weights


def _combine(per_direction: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    """Weighted (or plain) mean of a per-element, per-direction quantity."""
    if weights is None:
        return np.asarray(per_direction.mean(axis=1))
    return np.asarray((per_direction * weights).sum(axis=1) / weights.sum(axis=1))


def _auto_extinction(columns: np.ndarray, weights: Optional[np.ndarray]) -> float:
    """Extinction placing the median element at :data:`AUTO_TARGET_TRANSMITTANCE`.

    Calibrated on each element's mean column over directions — under the SAME
    weighting the final combination uses, so a hemisphere-weighted bake is not
    calibrated against a full-sphere population it never evaluates. The exact
    median transmittance lands slightly above the target (Jensen's inequality on
    the exponential). Close enough for a look parameter, and it is what makes the
    same call work unchanged on a 50k-point sketch and a 3M-splat fit.
    """
    median_column = float(np.median(_combine(columns.astype(np.float64), weights)))
    if median_column <= 0.0:
        # Every element sees an empty window: the data is sparser than `radius`,
        # so there is genuinely nothing to occlude with.
        return 0.0
    return float(-np.log(AUTO_TARGET_TRANSMITTANCE) / median_column)


def _validated_normals(
    normals: Optional[np.ndarray], n_elements: int
) -> Optional[np.ndarray]:
    """Normalize supplied normals to unit length, or pass ``None`` through."""
    if normals is None:
        return None
    arr = np.asarray(normals, dtype=np.float64)
    if arr.shape != (n_elements, 3):
        raise ValueError(f"normals must have shape ({n_elements}, 3), got {arr.shape}")
    lengths = np.linalg.norm(arr, axis=1, keepdims=True)
    # Zero-length rows are kept as zeros rather than rejected; a distance-field
    # gradient legitimately vanishes at a critical point, and
    # `_direction_weights` turns such a row into a full-sphere fallback.
    return np.asarray(
        np.divide(arr, lengths, out=np.zeros_like(arr), where=lengths > 1e-12)
    )


def _validated_mass(mass: Optional[np.ndarray], n_elements: int) -> np.ndarray:
    if mass is None:
        return np.ones(n_elements, dtype=np.float64)
    arr = np.asarray(mass, dtype=np.float64)
    if arr.shape != (n_elements,):
        raise ValueError(f"mass must have shape ({n_elements},), got {arr.shape}")
    if np.any(arr < 0.0):
        raise ValueError("mass must be non-negative")
    return arr


def _cell_size(local: np.ndarray, grid_cells: int) -> float:
    """One isotropic cell size, fixed across every sampled direction.

    Deriving it per direction from that direction's own rotated bounding box —
    the natural thing to write when there is only one direction — makes each
    direction integrate at its own scale and gives the mean a directional bias.
    """
    span = local.max(axis=0) - local.min(axis=0)
    longest = float(np.max(span))
    if longest <= 0.0:
        # Degenerate: every element at one point. Any positive cell works.
        return 1.0
    return longest / float(grid_cells)


def _window_cells(local: np.ndarray, radius: Optional[float], cell: float) -> int:
    """Occlusion window in whole cells, at least one."""
    if radius is None:
        span = local.max(axis=0) - local.min(axis=0)
        diagonal = float(np.linalg.norm(span))
        radius = DEFAULT_RADIUS_FRACTION * diagonal
    if radius <= 0.0:
        return 1
    return max(1, int(round(radius / cell)))
