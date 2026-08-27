"""View-independent ambient occlusion baked from a point-sampled density field.

Luxar's Points, Lines and GSplats are *emissive*: nothing in their shaders knows
that neighbouring geometry exists, so a dense shell accumulates into a flat glow
and the eye loses the shape. Ambient occlusion restores it by darkening elements
that sit inside the mass and leaving exposed ones bright.

Emissivity as a function of ambient illumination
------------------------------------------------
The useful way to read this is that it is not decoration bolted onto the
renderer, but the missing half of the transport it already implements.
Emission-absorption rendering has two terms. Luxar's blending modes supply the
attenuation one: radiance is absorbed on its way OUT to the eye. The emission
term is the other, and for matter that is **lit from outside** rather than
genuinely glowing, the physically correct source is ``albedo x incident
irradiance`` — and the incident irradiance at a point is precisely what ambient
occlusion measures, the fraction of the surrounding environment that can reach
it. So an emissive scene is best understood as one whose emissivity is a function
of ambient illumination, and this module computes that function.

Three consequences follow, and they are why the API looks the way it does. The
result belongs multiplied into the **emission** (colour x intensity), never into
opacity or absorption, which are the other term. It composes with an absorbing
blending mode rather than double-counting it, because in-scattered source and
outgoing attenuation are different halves of one equation. And ``strength``
stops being a taste knob: ``1 - strength`` is the *indirect* ambient, the
multiply-scattered light that reaches even a fully enclosed point, which is the
same quantity ``demo_volumetric_cloud`` spends three tuned radiance terms on and
that ``demo_mandelbulb`` writes as its ambient floor of 0.32.

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
3. Map that column to a per-direction transmittance — ``exp(-tau)`` for a medium,
   or a saturating ``max(0, 1 - tau)`` for an opaque surface (see ``occluder``).
   The ambient term is the mean over all directions, cosine-weighted into the
   hemisphere a normal faces when normals are supplied.

Which mapping matters more than it sounds. Under Beer-Lambert a one-cell-thick
shell — what a surface sampled as points is made of — attenuates only by
``exp(-k)``, so at any ``k`` gentle enough to keep solid regions readable a WALL
passes about half the light, and raising ``k`` until walls block properly
over-darkens everywhere thick. Saturation has no such trade: it reaches full
occlusion at one wall and stops. So volumetric data wants ``"density"`` and
surfaces want ``"opaque"``.

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

from typing import Iterator, Optional, Sequence, Tuple, Union

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

#: Opposed direction pairs used for a full-sphere bake. The cost is linear in
#: this count. Persistent direction arrays cost ``4 * N * D`` bytes for a
#: full-sphere bake or ``8 * N * D`` with hemisphere weights. The measured
#: ``O(N)`` indexing and grid temporaries add approximately 145 bytes per
#: element, and group assembly can transiently add ``4 * N_group * D`` bytes.
#: Shading and weighted combination limit each work array to 32 MiB rather than
#: another full array; at ``N=600k, D=24`` the measured weighted-shading peak is
#: 1.255x one full direction array. The rotated grid's worst-case volume is
#: approximately ``(sqrt(3) * grid_cells) ** 3`` cells.
DEFAULT_N_DIRECTIONS = 24

#: Hemisphere weighting converges more slowly than a plain sphere average, so
#: surface bakes use the measured 48-direction floor unless explicitly overridden.
DEFAULT_NORMAL_N_DIRECTIONS = 48

#: Byte budget for temporary ``(N, n_directions)`` work arrays. Row slabs are
#: sized from this budget while preserving each row's reduction order exactly.
_ROW_SLAB_BYTES = 32 * 1024 * 1024

#: ``extinction="auto"`` solves for the scale that puts the *median* element at
#: this transmittance, so the population lands in a useful range whatever the
#: mass units and however densely the object was sampled.
#:
#: Chosen by measurement, against two deliberately different regimes — a thin
#: gyroid shell read through the hemisphere path and a solid ball read through the
#: full sphere — since a target tuned on one shape would not be a general default.
#: Contrast is std/mean of the multiplier at ``strength=1.0``; p5 is the guard,
#: because contrast that climbs while p5 collapses is climbing by CRUSHING the
#: dark end to black rather than by revealing anything:
#:
#: ===============  ==================  ==================
#: target           shell contrast/p5   ball contrast/p5
#: ===============  ==================  ==================
#: 0.50             0.198 / 0.398       0.150 / 0.454
#: 0.35             0.273 / 0.273       0.247 / 0.304
#: **0.25**         **0.335 / 0.195**   **0.351 / 0.209**
#: 0.15             0.417 / 0.120       0.531 / 0.119
#: 0.10             0.474 / 0.083       0.688 / 0.076
#: ===============  ==================  ==================
#:
#: 0.25 roughly doubles the contrast of the original 0.5 in BOTH regimes while
#: holding p5 near 0.2 — dark but still carrying colour. Below it the gain comes
#: from clipping.
AUTO_TARGET_TRANSMITTANCE = 0.25


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
    local = np.ascontiguousarray(positions[:, dims], dtype=np.float64)
    if not np.all(np.isfinite(local)):
        raise ValueError("spatial positions must be finite")
    return local


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
    pos = np.asarray(positions)
    _validate_common_args(pos, radius=radius, grid_cells=grid_cells)
    local_all = _spatial_positions(pos, spatial_dims)
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
    occluder: str = "density",
    radius: Optional[float] = None,
    n_directions: Optional[int] = None,
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
            pure count density.

            **This is the only channel through which an element's own appearance
            enters.** Nothing here reads a node's radii, sharpness or opacity —
            per geometry type, the quantity to pass is:

            ============  ==================================================
            GSplats       ``amplitudes`` (times per-splat alpha if RGBA)
            Points        ``radii ** 3`` when radii vary; else leave ``None``
            Lines         ``widths ** 2 * segment_length`` per vertex
            Mesh          leave ``None`` — a vertex has no extent of its own
            ============  ==================================================

            Two things deliberately do NOT need folding in. A *uniform* factor —
            node opacity, or a constant radius or sharpness — cancels out
            entirely, because ``extinction="auto"`` calibrates against the
            population's own median. And sharpness only changes an element's
            profile SHAPE, which is a modest constant unless it varies per
            element, in which case fold it into ``mass`` yourself.

            One real approximation to know about: mass is deposited at each
            element's centre, so an element's extent is a weight and not a
            footprint. That holds while the render radius is small next to the
            occlusion grid cell (``extent / grid_cells``) — measured at 0.40,
            0.18 and 0.44 of a cell in the three bundled point demos at their
            default resolutions. Raise ``grid_cells``, or splat pre-spread mass
            yourself, if your elements are large enough to span cells.
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
        occluder: What the material is taken to BE, which decides how a column
            maps to transmittance.

            ``"density"`` (default) is Beer-Lambert, ``exp(-depth)`` — correct for
            a medium, where twice the material attenuates twice as much without
            limit. Right for a light-sheet fit, a cloud, a filled molecular
            complex.

            ``"opaque"`` is saturating, ``max(0, 1 - depth)`` — correct for a
            surface, where once a direction is blocked it cannot become more
            blocked, so a thick wall darkens exactly as much as a thin one. Prefer
            it whenever the subject is a **surface sampled as points**.

            The distinction is not cosmetic. Under Beer-Lambert a one-cell-thick
            shell — which is what a surface-sampled point cloud is made of — only
            attenuates by ``exp(-k)``, so at any ``k`` gentle enough to keep solid
            regions readable a WALL passes about half the light, and raising ``k``
            until walls block properly over-darkens everywhere thick. Saturation
            has no such trade. Measured on the shipped gyroid shell,
            ``"opaque"`` carries about 10% more contrast than ``"density"`` at
            the demo settings and about 13% more at the library defaults, at the
            cost of clipping the darkest directions.
        radius: World-space occlusion radius — the scale of structure AO
            responds to. Defaults to :data:`DEFAULT_RADIUS_FRACTION` of the
            bounding-box diagonal, and is the first thing to tune.
        n_directions: Sphere directions to average; rounded up to even. Defaults
            to 24 for a full-sphere bake and 48 when ``normals`` are supplied.
        grid_cells: Cells across the longest data axis.
        spatial_dims: Which three ``positions`` columns are the occluding axes.
            Everything else — time, channel — must be excluded, and is normally
            excluded via ``group_by`` as well.
        group_by: ``(N,)`` integer labels; geometry is integrated independently
            per group so occlusion never crosses a non-spatial axis, while cell
            size, radius and auto extinction stay shared across the population.
            Pass the timepoint index for a timelapse.
        extinction: Occlusion per cell of typical material traversed — an O(1)
            quantity, unlike the directional function's extinction, because the
            column here is normalized by the reference density. Useful explicit
            values sit around ``0.3`` – ``1.0``; much above that the field
            saturates against ``1 - strength`` and stops discriminating. Or
            ``"auto"`` to solve for the value putting the median element at
            :data:`AUTO_TARGET_TRANSMITTANCE`, which is the default because it
            makes the bake independent of the mass units and of how densely the
            object was sampled.
        strength: Fraction of the ambient illumination that is DIRECT, and so
            occludable; ``1 - strength`` is the indirect, multiply-scattered
            ambient that reaches even a fully enclosed element. ``0.0`` returns
            all ones (everything reached by indirect light alone).
        floor: Lower clamp, so fully enclosed elements keep some brightness.

    Returns:
        ``(N,)`` float32 in ``[floor, 1]``.
    """
    pos = np.asarray(positions)
    _validate_look_args(
        pos,
        occluder=occluder,
        radius=radius,
        grid_cells=grid_cells,
        strength=strength,
        floor=floor,
    )

    n_elements = pos.shape[0]
    mass_arr = _validated_mass(mass, n_elements)
    normals_arr = _validated_normals(normals, n_elements)
    if n_elements == 0:
        return np.empty(0, dtype=np.float32)

    resolved_n_directions = (
        DEFAULT_NORMAL_N_DIRECTIONS
        if n_directions is None and normals_arr is not None
        else DEFAULT_N_DIRECTIONS
        if n_directions is None
        else n_directions
    )
    directions = sphere_directions(resolved_n_directions)
    local_all = _spatial_positions(pos, spatial_dims)
    cell = _cell_size(local_all, grid_cells)
    window = _window_cells(local_all, radius, cell)
    if group_by is None:
        columns = _group_columns(
            local_all,
            mass_arr,
            directions=directions,
            cell=cell,
            window=window,
        )
    else:
        groups = np.asarray(group_by)
        if groups.shape != (n_elements,):
            raise ValueError(
                f"group_by must have shape ({n_elements},), got {groups.shape}"
            )
        try:
            groups_are_finite = bool(np.all(np.isfinite(groups)))
        except TypeError as exc:
            raise ValueError("group_by must contain finite numeric labels") from exc
        if not groups_are_finite:
            raise ValueError("group_by must be finite")
        columns = np.empty((n_elements, len(directions)), dtype=np.float32)
        for label in np.unique(groups):
            where = np.flatnonzero(groups == label)
            columns[where] = _group_columns(
                local_all[where],
                mass_arr[where],
                directions=directions,
                cell=cell,
                window=window,
            )

    weights = _direction_weights(normals_arr, directions)
    resolved_extinction = _resolve_extinction(extinction, columns, weights, occluder)
    return _shade_columns(
        columns,
        weights,
        occluder=occluder,
        extinction=resolved_extinction,
        strength=strength,
        floor=floor,
    )


def _group_columns(
    local_all: np.ndarray,
    mass: np.ndarray,
    *,
    directions: np.ndarray,
    cell: float,
    window: int,
) -> np.ndarray:
    """Build columns for one isolated group on a shared spatial calibration."""
    n_elements = len(local_all)
    n_axes = len(directions) // 2
    columns = np.empty((n_elements, len(directions)), dtype=np.float32)
    for axis_index in range(n_axes):
        local = local_all @ _direction_frame(directions[axis_index])
        grid, idx, reference = _mass_grid(local, mass, cell)
        forward, backward = _windowed_columns(grid, idx, window)
        scale = 1.0 / max(reference, 1e-12)
        columns[:, axis_index] = forward * scale
        columns[:, n_axes + axis_index] = backward * scale
    return columns


def _resolve_extinction(
    extinction: Union[float, str],
    columns: np.ndarray,
    weights: Optional[np.ndarray],
    occluder: str,
) -> float:
    if isinstance(extinction, str):
        if extinction != "auto":
            raise ValueError(
                f"extinction must be a float or 'auto', got {extinction!r}"
            )
        return _auto_extinction(columns, weights, occluder)
    if extinction < 0.0:
        raise ValueError(f"extinction must be >= 0, got {extinction}")
    return float(extinction)


def _shade_columns(
    columns: np.ndarray,
    weights: Optional[np.ndarray],
    *,
    occluder: str,
    extinction: float,
    strength: float,
    floor: float,
) -> np.ndarray:
    transmittance = np.empty(len(columns), dtype=np.float32)
    for start, stop in _row_slabs(len(columns), columns.shape[1]):
        mapped = columns[start:stop].copy()
        mapped *= extinction
        _transmittance(mapped, occluder)
        slab_weights = None if weights is None else weights[start:stop]
        transmittance[start:stop] = _combine(mapped, slab_weights)
    return np.asarray(
        np.clip(1.0 - strength * (1.0 - transmittance), floor, 1.0),
        dtype=np.float32,
    )


def _validate_look_args(
    positions: np.ndarray,
    *,
    occluder: str,
    radius: Optional[float],
    grid_cells: int,
    strength: float,
    floor: float,
) -> None:
    """Reject bad arguments up front, before any grid pass runs.

    Split out of :func:`bake_ambient_occlusion` to keep that function under the
    repo's complexity gate, and because failing fast matters here: a typo in
    ``occluder`` caught only where it is consumed would surface after paying for
    every direction's grid build.
    """
    _validate_common_args(positions, radius=radius, grid_cells=grid_cells)
    if occluder not in ("density", "opaque"):
        raise ValueError(f"occluder must be 'density' or 'opaque', got {occluder!r}")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    if strength < 0.0:
        raise ValueError(f"strength must be >= 0, got {strength}")


def _validate_common_args(
    positions: np.ndarray, *, radius: Optional[float], grid_cells: int
) -> None:
    if positions.ndim != 2:
        raise ValueError(f"positions must have shape (N, D), got {positions.shape}")
    if grid_cells < 4:
        raise ValueError(f"grid_cells must be >= 4, got {grid_cells}")
    if radius is not None and radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")


def _transmittance(depth: np.ndarray, occluder: str) -> np.ndarray:
    """Map a scaled column integral to per-direction transmittance.

    Both models take the SAME scaled column and differ only in the mapping, which
    is what lets one ``extinction`` knob serve both:

    ``"density"``
        ``exp(-depth)`` — Beer-Lambert. Correct for a medium: twice the material
        attenuates twice as much, without limit.
    ``"opaque"``
        ``max(0, 1 - depth)`` — saturating. Correct for a surface: once a
        direction is blocked it cannot become more blocked, so a thick wall
        darkens exactly as much as a thin one.

    The saturation is the whole point of the second mode, not an approximation of
    the first. Under Beer-Lambert a one-cell-thick shell — which is what a
    surface-sampled point cloud is made of — only attenuates by ``exp(-k)``, so
    at any ``k`` gentle enough to keep thick regions readable a WALL passes
    roughly half the light. Raising ``k`` until walls block properly then
    over-darkens everywhere the geometry is solid. A saturating mapping has no
    such trade: it reaches full occlusion at one wall and stops.
    """
    if occluder == "density":
        np.negative(depth, out=depth)
        np.exp(depth, out=depth)
        return np.asarray(depth)
    if occluder == "opaque":
        np.negative(depth, out=depth)
        depth += 1.0
        np.maximum(depth, 0.0, out=depth)
        return np.asarray(depth)
    raise ValueError(f"occluder must be 'density' or 'opaque', got {occluder!r}")


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

    normals32 = normals.astype(np.float32, copy=False)
    directions32 = directions.astype(np.float32, copy=False)
    weights = np.asarray(normals32 @ directions32.T)
    np.clip(weights, 0.0, None, out=weights)
    total = weights.sum(axis=1)
    degenerate = total <= 1e-12
    if np.any(degenerate):
        weights[degenerate] = 1.0
    return weights


def _combine(per_direction: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    """Weighted (or plain) mean of a per-element, per-direction quantity."""
    if weights is None:
        return np.asarray(per_direction.mean(axis=1))
    combined = np.empty(len(per_direction), dtype=per_direction.dtype)
    for start, stop in _row_slabs(len(per_direction), per_direction.shape[1]):
        slab_weights = weights[start:stop]
        combined[start:stop] = (per_direction[start:stop] * slab_weights).sum(
            axis=1
        ) / slab_weights.sum(axis=1)
    return combined


def _row_slabs(n_rows: int, n_directions: int) -> Iterator[Tuple[int, int]]:
    rows_per_slab = max(1, _ROW_SLAB_BYTES // (n_directions * 4))
    for start in range(0, n_rows, rows_per_slab):
        yield start, min(start + rows_per_slab, n_rows)


def _auto_extinction(
    columns: np.ndarray, weights: Optional[np.ndarray], occluder: str
) -> float:
    """Extinction placing the median element at :data:`AUTO_TARGET_TRANSMITTANCE`.

    Calibrated on each element's mean column over directions — under the SAME
    weighting the final combination uses, so a hemisphere-weighted bake is not
    calibrated against a full-sphere population it never evaluates. The realized
    median lands above the target because the mapping is applied per direction
    before averaging. The gap is shape-dependent and can be large on thin shells,
    especially when ``"opaque"`` saturates. This remains a look calibration, not
    a promise that the realized median equals the target exactly.

    Inverted per mode, so both land on the same declared target rather than one
    of them silently aiming somewhere else.
    """
    median_column = float(np.median(_combine(columns, weights)))
    if median_column <= 0.0:
        # Every element sees an empty window: the data is sparser than `radius`,
        # so there is genuinely nothing to occlude with.
        return 0.0
    if occluder == "density":
        # exp(-k * col) == target
        return float(-np.log(AUTO_TARGET_TRANSMITTANCE) / median_column)
    if occluder == "opaque":
        # 1 - k * col == target
        return float((1.0 - AUTO_TARGET_TRANSMITTANCE) / median_column)
    raise ValueError(f"occluder must be 'density' or 'opaque', got {occluder!r}")


def _validated_normals(
    normals: Optional[np.ndarray], n_elements: int
) -> Optional[np.ndarray]:
    """Normalize supplied normals to unit length, or pass ``None`` through."""
    if normals is None:
        return None
    arr = np.asarray(normals, dtype=np.float64)
    if arr.shape != (n_elements, 3):
        raise ValueError(f"normals must have shape ({n_elements}, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("normals must be finite")
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
    if not np.all(np.isfinite(arr)):
        raise ValueError("mass must be finite")
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
