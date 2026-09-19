def assert_only_geometry_leaves(group, geometry: str) -> None:
    forbidden = {
        "centers",
        "amplitudes",
        "cholesky_factors_diag",
        "cholesky_factors_offdiag",
    }
    for name in group.group_keys():
        child = group[name]
        if child.attrs.get("kind"):
            assert_only_geometry_leaves(child, geometry)
        else:
            assert child.attrs["type"] == geometry
            assert forbidden.isdisjoint(child.array_keys())
