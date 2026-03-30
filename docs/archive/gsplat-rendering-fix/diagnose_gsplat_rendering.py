"""
Diagnostic script to compare gsplat rendering between Python and GLSL.

This traces through the exact numerical calculations to identify discrepancies.
"""

import numpy as np
from scipy.linalg import cholesky


def diagnose_gsplat_rendering():
    """Compare Python and GLSL gsplat rendering calculations."""

    print("=" * 70)
    print("GSplat Rendering Diagnostic")
    print("=" * 70)

    # Test case: Single 3D Gaussian
    # Parameters matching a typical gsplat
    center = np.array([64.0, 64.0, 64.0])  # Voxel coordinates
    sigma = np.array([10.0, 8.0, 12.0])  # Different σ per axis
    amplitude = 1.0
    sharpness = 2.0  # Standard Gaussian

    # Build covariance matrix (diagonal for simplicity)
    Sigma = np.diag(sigma**2)

    # Cholesky factor
    L = cholesky(Sigma, lower=True)
    print("\n1. Input Gaussian:")
    print(f"   Center: {center}")
    print(f"   Sigma: {sigma}")
    print(f"   Amplitude: {amplitude}")
    print(f"   Sharpness: {sharpness}")
    print(f"\n   Covariance Σ:\n{Sigma}")
    print(f"\n   Cholesky L (Σ = L @ L.T):\n{L}")

    # Pack Cholesky (row-major lower triangular)
    packed = [L[0, 0], L[1, 0], L[1, 1], L[2, 0], L[2, 1], L[2, 2]]
    print(f"\n   Packed Cholesky: {packed}")

    # === Camera setup ===
    # Camera looking at center from distance d along -Z axis
    distance = 200.0  # Camera distance from origin
    fov_deg = 60.0  # Vertical FOV
    resolution = (800, 600)  # Width x Height

    fov_rad = np.radians(fov_deg)
    fy = resolution[1] / (2 * np.tan(fov_rad / 2))
    fx = resolution[0] / (
        2 * np.tan(fov_rad / 2)
    )  # Same formula (aspect ratio handled)

    print("\n2. Camera Setup:")
    print(f"   Distance: {distance}")
    print(f"   FOV: {fov_deg}°")
    print(f"   Resolution: {resolution}")
    print(f"   Focal lengths: fx={fx:.2f}, fy={fy:.2f}")

    # Camera at (64, 64, 64+200) looking at center
    # In camera space, center is at (0, 0, -200)
    center_cam = np.array([0.0, 0.0, -distance])

    # Rotation is identity (camera aligned with world)
    R = np.eye(3)

    # Transform Cholesky to camera space
    L_cam = R @ L
    Sigma_cam = L_cam @ L_cam.T

    print("\n3. Camera Space Transform:")
    print(f"   Center in camera space: {center_cam}")
    print(f"   Σ_cam (same as Σ for identity rotation):\n{Sigma_cam}")

    # === Perspective Jacobian ===
    z = -center_cam[2]  # Positive depth
    x, y = center_cam[0], center_cam[1]

    invZ = 1.0 / z
    invZ2 = invZ * invZ

    # Jacobian J = d(screen)/d(camera)
    # J is 2x3: [[fx/z, 0, fx*x/z²], [0, fy/z, fy*y/z²]]
    J = np.array([[fx * invZ, 0.0, fx * x * invZ2], [0.0, fy * invZ, fy * y * invZ2]])

    print("\n4. Perspective Jacobian at center:")
    print(f"   z (depth) = {z}")
    print(f"   J:\n{J}")

    # === 2D Covariance ===
    Sigma_2D = J @ Sigma_cam @ J.T

    print("\n5. 2D Projected Covariance:")
    print(f"   Σ_2D = J @ Σ_cam @ J.T:\n{Sigma_2D}")

    # Eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma_2D)
    print(f"\n   Eigenvalues: {eigenvalues}")
    print(f"   Sqrt(eigenvalues) (std devs in pixels): {np.sqrt(eigenvalues)}")

    # Quad extent with truncation radius 3.0
    truncate = 3.0
    extent = truncate * np.sqrt(eigenvalues)
    print(f"\n   Quad extent (3σ truncation): {extent} pixels")

    # === Ray Integration ===
    ray_dir = center_cam / np.linalg.norm(center_cam)  # Normalized ray direction
    sigma_ray_sq = ray_dir @ Sigma_cam @ ray_dir
    sigma_ray = np.sqrt(sigma_ray_sq)

    print("\n6. Ray Integration:")
    print(f"   Ray direction: {ray_dir}")
    print(f"   σ_ray² = ray @ Σ_cam @ ray = {sigma_ray_sq:.4f}")
    print(f"   σ_ray = {sigma_ray:.4f}")

    # Sharpness integral factor c(s)
    # For s=2: c(2) ≈ √(2π) ≈ 2.5066
    # GLSL approximation: 1.97 + 1.95 * exp(-0.64 * s)
    c_s_exact = np.sqrt(2 * np.pi)  # For s=2
    c_s_glsl = 1.97 + 1.95 * np.exp(-0.64 * sharpness)

    print("\n   Sharpness integral factor:")
    print(f"   c(s) exact (for s=2): √(2π) = {c_s_exact:.4f}")
    print(f"   c(s) GLSL approx: {c_s_glsl:.4f}")
    print(f"   Approximation error: {abs(c_s_glsl - c_s_exact) / c_s_exact * 100:.2f}%")

    # Ray integration boost
    ray_boost = sigma_ray * c_s_exact
    ray_boost_glsl = sigma_ray * c_s_glsl

    print("\n   Ray integration boost = σ_ray × c(s):")
    print(f"   Exact: {ray_boost:.4f}")
    print(f"   GLSL approx: {ray_boost_glsl:.4f}")

    # === Final amplitude ===
    amplitude_2D = amplitude * ray_boost

    print("\n7. Final 2D Amplitude:")
    print(f"   amplitude_2D = amplitude × ray_boost = {amplitude_2D:.4f}")

    # === Intensity at center pixel ===
    # At center: mahalSq = 0, exp(-0.5 * 0) = 1
    intensity_center = amplitude_2D * 1.0  # exp(-0.5 * 0) = 1

    print("\n8. Intensity at Splat Center:")
    print(f"   intensity = amplitude_2D × exp(-0.5 × 0) = {intensity_center:.4f}")

    # === Comparison with Python volume rendering ===
    print("\n" + "=" * 70)
    print("Comparison with Python Volume Rendering")
    print("=" * 70)

    # For Python volume rendering, the peak intensity at voxel center is:
    # amplitude × Gaussian(0) = amplitude × exp(-0.5 × 0) = amplitude = 1.0
    print("\nPython render_to_volume at center voxel:")
    print(f"   Peak intensity = amplitude = {amplitude}")

    # But napari sums along rays. For ray through center:
    # Sum = amplitude × integral of exp(-z²/(2σ_z²)) dz
    #     = amplitude × σ_z × √(2π)
    sigma_z = sigma[2]  # Z-axis sigma
    napari_ray_sum = amplitude * sigma_z * np.sqrt(2 * np.pi)

    print("\nNapari (additive volume) ray sum through center:")
    print(
        f"   Sum = amplitude × σ_z × √(2π) = {amplitude} × {sigma_z} × {np.sqrt(2 * np.pi):.4f}"
    )
    print(f"   Sum = {napari_ray_sum:.4f}")

    # Compare with Luxar
    print(f"\nLuxar gsplat center intensity: {intensity_center:.4f}")
    print(f"Napari ray sum: {napari_ray_sum:.4f}")

    ratio = napari_ray_sum / intensity_center
    print(f"\nRatio (napari / Luxar): {ratio:.4f}")

    if abs(ratio - 1.0) > 0.01:
        print("\n⚠️  MISMATCH DETECTED!")
        print(f"   The intensities should match but differ by factor of {ratio:.4f}")

        # Diagnose the source of mismatch
        print("\n   Diagnosis:")
        print(f"   - σ_ray (from Σ_cam) = {sigma_ray:.4f}")
        print(f"   - σ_z (diagonal of Σ) = {sigma_z:.4f}")
        print("   - For this axis-aligned case, σ_ray should equal σ_z")

        if abs(sigma_ray - sigma_z) > 0.01:
            print("\n   ❌ σ_ray ≠ σ_z")
            print(f"   This is because ray_dir is not [0,0,1] but {ray_dir}")
            print("   σ_ray = sqrt(ray @ Σ @ ray) depends on all elements of Σ")
    else:
        print("\n✓ Intensities match!")

    # === Additional diagnostics ===
    print("\n" + "=" * 70)
    print("Additional Diagnostics")
    print("=" * 70)

    # What if camera is looking straight down Z (ray_dir = [0, 0, -1])?
    ray_straight = np.array([0.0, 0.0, -1.0])
    sigma_ray_straight = np.sqrt(ray_straight @ Sigma @ ray_straight)

    print("\nIf ray direction were [0, 0, -1] (straight down Z):")
    print(f"   σ_ray = {sigma_ray_straight:.4f} (should equal σ_z = {sigma_z})")

    # But our actual ray direction is (0, 0, -200) normalized = (0, 0, -1)
    print(f"\nActual ray direction: {ray_dir}")
    print("This IS [0, 0, -1], so σ_ray SHOULD equal σ_z")

    print("\nSigma_cam is positive definite, so ray @ Σ @ ray is always positive")
    print(f"Checking: ray @ Σ_cam @ ray = {ray_dir @ Sigma_cam @ ray_dir:.4f}")
    print(f"sqrt(...) = {np.sqrt(ray_dir @ Sigma_cam @ ray_dir):.4f}")


def diagnose_perspective_distortion():
    """Analyze perspective distortion for off-center splats."""

    print("\n" + "=" * 70)
    print("Perspective Distortion Analysis")
    print("=" * 70)

    # Camera setup
    distance = 200.0
    fov_deg = 60.0
    resolution = (800, 600)
    fov_rad = np.radians(fov_deg)
    fy = resolution[1] / (2 * np.tan(fov_rad / 2))
    fx = resolution[0] / (2 * np.tan(fov_rad / 2))

    # Test isotropic Gaussian (σ = 10 in all dimensions)
    sigma = 10.0
    Sigma = sigma**2 * np.eye(3)
    _L = sigma * np.eye(3)  # noqa: F841 (kept for documentation)

    print(f"\nTest: Isotropic Gaussian with σ = {sigma}")
    print(f"Camera distance: {distance}, FOV: {fov_deg}°")
    print("\nComparing center vs off-center splats:\n")

    # Test positions: center and off-center
    test_positions = [
        ("Center", [0, 0, -distance]),
        ("Right (X=50)", [50, 0, -distance]),
        ("Right (X=100)", [100, 0, -distance]),
        ("Corner (X=Y=50)", [50, 50, -distance]),
    ]

    print(
        f"{'Position':<20} {'σ_x (px)':<12} {'σ_y (px)':<12} {'Ratio':<10} {'Area Ratio':<12}"
    )
    print("-" * 70)

    reference_area = None
    for name, pos in test_positions:
        center_cam = np.array(pos, dtype=float)
        z = -center_cam[2]
        x, y = center_cam[0], center_cam[1]
        invZ = 1.0 / z
        invZ2 = invZ * invZ

        # Jacobian
        J = np.array(
            [[fx * invZ, 0.0, fx * x * invZ2], [0.0, fy * invZ, fy * y * invZ2]]
        )

        # 2D Covariance
        Sigma_2D = J @ Sigma @ J.T

        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(Sigma_2D)
        sigma_x = np.sqrt(max(eigenvalues))
        sigma_y = np.sqrt(min(eigenvalues))
        ratio = sigma_x / sigma_y if sigma_y > 0 else float("inf")
        area = np.pi * sigma_x * sigma_y

        if reference_area is None:
            reference_area = area
        area_ratio = area / reference_area

        print(
            f"{name:<20} {sigma_x:<12.2f} {sigma_y:<12.2f} {ratio:<10.2f} {area_ratio:<12.2f}"
        )

    print("\nNote: A ratio > 1.0 indicates radial stretching due to perspective.")
    print("This effect doesn't occur in orthographic projection (napari).")

    # Orthographic comparison
    print("\n" + "-" * 70)
    print("Orthographic projection (napari-like):")
    print("-" * 70)
    # In orthographic, J = [[scale, 0, 0], [0, scale, 0]] (no z-dependence)
    scale = fx / distance  # Approximate scale at the reference distance
    J_ortho = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0]])
    Sigma_2D_ortho = J_ortho @ Sigma @ J_ortho.T
    eigenvalues_ortho = np.linalg.eigvalsh(Sigma_2D_ortho)
    sigma_ortho = np.sqrt(eigenvalues_ortho[0])
    print(f"All positions: σ_x = σ_y = {sigma_ortho:.2f} px (no distortion)")

    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print("""
Key findings:

1. PERSPECTIVE VS ORTHOGRAPHIC:
   - Luxar uses perspective projection → radial stretching for off-center splats
   - napari uses orthographic projection → no radial distortion
   - This explains the "elongated/streaky" appearance in Luxar

2. RECOMMENDATIONS:
   a) For napari comparison, view splats near the center of the screen
   b) Consider adding orthographic projection mode to Luxar
   c) Or accept the difference as a feature of 3D perspective viewing

3. SIZE DISCREPANCY:
   - The projected 2D size depends on camera distance and FOV
   - Different camera setups between napari and Luxar will cause different apparent sizes
   - Verify camera distance/zoom is equivalent between the two viewers
""")


def diagnose_actual_gsplat_data(path: str):
    """Load and diagnose actual gsplat data from a file."""
    import zarr

    print("\n" + "=" * 70)
    print(f"Diagnosing GSplat Data: {path}")
    print("=" * 70)

    # Try to load as GSplatData first
    try:
        from luxar.gsplats import GSplatData

        data = GSplatData.load(path)
    except ValueError:
        # If it's a scene zarr, load the gsplats data directly
        print("  (Loading as scene zarr...)")
        root = zarr.open(path, mode="r")

        # Find gsplats groups (by node_type or by having centers/cholesky_factors arrays)
        gsplats_groups = []

        def find_gsplats(group, grp_path=""):
            attrs = dict(group.attrs)
            # Check if this is a gsplats group by node_type or by having the required arrays
            is_gsplats = attrs.get("node_type") == "gsplats"
            has_gsplats_arrays = (
                "centers" in group.keys() and "cholesky_factors" in group.keys()
            )
            if is_gsplats or has_gsplats_arrays:
                gsplats_groups.append((grp_path, group))
            for key in group.keys():
                if isinstance(group[key], zarr.Group):
                    find_gsplats(group[key], f"{grp_path}/{key}" if grp_path else key)

        find_gsplats(root)

        if not gsplats_groups:
            print(f"  No gsplats groups found in {path}")
            return None

        print(f"  Found {len(gsplats_groups)} gsplats group(s)")
        group_path, group = gsplats_groups[0]
        print(f"  Analyzing: {group_path}")

        # Load arrays directly
        centers = np.array(group["centers"])
        amplitudes = np.array(group["amplitudes"]).astype(np.float32)
        cholesky_factors = np.array(group["cholesky_factors"])

        # Handle sharpness (might be 'sharpness' or 'sharpnesses', might be broadcasted)
        if "sharpnesses" in group:
            sharpnesses = np.array(group["sharpnesses"])
        elif "sharpness" in group:
            sharpnesses = np.array(group["sharpness"])
        else:
            sharpnesses = np.full(len(amplitudes), 2.0)

        # Broadcast if needed
        if len(sharpnesses) == 1:
            sharpnesses = np.full(len(amplitudes), sharpnesses[0])

        # Create a simple data object
        class SimpleData:
            pass

        data = SimpleData()
        data.centers = centers
        data.amplitudes = amplitudes
        data.cholesky_factors = cholesky_factors
        data.sharpnesses = sharpnesses

    n_splats = len(data.amplitudes)
    ndim = data.centers.shape[1]

    print("\nBasic Info:")
    print(f"  Number of splats: {n_splats:,}")
    print(f"  Dimensions: {ndim}D")

    # Centers analysis
    print("\nCenters (positions):")
    print(f"  Shape: {data.centers.shape}")
    print("  Range per axis:")
    for d in range(ndim):
        min_val = data.centers[:, d].min()
        max_val = data.centers[:, d].max()
        span = max_val - min_val
        print(f"    Axis {d}: [{min_val:.2f}, {max_val:.2f}] (span: {span:.2f})")

    # Compute average spacing between splats
    from scipy.spatial import cKDTree

    if n_splats > 1 and n_splats < 100000:
        tree = cKDTree(data.centers)
        distances, _ = tree.query(
            data.centers, k=2
        )  # k=2 to get nearest neighbor (excluding self)
        nn_distances = distances[:, 1]  # Second column is nearest neighbor
        print("\n  Nearest neighbor distances:")
        print(f"    Min: {nn_distances.min():.4f}")
        print(f"    Max: {nn_distances.max():.4f}")
        print(f"    Mean: {nn_distances.mean():.4f}")
        print(f"    Median: {np.median(nn_distances):.4f}")

    # Cholesky analysis
    print("\nCholesky factors:")
    print(f"  Shape: {data.cholesky_factors.shape}")

    # Unpack Cholesky and compute sigmas (diagonal elements are std devs for diagonal covariance)
    from luxar.gsplats.utils.trils import unpack_tril

    L_matrices = unpack_tril(data.cholesky_factors, ndim)  # (N, d, d)

    # Diagonal elements of L are related to standard deviations
    diag_elements = np.array([L_matrices[:, i, i] for i in range(ndim)])  # (d, N)

    print("\n  Diagonal elements of L (related to std devs):")
    for d in range(ndim):
        min_val = diag_elements[d].min()
        max_val = diag_elements[d].max()
        mean_val = diag_elements[d].mean()
        print(
            f"    L[{d},{d}]: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}"
        )

    # Compute actual covariances
    Sigmas = np.einsum("nij,nkj->nik", L_matrices, L_matrices)  # Σ = L @ L.T
    eigenvalues = np.linalg.eigvalsh(Sigmas)  # (N, d)
    # Clamp negative eigenvalues (numerical issues) before sqrt
    eigenvalues = np.maximum(eigenvalues, 0)
    sigmas = np.sqrt(eigenvalues)  # Standard deviations along principal axes

    print("\n  Standard deviations along principal axes:")
    for d in range(ndim):
        valid = ~np.isnan(sigmas[:, d])
        if valid.sum() > 0:
            min_val = sigmas[valid, d].min()
            max_val = sigmas[valid, d].max()
            mean_val = sigmas[valid, d].mean()
            print(
                f"    σ_{d}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}"
            )
        else:
            print(f"    σ_{d}: all NaN")

    # Compare splat size to spacing
    if n_splats > 1 and n_splats < 100000:
        avg_sigma = np.nanmean(sigmas)  # Average std dev across all splats and axes
        avg_spacing = nn_distances.mean()
        coverage_ratio = (2 * avg_sigma) / avg_spacing  # 2σ captures ~95% of Gaussian

        print("\n  Coverage Analysis:")
        print(f"    Average σ (all axes): {avg_sigma:.4f}")
        print(f"    Average splat spacing: {avg_spacing:.4f}")
        print(f"    Coverage ratio (2σ / spacing): {coverage_ratio:.4f}")

        if np.isnan(coverage_ratio):
            print("\n  ⚠️  Could not compute coverage ratio (NaN values)")
        elif coverage_ratio < 1.0:
            print("\n  ⚠️  WARNING: Coverage ratio < 1.0 means splats may not overlap!")
            print("     This could cause visible gaps in rendering.")
            print(
                f"     Splats would need σ ≈ {avg_spacing / 2:.4f} for smooth coverage."
            )
        elif coverage_ratio < 1.5:
            print("\n  ⚠️  Coverage ratio is marginal. May see some gaps.")
        else:
            print("\n  ✓ Good coverage ratio. Splats should overlap smoothly.")

    # Sharpness analysis
    print("\nSharpness:")
    print(f"  Shape: {data.sharpnesses.shape}")
    print(f"  Range: [{data.sharpnesses.min():.4f}, {data.sharpnesses.max():.4f}]")
    print(f"  Mean: {data.sharpnesses.mean():.4f}")
    if np.allclose(data.sharpnesses, 2.0, atol=0.01):
        print("  ✓ All sharpnesses ≈ 2.0 (standard Gaussian)")
    else:
        print("  Note: Sharpness varies from standard Gaussian (s=2)")

    # Amplitude analysis
    print("\nAmplitudes:")
    print(f"  Shape: {data.amplitudes.shape}")
    print(f"  Range: [{data.amplitudes.min():.6f}, {data.amplitudes.max():.6f}]")
    print(f"  Mean: {data.amplitudes.mean():.6f}")

    return data


def main():
    import sys

    diagnose_gsplat_rendering()
    diagnose_perspective_distortion()

    # If a path is provided, diagnose that file
    if len(sys.argv) > 1:
        path = sys.argv[1]
        diagnose_actual_gsplat_data(path)


if __name__ == "__main__":
    main()
