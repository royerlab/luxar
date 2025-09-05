# --- demo_gaussian_splats_full_torch_napari_compression_3d.py ---
import napari
import numpy as np
from arbol import aprint
from scipy import ndimage
from skimage import data

from luxar.gsplats.candidates import find_candidates_overcomplete_nd
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.gsplats_render import render_gaussians_full_numpy
from luxar.gsplats.utils.trils import tril_size, unpack_tril

# ======= Demo knobs =======
USE_POISSON = True  # True: Poisson deviance; False: MSE
L1_AMP = 0.001  # e.g. 1e-3 to encourage sparsity
N_ITERS = 400  # Reduced for 3D (more expensive)
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps:0"
N_FRAMES = 30  # number of compression steps (<= #splats)
TRUNCATE_SIG = 3.0  # rendering support truncation (≈ ±3σ)
SPACING = (1.0, 1.0, 1.0)  # (z, y, x); not needed for ranking here
# ==========================


# --- Helper: oriented 3D ellipsoid wireframe from covariance ---
def ellipsoid_wireframe_from_L(
    mu_zyx: np.ndarray, L: np.ndarray, t: float = 2.0, n_pts: int = 32
) -> np.ndarray:
    """
    Build a wireframe approximating the 3D ellipsoid corresponding to the level set
    (x-μ)^T Σ^{-1} (x-μ) = t^2, where Σ = L L^T (full cov in voxel units).

    Returns wireframe points as (n_wireframe_pts, 3) array of (z,y,x) coordinates.
    Creates circular wireframes along the three principal planes.
    """
    Sigma = L @ L.T  # (3,3)
    # Eigen-decompose Sigma for principal axes
    evals, evecs = np.linalg.eigh(Sigma)  # evals >= 0
    evals = np.clip(evals, 1e-12, None)
    # Radii along principal axes at level t: r_i = t * sqrt(lambda_i)
    radii = t * np.sqrt(evals)  # (3,)

    # Create wireframe circles in the three principal planes
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    wireframe_pts = []

    # XY plane (z=0 in principal coords)
    circle_xy = np.zeros((n_pts, 3))
    circle_xy[:, 0] = radii[0] * cos_theta  # x-axis in principal coords
    circle_xy[:, 1] = radii[1] * sin_theta  # y-axis in principal coords
    circle_xy[:, 2] = 0  # z-axis
    # Transform to data coordinates
    pts_xy = (evecs @ circle_xy.T).T + mu_zyx[None, :]
    wireframe_pts.append(pts_xy)

    # XZ plane (y=0 in principal coords)
    circle_xz = np.zeros((n_pts, 3))
    circle_xz[:, 0] = radii[0] * cos_theta
    circle_xz[:, 1] = 0
    circle_xz[:, 2] = radii[2] * sin_theta
    pts_xz = (evecs @ circle_xz.T).T + mu_zyx[None, :]
    wireframe_pts.append(pts_xz)

    # YZ plane (x=0 in principal coords)
    circle_yz = np.zeros((n_pts, 3))
    circle_yz[:, 0] = 0
    circle_yz[:, 1] = radii[1] * cos_theta
    circle_yz[:, 2] = radii[2] * sin_theta
    pts_yz = (evecs @ circle_yz.T).T + mu_zyx[None, :]
    wireframe_pts.append(pts_yz)

    # Combine all wireframes
    return np.vstack(wireframe_pts).astype(np.float32)


# 1) Create 3D volumetric "blobs" data (same approach as 2D version)
aprint("Creating 3D test volume...")
volume_size = 64  # Size for 3D demo
blobs = data.binary_blobs(
    length=volume_size, blob_size_fraction=0.08, n_dim=3, volume_fraction=0.15, rng=42
).astype(float)
V = ndimage.gaussian_filter(blobs, sigma=2.5).astype(np.float32)

aprint(f"Created 3D volume: {V.shape} = {V.size:,} voxels")
aprint(f"Volume range: [{V.min():.4f}, {V.max():.4f}]")

# 2) Overcomplete candidate centers (n-D generic)
aprint("Finding 3D candidate centers...")
centers = find_candidates_overcomplete_nd(
    V,
    scales=(0.6, 1.0, 1.5, 2.2, 3.0),  # Scales for 3D
    peaks_per_scale=500,  # Fewer candidates for 3D efficiency
    percentile_thresh=50,  # Slightly higher threshold
    min_dist=2.5,  # Larger spacing for 3D
    add_intensity_grid=False,  # Include grid sampling
    grid_step=[3, 3, 3],  # 3D grid step
)
aprint(f"Found {len(centers)} candidates")

if len(centers) == 0:
    raise RuntimeError("No candidates found; try lowering thresholds.")

# 3) Fit oriented (full-covariance) 3D Gaussians with PyTorch
aprint("Fitting 3D Gaussian splats...")
params_full, amps = fit_gaussian_splats(
    V,
    centers_overcomplete=centers,
    init_sigma_vox=1.4,  # Slightly smaller for 3D
    n_iters=N_ITERS,
    lr=0.15,  # Lower LR for stability in 3D
    loss_type=("poisson" if USE_POISSON else "mse"),
    l1_amp=L1_AMP,
    sigma_min_diag=[0.5, 0.5, 0.5],  # 3D minimum sigma constraints
    sigma_max_diag=[32.0, 32.0, 32.0],  # Maximum sigma to prevent huge splats
    truncate=TRUNCATE_SIG,
    device=DEVICE,
    verbose=True,
)

aprint(f"Fitted {len(amps)} splats successfully")

if len(amps) == 0:
    raise RuntimeError(
        "No splats were fitted; try lowering thresholds or increasing iterations."
    )

# ----- Compression ranking by approximate L2 energy -----
# For a 3D Gaussian, ||G||_2^2 = (sqrt(pi))^d * sqrt(det Σ).
# Here sqrt(det Σ) = prod(diag(L)) because Σ = L L^T.
d = 3
L_packed = params_full[:, d:]  # (N, 6) in 3D (triangular matrix has 6 elements)
L_full = unpack_tril(L_packed, d)  # (N, 3, 3)
diag_prod = np.prod(
    np.stack([L_full[:, 0, 0], L_full[:, 1, 1], L_full[:, 2, 2]], axis=1), axis=1
)  # ∏ diag(L)
energy_score = (amps**2) * (np.sqrt(np.pi) ** d) * diag_prod
order = np.argsort(-energy_score)  # descending

aprint("Energy scores computed for compression ranking")

# ----- Precompute reconstructions/residuals + wireframe ellipsoids per frame -----
N = len(amps)
keep_counts = np.unique(
    np.linspace(1, N, num=min(N_FRAMES, N), endpoint=True).astype(int)
)

aprint(f"Computing {len(keep_counts)} compression frames...")

stack_recon = np.zeros((len(keep_counts),) + V.shape, dtype=np.float32)
stack_resid = np.zeros_like(stack_recon)
rel_err_frames = np.zeros(len(keep_counts), dtype=np.float32)

# Bit accounting (float32 for all params + amps)
FLOAT_BITS = 32
FLOATS_PER_SPLAT = d + tril_size(d) + 1  # centers(3) + packed L(6) + amplitude(1) = 10
BITS_PER_SPLAT = FLOATS_PER_SPLAT * FLOAT_BITS
VOLUME_BITS = V.size * FLOAT_BITS
NUM_VOXELS = V.size

model_bits_frames = np.zeros(len(keep_counts), dtype=np.float64)
bpv_frames = np.zeros(len(keep_counts), dtype=np.float64)  # bits per voxel
bit_compression_pct = np.zeros(len(keep_counts), dtype=np.float64)

# 3D shapes data for each frame (wireframe ellipsoids)
wireframes_frames = []
centers_frames = []

for i, K in enumerate(keep_counts):
    if i % 5 == 0:
        aprint(f"  Processing frame {i + 1}/{len(keep_counts)} (K={K} splats)...")

    idx = order[:K]

    # Reconstruction & residual
    Vk = render_gaussians_full_numpy(
        V.shape, params_full[idx], amps[idx], truncate=TRUNCATE_SIG
    )
    stack_recon[i] = Vk
    stack_resid[i] = V - Vk
    rel_err_frames[i] = np.linalg.norm(V - Vk) / (np.linalg.norm(V) + 1e-12)

    # Bit cost for current model
    model_bits_frames[i] = int(K) * BITS_PER_SPLAT
    bpv_frames[i] = model_bits_frames[i] / NUM_VOXELS  # bits per voxel
    bit_compression_pct[i] = 100.0 * (1.0 - (model_bits_frames[i] / VOLUME_BITS))

    # Wireframe ellipsoids & centers (2σ contour)
    Lk = L_full[idx]  # (K, 3, 3)
    Ck = params_full[idx, :3]  # (K, 3) voxel centers (z,y,x)

    # Create wireframes for each ellipsoid (limit to reasonable number for visualization)
    max_wireframes = min(K, 50)  # Limit wireframes for performance
    wireframes = []
    for j in range(max_wireframes):
        wireframe = ellipsoid_wireframe_from_L(Ck[j], Lk[j], t=2.0, n_pts=24)
        wireframes.append(wireframe)

    # Combine all wireframes into single array for napari
    if wireframes:
        wireframes_combined = np.vstack(wireframes)
    else:
        wireframes_combined = np.zeros((0, 3))

    wireframes_frames.append(wireframes_combined)
    centers_frames.append(Ck)

aprint("3D rendering and compression analysis complete!")

# 4) Napari viewer with 3D volumes and "compression" slider
aprint("Launching 3D napari viewer...")
viewer = napari.Viewer(ndisplay=3)  # Force 3D display

# Add the input volume
viewer.add_image(
    V,
    name="input_volume",
    colormap="viridis",
    contrast_limits=[0, float(V.max())],
    opacity=0.8,
    rendering="mip",  # Maximum intensity projection for better 3D visualization
)

# Add reconstruction volume stack
lyr_recon = viewer.add_image(
    stack_recon,
    name="reconstruction (compression, 3D)",
    colormap="plasma",
    opacity=0.7,
    contrast_limits=[0, max(1e-12, float(stack_recon.max()))],
    rendering="mip",
)

# Add residual volume stack
lyr_resid = viewer.add_image(
    np.maximum(stack_resid, 0),
    name="residual (clipped ≥0, compression)",
    colormap="turbo",
    opacity=0.5,
    contrast_limits=[0, max(1e-12, float(stack_resid.max()))],
    rendering="mip",
)

# 3D wireframe ellipsoids that update with slider
wireframe_layer = viewer.add_points(
    np.zeros((0, 3)),
    name="ellipsoid wireframes (kept)",
    size=1,
    border_color="cyan",
    face_color="cyan",
    opacity=0.6,
)

# Centers that update with slider
pts = viewer.add_points(
    np.zeros((0, 3)),
    name="centers (kept)",
    size=4,
    border_color="yellow",
    face_color="transparent",
)

# Try to label axes (requires napari >= 0.4.18)
try:
    viewer.dims.axis_labels = ["compression", "z", "y", "x"]
except Exception:
    pass

# Set better 3D camera view
viewer.camera.angles = (15, 25, 120)  # Good 3D viewing angle
viewer.camera.zoom = 0.8


def _set_overlay_text_3d(t_index: int):
    """Update text overlay with 3D-specific information."""
    K = int(keep_counts[t_index])
    bits_model = int(model_bits_frames[t_index])
    bpv = float(bpv_frames[t_index])  # bits per voxel
    pct_bits = float(bit_compression_pct[t_index])
    rel = float(rel_err_frames[t_index])
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = (
        f"3D Splats: {K}/{N}  |  Model bits: {bits_model:,}  "
        f"|  Model bpv: {bpv:.3f} (raw=32.000)  |  Bit compression: {pct_bits:.1f}%  "
        f"|  rel L2 err: {rel:.4f}  |  Volume: {volume_size}³ voxels"
    )


def _update_layers_for_t_3d(t_index: int):
    """Update 3D layers for given time index."""
    # Update wireframe ellipsoids
    wireframe_layer.data = wireframes_frames[t_index]

    # Update centers
    pts.data = centers_frames[t_index]

    # Update text overlay
    _set_overlay_text_3d(t_index)


# Initialize with first frame
_update_layers_for_t_3d(0)


# Hook slider to updates
def _on_step_change_3d(event=None):
    t = viewer.dims.current_step[0]
    _update_layers_for_t_3d(int(t))


viewer.dims.events.current_step.connect(_on_step_change_3d)

# Console summary for 3D
aprint("\n3D Gaussian Splat Compression Analysis")
aprint(f"{'=' * 60}")
aprint(f"Raw volume bits (float32): {VOLUME_BITS:,}  |  raw bpv = 32.000")
aprint(f"Volume size: {volume_size}³ = {NUM_VOXELS:,} voxels")
aprint(f"Bits per splat: {BITS_PER_SPLAT} (3 centers + 6 covariance + 1 amplitude)")
aprint(f"{'=' * 60}")

for i, K in enumerate(keep_counts):
    aprint(
        f"Frame {i:02d} | keep {K:4d} | model_bits={int(model_bits_frames[i]):>10,d} "
        f"| bit_compression={bit_compression_pct[i]:6.1f}% | bpv={bpv_frames[i]:6.3f} "
        f"| relL2={rel_err_frames[i]:.4f}"
    )

aprint("\n🎮 3D Navigation Tips:")
aprint("  • Use mouse + Shift to rotate the 3D view")
aprint("  • Use the top slider (axis 0) to see compression progression")
aprint("  • Yellow points = splat centers")
aprint("  • Cyan wireframes = 2σ ellipsoid boundaries")
aprint("  • Toggle layers on/off to compare input vs reconstruction")

aprint("\n📊 Compression Insights:")
best_compression = np.max(bit_compression_pct)
final_error = rel_err_frames[-1]
aprint(f"  • Best compression: {best_compression:.1f}% bit reduction")
aprint(f"  • Final relative error: {final_error:.4f}")
aprint("  • 3D splats can achieve good compression on volumetric data!")

napari.run()
