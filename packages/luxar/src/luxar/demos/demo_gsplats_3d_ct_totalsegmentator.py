#!/usr/bin/env python3
"""GSplats Demo: CT Anatomical Atlas (TotalSegmentator) — Organs in Color

Gaussian-splats a real clinical CT scan (a neck-to-pelvis study — the fullest
common coverage in routine CT; head and distal limbs are outside the scan) and
colors every splat by the anatomical structure it belongs to, using the
TotalSegmentator dataset's 117-structure segmentation. The result is a glowing,
rotatable 3D atlas: the ivory skeleton, red great vessels, cyan lungs, and
colored abdominal organs all in their true 3D positions — the microscopy splat
pipeline applied to clinical radiology.

================================================================================
CT VOLUME + 117-LABEL SEGMENTATION → COLORED GAUSSIAN SPLATS
================================================================================

Each subject has a ``ct.nii.gz`` (Hounsfield-unit volume) and a per-structure
segmentation (one binary mask per organ). We combine the masks into one
label volume, window the CT to the body, fit oriented Gaussians to the
segmented anatomy, then sample the label at each splat center and map it to a
tissue-grouped color palette (bone = ivory, vessels = red, lungs = cyan, GI =
green, abdominal organs = warm, muscle = dim flesh). One fit + per-splat label
sampling — the same idea as the Visible Human head demo, but the sampled organ
label drives both the color and a hover tooltip (the specific structure name,
all 117 tissue types), and the splats are split into toggle-able Layers-panel
groups (Skeleton / Organs / Vessels & heart / Nervous system / Muscles).

DATA SOURCE & CITATION
----------------------
TotalSegmentator dataset (small 102-subject subset, v2.0.1, CC BY 4.0):
    https://zenodo.org/records/10047263
Wasserthal, J. et al. (2023). "TotalSegmentator: Robust Segmentation of 104
    Anatomic Structures in CT Images." Radiology: Artificial Intelligence.
    https://doi.org/10.1148/ryai.230024
Label scheme: the 117-class `total` map from the TotalSegmentator tool
    (Apache-2.0). Colors here are a tissue-grouped palette (no canonical LUT
    exists upstream).

SELF-CONTAINED / CACHING
------------------------
On a fresh machine this demo bootstraps itself with no manual steps:
  1. Fast path: a precomputed fit + per-splat organ labels shipped via Git LFS
     (``demos/data/gsplats_ct_totalsegmentator/``); loads instantly (colors,
     layers, and hover tooltips are all derived from the labels at scene build).
  2. If those assets aren't pulled, ``--recompute`` (or missing assets)
     AUTOMATICALLY downloads the 3.2 GB subset to
     ``~/.cache/luxar/gsplats_ct_totalsegmentator/`` (resumable), extracts one
     subject, combines its masks with ``nibabel``, fits on the GPU, samples the
     per-splat organ label, and caches.

USAGE
-----
    python demo_gsplats_3d_ct_totalsegmentator.py [--recompute] [--no-serve] [--serve-only]

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan,  'C': fly controls
"""

DEMO_META = {
    "key": "gsplats_3d_ct_totalsegmentator",
    "title": "CT Anatomical Atlas (TotalSegmentator) — Organs in Color",
    "description": "A clinical CT scan Gaussian-splatted and colored by 117 TotalSegmentator organ labels.",
    "category": "medical",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 7,
        "compute": "medium",
        "gpu": "optional",
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_ct_totalsegmentator"],
    "outputs": ["gsplats_3d_ct_totalsegmentator"],
}

from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import CameraConfig, Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import require_module
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import (
    detect_device,
    is_lfs_pointer,
    launch_viewer,
    load_precomputed_gsplats,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

SUBSET_URL = (
    "https://zenodo.org/api/records/10047263/files/"
    "Totalsegmentator_dataset_small_v201.zip/content"
)
SUBSET_SIZE = 3_244_617_817  # bytes
# The broadest-coverage subject in the subset (neck-thorax-abdomen-pelvis, ~109
# of the 117 structures present, 1.5 mm isotropic, ~668 mm neck→pelvis).
SUBJECT_ID = "s0720"

DEMO_NAME = "gsplats_ct_totalsegmentator"
FIT_FILE = "ct_atlas.gsplats.zarr.zip"
LABELS_FILE = "ct_atlas_labels.npz"

CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
CACHE_ZIP = CACHE_DIR / "totalsegmentator_small.zip"
CACHE_FIT = CACHE_DIR / FIT_FILE
CACHE_LABELS = CACHE_DIR / LABELS_FILE

DATA_DIR = Path(__file__).parent / "data" / DEMO_NAME
LFS_FIT = DATA_DIR / FIT_FILE
LFS_LABELS = DATA_DIR / LABELS_FILE

# CT windowing (Hounsfield units): soft tissue + bone. Below LO → 0, above HI → 1.
HU_LO = -150.0
HU_HI = 500.0

# Resample so the largest physical axis is this many voxels (cubic voxels). The
# dataset is 1.5 mm isotropic (native full-body grid ~450 voxels tall) — the
# resolution ceiling. `MAX_SPLATS` is only a ceiling: the progressive fitter
# self-culls to what the segmented anatomy needs (~0.66M splats at ~43 dB, well
# under 1M) regardless of the cap.
TARGET_MAX_DIM = 512
MAX_SPLATS = 2_500_000
MAX_SPLATS_PER_PASS = 400_000
ITERS_PER_PASS = 4_000
PSNR_PATIENCE = 0.1

SCENE_INTENSITY = 0.012  # Display brightness (dense body — dial down; see VH demo)

# TotalSegmentator v2 `total` task — 117 structures (label index → name).
CLASS_MAP = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_left_1",
    93: "rib_left_2",
    94: "rib_left_3",
    95: "rib_left_4",
    96: "rib_left_5",
    97: "rib_left_6",
    98: "rib_left_7",
    99: "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages",
}

# Tissue-group base colors (RGB in [0, 1]) — no canonical LUT exists upstream.
# Muscle is ~26% of splats (paraspinal/gluteus/iliopsoas) and bone ~38%, so both
# are kept as receding, low-saturation "context" tones (dim flesh, warm ivory)
# while the organs/vessels stay vivid, preserving contrast under the volumetric
# blend instead of drowning in a bright pink+ivory mush.
GROUP_COLORS = {
    "bone": (0.90, 0.90, 0.88),  # near-neutral white (no yellow cast)
    "muscle": (0.85, 0.45, 0.42),  # salmon flesh — visible but not garish
    "vessel": (0.95, 0.18, 0.18),  # arteries/veins — vivid red
    "heart": (0.90, 0.12, 0.30),  # crimson
    "lung": (0.35, 0.80, 0.92),  # cyan
    "gi": (0.55, 0.78, 0.28),  # green
    "abdominal_organ": (0.90, 0.52, 0.22),  # liver/spleen/pancreas — vivid amber-brown
    "urinary": (0.98, 0.78, 0.20),  # kidney/adrenal/bladder — gold
    "misc_organ": (0.30, 0.82, 0.60),  # trachea/thyroid/prostate — teal
    "brain": (0.98, 0.92, 0.50),  # pale yellow
    "spinal": (0.98, 0.82, 0.25),  # yellow
}

# Toggle-able Layers-panel groups: (name, member tissue groups, opacity,
# amplitude_boost). Order sets the Layers-panel order. Muscle has low CT
# amplitude (soft tissue) so it needs an amplitude boost to be visible — the
# boost multiplies the splat amplitudes at build (a true intensity gain, not
# capped at opacity=1); it stays semi-transparent so organs still read through.
SUPERGROUPS: list[tuple[str, tuple[str, ...], float, float]] = [
    ("Skeleton", ("bone",), 1.0, 1.0),
    ("Organs", ("lung", "gi", "abdominal_organ", "urinary", "misc_organ"), 1.0, 1.0),
    ("Vessels & heart", ("vessel", "heart"), 1.0, 1.0),
    ("Nervous system", ("brain", "spinal"), 1.0, 1.0),
    ("Muscles", ("muscle",), 0.55, 3.0),
]

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

Arbol.max_depth = 5
DEVICE = None


# =============================================================================
# Pure helpers (unit-tested)
# =============================================================================


def tissue_group(name: str) -> str:
    """Classify a TotalSegmentator structure name into a tissue group."""
    if "lung" in name:
        return "lung"
    if any(
        k in name
        for k in (
            "vertebrae",
            "sacrum",
            "hip",
            "femur",
            "humerus",
            "scapula",
            "clavicula",
            "skull",
            "rib",
            "sternum",
            "costal",
        )
    ):
        return "bone"
    if "heart" in name or "atrial" in name:
        return "heart"
    if any(
        k in name
        for k in (
            "aorta",
            "artery",
            "vein",
            "vena",
            "pulmonary",
            "brachiocephalic",
            "subclavian",
            "carotid",
            "iliac",
            "portal",
        )
    ):
        return "vessel"
    if any(k in name for k in ("gluteus", "iliopsoas", "autochthon")):
        return "muscle"
    if "brain" in name:
        return "brain"
    if "spinal" in name:
        return "spinal"
    if any(
        k in name for k in ("colon", "small_bowel", "duodenum", "stomach", "esophagus")
    ):
        return "gi"
    if any(k in name for k in ("kidney", "urinary_bladder", "adrenal")):
        return "urinary"
    if any(k in name for k in ("liver", "spleen", "pancreas", "gallbladder")):
        return "abdominal_organ"
    return "misc_organ"


def organ_palette() -> np.ndarray:
    """Build a (118, 3) float32 RGB palette indexed by label id (0 = background).

    Each structure takes its tissue-group base color, nudged in brightness by a
    small deterministic per-label offset so neighbouring/left-right structures
    (e.g. individual vertebrae or ribs) stay visually separable.
    """
    palette = np.zeros((118, 3), dtype=np.float32)
    palette[0] = (0.12, 0.12, 0.14)  # background — near-black
    for label, name in CLASS_MAP.items():
        base = np.array(GROUP_COLORS[tissue_group(name)], dtype=np.float32)
        # deterministic ±8% brightness ripple keyed on label id (keeps
        # left/right & adjacent structures separable without muddying the hue)
        f = 0.92 + 0.16 * (((label * 2654435761) % 1000) / 1000.0)
        palette[label] = np.clip(base * f, 0.0, 1.0)
    return palette


def window_ct(ct: np.ndarray, lo: float = HU_LO, hi: float = HU_HI) -> np.ndarray:
    """Window a Hounsfield-unit CT volume to [0, 1] (clip below lo, above hi)."""
    v = np.asarray(ct, dtype=np.float32)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def sample_labels(label_vol: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest-voxel label id for each splat center (centers in (z, y, x) voxels)."""
    zi = np.clip(np.round(centers[:, 0]).astype(int), 0, label_vol.shape[0] - 1)
    yi = np.clip(np.round(centers[:, 1]).astype(int), 0, label_vol.shape[1] - 1)
    xi = np.clip(np.round(centers[:, 2]).astype(int), 0, label_vol.shape[2] - 1)
    return label_vol[zi, yi, xi].astype(np.int32)


def label_colors(label_ids: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map per-splat label ids → (N, 3) float32 RGB via the palette."""
    ids = np.clip(np.asarray(label_ids, dtype=np.intp), 0, palette.shape[0] - 1)
    return palette[ids].astype(np.float32)


def _label_to_supergroup_index() -> dict[int, int]:
    """label id → SUPERGROUPS index (background label 0 → -1)."""
    group_to_super = {
        g: i for i, (_, groups, *_) in enumerate(SUPERGROUPS) for g in groups
    }
    return {lid: group_to_super[tissue_group(name)] for lid, name in CLASS_MAP.items()}


def splat_layer_indices(label_ids: np.ndarray) -> np.ndarray:
    """Per-splat SUPERGROUPS index (-1 for background/label 0)."""
    lut = np.full(118, -1, dtype=np.int32)
    for lid, sidx in _label_to_supergroup_index().items():
        lut[lid] = sidx
    return lut[np.clip(np.asarray(label_ids, dtype=np.intp), 0, 117)]


def organ_label_text(label_id: int) -> str:
    """Human-readable organ name for a hover tooltip (e.g. 'Kidney right')."""
    name = CLASS_MAP.get(int(label_id))
    if not name:
        return ""
    return name.replace("_", " ").capitalize()


def crop_to_content(
    mask: np.ndarray, pad: int = 2
) -> tuple[int, int, int, int, int, int]:
    """Bounding box (z0, z1, y0, y1, x0, x1) of nonzero voxels, padded/clamped."""
    if not mask.any():
        return (0, mask.shape[0], 0, mask.shape[1], 0, mask.shape[2])
    out = []
    for axis in range(3):
        idx = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
        lo = max(0, int(idx[0]) - pad)
        hi = min(mask.shape[axis], int(idx[-1]) + 1 + pad)
        out.extend([lo, hi])
    return tuple(out)  # type: ignore[return-value]


def _save_labels_u8(labels: np.ndarray, path: Path) -> None:
    """Persist the per-splat organ label id as uint8 (0-117)."""
    u8 = np.clip(np.asarray(labels), 0, 117).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, labels_u8=u8)
    tmp.rename(path)


def _load_labels(path: Path) -> np.ndarray:
    """Load the per-splat organ label id array (int32)."""
    with np.load(path) as data:
        return data["labels_u8"].astype(np.int32)


# =============================================================================
# Data loading (network / nibabel IO — not unit-tested)
# =============================================================================


def download_subset() -> Path:
    """Download the 3.2 GB TotalSegmentator subset (resumable)."""
    from luxar.utils.download import robust_download

    with asection("Downloading TotalSegmentator subset (~3.2 GB)"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        robust_download(
            SUBSET_URL,
            CACHE_ZIP,
            max_retries=5,
            timeout=3600,
            expected_size=SUBSET_SIZE,
        )
    return CACHE_ZIP


def extract_subject(zip_path: Path, subject_id: str) -> Path:
    """Extract one subject's ct + segmentations from the subset zip."""
    import zipfile

    with asection(f"Extracting subject {subject_id}"):
        dest = CACHE_DIR / "subjects"
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Subjects are top-level dirs in the subset zip (e.g. `s0720/...`);
            # also tolerate a nested wrapper (`.../s0720/...`) defensively.
            members = [
                m
                for m in zf.namelist()
                if (m.startswith(f"{subject_id}/") or f"/{subject_id}/" in m)
                and m.endswith(".nii.gz")
            ]
            if not members:
                raise FileNotFoundError(
                    f"Subject {subject_id} not found in {zip_path.name}"
                )
            zf.extractall(dest, members)
        matches = list(dest.rglob(f"{subject_id}/ct.nii.gz"))
        if not matches:
            raise FileNotFoundError(f"ct.nii.gz missing for {subject_id}")
        aprint(f"Subject dir: {matches[0].parent}")
        return matches[0].parent


def build_label_volume(subject_dir: Path, shape: tuple[int, ...]) -> np.ndarray:
    """Combine per-structure binary masks → one int label volume via CLASS_MAP."""
    nib = require_module("nibabel")

    name_to_id = {name: lid for lid, name in CLASS_MAP.items()}
    labels = np.zeros(shape, dtype=np.int32)
    seg_dir = subject_dir / "segmentations"
    n = 0
    for name, lid in name_to_id.items():
        mask_path = seg_dir / f"{name}.nii.gz"
        if not mask_path.exists():
            continue
        m = np.asanyarray(nib.load(mask_path).dataobj)
        labels[m > 0] = lid
        n += 1
    aprint(f"Combined {n} structure masks into label volume")
    return labels


def load_ct_and_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fit_volume, label_volume, spacing_mm) on a shared cubic grid.

    fit_volume is the windowed CT masked to segmented anatomy; label_volume is
    the co-registered organ labels (nearest-neighbour resampled to the same grid).
    """
    nib = require_module("nibabel")
    from scipy.ndimage import zoom

    zip_path = download_subset()
    subject_dir = extract_subject(zip_path, SUBJECT_ID)

    with asection("Reading CT + segmentation"):
        ct_nii = nib.load(subject_dir / "ct.nii.gz")
        ct = np.asarray(ct_nii.get_fdata(), dtype=np.float32)
        spacing = np.asarray(ct_nii.header.get_zooms()[:3], dtype=np.float64)
        labels = build_label_volume(subject_dir, ct.shape)
        aprint(f"CT {ct.shape} spacing {tuple(round(s, 2) for s in spacing)} mm")

    with asection("Windowing + cropping to segmented anatomy"):
        body = labels > 0
        z0, z1, y0, y1, x0, x1 = crop_to_content(body, pad=2)
        ct = ct[z0:z1, y0:y1, x0:x1]
        labels = labels[z0:z1, y0:y1, x0:x1]
        fit_vol = window_ct(ct) * (labels > 0)

    with asection("Resampling to cubic voxels"):
        # cubic voxels at the finest spacing; cap the largest axis at TARGET_MAX_DIM
        phys = np.array(fit_vol.shape, dtype=np.float64) * spacing
        target_iso = float(spacing.min())
        out_shape = np.maximum(1, np.round(phys / target_iso)).astype(int)
        scale = float(TARGET_MAX_DIM) / float(out_shape.max())
        if scale < 1.0:
            out_shape = np.maximum(1, np.round(out_shape * scale)).astype(int)
        factors = out_shape / np.array(fit_vol.shape, dtype=np.float64)
        if not np.allclose(factors, 1.0):
            fit_vol = zoom(fit_vol, factors, order=1)
            labels = zoom(labels, factors, order=0).astype(np.int32)
        aprint(f"Fit grid {fit_vol.shape} (cubic {target_iso:.2f} mm voxels)")

    return fit_vol.astype(np.float32), labels, spacing


# =============================================================================
# Fit
# =============================================================================


def fit_atlas(
    fit_vol: np.ndarray, label_vol: np.ndarray
) -> tuple[GSplatData, np.ndarray]:
    """Fit splats to the CT, sample the per-splat organ label, cache both."""
    global DEVICE
    if DEVICE is None:
        DEVICE = detect_device()

    from luxar.gsplats import fit_progressive_gaussian_splats

    with asection(f"Fitting GSplats ({fit_vol.shape}, max {MAX_SPLATS:,}, {DEVICE})"):
        result = fit_progressive_gaussian_splats(
            fit_vol,
            max_splats=MAX_SPLATS,
            max_splats_per_pass=MAX_SPLATS_PER_PASS,
            iters_per_pass=ITERS_PER_PASS,
            psnr_patience=PSNR_PATIENCE,
            device=DEVICE,
            verbose=True,
        )
        aprint(f"Fitted {len(result.amplitudes):,} splats")

    with asection("Sampling per-splat organ labels"):
        labels = sample_labels(label_vol, result.centers)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.save(
        CACHE_FIT,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=False,
        compress="zip",
        zip_deflate=True,
    )
    _save_labels_u8(labels, CACHE_LABELS)
    return result, labels


def load_or_build() -> tuple[GSplatData, np.ndarray]:
    """Return (fit, per-splat labels), self-contained on a fresh system."""
    if not RECOMPUTE:
        if CACHE_FIT.exists() and CACHE_LABELS.exists():
            precomputed = load_precomputed_gsplats(
                DEMO_NAME, [FIT_FILE], recompute=False
            )
            if precomputed is not None:
                return precomputed[0], _load_labels(CACHE_LABELS)
        if (
            LFS_FIT.exists()
            and LFS_LABELS.exists()
            and not is_lfs_pointer(LFS_FIT)
            and not is_lfs_pointer(LFS_LABELS)
        ):
            fit = GSplatData.load(LFS_FIT)
            return fit, _load_labels(LFS_LABELS)
        aprint(
            "Precomputed atlas not available (Git LFS assets not pulled). "
            "Falling back to download + fit (one-time; result is cached)."
        )

    warn_if_no_cuda_gpu()
    fit_vol, label_vol, _ = load_ct_and_labels()
    return fit_atlas(fit_vol, label_vol)


# =============================================================================
# Scene
# =============================================================================


def create_luxar_scene(fit: GSplatData, labels: np.ndarray, output_path: Path) -> Path:
    """Build the CT anatomical-atlas scene, split into per-tissue toggle layers."""
    with asection("Creating Luxar Scene"):
        centered = fit.center_at_centroid().scale_intensity(SCENE_INTENSITY)
        palette = organ_palette()
        colors = label_colors(labels, palette)
        layer_idx = splat_layer_indices(labels)
        # Per-splat hover text: the specific organ name (all 117 tissue types).
        name_lut = [organ_label_text(i) for i in range(118)]
        dims = Dimensions(
            [
                Dimension("x", unit="mm", display=True),
                Dimension("y", unit="mm", display=True),
                Dimension("z", unit="mm", display=True),
            ]
        )
        # Coronal (front) view: the body's long axis is z (last center column),
        # so put z up and look along -y. Auto-framing otherwise picks an
        # end-on axial view that foreshortens the ~668 mm tall body into a ring.
        c = centered.centers
        lo = np.percentile(c, 1, axis=0)
        hi = np.percentile(c, 99, axis=0)
        center = (lo + hi) / 2.0
        height = float(np.max(hi - lo))
        fov_deg = 45.0
        cam_dist = (height * 0.5) / np.tan(np.radians(fov_deg) / 2.0) * 1.2
        camera = CameraConfig(
            position=(float(center[0]), float(center[1] - cam_dist), float(center[2])),
            target=(float(center[0]), float(center[1]), float(center[2])),
            up=(0.0, 0.0, 1.0),
            fov=fov_deg,
            near=float(max(0.1, cam_dist * 0.01)),
            far=float(cam_dist * 10.0 + height * 5.0),
        )
        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(tone_mapping="ACES", camera=camera),
            )
            scene.attrs["title"] = "GSplats: CT Anatomical Atlas (TotalSegmentator)"
            # One toggle-able layer per tissue supergroup (Layers panel); each
            # splat keeps its specific organ name as a hover tooltip.
            centers = centered.centers
            amps = centered.amplitudes
            chol = centered.cholesky_factors
            for i, (layer_name, _groups, opacity, amp_boost) in enumerate(SUPERGROUPS):
                mask = layer_idx == i
                n = int(mask.sum())
                if n == 0:
                    continue
                lids = labels[mask]
                scene.add_gsplats(
                    name=layer_name,
                    centers=centers[mask],
                    amplitudes=(amps[mask] * amp_boost).astype(np.float32),
                    cholesky_factors=chol[mask],
                    colors=colors[mask].astype(np.float32),
                    labels=[name_lut[int(lid)] for lid in lids],
                    opacity=float(opacity),
                    absorption=1.0,
                    blending_mode="volumetric",
                    layer=True,
                )
                aprint(
                    f"Layer '{layer_name}': {n:,} splats "
                    f"(opacity {opacity}, ×{amp_boost} amplitude)"
                )
            scene.add_text(
                "CT Anatomical Atlas — organs in color",
                position=(0.02, 0.02),
                font_size=0.045,
                anchor="top-left",
                color="rgba(255,255,255,0.7)",
                blend_mode="difference",
            )
            scene.add_text(
                "TotalSegmentator • CT + 117-organ segmentation → Gaussian splats",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.5)",
            )
        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    aprint("=" * 70)
    aprint("GSplats Demo: CT Anatomical Atlas (TotalSegmentator)")
    aprint("=" * 70)

    output_path = get_demos_output_dir() / "gsplats_3d_ct_totalsegmentator.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    fit, labels = load_or_build()
    aprint(f"Splats: {len(fit.amplitudes):,}")
    scene_path = create_luxar_scene(fit, labels, output_path)

    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("Data credit: TotalSegmentator (Wasserthal et al. 2023; CC BY 4.0)")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
