> **⚠️ Archived — historical document, not maintained.** Kept for design history; it reflects the project state as of its original date and may not match current code. Do not treat it as current guidance. See [the archive README](../README.md) for status labels and retention policy.

The name **"Hierarchical Tensor-Gaussian Pursuit"** is **not a standard, pre-existing algorithm** in the literature. I coined the term during our discussion to describe the specific pipeline we constructed.

However, the **mathematical components** are well-established, and you are effectively synthesizing a novel "constructive" initialization method from three distinct fields:

1. **Non-negative Tensor Factorization (NTF/NCP):** Used heavily in **Hyperspectral Unmixing** and **Blind Source Separation**. In those fields, scientists use it to separate mixed signals into pure components (Rank-1 tensors).
2. **Multi-scale Blob Detection:** Standard in computer vision (e.g., Laplacian of Gaussian), used to find features at specific sizes.
3. **Gaussian Splatting:** A modern rendering technique that typically relies on Structure-from-Motion (SfM) points for initialization.

### Why this combination is unique (and clever)

Standard Gaussian Splatting papers (like 3DGS) initialize Gaussians using a **Sparse Point Cloud** from COLMAP (SfM). They do **not** typically use Tensor Decomposition to find the Gaussians constructively from the image volume itself.

Your approach is effectively **"Inverse Rendering via Tensor Factorization."** You are using the fact that a 3D image is a sum of light blobs, and a tensor decomposition is the mathematically optimal way to unmix sums of separable functions.

### The Algorithm's Lineage

While the *combination* is unique to your use case, the roots run deep in signal processing:

| Concept | Standard Name in Literature | Your Adaptation |
| --- | --- | --- |
| **The "What"** | **Blind Source Separation (BSS)** | Using NCP to unmix overlapping 3D Gaussians. |
| **The "Where"** | **Difference of Gaussians (DoG)** | Using Scale Decomposition to isolate blob sizes. |
| **The "Shape"** | **Rank-1 Approximation** | Using the outer product  to approximate axis-aligned blobs. |

### Visualizing the "Novelty"

Standard methods optimize parameters *descriptively* (trying to match the image). Your method finds parameters *constructively* (building the image from factors).

*This diagram shows how a Rank-1 tensor is formed by the outer product of vectors, which is the mathematical equivalent of an axis-aligned Gaussian.*

### Summary

* **Name:** "Hierarchical Tensor-Gaussian Pursuit" (Coined here).
* **Status:** A novel engineering synthesis of **NTF** (for separation) and **Hessian-based Detection** (for localization).
* **Closest Relative:** **Hyperspectral Unmixing**, where a 3D "cube" of data is factored into spatial maps and spectral signatures. You are simply treating the "spectral" dimension as just another spatial dimension (Z-axis).
