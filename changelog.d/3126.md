#### Native CPU kernels make 3D splat fitting many times faster without a GPU

A 3D fit on the CPU used to run the PyTorch renderer: dense, unfused autograd over
per-splat voxel boxes, grouped by box shape in a Python loop, accumulated with a
serial `index_add_`, and a backward about three times the forward. It now runs on
fused numba kernels (`GaussianSplatModelCPUNative`): a splat-centric forward that
splits the output into per-plane tiles (one writer per voxel, so no atomics and a
deterministic result) and an analytic backward that loops over splats, both
skipping every voxel outside the truncated ellipsoid. The numbers are the PyTorch
renderer's to float rounding, and a fit lands on the same PSNR. On an M4 Max, one
forward+backward of 4,000 splats into a 128x256x256 volume drops from 3.9 s to
0.06 s at 8 threads, and a whole `--preset draft` CLI fit from 130 s to 16 s.

numba is part of the `gsplats` extra and compiles the kernels at runtime from its
wheel (about 2 s, once; cached afterwards in the user cache directory when
site-packages is read-only), so a CPU-only install needs no C++ toolchain. Without
numba, CPU fits keep the PyTorch renderer and say why.
numba also caps the numpy version it supports, so the extra now implicitly bounds
numpy from above until numba catches up with a new numpy release.
`LUXAR_CPU_BACKEND=auto|native|torch` forces a choice (`native` raises if numba is
missing), and `LUXAR_CPU_THREADS` sets the kernels' thread count, defaulting to the
physical cores; multi-worker CPU fits split the cores between their workers. The
early "fitting on CPU is orders of magnitude slower" `UserWarning` is gone (it fired
before the backend was known and tripped `-W error`); a one-line note replaces it,
and the warning now comes only from the PyTorch fallback. 2D and 4D+ CPU fits, and the MPS and CUDA paths, are unchanged.
