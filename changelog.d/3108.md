#### CUDA splatting and NLM kernels without a build step (NVRTC)

A plain `pip install "luxar[gsplats]"` on Linux with an NVIDIA GPU now gets the
custom CUDA kernels. Until now they existed only as a torch C++ extension built
by `make build-cuda`, which needs a source checkout, `nvcc` and a C++ compiler,
so a pip install fitted on the PyTorch renderer: a 100-iteration 128³ fit
with 2,000 splats took 136 s instead of 8 s on an RTX 3070. The same kernel
sources ship in the wheel and are now compiled on first use by NVRTC, the CUDA runtime compiler inside the NVIDIA
wheels that a CUDA build of torch already installs, then loaded and launched
through the CUDA driver API with `ctypes`. Nothing new is installed and there
is no torch C++ ABI to match. Each dimensionality compiles on its own the first
time it is used, and the binary is cached under
`$XDG_CACHE_HOME/luxar/cuda/` (default `~/.cache`). A corrupt or unwritable
cache only costs a recompile. A GPU newer than the installed NVRTC gets PTX that
the driver compiles. On an RTX PRO 6000 (sm_120) and an RTX 3070 (sm_86) the
runtime-compiled kernels give the same gradients as the extension, bit for
bit, and forward volumes that differ only by float atomic-accumulation order
(relative L2 about 5e-8); fit speed matches. The parity suite also passes on a
DGX Spark GB10 (sm_121, aarch64) installed from the wheel.

The extension is still used first when it imports, so checkouts and HPC
installs that build it behave as before. `LUXAR_CUDA_BACKEND=aot|nvrtc|torch`
forces a path; a forced kernel path that cannot run raises instead of falling
back. The fit log names the kernels in use, and the warning when none can run
now gives the reason instead of telling pip users to run `make build-cuda`. The
non-local-means denoising kernels follow the same order.

`cholesky_to_conic` runs under `torch.compile`, which only compiles at its first
call. On a machine without a working C compiler or the Python headers Triton
needs, that call raised and aborted every fit on the custom-kernel path. It now
falls back to eager execution for the rest of the process and warns once.
