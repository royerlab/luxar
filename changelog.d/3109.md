#### Metal splatting kernels compile at runtime, no Xcode needed

On Apple silicon, the 3D Gaussian-splat kernels in `kernels.metal` are now compiled
on first use by `torch.mps.compile_shader`, i.e. by the Metal compiler built into
macOS, so `pip install "luxar[gsplats]"` gets the fast kernels with no Xcode and no
compiled extension tied to one torch ABI (torch >= 2.7; 2.6 through torch's private
API). The runtime path matches the ahead-of-time extension to float32 rounding and
is as fast or faster (1.5 ms vs 1.9 ms per 128³/2k-splat training step on an M4
Max; the PyTorch fallback takes 646 ms).

Importing the Metal package no longer tries to build the extension into the package
directory or prints an Xcode banner. The extension is now an explicit developer
build (`python setup.py build_ext --inplace --force`); when built and up to date it
is still preferred in `auto`, mirroring the CUDA backend order.
`LUXAR_METAL_BACKEND=auto|aot|runtime|torch` overrides the choice (`runtime`
selects the equivalent runtime path even when the extension is built); a forced
backend that cannot run raises, in fitting too, and a fit that falls back to
PyTorch on MPS warns with the reason.
