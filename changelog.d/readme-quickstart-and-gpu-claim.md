#### Correct the quick start and the fitting hardware requirement

The quick start ran `make setup-dev` and then `luxar demo`, but `setup-dev`
creates the Hatch environment without activating it, so the very next command
could not resolve. It now activates the environment first.

The prerequisites also said Gaussian-splat fitting "needs an NVIDIA CUDA GPU".
It does not: fitting runs on the CPU, and CUDA or Apple MPS make it faster. The
same README already said so in its GPU-acceleration table, so the two statements
disagreed.
