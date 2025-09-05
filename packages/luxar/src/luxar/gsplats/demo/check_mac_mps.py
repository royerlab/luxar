import torch

from luxar.gsplats.models.utils.lt_solver import solve_lower_triangular

dev = torch.device("mps")  # or "cpu" to compare
d = 2
K = 8
P = 4096

L = torch.zeros((K, d, d), device=dev)
L[:, 0, 0] = 1.2
L[:, 1, 1] = 0.9
L[:, 1, 0] = 0.3  # some off-diagonal

delta = torch.randn((K, d, P), device=dev) * 3.0


y = solve_lower_triangular(L, delta)  # should be finite
assert torch.isfinite(y).all(), "NaNs/Infs from triangular solve on MPS"
expo = (y * y).sum(1).clamp(0, 1e6)
g = torch.exp(-0.5 * expo)
assert torch.isfinite(g).all(), "NaNs/Infs from exp on MPS"
print("MPS path looks good ✅")
