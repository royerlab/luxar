import torch
from arbol import aprint

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    aprint(x)
else:
    aprint("MPS device not found.")
