from __future__ import annotations
import torch
from torch import nn
from jaxtyping import Float
from torch import Tensor

class LayerNorm(nn.Module):
    """Layer Normalization"""

    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gemma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x: Float[Tensor, "... n_embd"]) -> Float[Tensor, "... n_embd"]:
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_var = torch.var(x, dim=-1, keepdim=True)
        normalized = (x - x_mean) / torch.sqrt(x_var + self.eps)
        return self.gemma * normalized + self.beta