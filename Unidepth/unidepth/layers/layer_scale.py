import torch
import torch.nn as nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values = 1e-5,
        inplace = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
