import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NAC(nn.Module):
    """Implementation of the Neural Accumulator Cell (Trask et al., 2018).

    The NAC parametrises a weight matrix ``W`` constrained to ``[-1, 1]`` by
    applying ``tanh`` and ``sigmoid`` transforms to two unconstrained parameter
    tensors ``w_hat`` and ``m_hat``.  Initialising these tensors near zero keeps
    the effective weights close to linear addition/subtraction, which mirrors
    the behaviour described in the original paper.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.w_hat = nn.Parameter(torch.empty(output_dim, input_dim))
        self.m_hat = nn.Parameter(torch.empty(output_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise parameters with a small uniform distribution."""
        nn.init.xavier_uniform_(self.w_hat)
        nn.init.xavier_uniform_(self.m_hat)

    def effective_weights(self) -> torch.Tensor:
        return torch.tanh(self.w_hat) * torch.sigmoid(self.m_hat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weights())

class NALU(nn.Module):
    """Neural Arithmetic Logic Unit as defined in Trask et al. (2018)."""
    def __init__(self, input_dim: int, output_dim: int, eps: float = 1e-8,
                 log_floor: float = 1e-7, gate_bias: bool = True,
                 clamp_exp: float = 20.0):
        super().__init__()
        self.g = nn.Linear(input_dim, output_dim, bias=gate_bias)
        self.a = NAC(input_dim, output_dim)
        self.m = NAC(input_dim, output_dim)
        self.register_buffer("eps", torch.tensor(eps))
        self.log_floor = log_floor
        self.clamp_exp = clamp_exp

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        g = torch.sigmoid(self.g(x))
        a = self.a(x)
        ax = x.abs().clamp_min(self.log_floor)
        eps = self.eps.to(dtype=x.dtype, device=x.device)
        m_x = (ax + eps).log()
        m = torch.exp(self.m(m_x).clamp(-self.clamp_exp, self.clamp_exp))
        return g * a + (1 - g) * m
