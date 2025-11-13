import torch.nn as nn
import torch


class NAC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w_hat = nn.Parameter(torch.randn(input_dim, output_dim))
        self.m_hat = nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, x):
        W = torch.tanh(self.w_hat) * torch.sigmoid(self.m_hat)
        return torch.matmul(x, W)

class NALU(nn.Module):
    def __init__(self, input_dim, output_dim, eps=1e-10):
        super().__init__()
        self.g = nn.Linear(input_dim, output_dim, bias=False)
        self.a = NAC(input_dim, output_dim)
        self.m = NAC(input_dim, output_dim)
        self.eps = eps

    def forward(self, x):
        g = torch.sigmoid(self.g(x))
        a = self.a(x)
        m_x = torch.log(torch.abs(x) + self.eps)
        m = torch.exp(self.m(m_x))
        return g * a + (1 - g) * m