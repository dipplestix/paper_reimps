import torch.nn as nn
import torch.nn.functional as F
import torch


class ContrastiveLoss(nn.Module):
    def __init__(self, alpha=1, beta=1, m=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.m = m

    def forward(self, o1, o2, y):
        d = F.pairwise_distance(o1, o2)

        loss = self.alpha*y*torch.pow(d, 2)
        loss += self.beta*(1 - y)*torch.pow(torch.clamp(self.m - d, min=0.0), 2)

        return loss.mean()
