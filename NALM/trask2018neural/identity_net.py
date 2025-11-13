import torch.nn as nn
from nac import NAC

class IdentityNet(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            activation,
            nn.Linear(8, 8),
            activation,
            nn.Linear(8, 8),
            activation,
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


class IdentityNetRes(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            activation,
            nn.Linear(8, 8),
            activation,
            nn.Linear(8, 8),
            activation,
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return x + self.net(x)


class IdentityNetNAC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            NAC(1, 8),
            NAC(8, 8),
            NAC(8, 8),
            NAC(8, 1)
        )

    def forward(self, x):
        return self.net(x)
