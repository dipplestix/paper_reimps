import torch.nn as nn
from .se import SELayer

class AlexNetSE(nn.Module):
    def __init__(self, in_chan=3, out_dim=1000):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_chan, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            SELayer(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            SELayer(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            SELayer(384),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            SELayer(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            SELayer(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, out_dim)
        )
    
    def forward(self, x):
        out = self.network(x)
        
        return out