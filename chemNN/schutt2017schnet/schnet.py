import torch
import torch.nn as nn
import torch.nn.functional as F

class SchNet(nn.Module):
    def __init__(self, num_atoms, num_features, num_interactions, num_kernels):
        super().__init__()
        # Embedding layer to convert atom types to feature vectors
        self.embedding = nn.Embedding(num_atoms, num_features)
        self.interaction_blocks = nn.ModuleList([InteractionBlock(num_features, num_kernels) for _ in range(num_interactions)])
        self.output_block = nn.Sequential(
            nn.Linear(num_features, 32),
            ShiftedSoftplus(),
            nn.Linear(32, 1),
            nn.AdaptiveAvgPool1d(1)  # Sum pooling across atoms
        )

    def forward(self, data):
        x, r, batch = data.x, data.r, data.batch
        x = self.embedding(x)
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, r)
        x = self.output_block(x)
        return x.view(-1)
        

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.log(torch.tensor(2.0)))

        
    def forward(self, x):
        return F.softplus(x) - self.shift
    

class InteractionBlock(nn.Module):
    def __init__(self, num_features, num_kernels):
        super().__init__()
        self.num_features = num_features
        self.num_kernels = num_kernels

        # Linear transformation before CFConv
        self.atom_wise_linear = nn.Linear(num_features, num_features)
        # Continuous-filter convolution layer
        self.cfconv = CFConv(num_features, num_kernels)
        # Neural network after CFConv
        self.net = nn.Sequential(
            nn.Linear(num_features, num_features),
            ShiftedSoftplus(),
            nn.Linear(num_features, num_features)
        )

    def forward(self, x, r):
        o = self.atom_wise_linear(x)
        o = self.cfconv(o, r)
        o = self.net(o)
        return o + x
        

class CFConv(nn.Module):
    def __init__(self, num_features, num_kernels=301, gamma=10):
        super().__init__()
        self.num_features = num_features
        self.num_kernels = num_kernels

        self.gamma = gamma
        # Create a set of Gaussian kernels
        self.register_buffer('centers', torch.linspace(0, 30, num_kernels))

        # Neural network to process the expanded distances
        self.net = nn.Sequential(
            nn.Linear(num_kernels, num_features),
            ShiftedSoftplus(),
            nn.Linear(num_features, num_features),
            ShiftedSoftplus(),
        )

    def forward(self, x, r):
        # Calculate pairwise distances between atoms
        d = torch.cdist(r, r, p=2.0)
        # Expand distances using Gaussian kernels
        expanded = torch.exp(-self.gamma*torch.pow(d.unsqueeze(-1) - self.centers, 2))
        # Apply neural network to expanded distances
        w = self.net(expanded)

        # Perform the convolution operation
        o = torch.einsum('jk,ijk->ik', x, w)
        return o
