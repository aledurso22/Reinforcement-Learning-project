import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, X):
        return self.model(X)


