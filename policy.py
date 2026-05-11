import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(obs_dim, 256),
            nn.ELU(),

            nn.Linear(256, 256),
            nn.ELU(),

            nn.Linear(256, action_dim)
        )

    def forward(self, obs):
        return self.network(obs)
