import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Critic 1
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

        # Critic 2
        self.layer4 = nn.Linear(state_dim + action_dim, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        # Critic 1 forward
        q1 = torch.relu(self.layer1(sa))
        q1 = torch.relu(self.layer2(q1))
        q1 = self.layer3(q1)

        # Critic 2 forward
        q2 = torch.relu(self.layer4(sa))
        q2 = torch.relu(self.layer5(q2))
        q2 = self.layer6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = torch.relu(self.layer1(sa))
        q1 = torch.relu(self.layer2(q1))
        q1 = self.layer3(q1)
        return q1
