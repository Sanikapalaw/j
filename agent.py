import torch
import torch.nn.functional as F
import numpy as np
from model import DQN

class Agent:

    def __init__(self, state_dim, action_dim):

        self.model = DQN(state_dim, action_dim)
        self.target = DQN(state_dim, action_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995

        self.action_dim = action_dim

    def act(self, state):

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_vals = self.model(state)
        return torch.argmax(q_vals).item()

    def train_step(self, buffer, batch_size=32):

        if len(buffer) < batch_size:
            return

        s,a,r,ns,d = buffer.sample(batch_size)

        s = torch.FloatTensor(s)
        ns = torch.FloatTensor(ns)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)

        q_vals = self.model(s)
        next_q = self.target(ns).max(1)[0].detach()

        target = r + self.gamma * next_q
        current = q_vals.gather(1, a.unsqueeze(1)).squeeze()

        loss = F.mse_loss(current, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.eps_min,
                           self.epsilon * self.eps_decay)
