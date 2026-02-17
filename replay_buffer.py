import random
import numpy as np

class ReplayBuffer:

    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def push(self, s, a, r, ns, d):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,ns,d = map(np.array, zip(*batch))
        return s,a,r,ns,d

    def __len__(self):
        return len(self.buffer)
