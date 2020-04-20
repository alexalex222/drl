import torch

q = torch.zeros(10, 2)
idx = torch.randint(0, 2, (10,))

q[torch.arange(10), idx] = q[torch.arange(10), idx] + 1