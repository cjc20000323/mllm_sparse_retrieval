import torch

a = torch.rand(4)
l = [a, a]
b = torch.cat(l, dim=-1)
print(b)