import torch
a = torch.rand([2, 4])
b = torch.rand([3, 4])
c = torch.cat([a, b])
print(c.shape)
