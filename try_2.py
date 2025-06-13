import torch
a = [torch.tensor(1.), torch.tensor(3.), torch.tensor(8.), torch.tensor(10.)]
b = torch.tensor(a)
print(b)
print(b.mean())