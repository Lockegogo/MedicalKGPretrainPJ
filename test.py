import torch

a = torch.rand(5, 7)
b = torch.rand(5, 7)
c = (a * b).sum(-1).view(-1)
print(c.shape)

t = {"a": 1, "b": 2}
for k, v in t.items():
    print(k, v)
