import torch

x = torch.randn(4, 4)

print("Before chunking:")
print(x)

print("After chunking:")
for i in x.chunk(2, dim=1):
    print(i)