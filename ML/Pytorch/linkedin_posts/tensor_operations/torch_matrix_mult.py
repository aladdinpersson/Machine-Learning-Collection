import torch

# Create two random matrices
a = torch.rand(2**10, 2**10)
b = torch.rand(2**10, 2**10)

# using mm (note only works for 2D tensors)
c = torch.mm(a, b)

# using matmul 
c = torch.matmul(a, b)

# using @ operator (note exact same as matmul)
c = a @ b
