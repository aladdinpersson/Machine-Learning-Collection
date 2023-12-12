import torch

# Create a 3D tensor with shape (2, 3, 4)
tensor = torch.rand(2, 3, 4)

# We'll swap the first and second dimension using torch.transpose
transposed = tensor.transpose(0, 1)  # shape: 3x2x4

# Now let's permute the tensor dimensions with torch.permute
permuted = tensor.permute(2, 0, 1)  # shape: 4x2x3
permuted_like_transpose = tensor.permute(1, 0, 2) # shape: 3x2x4
