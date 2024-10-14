import torch

# ===== USING torch.cat =====
# torch.cat concatenates tensors along an existing axis.
t1 = torch.randn(2, 3) 
t2 = torch.randn(2, 3)

cat_dim0 = torch.cat((t1, t2), dim=0) # shape: (4, 3)
cat_dim1 = torch.cat((t1, t2), dim=1) # shape: (2, 6)

# ===== USING torch.stack =====
# torch.stack concatenates tensors along a new axis, not existing one.
stack_dim0 = torch.stack((t1, t2), dim=0) # shape: 2x2x3
stack_dim2 = torch.stack((t1, t2), dim=2) # shape: 2x3x2
