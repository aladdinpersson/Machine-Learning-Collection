import torch
import torch.nn as nn

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
x=torch.empty(size=(3,3)).uniform_(0,1)
y=torch.diag(torch.ones(3))
z=torch.ones(3)
print(x)
print(y)
print(z)
import numpy as np
a = np.array([1, 2, 3])
b = torch.from_numpy(a)#这里是将numpy数组转换为tensor
print(a)
print(b)
c=b.numpy()#这里是将tensor转换为numpy数组
print(c.dtype)
import torch
x = torch.tensor([1, 2, 3])
print(torch.diag(x))
# 输出:
# tensor([[1, 0, 0],
#         [0, 2, 0],
#         [0, 0, 3]])
A = torch.tensor([[1, 2], [3, 4]])
print(torch.diag(A))
# 输出: tensor([1, 4])
p=torch.rand(3, 4)
print(p)
q=torch.eye(4)
print(q)
z=torch.empty(3,4).normal_(mean=0,std=1)
print(z)
j=torch.arange(1,10,2)
print(j)
k=torch.empty(3,4)
print(k)