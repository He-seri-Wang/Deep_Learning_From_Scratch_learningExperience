import torch
import numpy as np
import math

# Create a tensor directly from data
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print("x:", x)

# Create a tensor of zeros
y = torch.zeros(2, 2)
print("y:", y)

# Create a tensor of ones
z = torch.ones(2, 2)
print("z:", z)

# Create a random tensor
w = torch.rand(2, 2)
print("w:", w)

# Create a tensor from a NumPy array
np_array = np.array([1,2,3])
x_np = torch.from_numpy(np_array)
print("x_np:", x_np)
tensor = torch.tensor([[1, 2, 3], [3, 4, 5]])

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Move the tensor to GPU if available
if torch.cuda.is_available():
  tensor = tensor.to("cuda")

# Standard numpy-like indexing and slicing
tensor = torch.tensor([[1,2,3], [3,4,5]])
print("First row: ", tensor[0])
print("First column: ", tensor[:,0])

# Matrix multiplication
tensor = torch.ones(3, 3)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
print("y1: ", y1)
print("y2: ", y2)

# Element wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)
print("z1: ", z1)
print("z2: ", z2)

# common functions
a = torch.rand(2, 4) * 2 - 1
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

# Reshape
a = torch.arange(4.)
a_reshaped = torch.reshape(a, (2, 2))
b = torch.tensor([[0, 1], [2, 3]])
b_reshaped =torch.reshape(b, (-1,))
print("a_reshaped", a_reshaped)
print("b_reshaped", b_reshaped)

x1 = torch.tensor([[1, 2, 3], [3, 4, 5]])
x2 = torch.tensor([2,2,2])
doubled = x1 * x2

print(doubled)