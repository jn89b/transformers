"""
This is just making sure I really do understand 
how the linear function from torch works

https://www.tutorialexample.com/understand-pytorch-f-linear-with-examples-pytorch-tutorial/
https://stackoverflow.com/questions/54916135/what-is-the-class-definition-of-nn-linear-in-pytorch

y = x*W^T + b
"""

import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.utils.data as data
import math 

#tensor toy example
x = torch.tensor([[1.0, -1.0],
                  [0.0,  1.0],
                  [0.0,  0.0]])

#this tensor is shaped 3x2
in_features = x.shape[1]  # = 2
out_features = 2
m = nn.Linear(in_features, out_features)

# therefore m is size 2 x 2 
print("weights", m.weight)
print("bias", m.bias)

# Now let's feed the tensor and conduct a linear transformation of it
y_short = m(x)
print("y short", y_short)

# this is the explicit demonstration of how this works
# y = x*W^T + b
y_explicit = torch.matmul(x, m.weight.T) + m.bias
print("y_explicit explicit", y_explicit)
print("type", type(y_explicit))