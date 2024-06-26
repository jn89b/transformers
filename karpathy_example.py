import torch 
import torch.nn as nn
from torch.nn import functional as F
"""
Refer to this video
https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy
"""

"""
SELF-ATTENTION MECHANISM when Karpathy is talking about wei he means
the attention value (the dot product application)
"""
## The key mechnaism of self-attention! 
torch.manual_seed(1337)
B,T,C = 4,8,32 #batch. time, and channels 
x = torch.randn(B,T,C)

# this returns the lower triangle of the matrix filled with ones
# what's going happen to our wei is from our positional encoding 
# we will have two vectors one for query and one for key
# we will apply the dot product of the individual query to each of our queries
# and the more similiar we are the more information we can obtain because it has
# a higher score or similiarity (more close to 1)
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
# now what we are gonna do is basically fill up our wei matrix
# inside the wei matrix we want to put -inf on the upper triangle of this matrix 
wei = wei.masked_fill(tril==0, float('-inf'))
print("wei", wei)
# we now apply a softmax to our wei matrix to normalize the values between 0 and 1
# what should happen is the upper triangle should be 0's 
wei = F.softmax(wei, dim=-1)
print("wei now", wei)
out = wei @ x
print("out", out)
