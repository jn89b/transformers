import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

"""
To start off I need to build the Multi Headed Attention network
Then build the encoder
THen build the decode
"""

m = nn.Linear(20,20)
stuff = torch.rand(128,20)
output = m(stuff)
print(output.size())
 
class MultiHeadAttention(nn.Module):
    """
    Refer to https://towardsdatascience.com/all-you-need-to-know-about-attention-and-transformers-in-depth-understanding-part-2-bf2403804ada
    https://towardsdatascience.com/all-you-need-to-know-about-attention-and-transformers-in-depth-understanding-part-1-552f0b41d021#9c93
    3.1.3 Multi-Head Attention
    
    dim_model (int) Determines the size of the input/output embeddings
    num_attention_heads (int)  Determines how many separate attention mechanisms will be used in parallel
    """
    def __init__(self, 
                 model_dim:int,
                 num_attention_heads:int) -> None:
        
        self.model_dim = model_dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = model_dim // num_attention_heads
        
        # The linear function takes in an input vector of input_dim and spits out model_dim
        # so 2in 2out uses y = x*W^T + b
        self.W_query = nn.Linear(self.model_dim, self.model_dim)
        self.W_key   = nn.Linear(self.model_dim, self.model_dim)
        self.W_value = nn.Linear(self.model_dim, self.model_dim)
        self.W_output = nn.Linear(self.model_dim, self.model_dim)
        
    def scaled_dot_production_attention(self, Q, K, V, mask=None) -> nn.Linear:
        """
        Scaled dot product application
        
        head_dim is d_k from the paper
        """
        attention_score = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        
        attention_soft = torch.softmax(attention_score)
        
        return torch.matmul(attention_soft, V)