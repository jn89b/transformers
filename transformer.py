import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

"""
To start off I need to build the Multi Headed Attention network
Then build the encoder
Then build the decode
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
    
    For the Attention is All You Need Paper refer to section 3.2
    
    dim_model (int) Determines the size of the input/output embeddings
    num_attention_heads (int)  Determines how many separate attention 
    mechanisms will be used in parallel
    
    """
    def __init__(self, 
                 model_dim:int,
                 num_attention_heads:int) -> None:
        
        self.model_dim = model_dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = model_dim // num_attention_heads
        
        # The linear function takes in an input vector of input_dim 
        # and spits out model_dim so 2in 2out uses y = x*W^T + b
        # W is shape of size (out_features,in_features)
        self.W_query = nn.Linear(self.model_dim, self.model_dim)
        self.W_key   = nn.Linear(self.model_dim, self.model_dim)
        self.W_value = nn.Linear(self.model_dim, self.model_dim)

        self.W_output = nn.Linear(self.model_dim, self.model_dim)
        
    def scaled_dot_production_attention(self, 
                                        Q:torch.tensor, 
                                        K:torch.tensor, 
                                        V:torch.tensor, 
                                        mask:bool=None) -> torch.Tensor:
        """
        In the Attention all you need paper refer to 3.2.1 subtitle
        
        - Need to figure out the size of this mug
        - Q and K must be dimension d_k
        - V must be dimension of d_v
        
        This is the magic equation used from the paper 
        to get the attention scores. When we train the network 
        we can mask out stuff put 0's stuff we don't care
        """
        attention_score = torch.matmul(Q, K.transpose(-2,-1)) / \
            math.sqrt(self.head_dim)
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        
        attention_soft = torch.softmax(attention_score)
        
        return torch.matmul(attention_soft, V)
    
    def split_heads(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, d_model = x.size()
        return x.transpose(1,2).contiguous().view(batch_size,
                                                  seq_length,
                                                  self.head_dim)
        
    def forward(self, Q:torch.tensor, K:torch.tensor, V:torch.tensor,
                mask=None) -> torch.Tensor:
        """
        This is the main pipeline of our forward pass for the multi-headed 
        attention mechanism refer to  
        """
        Q = self.split_heads(self.W_query(Q))
        K = self.split_heads(self.W_key(K))
        V = self.split_heads(self.W_value(V))
        
        attention_output = self.scaled_dot_production_attention(
            Q, K, V)
        output = self.W_output(self.combine_heads(attention_output))
        
        return output

class PositionWiseFeedForward():
    """
    In the Attention is All You Need paper refer to section 3.3:
    In addition to attention sub-layers, each of the layers in our encoder 
    and decoder contains a fully
    connected feed-forward network, which is applied to each 
    position separately and identically. This
    consists of two linear transformations with a ReLU activation in between.
    FFN(x) = max(0, xW1 + b1)W2 + b2 (2)

    While the linear transformations are the same across different positions, they use different parameters
    from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
    The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality
    df f = 2048.

    """
    def __init__(self, d_model:int,
                 d_ff:int) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.Relu()

    def forward(self, x:torch.tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
    
    
class PositionEncoding(nn.Module):
    """
    Refer to section 3.5 from Attention is All You need paper:
    Since our model contains no recurrence and no convolution, in order 
    for the model to make use of the
    order of the sequence, we must inject some information about 
    the relative or absolute position of the
    tokens in the sequence. To this end, we add "positional encodings" 
    to the input embeddings at the
    bottoms of the encoder and decoder stacks. The positional encodings 
    have the same dimension dmodel
    as the embeddings, so that the two can be summed. There are 
    many choices of positional encodings,
    learned and fixed [9].
    
    
    FUTURE WORK - ADJUST THIS and look at this for future reference:
    https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    Take home message is that this is customizable, the authors found this
    equation to work decently well but from hearing what Teddy said it seems 
    it didn't matter much from his work. 
    
    However it seems like 
    """
    def __init__(self, d_model:int, max_seq_length:int) -> None:
        super(PositionEncoding, self).__init__()
        
        pos_encoding = torch.zeros(max_seq_length, d_model)