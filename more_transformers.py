import torch 
import torch.nn as nn
import math as m
from torch.nn import functional as F

"""
Based on Andrew Karpathy notes as well as Teddy's code 
"""

# hyperparamters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device ='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.1
# ---- 

class SingleHeadAttention(nn.Module):
    """
    Single vector
    """
    def __init__(self, num_embed:int, head_size:int,
                 dropout=0.1, to_mask:bool=True) -> None:
        
        self.num_embed = num_embed
        self.head_size = head_size
        
        self.query  =  nn.Linear(self.num_embed, self.head_size, bias=False)
        self.key    =  nn.Linear(self.num_embed, self.head_size, bias=False)
        self.value  =  nn.Linear(self.num_embed, self.head_size, bias=False)
        self.output =  nn.Linear(self.num_embed, self.head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer('tril', torch.tril(torch.ones(num_embed, num_embed)))
        #set true or false to mask the attention
        self.to_mask = to_mask
        
        ## why drop out -> regularization
        self.dropout = nn.Dropout(dropout)        

    def forward(self, x:torch.Tensor,
                to_mask:bool=True)-> torch.Tensor:
        """
        The production of the query and keys are all done 
        in parallel without the need of one relying on the other
        (sequentially)
        Think of it like this:
        You have B number of batches computed all seperately

        There might be times when you won't need to mask the 
        matrix if you are doing sentiment analysis 
        
        """
        
        #B,T,C Batch, Time, and Channels
        # from paper that is the 
        B,T,C = x.shape 
        
        #grab the query,key, and values 
        q = self.query(x) #(B,T, head_size)
        k = self.key(x)   #(B,T, head_size)
        v = self.value(x)
        
        # we now compute the attention scores 
        # shape will be (B,T,C) @ (B,C,T) --> (B,T,T)
        # we divide by the square root because we zero mean variance 
        # if we don't do this the attention score can be higher than one 
        # what will happen is the soft max will become more like a one hot
        # encoding feature where it will create a sharpened distrubution
        # of values
        attention_score = torch.matmul(q,k.transpose(-2,-1)) / m.sqrt(self.head_size) 
        print("attention score shape", attention_score.shape)
        
        # now we apply the mask to succeeding tokens since we want to 
        # don't want our network to cheat and have the answer already 
        # and then the softmax to normalize our attention
        # if you think about it its a graph from the current token 
        # to the other preceding token (think linked list)
        """
        ie if its 4 x 4 x ...
        [[1,    mask, mask, mask],
         [0.5,  0.5,  mask, mask],
         [0.33, 0.33, 0.33, mask],
         [0.25, 0.25, 0.25, 0.25]]
        """
        tril = torch.nn(torch.ones(T,T))
        if to_mask:
            attention_score = attention_score.masked_fill(tril==0, float('-inf'))
        
        attention_score = F.softmax(attention_score, dim=-1)

        attention_score = self.dropout(attention_score)
        output = torch.matmul(attention_score, v)
        print("output shape is", output.shape)
        
        return output
        
class MultiHeadAttention(nn.Module):
    """
    Stacked versino of the SingleHeadAttention class 
    """
    def __init__(self, num_embed:int, head_size:int,
                 dropout=0.1, to_mask:bool=True) -> None:
        super().__init__()
        
        self.num_embed = num_embed
        self.head_size = head_size
        self.dropout   = nn.Dropout(dropout)
        self.to_mask   = to_mask
        
        self.heads = nn.ModuleList([SingleHeadAttention(num_embed,
                                                        head_size, 
                                                        dropout, 
                                                        False)] for _ in range(head_size))
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        
        return x 
    
class FeedForward(nn.Module):
    """
    Refer to section 3.3 from paper Attention is All You Need
    """
    def __init__(self, model_dim:int=512, dropout:float=0.1) -> None:
        super().__init__()
        self.model_dim = model_dim
        #multiply by 4 because in the paper that's what they did
        self.dim_ff = model_dim * 4
        self.linear_1 = nn.Linear(model_dim, self.dim_ff)
        self.linear_2 = nn.Linear(self.dim_ff, model_dim)
        self.relu     = nn.ReLU()
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.relu(x)
        output = self.linear_2(x)
        
        return output
    
class EncoderBlock(nn.Module):
    """
    This is the e
    """
    def __init__(self, num_embed:int, num_head:int,
                 dropout:float=0.1) -> None:
        super().__init__()
        
        self.num_embed = num_embed
        self.num_head  =  num_head 
        self.dropout = dropout
        # if you have num_embedding = 32 and you have 
        # 4 num_heads your head size will be 8
        self.head_size = num_embed // num_head
        
        #refer to paper on the order of mechanism
        self.multi_head = MultiHeadAttention(num_embed=num_embed,
                                             head_size=self.head_size,
                                             dropout=dropout)
        self.feedforward = FeedForward(num_embed, dropout)
        self.norm1 = nn.LayerNorm(num_embed)
        self.norm2 = nn.LayerNorm(num_embed)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Refer to 3.1 of paper 
        The encoder is composed of a stack of N = 6 identical layers. 
        Each layer has two sub-layers. The first is a multi-head 
        self-attention mechanism, and the second is a simple, 
        positionwise fully connected feed-forward network. 
        
        THIS IS RESIDUAL NEURAL NETWORK
        We employ a residual connection [11] around each of
        the two sub-layers, followed by layer normalization [1]. 
        
        THIS IS THE PART RIGHT HERE
        That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), 
        where Sublayer(x) is the function implemented by the sub-layer
        itself. To facilitate these residual connections, all sub-layers 
        in the model, as well as the embedding layers, 
        produce outputs of dimension dmodel = 512.
        
        NOTE we are deviating a little bit from the paper and applying
        pre normalization - Check out Andrej's video
        
        """
        # note the multi head already has a drop out
        # This is old school method
        # x  = self.norm1(x + self.multi_head(x))
        # output  =  self.norm2(x + self.feedforward(x))
        
        #this is the prenorm method
        x = x + self.multi_head(self.norm1(x))
        output = x + self.feedforward(self.norm2(x))

        return output
        