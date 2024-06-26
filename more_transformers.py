import torch 
import torch.nn as nn
import math as m
from torch.nn import functional as F

"""
Based on Andrew Karpathy notes as well as Teddy's code 
"""

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
        
        ## why drop out
        self.dropout = nn.Dropout(dropout)        

    def forward(self, x:torch.Tensor)-> torch.Tensor:
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
        if self.to_mask:
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