import torch.nn as nn
from src.layers.attention import ScaledDotProductAttention
from src.layers.attention import MultiHeadAttention
from src.layers.feedforward import FeedForward



class TransformerEncoderLayer(nn.Module):
    """ Assembly of a single Encoder layer which based on "Attention is all you Need"
    consists of two sub-layers, first is multihead attention and the second is a fully
    connected feed-forward network. In each sublayer we employ a residual connection followed
    by layer normalization.
    
        d_model : Dimensionality of input and output
        n_heads : Number of attention heads.
        d_ff : Dimensionality of the feed-forward layer
        dropout : dropout probability applied after any activation function
    
    Input: tensor X of shape (batch_size, seq_len, d_model)
           Optional attention masking , mask shape :
           (batch_size, seq_len) or (batch_size, seq_len, seq_len)  
           for padding mask and casual mask respectively
    
    Returns: Tensor of shape (batch_size, seq_len, d_model)
            after attention and feed-forward transformations
         """
    
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):

        super().__init__()

        self.multihead_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        
        # Sublayer1
        multihead_attn_output, _ = self.multihead_attn(X, X, X, mask=mask)
        multihead_attn_output = self.dropout(multihead_attn_output)
        X = self.norm1(X + multihead_attn_output) # Add + Normalize (Residual = (X + output)) 
        
        # Sublayer2
        feedforward_output = self.feedforward(X)
        feedforward_output = self.dropout(feedforward_output)
        X = self.norm2(X + feedforward_output) # Add + Normalize (Residual = (X + output)) 

        return X

