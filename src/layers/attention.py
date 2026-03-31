import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention based on
    "Attention is All You Need"
    Attention equation: Attention(Q,K,V) = softmax( (Q*K^T) / sqrt(d_k) ) * V
    Where: Q  : queries, shape (batch_size, seq_length_q, d_k)
           K : keys, shape(batch_size, seq_length_k, d_k)
           V : values, shape(batch_size, seq_length_v, d_v)
           mask: optional mask, shape(batch_size, seq_length_q, seq_len_k) where 0 is masked

    Returns:
        context: attended values, shape(batch_size, seq_length_v, d_v)
        attn_weights : attention weights, shape(batch_size, seq_length_q,
                       seq_length_k)
    """

    def __init__(self, dropout = 0.1):
        #calling methods from the constructor class nn.Module
        #Dropout at 0.1 based on the original transformer paper
        super().__init__()      
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1) 

        #Compute raw scores: Q*K^T / sqrt(d_k)
        #Q has shape (batch_size, seq_length_q, d_k)
        #K.transpose(-2,-1) has shape (batch_size, d_k, seq_length_k)
        #scores = Q*K^T has shape (batch_size, seq_length_q, seq_length_k)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / d_k**0.5


        #Apply Mask (if needed) before softmax
        if mask is not None:
            mask = mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        #Subtract max per query for numerical stability
        max_scores = scores.max(dim=-1, keepdim=True).values
        scores = scores - max_scores


        #Softmax over keys
        #we are applying softmax over the last dimension of keys=seq_len_k.
        #That turns each row of scores for a given query into probabilities over
        #the keys
        attn_weights = F.softmax(scores, dim=-1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, 0.0)

        #Dropout
        attn_weights = self.dropout(attn_weights)

        # Avoid 0/0 during autograd when masking or dropout removes an entire row.
        denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attn_weights = attn_weights / denom

        #Multiply by V to get context
        #Shape (batch_size, seq_length_q, d_v) since we take the
        #weighted sum of of V vectors weighted by attn_weights
        context = torch.matmul(attn_weights, V)

        return context, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention based on 'Attention is All you Need' 3.2.2
    """

    def __init__(self, d_model, n_heads, dropout = 0.1):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads   # per head dimension
        # Linear Projections W_Q, W_V, W_K 
        # nn.Linear : X * W^T + b
        self.W_Q = nn.Linear(d_model, d_model)  
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

     
    def forward(self, Q, K , V, mask=None, return_weights=False):
        # Q, K, V have shape (batch_size, seq_length, d_model)
        batch_size, query_length, _ = Q.size() 
        _, key_length, _ = K.size() 
        _, value_length, _ = V.size() 

        if key_length != value_length:
            raise ValueError("K and V must have the same sequence length")
        
        #projected inputs
        Q = self.W_Q(Q)  # shape (batch_size, query_length ,d_model)
        K = self.W_K(K)  # shape (batch_size, key_length ,d_model)
        V = self.W_V(V)  # shape (batch size, value_length ,d_model)
        

        #reshape and split heads from (batch_size, seq_length, d_model) to (batch_size, seq_length, n_heads, head_dim)
        #and then to (batch_size, n_heads, seq_length, head_dim)
        Q = Q.view(batch_size, query_length, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_length, self.n_heads, self.head_dim).transpose(1, 2) 
        V = V.view(batch_size, value_length, self.n_heads, self.head_dim).transpose(1, 2)  

        if mask is not None:
             # Padding Mask shape (batch_size, seq_length), Casual Mask (batch_size, seq_length, seq_length)

            if mask.dim() == 2:
                #(batch_size, seq_length) to (batch_size, 1, 1, seq_length)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                #(batch_size, seq_length, seq_length) to (batch_size, 1, seq_length, seq_length)
                mask = mask.unsqueeze(1)



        # Apply Scaled Dot Product Attention      
        context, attn_weights = self.attention(Q, K, V, mask)


        #Transpose axes seq_length <-> n_heads
        context = context.transpose(1,2) # (batch_size, seq_length, n_heads, head_dim)
        # Reshape by flattening n_heads x head_dim back into d_model
        context = context.reshape(batch_size, query_length, self.d_model) # (batch_size, query_length, d_model)

        # Final output projection
        output = self.W_O(context) # (batch_size, seq_length, d_model)

        return output, attn_weights
