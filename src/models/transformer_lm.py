import torch
import torch.nn as nn
import torch.nn.functional as F 
from src.models.encoder import TransformerEncoderLayer
from src.layers.positional import SinusoidalPositionalEncoding
from src.layers.positional import LearnablePositionalEncoding

class TransformerEncoderLM(nn.Module):
    """ Transformer LM build right after the encoder layers stack.
    LM projects embedded tokens back to the vocabulary space (Linear Transformation)
    by producing logits over all possible tokens."""
     
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        num_layers : int = 4,
        dropout : float = 0.1,
        pos_encoding = "sin", # can be "learned" as well 
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token Embedding: (B, T) -> (B, T, d_model)
        # nn.Embedding is a lookup table (matrix) with rows: vocab_size and columns:d_model
        # Each row is a learned vector (the embedding) for one token id
        # Say E is the learnable weight matrix E of shape (vocab_size, d_model)
        # Each row E corresponds to one token in the vocab, row E(i) is the embedding vector for token id i 
        # dim of each embedding vector = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Positional Ecoding "sinusoidal" or "Learned"
        if pos_encoding == "sin":
            self.pos = SinusoidalPositionalEncoding(d_model=d_model)
        else:
            self.pos = LearnablePositionalEncoding(d_model=d_model)

    
        # N Encoder layers
        # # we are using nn.ModuleList and not just a python list
        # because we want to hold multiple submodules inside our constructor class nn.Module
        # so by using it, all encoder layers are properly registered as submodules 
        # with their parameters included in training
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
            
        # Final normalization and LM head


        # nn.Layernorm normilizes its token's embedded vector 
        # compute mean and variance then normalize x^ = x - mean / std
        self.norm = nn.LayerNorm(d_model)


        # linear projection that maps each embedded vector of size d_model to a vector of size vocab_size 
        # input shape : (B, T, d_model) W shape (vocab_size, d_model)  b (vocab_size,)
        # Y = X * W^T + b (linear transformation of X to Y)
        # Y = output logits(unormalized scores)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)

    
    def forward(self, idx: torch.Tensor, targets : torch.Tensor | None = None):
        """
        idx: (B, T) tensor of token ids
        targets: (B, T) tensor of next-token ids"""
        
        # Unpacking the shape of the input tensor idx
        B, T = idx.shape

        # Token Embeddings : (B, T) -> (B, T, d_model)
        x = self.tok_emb(idx)

        # Add positional information
        x = self.pos(x)

        # Causal mask : attend to self and past but no future
        # Creating a (T,T) matrix of true values only
        # Converting to lower triangular (including diagonal because we want present and past)
        # Keeps desired positions as true and all "future" positions as False
        # Adding two single dimensions so shapes become (1,1,T,T) in order to:
        # Be broadcastable across batch dimension B and also across all attn heads
        mask = torch.ones((T,T), dtype=torch.bool).tril_().unsqueeze(0).unsqueeze(1) # (1,1,T,T)
        
        # Pass through the N encoder layers with the mask
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)   # (B, T, d_model)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # Return logits, or (logits, loss) if targets provided for training.
        if targets is None:
            return logits

        # CE over all positions (standard LM loss)
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            )
        return logits, loss