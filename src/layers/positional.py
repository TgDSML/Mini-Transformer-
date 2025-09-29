import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model,  max_len = 5000):
    
    
        """Creates sinusoidal positional encodings as described
        in "Attention is all you Need" 
        PE is a matrix with shape d_model x max_len
        """
        
        super().__init__()

        

        #creating a tensor of 0s
        PE = torch.zeros(max_len, d_model)

        #create position column
        # torch.arange creates a tensor ([0,1,...,max_len-1])
        # unsqueeze.(1) turnes it into a column
        pos = torch.arange(0, max_len).unsqueeze(1) 

        # divisor for positional encoding
        # shape is (d_model // 2) because we have one value per frequency
        # # For each k, we compute one frequency w_k = 1 / 10000^{2k/d_model}
        # This frequency is used for dimensions 2k (sin) and 2k+1 (cos)

        div_term =  torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        # calculate sin on even indices
        even_PE = torch.sin(pos * div_term) # shape : (max_len, d_model // 2)

        # calculate cos on odd indices
        odd_PE = torch.cos(pos * div_term) # shape : (max_len, d_model // 2)

        # stack and flatten to have pairs of sin,cos per frequency
        stacked = torch.stack([even_PE, odd_PE], dim = 2) # shape: (max_len , d_model//2, 2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2) # shape : (max_len, d_model) (d_model//2 * 2 = d_model)


        # in order to broadcast correctly over the batch dimension reshaping to 
        # (1 , max_len , d_model)
        PE = PE.unsqueeze(0)

        # Using a buffer because we want this tensor to be part of the model's
        # state , but dont want it to be trainable
        self.register_buffer("PE", PE)
    
    def forward(self, X):
        """
        X : tensor of shape (batch_size, seq_len, d_model)
        Returns:
        X + positional encoding (same as input shape)
        """
        # 1 indicates X second dimension which is seq_len
        seq_len = X.size(1) 

        # adding positional encoding to the embeddings
        X = X + self.PE[:, :seq_len, :]
        
        # Apply dropout to the sums of embeddings and the positional encodings based on the article
        return X
    

class LearnablePositionalEncoding(nn.Module):
    """ Instead of using the sinusodial PE authors suggested learned encodings
    that it was shown that had identical results with the first one.

    Learnable Positional Encoding using nn.Parameter.

    d_model: Dimensionality of the model
    max_len: Maximum sequence length
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()


        # Creating a trainable lookup table of positional vectors, initialized with zeroes
        # and ready to be learned by the model
        self.PE = nn.Parameter(torch.zeros(1, max_len, d_model))

        # Initialize the learnable positional encodings with small random values
        # from a normal distribution (mean = 0.0, std = 0.02)
        nn.init.normal_(self.PE, mean = 0.0, std = 0.02)

    def forward(self, X):
        """
        X: tensor of shape (batch_size, seq_len, d_model)
        Returns:
        Tensor with positional encoding added"""

        seq_len = X.size(1)
        X = X + self.PE[:, :seq_len, :]
        return X
        
