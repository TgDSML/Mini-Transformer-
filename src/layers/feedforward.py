import torch.nn as nn

class FeedForward(nn.Module):
    """ Implements the Feed Forward Network (FFN) used in Transformer encoder layers
    as described in "Attention is All you Need"
    
    This module has two linear transformations with a ReLU activation in between,
    applied intependently to each position in the sequence
    linear_transform1 shape: d_model -> d_ff (512 -> 2048)
    linear_transform2 shape: d_ff -> d_model (2048 -> 512)
    
    d_model : Input and output dimensionality of the model.
    d_ff : Hidden Layer dimensionality 
    dropout : dropout probability applied after the activation

    Input: X (Tensor) : Tensor of shape (batch_size, seq_length, d_model)
    Output: Tensor of the same shape
    """
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()


        # Creating a fully connected layer, and keeping it inside this object as linear_transform1
        self.linear_transform1 = nn.Linear(d_model, d_ff)
        # RELU actiavtion function that provides model with non-linearity by attending negative values to 0
        self.activation = nn.ReLU()
        self.linear_transform2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_transform1(x) #Shape: (batch_size, seq_len, d_ff)
        x = self.activation(x)
        # Based on "Attention is all you need" apply dropout after the activation function after the first linear transformation
        x = self.dropout(x)
        x = self.linear_transform2(x) # Shape : (batch_size, seq_len, d_model)

        return x
        