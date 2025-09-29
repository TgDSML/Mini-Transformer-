import torch
import pytest
from src.layers.attention import ScaledDotProductAttention
from src.layers.attention import MultiHeadAttention
from src.layers.feedforward import FeedForward
from src.layers.positional import SinusoidalPositionalEncoding
from src.layers.positional import LearnablePositionalEncoding
from src.models.encoder import TransformerEncoderLayer
from tests import test_attention



def run_tests():
    """ Run all test functions from test_attention.
    Prints Pass for each test, if test fails python will stop.
    """

    print("\n Running Attention Tests \n")

    #1) Testing Shapes on happy path
    print("Test_attention_shapes_happy_path()")
    test_attention.test_attention_shapes_happy_path()
    print("Pass")

    #2) Testing Shapes on assymetric
    print("Test_attention_shapes_assymetric")
    test_attention.test_attention_shapes_assymetric()
    
    #3) Testing Shapes on dk_diff_dv
    print("Test_attention_shapes_dk_diff_dv")
    test_attention.test_attention_shapes_dk_diff_dv()

    #4) Testing Shapes on minimum
    print("Test_attention_shapes_minimum")
    test_attention.test_attention_shapes_minimum()

    #5) Testing attention invalid dk
    print("Test_attentinon_invalid_dk()")
    test_attention.test_attention_invalid_dk()


    print("All tests passed.")


def demo_Scaled_Dot_Product_Attention():

    print("\n=== Attention Demo ===\n")

    batch_size = 2
    seq_length_q = seq_length_k = seq_length_v =  5
    d_k = 64
    d_v = 64

    Q = torch.randn(batch_size, seq_length_q, d_k)
    K = torch.randn(batch_size, seq_length_k, d_k)
    V = torch.randn(batch_size, seq_length_v, d_k)

    attention = ScaledDotProductAttention(dropout=0.1)
    context, weights = attention(Q, K, V)

  
def demo_Multi_Head_Attention():
    print("\n Multi Head Attention\n")

    d_model = 512
    batch_size = 2
    seq_length = 5
    n_heads = 8

    Q = torch.randn(batch_size, seq_length, d_model)
    K = torch.randn(batch_size, seq_length, d_model)
    V = torch.randn(batch_size, seq_length, d_model)

    # Demonstration with padding mask
    mask = torch.ones(batch_size, seq_length)
    mask[:, -1] = 0 
    
    model = MultiHeadAttention(d_model, n_heads, dropout = 0.1) 
    output = model(Q, K, V, mask=mask)



def demo_FeedForward():
    print("\n FeedForward Network Demo \n")

    batch_size = 2
    seq_len = 5
    d_model = 512
    d_ff = 2048

    X_in = torch.randn(batch_size, seq_len, d_model)
    ff = FeedForward(d_model, d_ff, dropout = 0.1)
    X_out = ff(X_in)

    print(f"Input shape: {X_in.shape}")
    print(f"Output shape: {X_out.shape}")

    # ReLU activation inbetween linear transformations
    X_pre_activation = torch.randn(batch_size, seq_len, d_ff)
    X_post_activation = ff.activation(X_pre_activation)

    print("Before ReLU (first sample, first 20 values):")
    print(X_pre_activation[0, 0, :20])

    print("After ReLU (first sample, first 20 values):")
    print(X_post_activation[0, 0, :20])


def demo_Sinusoidal_Positional():
    print("\n Sinusodial Positional Encoding Demo \n")

    batch_size = 2
    seq_len = 5
    d_model = 512

    X = torch.randn(batch_size, seq_len, d_model)

    pos_enc = SinusoidalPositionalEncoding(d_model=d_model)

    output = pos_enc(X) 


    print(f"Input shape : {X.shape}")
    print(f"Output shape: {output.shape}")

def demo_Learnable_Positional():
    print("\n Learnable Positional Encoding Demo \n")
   
    batch_size = 2
    seq_len = 5
    d_model = 512

    X = torch.randn(batch_size, seq_len, d_model)

    pos_enc = LearnablePositionalEncoding(d_model=d_model)

    output = pos_enc(X) 


    print(f"Input shape : {X.shape}")
    print(f"Output shape: {output.shape}")


    
def demo_encoder_layer():
    print("\n TransformerEncoderLayer — shape & gradient demo")

    # Small sizes to keep prints readable
    batch_size = 2
    seq_len = 5
    d_model = 512
    d_ff = 2048
    n_heads = 8
    dropout = 0.1

    # Dummy input (track grads to demonstrate backprop)
    X = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    # Instantiate the layer
    layer = TransformerEncoderLayer(
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        dropout=dropout
    )

    # Forward
    out = layer(X)

    # Shapes
    print(f"Input shape : {X.shape}")
    print(f"Output shape: {out.shape}")

    # Backward (dummy scalar loss)
    loss = out.sum()
    loss.backward()

    print("Gradient present on input:", X.grad is not None)


if __name__ == "__main__":
    run_tests()
    demo_Scaled_Dot_Product_Attention()
    demo_Multi_Head_Attention()
    demo_FeedForward()
    demo_Sinusoidal_Positional()
    demo_Learnable_Positional()
    demo_encoder_layer()

