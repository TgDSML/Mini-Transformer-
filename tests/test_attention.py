import torch
import pytest
from src.layers.attention import ScaledDotProductAttention
from src.layers.attention import MultiHeadAttention
from src.layers.feedforward import FeedForward 
from src.layers.positional import SinusoidalPositionalEncoding
from src.layers.positional import LearnablePositionalEncoding
from src.models.encoder import TransformerEncoderLayer


def test_attention_shapes_happy_path():
    batch_size = 2
    seq_len_q = 3
    seq_len_k = seq_len_v = 4
    d_k = 8
    d_v = 8


     # random tensors
    Q = torch.randn(batch_size, seq_len_q, d_k)
    K = torch.randn(batch_size, seq_len_k, d_k)
    V = torch.randn(batch_size, seq_len_v, d_v)

    attn = ScaledDotProductAttention(dropout=0.0)
    context, attn_weights = attn(Q, K, V)
    

    #NO mask assertions
    # check context shape
    assert context.shape == (batch_size, seq_len_q, d_v), \
        f"Expected context shape {(batch_size, seq_len_q, d_v)}, but got {context.shape}"
    
    # check attention weights shape
    assert attn_weights.shape == (batch_size, seq_len_q, seq_len_k), \
        f"Expected attention weights shape {(batch_size, seq_len_q, seq_len_k)}, but got {attn_weights.shape}"
    
    # check that attention weights sum to ~1 over keys
    sums = attn_weights.sum(dim=-1)
    ones = torch.ones_like(sums)
    assert torch.allclose(sums, ones, atol=1e-5), \
        "Attention weights do not sum to 1 over keys"
    
    print(f" Context shape OK: {context.shape} == {(batch_size, seq_len_q, d_v)}")
    print(f" Attention weights shape OK: {attn_weights.shape} == {(batch_size, seq_len_q, seq_len_k)}")
    print(" Attention weights sum to ~1 over keys")

    # WITH deterministic mask (mask last key)
    mask = torch.ones(batch_size, seq_len_q, seq_len_k)
    mask[:, :, -1] = 0  # mask last key

    context_masked, attn_weights_masked = attn(Q, K, V, mask)

    # shapes still correct
    assert context_masked.shape == (batch_size, seq_len_q, d_v)
    assert attn_weights_masked.shape == (batch_size, seq_len_q, seq_len_k)

    # masked positions ≈ 0
    assert torch.allclose(
        attn_weights_masked[:, :, -1],
        torch.zeros_like(attn_weights_masked[:, :, -1]),
        atol=1e-5
    ), "Masked positions do not have ~0 attention weight."

    # valid positions sum ≈ 1
    sums = attn_weights_masked.sum(dim=-1)
    ones = torch.ones_like(sums)
    assert torch.allclose(sums, ones, atol=1e-5), \
        "Attention weights (with mask) do not sum to 1 over valid keys."

    print("Happy_path: masked positions = 0, valid positions sum to 1")


    # random (seeded) mask
    torch.manual_seed(42)
    mask_random = torch.randint(0, 2, (batch_size, seq_len_q, seq_len_k))

    context_rand_masked, attn_weights_rand_masked = attn(Q, K, V, mask_random)

    # shape checks
    assert context_rand_masked.shape == (batch_size, seq_len_q, d_v)
    assert attn_weights_rand_masked.shape == (batch_size, seq_len_q, seq_len_k)

    # masked positions ≈ 0
    assert torch.all(attn_weights_rand_masked[mask_random == 0] < 1e-5), \
        "Masked positions (random) do not have ~0 weight."

    # valid weights sum ~1
    sums = attn_weights_rand_masked.sum(dim=-1)
    ones = torch.ones_like(sums)
    assert torch.allclose(sums, ones, atol=1e-5), \
        "Attention weights (random mask) do not sum to 1 over valid keys."
    
    print("happy_path (random mask): masked positions = 0, valid positions sum to 1")


def test_attention_numerical_stability_extreme_inputs():
    """
    Testing that attention does not produce NaNs on extreme inputs.
    Verifies that the attention outputs remain valid even when Q 
    and K contain very large values.
    """

    batch_size =  2
    seq_len_q = 3
    seq_len_k = seq_len_v = 4
    d_k = 8 
    d_v = 8

    #Q, K with extreme values
    extreme_value = 1e9
    Q = torch.ones(batch_size, seq_len_q, d_k) * extreme_value
    K = torch.ones(batch_size, seq_len_k, d_k) * extreme_value
    V = torch.randn(batch_size, seq_len_v, d_v)

    attn = ScaledDotProductAttention(dropout=0.0)
    context, attn_weights = attn(Q, K ,V)

    #Check that there are no NaNs in attention weights or context
    assert not torch.isnan(attn_weights).any(), \
        "Attention weights contain NaNs"
    assert not torch.isnan(context).any(), \
        "Context contains NaNs"
    
    #Check that attention weights still sum to ~1
    sums = attn_weights.sum(dim=-1)
    ones = torch.ones_like(sums)
    assert torch.allclose(sums, ones, atol=1e-5), \
        "Attention weights do not sum to 1"
    
    print("Numerical stability test passed: no NaNs, sum to ~1")


def test_attention_shapes_assymetric():
    #Queries and keys/values have different seq lengths
    batch_size = 2
    seq_len_q = 3
    seq_len_k = seq_len_v = 5
    d_k = 4
    d_v = 6

   # random tensors
    Q = torch.randn(batch_size, seq_len_q, d_k)
    K = torch.randn(batch_size, seq_len_k, d_k)
    V = torch.randn(batch_size, seq_len_v, d_v)

    attn = ScaledDotProductAttention(dropout=0.0)
    context, attn_weights = attn(Q, K, V)

    # check context shape
    assert context.shape == (batch_size, seq_len_q, d_v), \
        f"Expected context shape {(batch_size, seq_len_q, d_v)}, but got {context.shape}"
    
    
    # check attention weights shape
    assert attn_weights.shape == (batch_size, seq_len_q, seq_len_k), \
        f"Expected attention weights shape {(batch_size, seq_len_q, seq_len_k)}, but got {attn_weights.shape}"
    
    
    # check that attention weights sum to ~1 over keys
    sums = attn_weights.sum(dim=-1)
    ones = torch.ones_like(sums)
    assert torch.allclose(sums, ones, atol=1e-5), \
        "Attention weights do not sum to 1 over keys"
    


    print(f" Context shape OK: {context.shape} == {(batch_size, seq_len_q, d_v)}")
    print(f" Attention weights shape OK: {attn_weights.shape} == {(batch_size, seq_len_q, seq_len_k)}")
    print(" Attention weights sum to ~1 over keys")


def test_attention_shapes_dk_diff_dv():
    #key/query dimension is different from value dimension
    batch_size = 2
    seq_len_q = 3
    seq_len_k = seq_len_v = 4
    d_k = 16
    d_v = 32

     # random tensors
    Q = torch.randn(batch_size, seq_len_q, d_k)
    K = torch.randn(batch_size, seq_len_k, d_k)
    V = torch.randn(batch_size, seq_len_v, d_v)

    attn = ScaledDotProductAttention(dropout=0.0)
    context, attn_weights = attn(Q, K, V)

    # check context shape
    assert context.shape == (batch_size, seq_len_q, d_v), \
        f"Expected context shape {(batch_size, seq_len_q, d_v)}, but got {context.shape}"
    
    # check attention weights shape
    assert attn_weights.shape == (batch_size, seq_len_q, seq_len_k), \
        f"Expected attention weights shape {(batch_size, seq_len_q, seq_len_k)}, but got {attn_weights.shape}"
    
    
    # check that attention weights sum to ~1 over keys
    sums = attn_weights.sum(dim=-1)
    ones = torch.ones_like(sums)
    assert torch.allclose(sums, ones, atol=1e-5), \
        "Attention weights do not sum to 1 over keys"
    
    
    print(f" Context shape OK: {context.shape} == {(batch_size, seq_len_q, d_v)}")
    print(f" Attention weights shape OK: {attn_weights.shape} == {(batch_size, seq_len_q, seq_len_k)}")
    print(" Attention weights sum to ~1 over keys")



def test_attention_shapes_minimum():
     #Smallest possible valid input
         
    batch_size = 1
    seq_len_q = 1
    seq_len_k = seq_len_v = 1
    d_k = 1
    d_v = 1
  # random tensors
    Q = torch.randn(batch_size, seq_len_q, d_k)
    K = torch.randn(batch_size, seq_len_k, d_k)
    V = torch.randn(batch_size, seq_len_v, d_v)

    attn = ScaledDotProductAttention(dropout=0.0)
    context, attn_weights = attn(Q, K, V)

    # check context shape
    assert context.shape == (batch_size, seq_len_q, d_v), \
        f"Expected context shape {(batch_size, seq_len_q, d_v)}, but got {context.shape}"
    
    # check attention weights shape
    assert attn_weights.shape == (batch_size, seq_len_q, seq_len_k), \
        f"Expected attention weights shape {(batch_size, seq_len_q, seq_len_k)}, but got {attn_weights.shape}"
    
    
    # check that attention weights sum to ~1 over keys
    sums = attn_weights.sum(dim=-1)
    ones = torch.ones_like(sums)
    assert torch.allclose(sums, ones, atol=1e-5), \
        "Attention weights do not sum to 1 over keys"
    
    print(f" Context shape OK: {context.shape} == {(batch_size, seq_len_q, d_v)}")
    print(f" Attention weights shape OK: {attn_weights.shape} == {(batch_size, seq_len_q, seq_len_k)}")
    print(" Attention weights sum to ~1 over keys")

def test_attention_invalid_dk():
    # Scenario: Q and K have different d_k should fail
    batch_size = 1
    seq_len_q = 2
    seq_len_k = seq_len_v = 2
    d_q = 4
    d_k = 5  # different!
    d_v = 6

    Q = torch.ones(batch_size, seq_len_q, d_q)
    K = torch.ones(batch_size, seq_len_k, d_k)
    V = torch.ones(batch_size, seq_len_v, d_v)

    attn = ScaledDotProductAttention(dropout=0.0)

    # Check that it raises an error
    with pytest.raises(RuntimeError):
        context, _ = attn(Q, K, V,)

    print(" Invalid d_k test (expected failure) passed.")  


def test_attention_train_vs_eval_modes():

    """
    Verifies that in training mode with dropout > 0, attention weights are dropped & renormalized,
    and in eval mode, dropout is inactive and weights sum to 1.
    """

    batch_size = 2
    seq_len_q = 3
    seq_len_k = seq_len_v = 4
    d_k = 8
    d_v = 8

    Q = torch.randn(batch_size, seq_len_q, d_k)
    K = torch.randn(batch_size, seq_len_k, d_k)
    V = torch.randn(batch_size, seq_len_v, d_v)

    attn = ScaledDotProductAttention(dropout=0.1)  # > 0 so we can see dropout

    # TRAIN mode
    attn.train()
    context_train, weights_train = attn(Q, K, V)
    sums_train = weights_train.sum(dim=-1)
    ones = torch.ones_like(sums_train)

    # ensure weights sum to ~1 even after dropout and renormalization
    assert torch.allclose(sums_train, ones, atol=1e-5), \
    "In training mode, weights do not sum to 1 after dropout & renormalization"

    # check that some weights have been dropped (are now zero)
    assert (weights_train == 0).any(), \
    "In training mode, no weights were dropped although expected some to be zero"

    #  EVAL mode
    attn.eval()
    context_eval, weights_eval = attn(Q, K, V)
    sums_eval = weights_eval.sum(dim=-1)

    # ensure weights sum to ~1 in eval too
    assert torch.allclose(sums_eval, ones, atol=1e-5), \
    "In eval mode, weights do not sum to 1"

    # check that no weights are zero (dropout inactive)
    assert not (weights_eval == 0).any(), \
    "In eval mode, found zeroed weights (dropout should be off)"

    print("Train vs Eval mode test passed: weights properly dropped & renormalized in train, untouched in eval.")


def test_multihead_attention_output_shape():
    batch_size =2
    seq_length = 5
    d_model = 512
    n_heads = 8

    Q = torch.randn(batch_size, seq_length, d_model)
    K = torch.randn(batch_size, seq_length, d_model)
    V = torch.randn(batch_size, seq_length, d_model)

    model = MultiHeadAttention(d_model=d_model, n_heads=n_heads,dropout=0.0)

    output, _ = model(Q, K, V)

    assert output.shape == (batch_size, seq_length, d_model), \
    f"Expected output shape {(batch_size, seq_length, d_model)}, but got {output.shape}"
        
    print(f" MultiHeadAttention output shape OK : {output.shape}")
    


def test_attention_weights_sum_per_head():
    batch_size = 2
    seq_length = 5
    d_model = 512
    n_heads = 8

    Q = torch.randn(batch_size, seq_length, d_model)
    K = torch.randn(batch_size, seq_length, d_model)
    V = torch.randn(batch_size, seq_length, d_model)

    model = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    
    # Get output and attention weights
    output, attn_weights = model(Q, K, V, return_weights=True)

    # attn_weights shape: (batch_size, n_heads, seq_length_q, seq_length_k)
    sums = attn_weights.sum(dim=-1)  # sum over keys
    ones = torch.ones_like(sums)

    assert torch.allclose(sums, ones, atol=1e-5), \
        "Attention weights do not sum to 1 over keys per head."

    print("Attention weights per head sum to 1 as expected.")


def test_multihead_attention_mask_propagation():
    batch_size = 2
    seq_length = 5
    d_model = 512
    n_heads = 8

    Q = torch.randn(batch_size, seq_length, d_model)
    K = torch.randn(batch_size, seq_length, d_model)
    V = torch.randn(batch_size, seq_length, d_model)

    model = MultiHeadAttention(d_model, n_heads, dropout=0.0)
   
    # Mask that masks the last position
    # Shape (batch_size, seq_length)
    mask = torch.ones(batch_size, seq_length)
    mask[:, -1] = 0 # masking last position in each sequence

    output, attn_weights = model(Q, K, V, mask=mask, return_weights=True)

    # assertion : last column in weights is all zeroes
    assert torch.allclose(attn_weights[:, : ,:, -1], torch.zeros_like(attn_weights[:, :, :, -1]), atol=1e-5), \
        "masked positions are not zeroed out in attention weights."
    
    print("Mask propagation accross heads works fine.")

def test_ffn_output_shape():
    batch_size = 2
    seq_len = 5
    d_model = 512
    d_ff = 2048

    ff = FeedForward(d_model, d_ff)
    X_in = torch.randn(batch_size, seq_len, d_model)
    X_out = ff(X_in)

    assert X_out.shape == (batch_size, seq_len, d_model), \
    f"Expected X_out shape {X_in.shape} but got {X_out.shape}"

    print(f" X_out shape after FFN is OK : {X_out.shape}")

def test_ffn_nonlinearity_behaviour():
    batch_size = 2
    seq_len = 5
    d_model = 512
    d_ff = 2048

    ff = FeedForward(d_model, d_ff)
    X_in = torch.randn(batch_size, seq_len, d_model)

    X_linear1 = ff.linear_transform1(X_in)
    X_activated = ff.activation(X_linear1)

    # 1st testing method
    # computing element-wise difference between x_linear and x_activated tensors
    # taking the absolute value of each element in the result
    # adding up all the absolute differences into a single tensor
    #converting the tensor to a python float for assertion usage
    activation_check = (X_linear1 - X_activated).abs().sum().item()

    assert activation_check > 1e-5, "Activation function does not modify the linearity of the output"
    print(f"Nonlinearity applied correctly. Total absolute difference after activation: {activation_check:.2f}")


    #2nd testing method
    assert not torch.equal(X_linear1, X_activated), \
     "Activation function does not modify the linearity of the output"
    print(f"Nonlinearity applied correctly.")



def test_positional_encoding_shape():

    batch_size = 2
    seq_len = 5
    d_model = 512

    X = torch.randn(batch_size, seq_len, d_model)

    pos_enc = SinusoidalPositionalEncoding(d_model=d_model)

    output = pos_enc(X)

    print("Input Shape:", X.shape)
    print("Output shape:", output.shape)

    assert output.shape == X.shape , "Shape missmatch after positional encoding."
    print("Test passed: Output shape matches input shape")


def test_learnable_positional_encoding_shape():
    batch_size = 2
    seq_len = 5
    d_model = 512

    X = torch.randn(batch_size, seq_len, d_model)

    pos_enc = LearnablePositionalEncoding(d_model=d_model)

    output = pos_enc(X)

    print("Input Shape:", X.shape)
    print("Output shape:", output.shape)

    assert output.shape == X.shape , "Shape missmatch after learnable positional encoding."
    print("Test passed: Output shape matches input shape")

def test_transformer_encoder_end_to_end():
    batch_size = 2
    seq_len = 5
    d_model = 512
    d_ff = 2048
    n_heads = 8
    dropout = 0.1

    transformer_encoder_layer = TransformerEncoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    dropout=dropout
        )

    X = torch.randn(batch_size, seq_len, d_model)
    output = transformer_encoder_layer(X)

    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected output shape {(batch_size, seq_len, d_model)}, but got {output.shape}"
        
    print("Encoder output shape is correct.")

def test_transformer_encoder_gradient_flow():
    batch_size = 2
    seq_len = 5
    d_model = 512
    d_ff = 2048
    n_heads = 8
    dropout = 0.1


    # requires_grad=True tracks all operations in order to compute its gradients during backpropagation
    X = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    transformer_encoder_layer = TransformerEncoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    dropout=dropout
    )

    output = transformer_encoder_layer(X)

    # Single final resault - scalar loss - that triggers backpropagation.
    # Sum all elements from the output matrix. Its abstract but we just want 
    # to test if gradients flow backward through the model without errors.
    loss = output.sum()

    # Backpropagation
    loss.backward()

    assert X.grad is not None, "Gradient did not flow back to input"
    print("Gradient successfully flowed through the encoder layer.")

    







   
   
   