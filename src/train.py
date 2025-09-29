import torch 
from src.models import transformer_lm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def build_model(cfg, vocab_size):
    model = transformer_lm(
        vocab_size=vocab_size,
        d_model = cfg["d_model"],
        n_heads = cfg["n_heads"],
        d_ff = cfg["d_ff"],
        num_layers = cfg["n_layers"],
        dropout = cfg["dropout"],
        pos_encoding = cfg["positional_encoding"],
    )

    return model

def build_optimizer0(cfg, model):
    return torch.optim.AdamW(            # AdamW optimizer
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )


def build_scheduler(cfg, optimizer, steps_per_epoch: int):
    
    # total steps = min(num_epochs * steps_per_epoch, max_steps if set)
    implied = cfg["num_epochs"] * max(1, steps_per_epoch)

    