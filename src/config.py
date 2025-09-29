from pathlib import Path
import re

def get_config():
    return {
            #Data
    "Dataset": "tinyshakespeare",
    "data_file":"tiny_shakespeare.txt",
    "seq_len": 3,
    "batch_size": 2,
    "tokenizer": "char",
    "train_val_split": 0.8,
    "shuffle": True,

    #Model
    "vocab_size": 16,
    "d_model": 512,
    "n_layers": 6, #Encoder Layer stack
    "n_heads": 8,   #Number of heads in MHA
    "d_ff": 2048, # Dimension of FFN
    "positional_encoding": "sin",
    "dropout": 0.1,

    #Optimization
    "optimizer": "adamw",
    "learning_rate": 3e-4,
    "scheduler": "cosine",
    "betas": [0.9,0.95],
    "weight_decay": 0.01,
    "warmup_steps": 2000,
    "max_steps": 200_000,
    "grad_clip": 1.0,
    "num_epochs": 20,


    #Checkpointing
    "out_dir": "checkpoints/mini_transformer",
    "model_folder": "weights",
    "model_basename": "encoder_lm_",
    "save_every": 1,  #save each epoch
    "keep_last_k": 3, #every time we save a new checkpoint, we keep the most recent k files (here 3)


    #Logging
    "use_tb": True, #Tensorboard: local dashboard that shows training curves (loss, accuracy, perplexity)
    "use_wandb": False, # if True: log metrics to Weights & Biases (cloud); if False: skip
    "Project": "Mini Transformer",
    "run_name": "Model Train",
    "log_every" : 100,  # log training loss & LR every 100 steps
    "eval_every": 1000  # run validation (val loss / perplexity) every 1000 steps

    }


def weights_path(cfg, epoch:int) -> str:
    root = Path(cfg["out_dir"]) / cfg["model_folder"]  # cfg["out_dir"] = the base directory for all runs, cfg["model_folder"]=subfolder for model weights
    root.mkdir(parents=True, exist_ok=True) # ensures folder exists before saving, parents=True: creates any missing parent dirs in the path
    fname = f"{cfg['model_basename']}{epoch:02d}.pt"

    return str(root / fname)   # full string path to where the checkpoint is being saved


def latest_weights_path(cfg) -> str | None:
    root = Path(cfg["out_dir"]) / cfg["model_folder"]  # Builds the directory where checkpoints live
    if not root.exists():
        return None
    files = sorted(root.glob(f"{cfg['model_basename']}*.pt"))  # finds all files whose names start with basename and end with .pt
                                                               #glob returns an iterator of path objects, sorted() sorts them lexicographically
    return str(files[-1]) if files else None 

