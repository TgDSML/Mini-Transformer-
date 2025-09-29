	
import os, io, glob 
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple 

class CharTokenizer:
    def __init__(self, path_: str = "data/tiny_shakespeare.txt"):
        with open(path_) as f:
            self.text = f.read()

        # Sorting unique characters in deterministic order
        self.chars = sorted(set(self.text))
        

        # Vocabulary mappings
        self.stoi = {ch: i for i , ch in enumerate(self.chars)} # char -> id
        self.itos = {i : ch for ch, i in self.stoi.items()} # id -> char

    def encode(self, s: str):
        return[self.stoi[c] for c in s ]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def vocab_size(self):
        return len(self.chars)
            

class CharLMSequenceDataset(Dataset):
        """ Character-level dataset for next-token prediction 
            Each sample is:
            x = [t_i,t_{i+1},.....,t_{i+seq_len-1}]
            y = [t_{i+1}, t_{i+2},....,t_{i+seq_len}]"""
        
        def __init__(self, encoded_ids, seq_len:int):
            # Store the whole text as a single LongTensor of token IDs
            self.ids = torch.tensor(list(encoded_ids), dtype = torch.long)
            self.seq_len = int(seq_len)

        def __len__(self) -> int:
            N = len(self.ids)
            T = self.seq_len
            
            # x and y have the same shape but y has one larger index 
            # y must include index i + T so i + T =< N - 1 hence i =< N - T - 1
            # valid start indices are 0,1,...., N - T -1 
            #count of integers in [0, N-T-1] is b-a+1 where b = N-T-1, a = 0 
            # hence N - T
                
            return max(0, N - T)

        def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
            # indexing the dataset with an integer idx 
            # returns : a tuple of two tensors (x, y)
            x = self.ids[idx: idx + self.seq_len]
            y = self.ids[idx + 1: idx +1 + self.seq_len]
            return x, y 
        

def make_loaders(
    seq_len: int,
    batch_size: int,
    path_ : str = "data/tiny_shakespeare.txt",
    split: float = 0.8,
    shuffle: bool = True,
    ):

    text = open("data/tiny_shakespeare.txt")
    tok = CharTokenizer(path_)
    ids = tok.encode(tok.text)


    # spliting the sample for train and val at 80% and 20%
    # train_ids is from start index until N_train-1 and 
    # val_ids are N_train till the end
    N = len(ids)
    N_train = int( N * split)
    train_ids, val_ids = ids[:N_train], ids[N_train:] 

    # Datasets 
    train_ds = CharLMSequenceDataset(train_ids, seq_len)
    val_ds = CharLMSequenceDataset(val_ids, seq_len)

    # Loaders
    train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=shuffle,
    )

    val_loader = DataLoader(
    val_ds, batch_size=batch_size,shuffle=False
    )

    train_batches = int(np.ceil(len(train_ds)/batch_size)) 
    val_batches = int(np.ceil(len(val_ds)/batch_size))

    return (
        tok,
        ids,
        train_ids,
        val_ids,
        train_loader,
        val_loader,
        train_ds,
        val_ds,
        train_batches,
        val_batches,
        N,
        N_train
        )
    


        



    
