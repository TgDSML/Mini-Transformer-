from text import CharTokenizer, CharLMSequenceDataset, make_loaders  # Local import

def main():
    path_ = "data/tiny_shakespeare.txt"
    seq_len = 3
    batch_size = 2
    split = 0.8

    tok, ids, train_ids, val_ids, train_loader, val_loader, train_ds, val_ds, train_batches, val_batches, N, N_train = make_loaders(
        seq_len=seq_len, batch_size=batch_size, path_=path_, split=split, shuffle=True
    )

    print("\nTokenizer!")
    print(f"Dummy Text: {tok.text}")
    print(f"Chars: {tok.chars}")
    print(f"vocab_size(number of distinct chars): {tok.vocab_size()}")
    print("Stoi representation:", [(ch, tok.stoi[ch]) for ch in tok.chars])
    print("Itos representation:", [(i, tok.itos[i]) for i in range(tok.vocab_size())])

    ds_full = CharLMSequenceDataset(ids, seq_len=3)
    print("DATASET LENGTH (window pairs) =", len(ds_full))
    print(f"N(total number of tokens in the corpus) = {len(ids)},  T(seq_len) = {seq_len=}")
    print(f"It was expected since ds length = N - T = {max(0, len(ids) - seq_len)} , T = seq_len = 3")
    print("ids: a list of integers where each integer corresponds to one character ", ids)

    print("===CharLM DATASET===")
    for i in range(len(ds_full)):
        x, y = ds_full[i]
        print(f"i={i} | x={x.tolist()}| y={y.tolist()}")

    print("==Splitting CharLM Dataset for train and val purposes==")
    print(f"split = {split} , N_train = {N_train}, N_val = {N - N_train}")
    print(f"train_ids len={len(train_ids)}")
    print(f"val_ids   len={len(val_ids)}")
    print("The first 21 token IDs of the corpus")
    print("train_ids ", train_ids)
    print("The last 6 token IDs of the corpus")
    print("val_ids  ", val_ids)
    print("These are the raw tokens that will be used to build separate Datasets")

    print("Training Dataset")
    for i in range(len(train_ds)):
        x, y = train_ds[i]
        print(f"i={i} | x={x.tolist()} | y={y.tolist()}")

    print("Val Dataset")
    for i in range(len(val_ds)):
        x, y = val_ds[i]
        print(f"i={i} | x={x.tolist()} | y={y.tolist()}")

    print("=== Train Loader Example ===")
    for x, y in train_loader:
        print("input batch:", x)
        print("target batch:", y)
    print("Train batches:", train_batches)

    print("=== Val Loader ===")
    for x, y in val_loader:
        print("input batch:", x)
        print("target batch:", y)
    print("Val batches:", val_batches)

if __name__ == "__main__":
    main()

