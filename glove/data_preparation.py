import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import re

def read_file():
    data_dir = "glove/dataset.txt"
    with open(data_dir, 'r') as f:
        return f.read().lower().strip().replace(',', '').replace(';', '').replace(':', '').replace('!', '.').replace('?', '.')

def tokenizer():
    raw = read_file()
    tok = get_tokenizer("basic_english")
    sentences = [s for s in re.split(r'[.\n\t]', raw) if s.strip()]
    tokenized = [tok(s) for s in sentences]
    
    return tokenized

def dictionary():
    tokenized = tokenizer()
    specials = ["<unknown>"]
    min_freq = 4
    vocab = build_vocab_from_iterator(tokenized, min_freq=min_freq, specials=specials, special_first=True)
    vocab.set_default_index(vocab["<unknown>"])
    
    id_vocab = [vocab[token] for tokens in tokenized for token in tokens]
    
    return vocab, id_vocab

def cooccurrence_matrix(window_length=4):
    vocab, _ = dictionary()
    V = len(vocab)
    X = torch.tensor(torch.zeros([V, V]), dtype=torch.float32)
    sentences = tokenizer()
    
    for sentence in sentences:
        for i, word in  enumerate(sentence):
            for j in range(1, window_length//2):
                if i - j >= 0:
                    X[vocab[word], vocab[sentence[i-j]]] = X[vocab[word], vocab[sentence[i-j]]] + (1 / j)
                if i + j < len(sentence):
                    X[vocab[word], vocab[sentence[i+j]]] = X[vocab[word], vocab[sentence[i+j]]] + (1 / j)
    
    return X
            
if __name__ == "__main__":
    import torch

    X = cooccurrence_matrix(window_length=4)

    print("Co-occurrence shape:", tuple(X.shape))
    print("dtype:", X.dtype)
    print("nnz (nonzeros):", int((X != 0).sum().item()))

    # symmetry check (should be ~0 if you symmetrize)
    sym_err = (X - X.T).abs().sum().item()
    print("symmetry error (L1):", sym_err)

    # peek at a few values
    uniq = torch.unique(X)
    print("unique values (sample):", uniq[:10].tolist(), " ... total:", uniq.numel())

    has_fraction = torch.any((uniq - torch.floor(uniq)) != 0).item()
    print("has fractional weights:", bool(has_fraction))
