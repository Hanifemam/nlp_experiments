from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
import torch

def tokenizer():
    tokenizer = get_tokenizer("basic_english")
    with open("word2Vec/dataset.txt", "r", encoding="utf-8") as f:
        raw = f.read().split('\n')
        
    sentence = [s for s in raw.split('.') if s.strip()]
    tokenized = [tokenizer(s) for s in sentence]
    
    
    
    return tokenized

def dictionary():
    tokenized = tokenizer()
    specials = ["<unknown>"]
    min_freq = 4
    vocab = build_vocab_from_iterator(tokenized, min_freq=min_freq, specials=specials, special_first=True)
    vocab.set_default_index(vocab["<unknown>"])
    
    id_vocab = [vocab[tokens] for tokens in tokenized]
    
    return vocab, id_vocab

def negative_samples():
    tokenized = tokenizer()
    vocab, _ = dictionary()
    
    freq = Counter([t for s in tokenized for t in s])
    
    V = len(vocab)
    counts = torch.zeros(V, dtype=torch.float)
    stoi = vocab.get_stoi()
    for tok, c in freq.items():
        if tok in stoi:
            counts[stoi[tok]] = float(c)
    unk_id = vocab["<unknown>"]
    counts[unk_id] = 0.0

    probs = counts.pow(0.75)
    probs = probs / probs.sum().clamp(min=1e-12)   # safe normalize
