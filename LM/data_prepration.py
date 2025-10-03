from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import re

def tokenizer():
        tokenizer = get_tokenizer("basic_english")
        with open("LM/dataset.txt", "r", encoding="utf-8") as f:
            raw = f.read().lower().strip().replace(',', '').replace(';', '').replace(':', '').replace('!', '.').replace('?', '.')
            sentneces = [s.strip() for s in re.split(r'[.\n\t]', raw) if s.strip]
            return [tokenizer(s) for s in sentneces]
        
def vocab():
    tokenized = tokenizer()
    specials = ["<unknown>"]
    min_freq = 4
    vocab = build_vocab_from_iterator(tokenized, min_freq=min_freq, specials=specials, special_first=True)
    vocab.set_default_index(vocab["<unknown>"])
    
    return vocab.get_stoi(), vocab.get_itos

print(tokenizer())
print(vocab())