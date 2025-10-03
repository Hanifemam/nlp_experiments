from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import re
from collections import Counter
import numpy as np

class data_prepration():
    def __init__(self):
        self.tokens = self.tokenizer()
        
    def tokenizer(self):
        tokenizer = get_tokenizer("basic_english")
        with open("LM/dataset.txt", "r", encoding="utf-8") as f:
            raw = f.read().lower().strip().replace(',', '').replace(';', '').replace(':', '').replace('!', '.').replace('?', '.')
            sentneces = [s.strip() for s in re.split(r'[.\n\t]', raw) if s.strip]
            return [tokenizer(s) for s in sentneces]
        
    def unigrams(self):
        flat = (w for sent in self.tokens for w in sent)
        return dict(Counter(flat))
    
    def bigrams(self):
        flat = [w for sent in self.tokens for w in sent]
        pairs = [(flat[i], flat[i + 1]) for i in range(len(flat) - 1)]
        return dict(Counter(pairs))
    
    def __call__(self):
        return self.unigrams(), self.bigrams()
        

def NGram(string):
    unigrams, bigrams = data_prepration()()  
    log_probs = 0.0
    words = string.lower().split()
    total_words = sum(unigrams.values())
    
    for i, w in enumerate(words):
        if w in unigrams:
            if i == 0: 
                log_probs += np.log(unigrams[w] / total_words)
            else:   
                prev = words[i-1]
                log_probs += np.log(bigrams.get((prev, w), 1) / unigrams.get(prev, 1))
    return log_probs
        
print(np.exp(NGram("hello torch is great")))