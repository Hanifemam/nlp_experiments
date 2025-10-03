from collections import Counter
import numpy as np

from data_prepration import tokenizer

class data_prepration():
    def __init__(self):
        self.tokens = tokenizer()
        
    
        
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