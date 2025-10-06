import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from data_preparation import tokenizer, vocab

class Model():
    def __init__(self):
        tokens = tokenizer()
        vocab_obj = vocab()
        flattened = [tok for sent in tokens for tok in sent]
        tok_id = [vocab_obj[tok] for tok in flattened]
        
class BiRNN(Dataset):
    def __init__(self, tokens, seq_len):
        super().__init__()
        self.dataset = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x = torch.tensor(self.dataset[index : index + self.seq_len], dtype=torch.long) # (T, )
        y = torch.tensor(self.dataset[index + 1 : index + self.seq_len + 1], dtype=torch.long) #(T, )
        return x, y