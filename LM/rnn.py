import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from collections import Counter

from data_preparation import tokenizer, vocab

class Model(nn.Module):
    def __init__(self):
        flat_tokens = [tok for sent in tokenizer() for tok in sent]
        vocab_stoi = vocab().get_stoi
        token_ids = [vocab_stoi[token] for token in flat_tokens]
        dataset_class = LanguageModelDataset(token_ids, 4)
        dataset = dataset_class()
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    def __call__(self):
        pass
    
class LanguageModelDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.data = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx: idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1: idx + 1 + self.seq_len], dtype=torch.long)
        return x, y
    
        
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_xh = nn.Parameter(torch.empty(input_size, hidden_size))
        self.b_h  = nn.Parameter(torch.zeros(hidden_size))

        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.W_hy = nn.Parameter(torch.empty(hidden_size, output_size))
        self.b_y  = nn.Parameter(torch.zeros(output_size))

        init.xavier_uniform_(self.W_xh)    
        init.xavier_uniform_(self.W_hy)    

        init.orthogonal_(self.W_hh)
        
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, device=self.device)
    
    def forward(self, X):
        B, T, I = X.shape
        h = self.init_hidden(batch_size=B, device=self.device) # (B, H)
        outs = []
        
        for t in range(T):
            x_t = X[:, t, :] # [B,I]
            a = x_t @ self.W_xh + h @ self.W_hh + self.b_h
            h = torch.tanh(a)                # (B, H)
            y_t = h @ self.W_hy + self.b_y   # (B, O)
            outs.append(y_t)
        Y = torch.stack(outs, dim=1)         # (B, T, O)
        return Y, h
        

        