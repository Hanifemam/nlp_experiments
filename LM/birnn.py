import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from data_preparation import tokenizer, vocab

class Model(nn.Module):
    def __init__(self, seq_len, batch_size=16, embedding_dim=512, epochs=16, shuffle=True, drop_last=True, lr=1e-3):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        tokens = tokenizer()
        vocab_obj = vocab()
        flattened = [tok for sent in tokens for tok in sent]
        tok_id = [vocab_obj[tok] for tok in flattened]
        
        self.seq_len = seq_len
        self.vocab = vocab_obj
        self.vocab_size = len(vocab_obj)
        
        dataset = BiRNNDataSet(tok_id, self.seq_len)
        self.loader = DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)
        
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim)
        self.birnn = BiRNN(input_size=embedding_dim, h_b_size=256, h_f_size=256, output_size=self.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        
        self.epochs = epochs
        self.batch_size = batch_size
        
    def forward(self, x_ids):
        X = self.embed(x_ids)       # (B, T, E)
        logits, _ = self.birnn(X)     # (B, T, V)
        return logits
    
    def train_loop(self):
        self.train()
        

        
        
        
class BiRNNDataSet(Dataset):
    def __init__(self, tokens, seq_len):
        super().__init__()
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, index):
        x = torch.tensor(self.tokens[index : index + self.seq_len], dtype=torch.long) # (T, )
        y = torch.tensor(self.tokens[index + 1 : index + self.seq_len + 1], dtype=torch.long) #(T, )
        return x, y
    
class BiRNN(nn.Module):
    def __init__(self, input_size, output_size, h_b_size=20, h_f_size=20):
        super().__init__()
        self.h_b_size = h_b_size
        self.h_f_size = h_f_size
        
        
        self.W_hh_b = nn.Parameter(torch.empty([self.h_b_size, self.h_b_size]))
        self.W_hh_f = nn.Parameter(torch.empty([self.h_f_size, self.h_f_size]))
        self.W_xh_b = nn.Parameter(torch.empty([input_size, self.h_b_size]))
        self.W_xh_f = nn.Parameter(torch.empty([input_size, self.h_f_size]))
        self.b_b = nn.Parameter(torch.zeros(self.h_b_size))
        self.b_f = nn.Parameter(torch.zeros(self.h_f_size))
        
        self.W_hy = nn.Parameter(torch.empty([self.h_b_size + self.h_f_size, output_size]))
        self.b_y = nn.Parameter(torch.zeros(output_size))
        
        init.xavier_uniform_(self.W_xh_b)
        init.xavier_uniform_(self.W_xh_f)
        init.xavier_uniform_(self.W_hy)
        init.orthogonal_(self.W_hh_b)
        init.orthogonal_(self.W_hh_f)
        
    def forward(self, X):
        B, T, E = X.shape
        h_b = X.new_zeros(B, self.h_b_size)
        h_f = X.new_zeros(B, self.h_f_size)
        assert E == self.W_xh_f.size(0) == self.W_xh_b.size(0)
        assert self.W_hy.size(0) == self.h_f_size + self.h_b_size
        
        outs = []
        h_b_list = []
        h_f_list = []
        for t in range(T):
            x_t_f = X[:, t, :]   
            h_f_t = torch.tanh(x_t_f @ self.W_xh_f + h_f @ self.W_hh_f + self.b_f) # (B, H)
            h_f = h_f_t
            h_f_list.append(h_f_t)
            
            x_t_b = X[:, T - t - 1,:]
            h_b_t = torch.tanh(x_t_b @ self.W_xh_b + h_b @ self.W_hh_b + self.b_b) # (B, H)
            h_b = h_b_t
            h_b_list.append(h_b_t)
        h_b_list = list(reversed(h_b_list))
        
        
            
        for h_f_next, h_b_pre in zip(h_f_list, h_b_list):
            h = torch.cat((h_f_next, h_b_pre), dim=1)
            y_t = h @ self.W_hy + self.b_y                 # (B, V)
            outs.append(y_t)
        
        return torch.stack(outs, dim=1), h_f, h_b
            
        
        
        
        
        
