
import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from data_preparation import tokenizer, vocab

class LSTMDataset(Dataset):
    def __init__(self, tok_id, seq_len):
        super().__init__()
        self.tok_id = tok_id
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.tok_id) - self.seq_len
    
    def __getitem__(self, index):
        x = torch.tensor(self.tok_id[index: index + self.seq_len], dtype=torch.float32)          #(T,)
        y = torch.tensor(self.tok_id[index + 1: index + self.seq_len + 1], dtype=torch.float32)  #(T,)
        return x, y
        
class Model(nn.Module):
    def __init__(self, batch_size, embedding_dim=512, shuffle=True, drop_last=True):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        tokens = tokenizer()
        vocab_obj = vocab()
        flattened = [tok for sent in tokens for tok in sent]
        tok_id = [vocab_obj[tok] for tok in flattened]
        
        dataset = LSTMDataset(tok_id, self.seq_len)
        self.loader = DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)
        
        vocab_size = len(vocab_obj)
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.model = LSTM(input_size=embedding_dim, output_size=vocab_size, hidden_size=256)
        
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        
        I, H = input_size, hidden_size
        self.h_size = hidden_size
        # input -> gate
        self.U_i = nn.Parameter(torch.empty(I, H))
        self.U_f = nn.Parameter(torch.empty(I, H))
        self.U_g = nn.Parameter(torch.empty(I, H))
        self.U_o = nn.Parameter(torch.empty(I, H))

        # hidden -> gate
        self.W_i = nn.Parameter(torch.empty(H, H))
        self.W_f = nn.Parameter(torch.empty(H, H))
        self.W_g = nn.Parameter(torch.empty(H, H))
        self.W_o = nn.Parameter(torch.empty(H, H))

        # biases
        self.b_i = nn.Parameter(torch.zeros(H))
        self.b_f = nn.Parameter(torch.zeros(H))
        self.b_g = nn.Parameter(torch.zeros(H))
        self.b_o = nn.Parameter(torch.zeros(H))

        # optional output head
        if output_size is not None:
            self.W_hy = nn.Parameter(torch.empty(H, output_size))
            self.b_y  = nn.Parameter(torch.zeros(output_size))

        # init: xavier for input mats, orthogonal for recurrent
        for p in [self.U_i, self.U_f, self.U_g, self.U_o, getattr(self, "W_hy", None)]:
            if p is not None: nn.init.xavier_uniform_(p)
        for p in [self.W_i, self.W_f, self.W_g, self.W_o]:
            nn.init.orthogonal_(p)
        
    def forward(self, X):
        B, T, E = X.shape
        H = self.h_size
        h = X.new_zeros(B, H)
        c = X.new_zeros(B, H)
        
        ys = []
        hs = [] 
        for t in range(T):
            x = X[:, t, :]
            i_t = torch.sigmoid(x @ self.U_i + h @ self.W_i + self.b_i)  # (B, H)
            f_t = torch.sigmoid(self.b_f + x @ self.U_f  + h @ self.W_f)
            g_t = torch.tanh(self.b_g + x @ self.U_g + h @ self.W_g)
            c = f_t * c + i_t * g_t                 # (B, H)
            o_t = torch.sigmoid(self.b_o + x @ self.U_o + h @ self.W_o)
            h = o_t * torch.tanh(c)   
            
            if hasattr(self, "W_hy"):
                y_t = h @ self.W_hy + self.b_y      # (B, O)
                ys.append(y_t)
            else:
                hs.append(h)

        if hasattr(self, "W_hy"):
            Y = torch.stack(ys, dim=1)              # (B, T, O)
        else:
            Y = torch.stack(hs, dim=1)              # (B, T, H)

        return Y, h, c