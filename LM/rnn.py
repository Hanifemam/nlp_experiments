import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm

from data_prepration import tokenizer

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
        

        