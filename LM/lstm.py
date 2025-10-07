
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
        x = torch.tensor(self.tok_id[index:index+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tok_id[index+1:index+1+self.seq_len], dtype=torch.long)
        return x, y
        
class Model(nn.Module):
    def __init__(self, batch_size=64, embedding_dim=512, lr=1e-3, seq_len=16, epochs=1, shuffle=True, drop_last=True):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        tokens = tokenizer()
        vocab_obj = vocab()
        flattened = [tok for sent in tokens for tok in sent]
        tok_id = [vocab_obj[tok] for tok in flattened]
        
        dataset = LSTMDataset(tok_id, self.seq_len)
        self.loader = DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)
        
        self.vocab_size = len(vocab_obj)
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim)
        self.model = LSTM(input_size=embedding_dim, output_size=self.vocab_size, hidden_size=256)
        
        self.criterion = nn.CrossEntropyLoss()
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        
    def forward(self, x_ind):
        x = self.embed(x_ind)
        logit, _, _ = self.model(x)
        return logit
    
    def train_loop(self):
        self.train()
        for epoch in range(1, self.epochs + 1):
            total_loss, batches = 0.0, 0
            for x, y in tqdm(self.loader, desc=f"Epoch {epoch}/{self.epochs}",
                             unit="batch", dynamic_ncols=True, leave=False):
                x = x.to(self.device)  # (B, T)
                y = y.to(self.device)  # (B, T)

                logits = self.forward(x)             # (B, T, V)
                B, T, V = logits.shape
                loss = self.criterion(
                    logits.reshape(B * T, V),
                    y.reshape(B * T)
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

            avg = total_loss / max(1, batches)
            print(f"epoch {epoch}/{self.epochs} - loss: {avg:.4f}")
        
        
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
        with torch.no_grad():
            self.b_f.fill_(1.0)
        return Y, h, c
    
    
# ===== test script =====

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(42)

    # --- hyperparams for a quick smoke test ---
    SEQ_LEN      = 8
    BATCH_SIZE   = 64
    EMBED_DIM    = 64
    EPOCHS       = 1
    LR           = 1e-3

    # instantiate your model
    model = Model(
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        embedding_dim=EMBED_DIM,
        epochs=EPOCHS,
        lr=LR,
        shuffle=True,
        drop_last=True,
    )

    print(f"device: {model.device}")
    print(f"vocab size: {model.vocab_size}")
    print(f"num batches: {len(model.loader)}")

    # --- quick forward pass sanity check on one mini-batch ---
    model.eval()
    with torch.no_grad():
        for x_ids, y_ids in model.loader:
            x_ids = x_ids.to(model.device)   # (B, T)
            y_ids = y_ids.to(model.device)   # (B, T)
            logits = model(x_ids)            # (B, T, V)
            B, T, V = logits.shape
            print(f"forward OK -> logits shape: {logits.shape} (B={B}, T={T}, V={V})")
            # show top-3 next-token predictions for the first example, first timestep
            topk = logits[0, 0].softmax(-1).topk(3)
            print(f"top-3 probs @ t=0: {topk.values.tolist()} | idx: {topk.indices.tolist()}")
            break

    # --- train for a couple of epochs ---
    print("\nstarting training…")
    model.train_loop()

    # --- optional: save a checkpoint ---
    ckpt_path = "birnn_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": model.vocab_size,
            "embed_dim": EMBED_DIM,
            "seq_len": SEQ_LEN,
        },
        ckpt_path,
    )
    print(f"saved checkpoint to {ckpt_path}")

