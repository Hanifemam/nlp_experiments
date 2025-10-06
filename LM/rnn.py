import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

from data_preparation import tokenizer, vocab  # vocab() returns a torchtext Vocab with default_index set


class LanguageModelDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.data = tokens            # 1D list[int] token IDs
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.seq_len], dtype=torch.long)           # (T,)
        y = torch.tensor(self.data[idx + 1 : idx + 1 + self.seq_len], dtype=torch.long)   # (T,)
        return x, y


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size   # embedding dim E
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

    def forward(self, X):
        """
        X: (B, T, E) float
        returns: (B, T, V) logits, (B, H) final hidden
        """
        B, T, E = X.shape
        H = self.hidden_size
        h = X.new_zeros(B, H)  # device/dtype from X

        outs = []
        for t in range(T):
            x_t = X[:, t, :]                               # (B, E)
            a = x_t @ self.W_xh + h @ self.W_hh + self.b_h # (B, H)
            h = torch.tanh(a)                              # (B, H)
            y_t = h @ self.W_hy + self.b_y                 # (B, V)
            outs.append(y_t)

        Y = torch.stack(outs, dim=1)                       # (B, T, V)
        return Y, h


class Model(nn.Module):
    def __init__(self, seq_len=4, batch_size=2, epochs=10,
                 shuffle=True, drop_last=True, embedding_dim=256,
                 hidden_size=128, lr=1e-3):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- data ---
        tokenized_sents = tokenizer()            # list[list[str]]
        vocab_obj = vocab()                      # torchtext Vocab with default index set
        flat_tokens = [tok for sent in tokenized_sents for tok in sent]
        token_ids = [vocab_obj[tok] for tok in flat_tokens]

        self.dataset = LanguageModelDataset(token_ids, seq_len)
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )

        # --- model components ---
        self.vocab = vocab_obj
        self.vocab_size = len(vocab_obj)

        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim)
        self.rnn   = RNN(input_size=embedding_dim, hidden_size=hidden_size, output_size=self.vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # optimize ALL params (embed + rnn)

        self.epochs = epochs
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.to(self.device)

    def forward(self, x_ids):
        """
        x_ids: (B, T) Long
        returns logits: (B, T, V)
        """
        X = self.embed(x_ids)       # (B, T, E)
        logits, _ = self.rnn(X)     # (B, T, V)
        return logits

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
            
if __name__ == "__main__":
    torch.manual_seed(42)

    # Create and train the model
    model = Model(
        seq_len=8,        # context window size
        batch_size=64,     # tiny batch for testing
        epochs=1,         # just a couple of epochs for a smoke test
        embedding_dim=64, # smaller embedding to speed up test
        hidden_size=64,   # smaller hidden for speed
        lr=1e-3
    )

    print(f"Device: {model.device}")
    print(f"Vocab size: {len(model.vocab)} tokens")

    model.train_loop()   # runs the training loop and prints epoch losses

