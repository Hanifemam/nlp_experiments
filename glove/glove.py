import torch
from torch import nn
import data_loader
from tqdm.auto import tqdm

class Model():
    def __init__(self, embedding_size=128, epochs=10, lr=1e-3, x_max=100, alpha=3/4):
        self.loader = data_loader.loader
        self.x_max = float(x_max)
        self.alpha = float(alpha)
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.lr = lr
        
        max_id = -1
        for words, _ in self.loader:
            max_id = max(max_id, int(words.max().item()))
        vocab_size = max_id + 1
        
        
        self.model = GloVe(vocab_size, self.embedding_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def __call__(self):
        self.model.train()
        for i in range(self.epochs):
            total_loss, batches = 0.0, 0
            for words, x_ij in tqdm(self.loader, desc=f"Epoch {i+1}/{self.epochs}", unit="batch", leave=False):
                words = words.to(self.device)
                words = words.to(self.device, non_blocking=True).long()
                x_ij = x_ij.to(self.device, non_blocking=True).float()
                logit = self.model(words)
                loss = self.loss(x_ij, logit)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                batches += 1

            avg = total_loss / max(1, batches)
            print(f"epoch {i+1}/{self.epochs} - loss: {avg:.4f}")

    
    def loss(self, x_ij, output):
        f_n = torch.where(x_ij < self.x_max,
                          (x_ij / self.x_max).pow(self.alpha),
                          torch.ones_like(x_ij))
        return (f_n * (output - x_ij.clamp_min(1e-12).log()).pow(2)).mean()
        
class GloVe(nn.Module):
    def __init__(self, input_size,  embedding_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        dev = 1.0 / (self.embedding_size ** 0.5)
        self.W_in = nn.Parameter(torch.empty([self.input_size, self.embedding_size], dtype=torch.float32).uniform_(-dev, dev))
        self.W_out = nn.Parameter(torch.empty([self.input_size, self.embedding_size], dtype=torch.float32).uniform_(-dev, dev))
        self.bias_in = nn.Parameter(torch.zeros(self.input_size))
        self.bias_out = nn.Parameter(torch.zeros(self.input_size))
        
    def forward(self, words:torch.tensor):
        if words.dim() == 1 and words.numel() == 2:
            words = words.unsqueeze(0)  
        i_idx = words[:, 0]                
        j_idx = words[:, 1]                 

        wi = self.W_in[i_idx]               
        wj = self.W_out[j_idx]               
        bi = self.bias_in[i_idx]             
        bj = self.bias_out[j_idx]           

        s = (wi * wj).sum(dim=-1) + bi + bj  
        return s
    
if __name__ == "__main__":
    def main():
        import torch
        torch.manual_seed(0)

        # build model (use a short run for smoke test)
        m = Model(embedding_size=64, epochs=1, lr=1e-3, x_max=100, alpha=0.75)

        # --- sanity check first batch ---
        words, x_ij = next(iter(m.loader))   # expects words: [B,2], x_ij: [B]
        print("First batch — words:", tuple(words.shape), words.dtype,
              "| x_ij:", tuple(x_ij.shape), x_ij.dtype)
        assert words.dim() == 2 and words.size(1) == 2, "words must be [B, 2] (i_idx, j_idx)"

        # one manual step to confirm forward/backward works
        words = words.to(m.device)
        x_ij  = x_ij.to(m.device).float()
        s = m.model(words)                   # [B]
        loss = m.loss(x_ij, s)               # scalar
        print("Initial loss:", float(loss.item()))
        m.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        m.optimizer.step()

        # --- run your training loop (1 epoch here) ---
        m()

        # inspect final embeddings (W_in + W_out is common)
        E = (m.model.W_in + m.model.W_out).detach().cpu()
        print("Embeddings shape:", tuple(E.shape))  # (V, D)

    main()
